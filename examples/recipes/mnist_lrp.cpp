/** @file mnist_lrp.cpp
 *  @brief Recipe: train a Conv2D -> ReLU -> Flatten -> Linear classifier on real MNIST, then
 *         explain held-out predictions with epsilon-rule LRP. Paired with
 *         docs/recipes/interpretability/mnist_lrp.md.
 *  @note Needs the real MNIST IDX files in data/MNIST/raw/ -- run `tools/fetch_mnist.py` first
 *        (same requirement as examples/mnist_training_demo.cpp).
 *  @note The explained quantity is a *contrast*, not the raw logit: logit(pred) minus the mean of
 *        the other nine logits, built as a one-output LinearModule over the shared features. The
 *        winning raw logit of a trained network can be negative, and seeding epsilon-LRP with a
 *        negative score flips the sign of the whole heatmap; the contrast is positive whenever
 *        the prediction wins.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/mnist_loader.hpp"
#include "pulsatrix/relu_module.hpp"

namespace {
constexpr int64_t kTrainSubsetSize = 10000;
constexpr int64_t kTestSubsetSize = 1000;
constexpr int kEpochs = 3;
constexpr int64_t kNumClasses = 10;
constexpr int64_t kFeatures = 8 * 24 * 24;  // Conv2D(1,8,5,5) on 28x28 -> 8 x 24 x 24
}  // namespace

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    MnistDataset train;
    MnistDataset test;
    try {
        train = MnistIdxLoader::Load("data/MNIST/raw/train-images-idx3-ubyte",
                                     "data/MNIST/raw/train-labels-idx1-ubyte", &backend, kTrainSubsetSize);
        test = MnistIdxLoader::Load("data/MNIST/raw/t10k-images-idx3-ubyte", "data/MNIST/raw/t10k-labels-idx1-ubyte",
                                    &backend, kTestSubsetSize);
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr,
                     "Could not load real MNIST data: %s\n"
                     "Run `py -3.11 tools/fetch_mnist.py` first to fetch it to data/MNIST/raw/.\n",
                     e.what());
        return 1;
    }

    // Layers held directly (not through MnistConvNet) so each one's propagate_relevance() is reachable.
    Conv2DModule conv(1, 8, 5, 5, &backend);
    ReluModule relu(&backend);
    FlattenModule flatten(&backend);
    LinearModule classifier(kFeatures, kNumClasses, &backend);
    CrossEntropyLoss loss(&backend);

    // Same small-uniform random init as MnistConvNet.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weight_dist(-0.1f, 0.1f);
    for (Module* m : {static_cast<Module*>(&conv), static_cast<Module*>(&classifier)}) {
        for (const auto& p : m->parameters()) {
            for (int64_t i = 0; i < p.value->numel(); ++i) {
                p.value->data()[i] = weight_dist(rng);
            }
        }
    }
    AdamOptimizer optimizer(0.001f, &backend);

    auto features = [&](const Tensor& image) { return flatten.forward(relu.forward(conv.forward(image))); };
    auto argmax = [](const Tensor& t) {
        int64_t best = 0;
        for (int64_t i = 1; i < t.numel(); ++i) {
            if (t.data()[i] > t.data()[best]) best = i;
        }
        return best;
    };
    auto accuracy = [&]() {
        int64_t correct = 0;
        for (size_t i = 0; i < test.images.size(); ++i) {
            correct += argmax(classifier.forward(features(test.images[i]))) == test.labels[i];
        }
        return 100.0f * static_cast<float>(correct) / static_cast<float>(test.images.size());
    };

    std::printf("MNIST LRP recipe -- Conv2D(1,8,5,5)->ReLU->Flatten->Linear(4608,10), Adam(lr=0.001)\n");
    std::printf("Training on %zu real images, evaluating on %zu held-out images, %d epochs\n\n", train.images.size(),
                test.images.size(), kEpochs);
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < train.images.size(); ++i) {
            optimizer.zero_grad(conv);
            optimizer.zero_grad(classifier);
            total_loss += loss.forward(classifier.forward(features(train.images[i])), train.labels[i]);
            (void)conv.backward(relu.backward(flatten.backward(classifier.backward(loss.backward()))));
            optimizer.step(conv);
            optimizer.step(classifier);
        }
        std::printf("epoch %d | mean train loss %.4f | test accuracy %.1f%%\n", epoch,
                    total_loss / static_cast<float>(train.images.size()), accuracy());
    }

    std::printf("\nepsilon-LRP on the first held-out example of each digit\n");
    std::printf("(explained score = logit(pred) - mean of the other nine logits)\n\n");
    std::printf("%-6s %-6s %10s %10s\n", "label", "pred", "score", "sum(R)");

    bool explained[kNumClasses] = {false};
    std::vector<float> first_relevance;
    for (size_t i = 0; i < test.images.size(); ++i) {
        const int64_t label = test.labels[i];
        if (explained[label]) continue;
        explained[label] = true;

        const Tensor& image = test.images[i];
        const int64_t pred = argmax(classifier.forward(features(image)));

        // Contrast head: one output whose weights are w_pred - mean(w_others). Weight layout is (in, out).
        std::vector<float> contrast_w(kFeatures);
        for (int64_t a = 0; a < kFeatures; ++a) {
            float others = 0.0f;
            for (int64_t k = 0; k < kNumClasses; ++k) {
                if (k != pred) others += classifier.weight().data()[a * kNumClasses + k];
            }
            contrast_w[a] = classifier.weight().data()[a * kNumClasses + pred] - others / (kNumClasses - 1);
        }
        float other_bias = 0.0f;
        for (int64_t k = 0; k < kNumClasses; ++k) {
            if (k != pred) other_bias += classifier.bias().data()[k];
        }
        LinearModule contrast(kFeatures, 1, &backend);
        contrast.set_weight(contrast_w);
        contrast.set_bias(std::vector<float>{classifier.bias().data()[pred] - other_bias / (kNumClasses - 1)});

        // Fresh forward so every layer's cached activations belong to this image, then LRP in reverse order.
        Tensor score = contrast.forward(features(image));
        LRPRuleConfig config;
        Tensor relevance = conv.propagate_relevance(
            relu.propagate_relevance(flatten.propagate_relevance(contrast.propagate_relevance(score, config), config),
                                     config),
            config);

        float sum_r = 0.0f;
        for (int64_t k = 0; k < relevance.numel(); ++k) sum_r += relevance.data()[k];
        std::printf("%-6lld %-6lld %10.3f %10.3f%s\n", static_cast<long long>(label), static_cast<long long>(pred),
                    score.data()[0], sum_r, label == pred ? "" : "   <- misclassified");
        if (label == 0) first_relevance.assign(relevance.data(), relevance.data() + relevance.numel());
    }

    // ASCII view of the digit-0 heatmap: '+' evidence for, '-' evidence against, '.' near zero.
    float max_abs = 0.0f;
    for (float r : first_relevance) {
        max_abs = std::max(max_abs, std::abs(r));
    }
    std::printf("\nrelevance heatmap for the first held-out 0 (28x28):\n");
    for (int64_t y = 0; y < 28; ++y) {
        for (int64_t x = 0; x < 28; ++x) {
            const float r = first_relevance[static_cast<size_t>(y * 28 + x)] / max_abs;
            std::putchar(r > 0.1f ? '+' : (r < -0.1f ? '-' : '.'));
        }
        std::putchar('\n');
    }
    std::printf("\nsum(R) tracks the explained score. The gap is relevance held by conv units that are active\n"
                "only through their bias (all-zero input patch), which has no input to flow back to.\n");
    return 0;
}
