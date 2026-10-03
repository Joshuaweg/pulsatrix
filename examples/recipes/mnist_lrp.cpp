/** @file mnist_lrp.cpp
 *  @brief Recipe: train MnistConvNet (Conv2D -> ReLU -> Flatten -> Linear) on real MNIST, then
 *         explain held-out predictions with whole-model LRP (LRP::explain). Paired with
 *         docs/recipes/interpretability/mnist_lrp.md.
 *  @note Needs the real MNIST IDX files in data/MNIST/raw/ -- run `tools/fetch_mnist.py` first
 *        (same requirement as examples/mnist_training_demo.cpp).
 *  @note The explained question is contrastive: "why this class rather than the runner-up?" --
 *        LRPTarget{{pred}, {runner_up}}, seeded with the two logits, so the relevance sums to the
 *        logit margin, which is positive whenever the prediction wins. Explaining a raw logit
 *        instead is fragile here: a trained logit can be positive only because of its bias, and
 *        the epsilon rule redistributes over the pre-bias sum, so the heatmap's sign can flip.
 *        Each digit is explained twice: epsilon on every layer, and the Zennit EpsilonPlus
 *        composite (ZPlus on the conv layer, epsilon with the bias in the denominator on the
 *        classifier).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/mnist_classifier_example.hpp"
#include "pulsatrix/mnist_loader.hpp"

namespace {
constexpr int64_t kTrainSubsetSize = 10000;
constexpr int64_t kTestSubsetSize = 1000;
constexpr int kEpochs = 3;
constexpr int64_t kNumClasses = 10;
constexpr int64_t kSide = 28;

// ASCII view of a 28x28 relevance map: '+' evidence for, '-' evidence against, '.' near zero.
std::vector<std::string> AsciiHeatmap(const std::vector<float>& relevance) {
    float max_abs = 0.0f;
    for (float r : relevance) {
        max_abs = std::max(max_abs, std::abs(r));
    }
    std::vector<std::string> rows;
    for (int64_t y = 0; y < kSide; ++y) {
        std::string row;
        for (int64_t x = 0; x < kSide; ++x) {
            const float r = max_abs > 0.0f ? relevance[static_cast<size_t>(y * kSide + x)] / max_abs : 0.0f;
            row += r > 0.1f ? '+' : (r < -0.1f ? '-' : '.');
        }
        rows.push_back(row);
    }
    return rows;
}
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
                     "Run `python3 tools/fetch_mnist.py` first to fetch it to data/MNIST/raw/.\n",
                     e.what());
        return 1;
    }

    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.001f, &backend);
    NoOpMetricsSink sink;

    auto accuracy = [&]() {
        int64_t correct = 0;
        for (size_t i = 0; i < test.images.size(); ++i) {
            correct += net.predict(test.images[i]) == test.labels[i];
        }
        return 100.0f * static_cast<float>(correct) / static_cast<float>(test.images.size());
    };

    std::printf("MNIST LRP recipe -- MnistConvNet: Conv2D(1,8,5,5)->ReLU->Flatten->Linear(4608,10), Adam(lr=0.001)\n");
    std::printf("Training on %zu real images, evaluating on %zu held-out images, %d epochs\n\n", train.images.size(),
                test.images.size(), kEpochs);
    int step = 0;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < train.images.size(); ++i) {
            total_loss += net.train_step(train.images[i], train.labels[i], optimizer, sink, step++);
        }
        std::printf("epoch %d | mean train loss %.4f | test accuracy %.1f%%\n", epoch,
                    total_loss / static_cast<float>(train.images.size()), accuracy());
    }

    // The explainers run over the trained layers through an ExplainerContext, on a batched
    // (1, 1, 28, 28) image so the network's output is the rank-2 (1, 10) LRP::explain expects.
    ExplainerContext ctx(net.modules());
    const LRP epsilon;                            // epsilon rule on every layer
    const LRP epsilon_plus = LRP::epsilon_plus();  // ZPlus on Conv2D, Zennit's epsilon on Linear
    auto batched = [&](const Tensor& image) {
        return Tensor(Shape({1, 1, kSide, kSide}), &backend, image.to_host_vector());
    };

    std::printf("\nLRP on the first held-out example of each digit: why the prediction rather than the runner-up\n");
    std::printf("(target = prediction, contrast = runner-up class, seeded with the two logits)\n\n");
    std::printf("%-6s %-5s %-9s %8s %12s %16s\n", "label", "pred", "runner-up", "margin", "sum(R) eps",
                "sum(R) eps-plus");

    bool explained[kNumClasses] = {false};
    std::vector<float> epsilon_map, epsilon_plus_map;
    for (size_t i = 0; i < test.images.size(); ++i) {
        const int64_t label = test.labels[i];
        if (explained[label]) continue;
        explained[label] = true;

        const Tensor x = batched(test.images[i]);
        const std::vector<float> logits = ctx.forward_pass(x).to_host_vector();
        std::vector<int64_t> order(kNumClasses);
        for (int64_t k = 0; k < kNumClasses; ++k) order[static_cast<size_t>(k)] = k;
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
        });
        const int64_t pred = order[0];
        const int64_t runner_up = order[1];
        const float margin = logits[static_cast<size_t>(pred)] - logits[static_cast<size_t>(runner_up)];

        // LRP is linear in the seed, so a contrastive target is exactly
        // explanation(pred) - explanation(runner-up), and it sums to the logit margin.
        const LRPTarget target{{pred}, {runner_up}};
        const Attribution eps = epsilon.explain(ctx, x, target, &backend);
        const Attribution eps_plus = epsilon_plus.explain(ctx, x, target, &backend);

        std::printf("%-6lld %-5lld %-9lld %8.3f %12s %16s%s\n", static_cast<long long>(label),
                    static_cast<long long>(pred), static_cast<long long>(runner_up), margin,
                    eps.metadata.at("relevance_in_sum").c_str(), eps_plus.metadata.at("relevance_in_sum").c_str(),
                    label == pred ? "" : "   <- misclassified");
        if (label == 0) {
            epsilon_map = eps.values.to_host_vector();
            epsilon_plus_map = eps_plus.values.to_host_vector();
        }
    }

    const std::vector<std::string> left = AsciiHeatmap(epsilon_map);
    const std::vector<std::string> right = AsciiHeatmap(epsilon_plus_map);
    std::printf("\nwhy the first held-out 0 is a 0 rather than its runner-up ('+' for, '-' against):\n");
    std::printf("%-28s    %s\n", "epsilon", "EpsilonPlus");
    for (int64_t y = 0; y < kSide; ++y) {
        std::printf("%s    %s\n", left[static_cast<size_t>(y)].c_str(), right[static_cast<size_t>(y)].c_str());
    }
    std::printf("\nsum(R) tracks the margin. With epsilon, the gap is relevance held by conv units active only\n"
                "through their bias (all-zero input patch), which has no input to flow back to. EpsilonPlus also\n"
                "keeps the classifier's bias in the denominator, so the bias's share of the margin is absorbed.\n");
    return 0;
}
