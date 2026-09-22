/** @file mnist_training_demo.cpp
 *  @brief Standalone demo: trains MnistConvNet on real MNIST digit images and prints
 *         real measured test-set accuracy -- closes grad_cam_mnist_demo.cpp's own
 *         documented gap ("Training on real MNIST data is a genuinely larger future
 *         task... not built here").
 *  @note Requires real MNIST data: run `py -3.11 tools/fetch_mnist.py` first to populate
 *        data/MNIST/raw/ (gitignored, not part of this repo's tracked source).
 *  @note Not a test -- tests/mnist_classifier_example_test.cpp is the actual TDD
 *        acceptance criterion for the network's wiring, using synthetic images so it
 *        stays fast. This exists so a human can watch training happen on real data and
 *        see a real accuracy number, run manually via the `mnist_training_demo` target.
 */
#include <cstdio>
#include <stdexcept>

#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/metrics_sink.hpp"
#include "exai/mnist_classifier_example.hpp"
#include "exai/mnist_loader.hpp"

int main() {
    using namespace exai;

    CPUBackend backend;

    constexpr int64_t kTrainSubsetSize = 2000;
    constexpr int64_t kTestSubsetSize = 500;
    constexpr int kEpochs = 3;

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

    std::printf("MNIST training demo -- Conv2D(1,8,5,5)->ReLU->Flatten->Linear(4608,10), Adam(lr=0.001)\n");
    std::printf("Training on %zu real images, evaluating on %zu real held-out images, %d epochs\n\n",
                train.images.size(), test.images.size(), kEpochs);

    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.001f, &backend);
    NoOpMetricsSink sink;

    auto evaluate_accuracy = [&](const MnistDataset& dataset) {
        int64_t correct = 0;
        for (size_t i = 0; i < dataset.images.size(); ++i) {
            if (net.predict(dataset.images[i]) == dataset.labels[i]) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / static_cast<float>(dataset.images.size());
    };

    std::printf("epoch %2d | test accuracy %.1f%%\n", 0, evaluate_accuracy(test) * 100.0f);

    int step = 0;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < train.images.size(); ++i) {
            total_loss += net.train_step(train.images[i], train.labels[i], optimizer, sink, step++);
        }
        float mean_loss = total_loss / static_cast<float>(train.images.size());
        float test_accuracy = evaluate_accuracy(test);
        std::printf("epoch %2d | mean train loss %.4f | test accuracy %.1f%%\n", epoch, mean_loss,
                    test_accuracy * 100.0f);
    }

    float final_accuracy = evaluate_accuracy(test);
    std::printf("\nFinal test accuracy: %.1f%% on %zu real held-out MNIST images (chance = 10%%)\n",
                final_accuracy * 100.0f, test.images.size());

    return 0;
}
