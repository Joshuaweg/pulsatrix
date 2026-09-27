/** @file mnist_dataloader_demo.cpp
 *  @brief Standalone demo: trains MnistConvNet on real MNIST digit images, but pulled
 *         through the new Dataset/DataLoader pipeline (MnistDatasetAdapter) instead of
 *         iterating MnistDataset's raw vectors directly -- proves the data_pipeline
 *         campaign's Phase 1 plumbing against real data end-to-end
 *         (campaign_exai_dl_library_data_pipeline, Mission 4).
 *  @note Requires real MNIST data: run `py -3.11 tools/fetch_mnist.py` first to populate
 *        data/MNIST/raw/ (gitignored, not part of this repo's tracked source).
 *  @note batch_size=1 deliberately: MnistConvNet::train_step/predict are still
 *        single-image, unbatched (this campaign builds the loading/batching plumbing;
 *        retrofitting every existing Module's training loop to consume N>1 batches
 *        end-to-end is out of this campaign's scope, per its Scope section).
 */
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/mnist_classifier_example.hpp"
#include "pulsatrix/mnist_dataset_adapter.hpp"
#include "pulsatrix/mnist_loader.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    constexpr int64_t kTrainSubsetSize = 2000;
    constexpr int64_t kTestSubsetSize = 500;
    constexpr int kEpochs = 3;

    std::shared_ptr<MnistDatasetAdapter> train_adapter;
    std::shared_ptr<MnistDatasetAdapter> test_adapter;
    try {
        MnistDataset train = MnistIdxLoader::Load("data/MNIST/raw/train-images-idx3-ubyte",
                                                   "data/MNIST/raw/train-labels-idx1-ubyte", &backend,
                                                   kTrainSubsetSize);
        MnistDataset test = MnistIdxLoader::Load("data/MNIST/raw/t10k-images-idx3-ubyte",
                                                  "data/MNIST/raw/t10k-labels-idx1-ubyte", &backend, kTestSubsetSize);
        train_adapter = std::make_shared<MnistDatasetAdapter>(std::move(train), &backend);
        test_adapter = std::make_shared<MnistDatasetAdapter>(std::move(test), &backend);
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr,
                      "Could not load real MNIST data: %s\n"
                      "Run `py -3.11 tools/fetch_mnist.py` first to fetch it to data/MNIST/raw/.\n",
                      e.what());
        return 1;
    }

    std::printf("MNIST DataLoader demo -- same network as mnist_training_demo.cpp, but images/labels\n");
    std::printf("pulled through Dataset/DataLoader (MnistDatasetAdapter) instead of raw vector iteration.\n");
    std::printf("Training on %lld real images, evaluating on %lld real held-out images, %d epochs\n\n",
                static_cast<long long>(train_adapter->size()), static_cast<long long>(test_adapter->size()), kEpochs);

    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.001f, &backend);
    NoOpMetricsSink sink;

    DataLoaderOptions options;
    options.batch_size = 1;

    auto evaluate_accuracy = [&]() {
        DataLoader eval_loader(test_adapter, &backend, options);
        int64_t correct = 0;
        int64_t total = 0;
        while (auto batch = eval_loader.next_batch()) {
            int64_t predicted = net.predict(batch->fields[0]);
            int64_t target = static_cast<int64_t>(batch->fields[1].data()[0]);
            correct += (predicted == target) ? 1 : 0;
            ++total;
        }
        return static_cast<float>(correct) / static_cast<float>(total);
    };

    std::printf("epoch %2d | test accuracy %.1f%%\n", 0, evaluate_accuracy() * 100.0f);

    int step = 0;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        DataLoader train_loader(train_adapter, &backend, options);
        float total_loss = 0.0f;
        int64_t batches = 0;
        while (auto batch = train_loader.next_batch()) {
            int64_t target = static_cast<int64_t>(batch->fields[1].data()[0]);
            total_loss += net.train_step(batch->fields[0], target, optimizer, sink, step++);
            ++batches;
        }
        float mean_loss = total_loss / static_cast<float>(batches);
        float test_accuracy = evaluate_accuracy();
        std::printf("epoch %2d | mean train loss %.4f | test accuracy %.1f%%\n", epoch, mean_loss,
                    test_accuracy * 100.0f);
    }

    std::printf("\nFinal test accuracy: %.1f%% on %lld real held-out MNIST images (chance = 10%%)\n",
                evaluate_accuracy() * 100.0f, static_cast<long long>(test_adapter->size()));

    return 0;
}
