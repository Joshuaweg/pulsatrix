#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/mnist_dataset_adapter.hpp"
#include "pulsatrix/mnist_loader.hpp"

// Same GTEST_SKIP() convention as mnist_loader_test.cpp -- real MNIST data is gitignored,
// fetched via tools/fetch_mnist.py, not part of this repo's tracked source.
namespace pulsatrix {
namespace {

const std::string kTrainImages = std::string(PULSATRIX_TEST_DATA_DIR) + "/MNIST/raw/train-images-idx3-ubyte";
const std::string kTrainLabels = std::string(PULSATRIX_TEST_DATA_DIR) + "/MNIST/raw/train-labels-idx1-ubyte";

bool RealMnistDataPresent() {
    std::ifstream f(kTrainImages, std::ios::binary);
    return f.good();
}

class MnistDatasetAdapterTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!RealMnistDataPresent()) {
            GTEST_SKIP() << "data/MNIST/raw/ not present -- run tools/fetch_mnist.py first";
        }
    }

    CPUBackend backend;
};

TEST_F(MnistDatasetAdapterTest, SizeMatchesUnderlyingDatasetImageCount) {
    MnistDataset dataset = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, /*max_count=*/50);
    MnistDatasetAdapter adapter(std::move(dataset), &backend);
    EXPECT_EQ(adapter.size(), 50);
}

TEST_F(MnistDatasetAdapterTest, GetReturnsSameImageAndLabelAsDirectLoader) {
    MnistDataset reference = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, /*max_count=*/10);
    MnistDataset for_adapter = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, /*max_count=*/10);
    MnistDatasetAdapter adapter(std::move(for_adapter), &backend);

    for (int64_t i = 0; i < 10; ++i) {
        Sample sample = adapter.get(i);
        ASSERT_EQ(sample.fields.size(), 2u);
        EXPECT_EQ(sample.fields[0].shape(), reference.images[static_cast<size_t>(i)].shape());
        for (int64_t p = 0; p < sample.fields[0].numel(); ++p) {
            EXPECT_FLOAT_EQ(sample.fields[0].data()[p], reference.images[static_cast<size_t>(i)].data()[p]);
        }
        EXPECT_FLOAT_EQ(sample.fields[1].data()[0], static_cast<float>(reference.labels[static_cast<size_t>(i)]));
    }
}

TEST_F(MnistDatasetAdapterTest, GetThrowsOnOutOfRangeIndex) {
    MnistDataset dataset = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, /*max_count=*/5);
    MnistDatasetAdapter adapter(std::move(dataset), &backend);
    EXPECT_THROW(adapter.get(5), std::out_of_range);
    EXPECT_THROW(adapter.get(-1), std::out_of_range);
}

// Regression proof (Mission 4's exit gate): DataLoader + MnistDatasetAdapter must produce
// batches bit-for-bit-equivalent to MnistIdxLoader's direct per-image output -- not just
// "compiles", a real field-by-field comparison across a full pass.
TEST_F(MnistDatasetAdapterTest, DataLoaderBatchesMatchDirectLoaderOutputAcrossFullPass) {
    constexpr int64_t kCount = 20;
    MnistDataset reference = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, kCount);
    MnistDataset for_loader = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, kCount);

    auto adapter = std::make_shared<MnistDatasetAdapter>(std::move(for_loader), &backend);
    DataLoaderOptions options;
    options.batch_size = 1;  // MnistConvNet::train_step/predict are single-image, unbatched
    DataLoader loader(adapter, &backend, options);

    int64_t seen = 0;
    while (auto batch = loader.next_batch()) {
        ASSERT_EQ(batch->size(), 1);
        const Tensor& image = batch->fields[0];
        const Tensor& reference_image = reference.images[static_cast<size_t>(seen)];
        EXPECT_EQ(image.shape(), reference_image.shape());
        for (int64_t p = 0; p < image.numel(); ++p) {
            EXPECT_FLOAT_EQ(image.data()[p], reference_image.data()[p]);
        }
        EXPECT_FLOAT_EQ(batch->fields[1].data()[0], static_cast<float>(reference.labels[static_cast<size_t>(seen)]));
        ++seen;
    }
    EXPECT_EQ(seen, kCount);
}

}  // namespace
}  // namespace pulsatrix
