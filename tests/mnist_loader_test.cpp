#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "exai/cpu_backend.hpp"
#include "exai/mnist_loader.hpp"

// MnistIdxLoader parses real MNIST IDX files (tools/fetch_mnist.py, data/MNIST/raw/,
// gitignored -- not part of this repo's tracked source). Every test here GTEST_SKIP()s
// when that directory is absent, the same environment-conditional pattern Phase 1.5's
// CUDA-hardware-gated tests already established -- skip, don't fail, when a real external
// dependency isn't present on this machine.
namespace exai {
namespace {

const std::string kTrainImages = std::string(EXAI_TEST_DATA_DIR) + "/MNIST/raw/train-images-idx3-ubyte";
const std::string kTrainLabels = std::string(EXAI_TEST_DATA_DIR) + "/MNIST/raw/train-labels-idx1-ubyte";
const std::string kTestImages = std::string(EXAI_TEST_DATA_DIR) + "/MNIST/raw/t10k-images-idx3-ubyte";
const std::string kTestLabels = std::string(EXAI_TEST_DATA_DIR) + "/MNIST/raw/t10k-labels-idx1-ubyte";

bool RealMnistDataPresent() {
    std::ifstream f(kTrainImages, std::ios::binary);
    return f.good();
}

class MnistIdxLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!RealMnistDataPresent()) {
            GTEST_SKIP() << "data/MNIST/raw/ not present -- run tools/fetch_mnist.py first";
        }
    }

    CPUBackend backend;
};

// Reference values cross-checked directly from the raw IDX bytes via a standalone Python
// script (struct.unpack, no torchvision decoding involved) -- these are also MNIST's
// well-known, publicly documented values (train[0] is the canonical "5", test[0] is "7"),
// an independent sanity confirmation beyond just "the two readers agree."
TEST_F(MnistIdxLoaderTest, LoadsTrainingSetWithKnownReferenceValues) {
    MnistDataset dataset = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend);

    ASSERT_EQ(dataset.images.size(), 60000u);
    ASSERT_EQ(dataset.labels.size(), 60000u);

    EXPECT_EQ(dataset.labels[0], 5);
    EXPECT_EQ(dataset.images[0].shape(), Shape({1, 1, 28, 28}));
    EXPECT_FLOAT_EQ(dataset.images[0].data()[0 * 28 + 0], 0.0f);          // pixel(0,0) == 0
    EXPECT_FLOAT_EQ(dataset.images[0].data()[14 * 28 + 14], 240.0f / 255.0f);  // pixel(14,14) == 240
}

TEST_F(MnistIdxLoaderTest, LoadsTestSetWithKnownReferenceValues) {
    MnistDataset dataset = MnistIdxLoader::Load(kTestImages, kTestLabels, &backend);

    ASSERT_EQ(dataset.images.size(), 10000u);
    ASSERT_EQ(dataset.labels.size(), 10000u);

    EXPECT_EQ(dataset.labels[0], 7);
    EXPECT_FLOAT_EQ(dataset.images[0].data()[14 * 28 + 14], 0.0f);  // pixel(14,14) == 0
}

TEST_F(MnistIdxLoaderTest, MaxCountTruncatesToRequestedSize) {
    MnistDataset dataset = MnistIdxLoader::Load(kTrainImages, kTrainLabels, &backend, /*max_count=*/17);

    EXPECT_EQ(dataset.images.size(), 17u);
    EXPECT_EQ(dataset.labels.size(), 17u);
    EXPECT_EQ(dataset.labels[0], 5);  // first 17 must still match the untruncated read
}

TEST_F(MnistIdxLoaderTest, ThrowsOnMissingFile) {
    EXPECT_THROW({ (void)MnistIdxLoader::Load("data/MNIST/raw/does-not-exist", kTrainLabels, &backend); },
                 std::runtime_error);
}

}  // namespace
}  // namespace exai
