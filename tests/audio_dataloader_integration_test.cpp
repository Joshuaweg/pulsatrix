#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

#include "pulsatrix/audio_collate.hpp"
#include "pulsatrix/audio_folder_dataset.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "wav_test_helper.hpp"

namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using ::pulsatrix::test::WriteWav;

class AudioDataLoaderIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::path(::testing::TempDir()) / "pulsatrix_audio_dataloader_test";
        fs::remove_all(root_);
        fs::create_directories(root_ / "bark");
        fs::create_directories(root_ / "meow");
        WriteWav((root_ / "bark" / "a.wav").string(), 8000, 1, {1, 2, 3});
        WriteWav((root_ / "meow" / "a.wav").string(), 8000, 1, {4});
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
    CPUBackend backend;
};

TEST_F(AudioDataLoaderIntegrationTest, ProducesCorrectlyPaddedBatchesAcrossVaryingLengthClips) {
    auto dataset = std::make_shared<AudioFolderDataset>(root_.string(), &backend);
    DataLoaderOptions options;
    options.batch_size = 2;  // whole dataset (lengths 3 and 1) in one batch
    options.collate_fn = AudioPadCollate();
    DataLoader loader(dataset, &backend, options);

    auto batch = loader.next_batch();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->size(), 2);
    EXPECT_EQ(batch->fields[0].shape(), Shape({2, 1, 3}));
    ASSERT_EQ(batch->fields[1].shape(), Shape({2}));

    // "bark" sorts before "meow", so bark's 3-sample clip is row 0, meow's 1-sample clip is row 1.
    EXPECT_FLOAT_EQ(batch->fields[1].data()[0], 3.0f);
    EXPECT_FLOAT_EQ(batch->fields[1].data()[1], 1.0f);
    EXPECT_FLOAT_EQ(batch->fields[0].at({1, 0, 1}), 0.0f);  // padding
    EXPECT_FLOAT_EQ(batch->fields[0].at({1, 0, 2}), 0.0f);  // padding

    EXPECT_FALSE(loader.next_batch().has_value());
}

}  // namespace
}  // namespace pulsatrix
