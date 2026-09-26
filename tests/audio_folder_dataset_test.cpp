#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/audio_folder_dataset.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "wav_test_helper.hpp"

namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using ::pulsatrix::test::WriteWav;

class AudioFolderDatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::path(::testing::TempDir()) / "pulsatrix_audio_folder_test";
        fs::remove_all(root_);
        fs::create_directories(root_ / "bark");
        fs::create_directories(root_ / "meow");
        WriteWav((root_ / "bark" / "a.wav").string(), 8000, 1, {100, 200, 300});
        WriteWav((root_ / "bark" / "b.wav").string(), 8000, 1, {400, 500});
        WriteWav((root_ / "meow" / "a.wav").string(), 8000, 1, {600});
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
    CPUBackend backend;
};

TEST_F(AudioFolderDatasetTest, DiscoversClassesInSortedOrder) {
    AudioFolderDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.classes(), (std::vector<std::string>{"bark", "meow"}));
}

TEST_F(AudioFolderDatasetTest, SizeCountsAllFilesAcrossClasses) {
    AudioFolderDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.size(), 3);
}

TEST_F(AudioFolderDatasetTest, GetAssignsLabelMatchingSortedClassIndexAndDecodesWaveform) {
    AudioFolderDataset dataset(root_.string(), &backend);

    Sample bark_a = dataset.get(0);
    EXPECT_FLOAT_EQ(bark_a.fields[1].data()[0], 0.0f);  // "bark" = class 0
    EXPECT_EQ(bark_a.fields[0].shape(), Shape({1, 1, 3}));
    EXPECT_FLOAT_EQ(bark_a.fields[0].at({0, 0, 0}), 100.0f / 32768.0f);

    Sample meow_a = dataset.get(2);
    EXPECT_FLOAT_EQ(meow_a.fields[1].data()[0], 1.0f);  // "meow" = class 1
}

TEST_F(AudioFolderDatasetTest, GetThrowsOnOutOfRangeIndex) {
    AudioFolderDataset dataset(root_.string(), &backend);
    EXPECT_THROW(dataset.get(3), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(AudioFolderDatasetTest, ConstructorThrowsOnMissingRootDirectory) {
    EXPECT_THROW(AudioFolderDataset((root_ / "does_not_exist").string(), &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
