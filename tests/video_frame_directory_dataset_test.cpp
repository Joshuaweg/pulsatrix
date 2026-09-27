#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "bmp_test_helper.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/video_frame_directory_dataset.hpp"

namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using ::pulsatrix::test::WriteBmp;

// Solid-color 2x2 BMP frame -- color identifies which frame it is.
void WriteFrame(const std::string& path, unsigned char value) {
    std::vector<unsigned char> rgb;
    for (int i = 0; i < 4; ++i) {
        rgb.push_back(value);
        rgb.push_back(value);
        rgb.push_back(value);
    }
    WriteBmp(path, /*width=*/2, /*height=*/2, rgb);
}

class VideoFrameDirectoryDatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::path(::testing::TempDir()) / "pulsatrix_video_frame_dir_test";
        fs::remove_all(root_);
        // cat/clip1 has 3 frames; dog/clip1 has 2 frames.
        fs::create_directories(root_ / "cat" / "clip1");
        fs::create_directories(root_ / "dog" / "clip1");
        WriteFrame((root_ / "cat" / "clip1" / "frame0.bmp").string(), 10);
        WriteFrame((root_ / "cat" / "clip1" / "frame1.bmp").string(), 20);
        WriteFrame((root_ / "cat" / "clip1" / "frame2.bmp").string(), 30);
        WriteFrame((root_ / "dog" / "clip1" / "frame0.bmp").string(), 40);
        WriteFrame((root_ / "dog" / "clip1" / "frame1.bmp").string(), 50);
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
    CPUBackend backend;
};

TEST_F(VideoFrameDirectoryDatasetTest, DiscoversClassesInSortedOrder) {
    VideoFrameDirectoryDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.classes(), (std::vector<std::string>{"cat", "dog"}));
}

TEST_F(VideoFrameDirectoryDatasetTest, SizeCountsClipsAcrossClasses) {
    VideoFrameDirectoryDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.size(), 2);
}

TEST_F(VideoFrameDirectoryDatasetTest, GetStacksFramesInOrderAndAssignsLabel) {
    VideoFrameDirectoryDataset dataset(root_.string(), &backend);

    Sample cat_clip = dataset.get(0);
    EXPECT_FLOAT_EQ(cat_clip.fields[1].data()[0], 0.0f);  // "cat" = class 0
    EXPECT_EQ(cat_clip.fields[0].shape(), Shape({3, 3, 2, 2}));
    EXPECT_FLOAT_EQ(cat_clip.fields[0].at({0, 0, 0, 0}), 10.0f / 255.0f);
    EXPECT_FLOAT_EQ(cat_clip.fields[0].at({1, 0, 0, 0}), 20.0f / 255.0f);
    EXPECT_FLOAT_EQ(cat_clip.fields[0].at({2, 0, 0, 0}), 30.0f / 255.0f);

    Sample dog_clip = dataset.get(1);
    EXPECT_FLOAT_EQ(dog_clip.fields[1].data()[0], 1.0f);  // "dog" = class 1
    EXPECT_EQ(dog_clip.fields[0].shape(), Shape({2, 3, 2, 2}));
}

TEST_F(VideoFrameDirectoryDatasetTest, GetThrowsOnOutOfRangeIndex) {
    VideoFrameDirectoryDataset dataset(root_.string(), &backend);
    EXPECT_THROW(dataset.get(2), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(VideoFrameDirectoryDatasetTest, ConstructorThrowsOnMissingRootDirectory) {
    EXPECT_THROW(VideoFrameDirectoryDataset((root_ / "does_not_exist").string(), &backend), std::runtime_error);
}

TEST_F(VideoFrameDirectoryDatasetTest, ConstructorThrowsOnEmptyClipDirectory) {
    fs::create_directories(root_ / "cat" / "empty_clip");
    EXPECT_THROW(VideoFrameDirectoryDataset(root_.string(), &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
