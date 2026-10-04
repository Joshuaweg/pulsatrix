#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "bmp_test_helper.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_folder_dataset.hpp"

namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using ::pulsatrix::test::WriteBmp;

// Solid-color 2x2 BMP -- content doesn't matter for these tests, only that it decodes.
void WriteSolidColorImage(const std::string& path, unsigned char r, unsigned char g, unsigned char b) {
    std::vector<unsigned char> rgb;
    for (int i = 0; i < 4; ++i) {
        rgb.push_back(r);
        rgb.push_back(g);
        rgb.push_back(b);
    }
    WriteBmp(path, /*width=*/2, /*height=*/2, rgb);
}

class ImageFolderDatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // One directory per test: ctest runs tests in parallel processes, and a shared one races.
        root_ = fs::path(::testing::TempDir()) /
                ("pulsatrix_image_folder_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        fs::remove_all(root_);
        fs::create_directories(root_ / "cat");
        fs::create_directories(root_ / "dog");
        WriteSolidColorImage((root_ / "cat" / "a.bmp").string(), 255, 0, 0);
        WriteSolidColorImage((root_ / "cat" / "b.bmp").string(), 254, 0, 0);
        WriteSolidColorImage((root_ / "dog" / "a.bmp").string(), 0, 255, 0);
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
    CPUBackend backend;
};

TEST_F(ImageFolderDatasetTest, DiscoversClassesInSortedOrder) {
    ImageFolderDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.classes(), (std::vector<std::string>{"cat", "dog"}));
}

TEST_F(ImageFolderDatasetTest, SizeCountsAllFilesAcrossClasses) {
    ImageFolderDataset dataset(root_.string(), &backend);
    EXPECT_EQ(dataset.size(), 3);
}

TEST_F(ImageFolderDatasetTest, GetAssignsLabelMatchingSortedClassIndex) {
    ImageFolderDataset dataset(root_.string(), &backend);

    // "cat" (index 0) files come first (sorted class dirs, then sorted filenames within).
    Sample cat_a = dataset.get(0);
    EXPECT_FLOAT_EQ(cat_a.fields[1].data()[0], 0.0f);
    Sample cat_b = dataset.get(1);
    EXPECT_FLOAT_EQ(cat_b.fields[1].data()[0], 0.0f);

    // "dog" (index 1).
    Sample dog_a = dataset.get(2);
    EXPECT_FLOAT_EQ(dog_a.fields[1].data()[0], 1.0f);
    EXPECT_EQ(dog_a.fields[0].shape(), Shape({1, 3, 2, 2}));
}

TEST_F(ImageFolderDatasetTest, GetThrowsOnOutOfRangeIndex) {
    ImageFolderDataset dataset(root_.string(), &backend);
    EXPECT_THROW(dataset.get(3), std::out_of_range);
    EXPECT_THROW(dataset.get(-1), std::out_of_range);
}

TEST_F(ImageFolderDatasetTest, ConstructorThrowsOnMissingRootDirectory) {
    EXPECT_THROW(ImageFolderDataset((root_ / "does_not_exist").string(), &backend), std::runtime_error);
}

TEST_F(ImageFolderDatasetTest, ConstructorThrowsWhenNoClassSubdirectoriesExist) {
    fs::path empty_root = fs::path(::testing::TempDir()) / "pulsatrix_image_folder_empty_test";
    fs::remove_all(empty_root);
    fs::create_directories(empty_root);
    EXPECT_THROW(ImageFolderDataset(empty_root.string(), &backend), std::runtime_error);
    fs::remove_all(empty_root);
}

}  // namespace
}  // namespace pulsatrix
