#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "bmp_test_helper.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_decoder.hpp"

// campaign_exai_dl_library_data_pipeline, Phase 2 Mission 5: no binary image fixtures are
// checked into tests/fixtures/ (unlike the CSV fixtures) -- BMP's uncompressed, trivially
// hand-computable format makes generating a small known-pixel-value fixture at test time
// both simpler and more self-contained than committing a binary blob. See
// bmp_test_helper.hpp (shared with image_transforms_test.cpp/image_folder_dataset_test.cpp).
namespace pulsatrix {
namespace {

using ::pulsatrix::test::WriteBmp;

class ImageDecoderTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(ImageDecoderTest, DecodesSmallKnownPixelBmpCorrectly) {
    // 3x2 image (width=3, height=2), row-major top-down, distinct R/G/B per pixel.
    std::vector<unsigned char> rgb = {
        255, 0,   0,    // (0,0) red
        0,   255, 0,    // (1,0) green
        0,   0,   255,  // (2,0) blue
        128, 128, 128,  // (0,1) gray
        255, 255, 0,    // (1,1) yellow
        0,   255, 255,  // (2,1) cyan
    };
    std::string path = ::testing::TempDir() + "pulsatrix_image_decoder_test.bmp";
    WriteBmp(path, /*width=*/3, /*height=*/2, rgb);

    Tensor image = ImageDecoder::DecodeFile(path, &backend, /*desired_channels=*/3);

    EXPECT_EQ(image.shape(), Shape({1, 3, 2, 3}));
    EXPECT_FLOAT_EQ(image.at({0, 0, 0, 0}), 255.0f / 255.0f);  // R at (0,0)
    EXPECT_FLOAT_EQ(image.at({0, 1, 0, 0}), 0.0f / 255.0f);    // G at (0,0)
    EXPECT_FLOAT_EQ(image.at({0, 2, 0, 0}), 0.0f / 255.0f);    // B at (0,0)
    EXPECT_FLOAT_EQ(image.at({0, 0, 1, 1}), 255.0f / 255.0f);  // R at (1,1) -- yellow
    EXPECT_FLOAT_EQ(image.at({0, 1, 1, 1}), 255.0f / 255.0f);  // G at (1,1)
    EXPECT_FLOAT_EQ(image.at({0, 2, 1, 1}), 0.0f / 255.0f);    // B at (1,1)
    EXPECT_FLOAT_EQ(image.at({0, 0, 1, 0}), 128.0f / 255.0f);  // R at (0,1) -- gray
}

TEST_F(ImageDecoderTest, ThrowsOnMissingFile) {
    EXPECT_THROW(ImageDecoder::DecodeFile(::testing::TempDir() + "does_not_exist.bmp", &backend), std::runtime_error);
}

TEST_F(ImageDecoderTest, ThrowsOnCorruptFile) {
    std::string path = ::testing::TempDir() + "pulsatrix_image_decoder_corrupt_test.bmp";
    std::ofstream out(path, std::ios::binary);
    out << "this is not a real image file";
    out.close();

    EXPECT_THROW(ImageDecoder::DecodeFile(path, &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
