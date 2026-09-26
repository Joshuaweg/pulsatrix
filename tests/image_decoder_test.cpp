#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_decoder.hpp"

// campaign_exai_dl_library_data_pipeline, Phase 2 Mission 5: no binary image fixtures are
// checked into tests/fixtures/ (unlike the CSV fixtures) -- BMP's uncompressed, trivially
// hand-computable format makes generating a small known-pixel-value fixture at test time
// both simpler and more self-contained than committing a binary blob.
namespace pulsatrix {
namespace {

void WriteU16(std::vector<unsigned char>& buf, size_t offset, uint16_t value) {
    buf[offset] = static_cast<unsigned char>(value & 0xFF);
    buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

void WriteU32(std::vector<unsigned char>& buf, size_t offset, uint32_t value) {
    buf[offset] = static_cast<unsigned char>(value & 0xFF);
    buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buf[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buf[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

// Writes a minimal uncompressed 24-bit BMP. `rgb` is row-major, top-down, 3 bytes/pixel
// (R,G,B), size width*height*3.
void WriteBmp(const std::string& path, int width, int height, const std::vector<unsigned char>& rgb) {
    int row_size = ((width * 3 + 3) / 4) * 4;
    int pixel_data_size = row_size * height;
    int file_size = 54 + pixel_data_size;

    std::vector<unsigned char> buf(static_cast<size_t>(file_size), 0);
    buf[0] = 'B';
    buf[1] = 'M';
    WriteU32(buf, 2, static_cast<uint32_t>(file_size));
    WriteU32(buf, 10, 54);  // pixel data offset
    WriteU32(buf, 14, 40);  // BITMAPINFOHEADER size
    WriteU32(buf, 18, static_cast<uint32_t>(width));
    WriteU32(buf, 22, static_cast<uint32_t>(height));  // positive height = bottom-up rows
    WriteU16(buf, 26, 1);                              // planes
    WriteU16(buf, 28, 24);                              // bits per pixel
    WriteU32(buf, 30, 0);                                // no compression
    WriteU32(buf, 34, static_cast<uint32_t>(pixel_data_size));

    for (int y = 0; y < height; ++y) {
        int src_y = height - 1 - y;  // bottom-up: first row written is the source's last row
        for (int x = 0; x < width; ++x) {
            size_t src_idx = (static_cast<size_t>(src_y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
            size_t dst_idx = 54 + static_cast<size_t>(y) * static_cast<size_t>(row_size) + static_cast<size_t>(x) * 3;
            buf[dst_idx + 0] = rgb[src_idx + 2];  // B
            buf[dst_idx + 1] = rgb[src_idx + 1];  // G
            buf[dst_idx + 2] = rgb[src_idx + 0];  // R
        }
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

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
