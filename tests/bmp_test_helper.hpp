/** @file bmp_test_helper.hpp
 *  @brief Test-only minimal uncompressed BMP writer -- generates image fixtures at test
 *         run time instead of checking in binary blobs (campaign_exai_dl_library_data_pipeline,
 *         Phase 2). Shared by image_decoder_test.cpp, image_transforms_test.cpp,
 *         image_folder_dataset_test.cpp.
 */
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace pulsatrix::test {

inline void WriteU16(std::vector<unsigned char>& buf, size_t offset, uint16_t value) {
    buf[offset] = static_cast<unsigned char>(value & 0xFF);
    buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
}

inline void WriteU32(std::vector<unsigned char>& buf, size_t offset, uint32_t value) {
    buf[offset] = static_cast<unsigned char>(value & 0xFF);
    buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buf[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buf[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

/**
 * @brief Writes a minimal uncompressed 24-bit BMP. `rgb` is row-major, top-down, 3
 *        bytes/pixel (R,G,B), size width*height*3.
 */
inline void WriteBmp(const std::string& path, int width, int height, const std::vector<unsigned char>& rgb) {
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

}  // namespace pulsatrix::test
