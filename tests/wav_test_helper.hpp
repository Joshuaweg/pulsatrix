/** @file wav_test_helper.hpp
 *  @brief Test-only minimal 16-bit PCM WAV writer -- generates audio fixtures at test run
 *         time instead of checking in binary blobs, mirroring bmp_test_helper.hpp
 *         (campaign_exai_dl_library_data_pipeline, Phase 4).
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace pulsatrix::test {

/**
 * @brief Writes a minimal 16-bit PCM WAV file.
 * @param interleaved_samples Samples interleaved per frame (e.g. for stereo: L0,R0,L1,R1,...).
 */
inline void WriteWav(const std::string& path, int sample_rate, int channels,
                      const std::vector<int16_t>& interleaved_samples) {
    constexpr int kBitsPerSample = 16;
    uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * kBitsPerSample / 8);
    uint16_t block_align = static_cast<uint16_t>(channels * kBitsPerSample / 8);
    uint32_t data_size = static_cast<uint32_t>(interleaved_samples.size() * sizeof(int16_t));
    uint32_t chunk_size = 36 + data_size;

    std::vector<unsigned char> buf(44 + static_cast<size_t>(data_size), 0);

    auto write_bytes = [&](size_t offset, const char* bytes, size_t len) {
        std::memcpy(&buf[offset], bytes, len);
    };
    auto write_u32 = [&](size_t offset, uint32_t value) {
        buf[offset] = static_cast<unsigned char>(value & 0xFF);
        buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
        buf[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFF);
        buf[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFF);
    };
    auto write_u16 = [&](size_t offset, uint16_t value) {
        buf[offset] = static_cast<unsigned char>(value & 0xFF);
        buf[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    };

    write_bytes(0, "RIFF", 4);
    write_u32(4, chunk_size);
    write_bytes(8, "WAVE", 4);
    write_bytes(12, "fmt ", 4);
    write_u32(16, 16);  // fmt subchunk size (PCM)
    write_u16(20, 1);   // audio format: PCM
    write_u16(22, static_cast<uint16_t>(channels));
    write_u32(24, static_cast<uint32_t>(sample_rate));
    write_u32(28, byte_rate);
    write_u16(32, block_align);
    write_u16(34, kBitsPerSample);
    write_bytes(36, "data", 4);
    write_u32(40, data_size);

    for (size_t i = 0; i < interleaved_samples.size(); ++i) {
        uint16_t s = static_cast<uint16_t>(interleaved_samples[i]);
        buf[44 + i * 2] = static_cast<unsigned char>(s & 0xFF);
        buf[44 + i * 2 + 1] = static_cast<unsigned char>((s >> 8) & 0xFF);
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

}  // namespace pulsatrix::test
