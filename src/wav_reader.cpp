#include "pulsatrix/wav_reader.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pulsatrix {

namespace {

uint32_t ReadU32(const std::string& content, size_t offset) {
    return static_cast<uint32_t>(static_cast<unsigned char>(content[offset])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(content[offset + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(content[offset + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(content[offset + 3])) << 24);
}

uint16_t ReadU16(const std::string& content, size_t offset) {
    return static_cast<uint16_t>(static_cast<unsigned char>(content[offset]) |
                                  (static_cast<unsigned char>(content[offset + 1]) << 8));
}

}  // namespace

WavData WavReader::Load(const std::string& path, DeviceBackend* backend) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("WavReader::Load: failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    if (content.size() < 44) {
        throw std::runtime_error("WavReader::Load: file too short to be a valid WAV: " + path);
    }
    if (content.compare(0, 4, "RIFF") != 0 || content.compare(8, 4, "WAVE") != 0) {
        throw std::runtime_error("WavReader::Load: not a RIFF/WAVE file: " + path);
    }
    if (content.compare(12, 4, "fmt ") != 0) {
        throw std::runtime_error("WavReader::Load: missing fmt chunk: " + path);
    }

    uint16_t audio_format = ReadU16(content, 20);
    uint16_t channels = ReadU16(content, 22);
    uint32_t sample_rate = ReadU32(content, 24);
    uint16_t bits_per_sample = ReadU16(content, 34);

    if (audio_format != 1) {
        throw std::runtime_error("WavReader::Load: only uncompressed PCM WAV is supported: " + path);
    }
    if (bits_per_sample != 16) {
        throw std::runtime_error("WavReader::Load: only 16-bit PCM WAV is supported (got " +
                                  std::to_string(bits_per_sample) + "-bit): " + path);
    }
    if (content.compare(36, 4, "data") != 0) {
        throw std::runtime_error("WavReader::Load: missing data chunk: " + path);
    }

    uint32_t data_size = ReadU32(content, 40);
    if (content.size() < 44 + static_cast<size_t>(data_size)) {
        throw std::runtime_error("WavReader::Load: file shorter than its declared data size: " + path);
    }

    int64_t total_samples = static_cast<int64_t>(data_size) / 2;  // 2 bytes per int16 sample
    int64_t num_samples_per_channel = total_samples / channels;

    Tensor waveform(Shape({1, channels, num_samples_per_channel}), backend);
    const unsigned char* data_ptr = reinterpret_cast<const unsigned char*>(content.data()) + 44;
    for (int64_t s = 0; s < num_samples_per_channel; ++s) {
        for (int64_t c = 0; c < channels; ++c) {
            size_t byte_offset = static_cast<size_t>((s * channels + c) * 2);
            int16_t raw = static_cast<int16_t>(static_cast<uint16_t>(data_ptr[byte_offset]) |
                                                (static_cast<uint16_t>(data_ptr[byte_offset + 1]) << 8));
            waveform.at({0, c, s}) = static_cast<float>(raw) / 32768.0f;
        }
    }

    return WavData{std::move(waveform), static_cast<int>(sample_rate)};
}

}  // namespace pulsatrix
