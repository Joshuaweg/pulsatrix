#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/wav_reader.hpp"
#include "wav_test_helper.hpp"

namespace pulsatrix {
namespace {

using ::pulsatrix::test::WriteWav;

class WavReaderTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(WavReaderTest, DecodesMonoSamplesCorrectly) {
    std::string path = ::testing::TempDir() + "pulsatrix_wav_mono_test.wav";
    WriteWav(path, /*sample_rate=*/8000, /*channels=*/1, {0, 16384, -16384, 32767});

    WavData data = WavReader::Load(path, &backend);

    EXPECT_EQ(data.sample_rate, 8000);
    EXPECT_EQ(data.waveform.shape(), Shape({1, 1, 4}));
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 1}), 16384.0f / 32768.0f);
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 2}), -16384.0f / 32768.0f);
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 3}), 32767.0f / 32768.0f);
}

TEST_F(WavReaderTest, DeinterleavesStereoChannelsCorrectly) {
    std::string path = ::testing::TempDir() + "pulsatrix_wav_stereo_test.wav";
    // Frame 0: L=100, R=200. Frame 1: L=300, R=400.
    WriteWav(path, /*sample_rate=*/16000, /*channels=*/2, {100, 200, 300, 400});

    WavData data = WavReader::Load(path, &backend);

    EXPECT_EQ(data.waveform.shape(), Shape({1, 2, 2}));
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 0}), 100.0f / 32768.0f);  // left, frame 0
    EXPECT_FLOAT_EQ(data.waveform.at({0, 1, 0}), 200.0f / 32768.0f);  // right, frame 0
    EXPECT_FLOAT_EQ(data.waveform.at({0, 0, 1}), 300.0f / 32768.0f);  // left, frame 1
    EXPECT_FLOAT_EQ(data.waveform.at({0, 1, 1}), 400.0f / 32768.0f);  // right, frame 1
}

TEST_F(WavReaderTest, ThrowsOnMissingFile) {
    EXPECT_THROW(WavReader::Load(::testing::TempDir() + "does_not_exist.wav", &backend), std::runtime_error);
}

TEST_F(WavReaderTest, ThrowsOnNonRiffFile) {
    std::string path = ::testing::TempDir() + "pulsatrix_wav_not_riff_test.wav";
    std::ofstream out(path, std::ios::binary);
    out << "this is not a wav file at all, not even close to 44 bytes of one";
    out.close();
    EXPECT_THROW(WavReader::Load(path, &backend), std::runtime_error);
}

TEST_F(WavReaderTest, ThrowsOnUnsupportedBitDepth) {
    std::string path = ::testing::TempDir() + "pulsatrix_wav_8bit_test.wav";
    WriteWav(path, 8000, 1, {0, 1});
    // Patch the bits-per-sample field (offset 34) from 16 to 8 after writing a valid 16-bit file.
    {
        std::fstream patch(path, std::ios::binary | std::ios::in | std::ios::out);
        patch.seekp(34);
        char bits8 = 8;
        char zero = 0;
        patch.write(&bits8, 1);
        patch.write(&zero, 1);
    }
    EXPECT_THROW(WavReader::Load(path, &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
