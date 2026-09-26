/** @file wav_reader.hpp
 *  @brief Hand-rolled 16-bit PCM WAV decoder.
 *  @ingroup dl_modules
 */
#pragma once

#include <string>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One decoded WAV file's contents: waveform Tensor plus its sample rate. */
struct WavData {
    /** @brief Shape (1, channels, num_samples), float32, normalized to [-1,1]. */
    Tensor waveform;
    int sample_rate;
};

/**
 * @brief Reads uncompressed 16-bit PCM WAV files directly -- pulsatrix's first audio
 *        primitive (campaign_exai_dl_library_data_pipeline, Phase 4, Decision Point 4:
 *        hand-rolled over libsndfile, same "narrowest thing that satisfies the exit gate"
 *        reasoning as Decision Points 1/2/6).
 * @note Assumes the canonical fixed-order RIFF/WAVE/fmt/data chunk layout (no support for
 *       extra chunks like LIST/INFO before the data chunk) -- a real, documented
 *       limitation, matching MnistIdxLoader's own fixed-header-layout precedent rather
 *       than a general chunk scanner.
 */
class WavReader {
public:
    /**
     * @param path Path to a WAV file.
     * @param backend Backend to allocate the output waveform Tensor through. Not owned.
     * @return The decoded waveform and its sample rate.
     * @throws std::runtime_error if the file can't be opened, isn't a valid RIFF/WAVE
     *         file, isn't PCM, isn't 16-bit, or is shorter than its own declared data
     *         size -- external boundary (file content, not an internal invariant).
     */
    [[nodiscard]] static WavData Load(const std::string& path, DeviceBackend* backend);
};

}  // namespace pulsatrix
