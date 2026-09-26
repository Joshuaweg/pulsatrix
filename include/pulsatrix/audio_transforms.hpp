/** @file audio_transforms.hpp
 *  @brief Sample-level audio Transforms -- linear-interpolation resampling.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/transform.hpp"

namespace pulsatrix {

/**
 * @brief Resamples a waveform (sample.fields[0], shape (1, channels, num_samples) --
 *        WavReader's/AudioFolderDataset's convention) from source_sample_rate to
 *        target_sample_rate via linear interpolation.
 * @note Waveform-only representation (campaign_exai_dl_library_data_pipeline Decision
 *       Point 4's Mission 12 sub-decision) -- a spectrogram transform is explicitly
 *       descoped, not silently dropped: a real STFT/mel-spectrogram needs an FFT
 *       implementation, a meaningfully larger undertaking than resampling.
 * @note Needs a DeviceBackend to allocate the differently-shaped output Tensor, matching
 *       ResizeTransform/CenterCropTransform's (Phase 2) precedent.
 */
class ResampleTransform : public Transform {
public:
    ResampleTransform(int target_sample_rate, int source_sample_rate, DeviceBackend* backend)
        : target_sample_rate_(target_sample_rate), source_sample_rate_(source_sample_rate), backend_(backend) {}

    [[nodiscard]] Sample apply(Sample sample) const override;

private:
    int target_sample_rate_;
    int source_sample_rate_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
