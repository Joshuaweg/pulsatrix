#include "pulsatrix/audio_collate.hpp"

#include <algorithm>
#include <stdexcept>

namespace pulsatrix {

namespace {
Batch AudioPadCollateImpl(std::vector<Sample> samples, DeviceBackend* backend) {
    if (samples.empty()) {
        throw std::invalid_argument("AudioPadCollate: samples must not be empty");
    }

    int64_t channels = samples[0].fields[0].shape().dim(1);
    int64_t max_len = 0;
    for (const Sample& sample : samples) {
        if (sample.fields[0].shape().dim(1) != channels) {
            throw std::invalid_argument("AudioPadCollate: all samples must have the same channel count");
        }
        max_len = std::max(max_len, sample.fields[0].shape().dim(2));
    }
    int64_t n = static_cast<int64_t>(samples.size());

    Tensor padded(Shape({n, channels, max_len}), backend);
    Tensor lengths(Shape({n}), backend);
    padded.fill(0.0f);  // silence

    for (int64_t i = 0; i < n; ++i) {
        const Tensor& waveform = samples[static_cast<size_t>(i)].fields[0];
        int64_t len = waveform.shape().dim(2);
        lengths.at({i}) = static_cast<float>(len);
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t t = 0; t < len; ++t) {
                padded.at({i, c, t}) = waveform.at({0, c, t});
            }
        }
    }

    Batch batch;
    batch.fields.push_back(std::move(padded));
    batch.fields.push_back(std::move(lengths));
    return batch;
}
}  // namespace

CollateFn AudioPadCollate() {
    return [](std::vector<Sample> samples, DeviceBackend* backend) {
        return AudioPadCollateImpl(std::move(samples), backend);
    };
}

}  // namespace pulsatrix
