#include "pulsatrix/audio_transforms.hpp"

#include <utility>

namespace pulsatrix {

Sample ResampleTransform::apply(Sample sample) const {
    const Tensor& src = sample.fields[0];
    int64_t channels = src.shape().dim(1);
    int64_t num_samples = src.shape().dim(2);

    int64_t new_num_samples = (num_samples * static_cast<int64_t>(target_sample_rate_)) /
                               static_cast<int64_t>(source_sample_rate_);
    double ratio = static_cast<double>(source_sample_rate_) / static_cast<double>(target_sample_rate_);

    Tensor dst(Shape({1, channels, new_num_samples}), backend_);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t i = 0; i < new_num_samples; ++i) {
            double src_pos = static_cast<double>(i) * ratio;
            int64_t floor_idx = static_cast<int64_t>(src_pos);
            double frac = src_pos - static_cast<double>(floor_idx);

            float value;
            if (floor_idx + 1 >= num_samples) {
                value = src.at({0, c, num_samples - 1});
            } else {
                float a = src.at({0, c, floor_idx});
                float b = src.at({0, c, floor_idx + 1});
                value = static_cast<float>(a * (1.0 - frac) + b * frac);
            }
            dst.at({0, c, i}) = value;
        }
    }

    sample.fields[0] = std::move(dst);
    return sample;
}

}  // namespace pulsatrix
