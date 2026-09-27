#include "pulsatrix/video_transforms.hpp"

#include <stdexcept>
#include <utility>

namespace pulsatrix {

Sample UniformFrameSampleTransform::apply(Sample sample) const {
    const Tensor& frames = sample.fields[0];
    int64_t total_frames = frames.shape().dim(0);
    int64_t channels = frames.shape().dim(1);
    int64_t height = frames.shape().dim(2);
    int64_t width = frames.shape().dim(3);

    if (num_frames_ > total_frames) {
        throw std::invalid_argument("UniformFrameSampleTransform: num_frames exceeds available frames");
    }

    Tensor dst(Shape({num_frames_, channels, height, width}), backend_);
    for (int64_t i = 0; i < num_frames_; ++i) {
        int64_t src_frame = (i * total_frames) / num_frames_;
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t y = 0; y < height; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    dst.at({i, c, y, x}) = frames.at({src_frame, c, y, x});
                }
            }
        }
    }

    sample.fields[0] = std::move(dst);
    return sample;
}

}  // namespace pulsatrix
