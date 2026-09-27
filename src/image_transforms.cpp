#include "pulsatrix/image_transforms.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace pulsatrix {

Sample ResizeTransform::apply(Sample sample) const {
    const Tensor& src = sample.fields[0];
    int64_t channels = src.shape().dim(1);
    int64_t src_h = src.shape().dim(2);
    int64_t src_w = src.shape().dim(3);

    Tensor dst(Shape({1, channels, target_height_, target_width_}), backend_);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < target_height_; ++y) {
            int64_t src_y = std::min(src_h - 1, (y * src_h) / target_height_);
            for (int64_t x = 0; x < target_width_; ++x) {
                int64_t src_x = std::min(src_w - 1, (x * src_w) / target_width_);
                dst.at({0, c, y, x}) = src.at({0, c, src_y, src_x});
            }
        }
    }
    sample.fields[0] = std::move(dst);
    return sample;
}

Sample CenterCropTransform::apply(Sample sample) const {
    const Tensor& src = sample.fields[0];
    int64_t channels = src.shape().dim(1);
    int64_t src_h = src.shape().dim(2);
    int64_t src_w = src.shape().dim(3);

    if (crop_height_ > src_h || crop_width_ > src_w) {
        throw std::invalid_argument("CenterCropTransform: crop size exceeds image size");
    }

    int64_t top = (src_h - crop_height_) / 2;
    int64_t left = (src_w - crop_width_) / 2;

    Tensor dst(Shape({1, channels, crop_height_, crop_width_}), backend_);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < crop_height_; ++y) {
            for (int64_t x = 0; x < crop_width_; ++x) {
                dst.at({0, c, y, x}) = src.at({0, c, top + y, left + x});
            }
        }
    }
    sample.fields[0] = std::move(dst);
    return sample;
}

Sample NormalizeTransform::apply(Sample sample) const {
    Tensor& image = sample.fields[0];
    int64_t channels = image.shape().dim(1);
    if (static_cast<int64_t>(mean_.size()) != channels || static_cast<int64_t>(std_.size()) != channels) {
        throw std::invalid_argument("NormalizeTransform: mean/std size must match the image's channel count");
    }
    int64_t h = image.shape().dim(2);
    int64_t w = image.shape().dim(3);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w; ++x) {
                float& value = image.at({0, c, y, x});
                value = (value - mean_[static_cast<size_t>(c)]) / std_[static_cast<size_t>(c)];
            }
        }
    }
    return sample;
}

Sample HorizontalFlipTransform::apply(Sample sample) const {
    Tensor& image = sample.fields[0];
    int64_t channels = image.shape().dim(1);
    int64_t h = image.shape().dim(2);
    int64_t w = image.shape().dim(3);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < h; ++y) {
            for (int64_t x = 0; x < w / 2; ++x) {
                std::swap(image.at({0, c, y, x}), image.at({0, c, y, w - 1 - x}));
            }
        }
    }
    return sample;
}

}  // namespace pulsatrix
