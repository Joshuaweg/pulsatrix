/** @file image_transforms.hpp
 *  @brief Sample-level image Transforms -- resize, center-crop, normalize, horizontal flip.
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/transform.hpp"

namespace pulsatrix {

/**
 * @brief Every transform in this file operates on sample.fields[0], assumed to be an image
 *        Tensor of shape (1, channels, height, width) -- the convention MnistDatasetAdapter,
 *        CsvDataset, and ImageDecoder all already share (image/features first, label last).
 */

/**
 * @brief Nearest-neighbor resize to (target_height, target_width).
 * @note Needs a DeviceBackend to allocate the differently-shaped output Tensor -- unlike
 *       NormalizeTransform/HorizontalFlipTransform, which modify a same-shaped buffer
 *       in place and need no backend at all.
 */
class ResizeTransform : public Transform {
public:
    ResizeTransform(int64_t target_height, int64_t target_width, DeviceBackend* backend)
        : target_height_(target_height), target_width_(target_width), backend_(backend) {}

    [[nodiscard]] Sample apply(Sample sample) const override;

private:
    int64_t target_height_;
    int64_t target_width_;
    DeviceBackend* backend_;
};

/**
 * @brief Crops the centered (crop_height, crop_width) region of the image.
 * @throws std::invalid_argument if the crop size exceeds the image's actual size --
 *         external boundary: crop dimensions are caller-supplied configuration.
 */
class CenterCropTransform : public Transform {
public:
    CenterCropTransform(int64_t crop_height, int64_t crop_width, DeviceBackend* backend)
        : crop_height_(crop_height), crop_width_(crop_width), backend_(backend) {}

    [[nodiscard]] Sample apply(Sample sample) const override;

private:
    int64_t crop_height_;
    int64_t crop_width_;
    DeviceBackend* backend_;
};

/**
 * @brief Per-channel normalization: pixel = (pixel - mean[c]) / std[c]. In-place, no
 *        backend needed (same shape in and out).
 * @throws std::invalid_argument if mean/std size doesn't match the image's channel count.
 */
class NormalizeTransform : public Transform {
public:
    NormalizeTransform(std::vector<float> mean, std::vector<float> std) : mean_(std::move(mean)), std_(std::move(std)) {}

    [[nodiscard]] Sample apply(Sample sample) const override;

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

/** @brief Mirrors the image left-right. In-place, no backend needed (same shape). */
class HorizontalFlipTransform : public Transform {
public:
    [[nodiscard]] Sample apply(Sample sample) const override;
};

}  // namespace pulsatrix
