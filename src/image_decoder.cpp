// STB_IMAGE_IMPLEMENTATION must be defined in exactly one translation unit in the whole
// program -- this is that one. image_decoder.hpp never includes stb_image.h, so no other
// TU can accidentally re-define it.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "pulsatrix/image_decoder.hpp"

#include <stdexcept>

namespace pulsatrix {

Tensor ImageDecoder::DecodeFile(const std::string& path, DeviceBackend* backend, int desired_channels) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels_in_file, desired_channels);
    if (pixels == nullptr) {
        throw std::runtime_error("ImageDecoder::DecodeFile: failed to decode image: " + path + " (" +
                                  stbi_failure_reason() + ")");
    }

    int channels = desired_channels > 0 ? desired_channels : channels_in_file;
    Tensor image(Shape({1, channels, height, width}), backend);

    // stb_image returns HWC-interleaved bytes; Tensor wants CHW, matching Conv2DModule's
    // (N, C, H, W) contract.
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int src_idx = (y * width + x) * channels + c;
                image.at({0, c, y, x}) = static_cast<float>(pixels[src_idx]) / 255.0f;
            }
        }
    }
    stbi_image_free(pixels);
    return image;
}

}  // namespace pulsatrix
