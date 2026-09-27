/** @file image_decoder.hpp
 *  @brief Decodes image files (PNG/JPEG/BMP/etc.) into Tensors via stb_image.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Decodes an image file into a Tensor -- pulsatrix's generalization beyond
 *        MnistIdxLoader's IDX-format-only precedent (campaign_exai_dl_library_data_pipeline,
 *        Phase 2). Powered by stb_image (public-domain, single-header, vendored via
 *        FetchContent -- Decision Point 2): supports PNG/JPEG/BMP/GIF/TGA/HDR and more.
 * @note stb_image.h itself is included only in image_decoder.cpp -- this header never
 *       exposes stb's own types/symbols, keeping it out of pulsatrix's public API surface.
 */
class ImageDecoder {
public:
    /**
     * @param path Path to an image file, any stb_image-supported format.
     * @param backend Backend to allocate the output Tensor through. Not owned.
     * @param desired_channels If > 0, forces the decoded image to this many channels
     *        (stb_image converts as needed -- e.g. an RGB source file with
     *        desired_channels=1 is converted to grayscale). 0 (default) keeps the file's
     *        own channel count.
     * @return Tensor of shape (1, channels, height, width) -- matching MnistIdxLoader's
     *         existing per-sample batch-of-one convention -- float32, pixel values
     *         normalized to [0,1].
     * @throws std::runtime_error if the file can't be opened or decoded -- external
     *         boundary (file content, not an internal invariant).
     */
    [[nodiscard]] static Tensor DecodeFile(const std::string& path, DeviceBackend* backend, int desired_channels = 0);
};

}  // namespace pulsatrix
