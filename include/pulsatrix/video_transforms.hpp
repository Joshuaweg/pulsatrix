/** @file video_transforms.hpp
 *  @brief Frame-sampling Transform -- selects a fixed number of evenly-spaced frames.
 *  @ingroup data_pipeline
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/transform.hpp"

namespace pulsatrix {

/**
 * @brief Samples num_frames evenly-spaced frames from a clip Tensor (sample.fields[0],
 *        shape (T, C, H, W) -- VideoFrameDirectoryDataset's convention), producing an
 *        (N, C, H, W) Tensor -- deliberately identical in shape convention to Phase 2's
 *        image Dataset/Transform outputs (ImageFolderDataset, ResizeTransform, etc.): a
 *        video clip, once frame-sampled, is just a batch of images from this codebase's
 *        perspective.
 * @note Needs a DeviceBackend to allocate the differently-shaped output Tensor, matching
 *       ResizeTransform/CenterCropTransform's (Phase 2) and ResampleTransform's (Phase 4)
 *       precedent.
 */
class UniformFrameSampleTransform : public Transform {
public:
    UniformFrameSampleTransform(int64_t num_frames, DeviceBackend* backend)
        : num_frames_(num_frames), backend_(backend) {}

    /**
     * @throws std::invalid_argument if num_frames exceeds the clip's actual frame count --
     *         external boundary: num_frames is caller-supplied configuration.
     */
    [[nodiscard]] Sample apply(Sample sample) const override;

private:
    int64_t num_frames_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
