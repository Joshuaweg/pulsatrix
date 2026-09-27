/** @file video_frame_directory_dataset.hpp
 *  @brief Directory-of-pre-extracted-frames video Dataset (reduced-scope stub, no codec decode).
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief Dataset over a directory tree of the form
 *        root_dir/<class_name>/<clip_name>/<sequentially-named-frame-image>, decoding each
 *        clip's frames (via ImageDecoder) into one (T, C, H, W) Tensor per sample.
 * @note Reduced-scope stub (campaign_exai_dl_library_data_pipeline, Phase 5, Decision
 *       Point 3, operator-directed): no real video container/codec decode -- frames must
 *       already be extracted into per-clip directories. Real container/codec decode
 *       (FFmpeg or otherwise) is a named, tracked follow-up, not implemented here.
 */
class VideoFrameDirectoryDataset : public Dataset {
public:
    /**
     * @param root_dir Directory containing one subdirectory per class, each containing
     *        one subdirectory per clip, each containing that clip's frame image files.
     * @param backend Backend to allocate decoded frame Tensors through. Not owned.
     * @throws std::runtime_error if root_dir doesn't exist/isn't a directory, contains no
     *         class subdirectories, or any clip directory contains no frame files.
     */
    VideoFrameDirectoryDataset(const std::string& root_dir, DeviceBackend* backend);

    [[nodiscard]] int64_t size() const override;

    /**
     * @throws std::out_of_range if index is out of bounds.
     * @return {frames (T,C,H,W), label (1,) float, the class's sorted-order index}.
     */
    [[nodiscard]] Sample get(int64_t index) const override;

    /** @brief Sorted class names; classes()[i] is the human-readable name for label i. */
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }

private:
    struct Entry {
        std::string clip_dir;
        int64_t label;
    };

    std::vector<Entry> entries_;
    std::vector<std::string> classes_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
