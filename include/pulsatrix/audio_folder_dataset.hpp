/** @file audio_folder_dataset.hpp
 *  @brief Directory-of-class-subfolders audio Dataset, mirroring ImageFolderDataset.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief Dataset over a directory tree of the form root_dir/<class_name>/<audio_file.wav>,
 *        structurally identical to ImageFolderDataset (Phase 2): sorted subdirectory names
 *        are classes, sorted filenames within each are samples, decoded lazily per get()
 *        via WavReader.
 */
class AudioFolderDataset : public Dataset {
public:
    /**
     * @throws std::runtime_error if root_dir doesn't exist/isn't a directory, or contains
     *         no class subdirectories.
     */
    AudioFolderDataset(const std::string& root_dir, DeviceBackend* backend);

    [[nodiscard]] int64_t size() const override;

    /**
     * @throws std::out_of_range if index is out of bounds.
     * @return {waveform (1,channels,num_samples), label (1,) float, the class's sorted-order index}.
     */
    [[nodiscard]] Sample get(int64_t index) const override;

    /** @brief Sorted class names; classes()[i] is the human-readable name for label i. */
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }

private:
    struct Entry {
        std::string path;
        int64_t label;
    };

    std::vector<Entry> entries_;
    std::vector<std::string> classes_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
