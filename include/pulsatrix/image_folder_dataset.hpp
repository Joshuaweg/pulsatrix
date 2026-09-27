/** @file image_folder_dataset.hpp
 *  @brief Directory-of-class-subfolders image Dataset, mirroring torchvision's ImageFolder.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief Dataset over a directory tree of the form root_dir/<class_name>/<image_file>,
 *        mirroring torchvision's ImageFolder convention. Class names are the sorted
 *        subdirectory names; each class's label is its index in that sorted order.
 *        Images are decoded lazily (per get() call) via ImageDecoder.
 */
class ImageFolderDataset : public Dataset {
public:
    /**
     * @param root_dir Directory containing one subdirectory per class.
     * @param backend Backend to allocate decoded image Tensors through. Not owned.
     * @throws std::runtime_error if root_dir doesn't exist/isn't a directory, or contains
     *         no class subdirectories -- external boundary: a caller-supplied path.
     */
    ImageFolderDataset(const std::string& root_dir, DeviceBackend* backend);

    [[nodiscard]] int64_t size() const override;

    /**
     * @throws std::out_of_range if index is out of bounds.
     * @throws std::runtime_error if the image file fails to decode (see ImageDecoder).
     * @return {image (1,C,H,W), label (1,) float, the class's sorted-order index}.
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
