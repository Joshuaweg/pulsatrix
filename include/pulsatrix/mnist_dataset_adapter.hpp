/** @file mnist_dataset_adapter.hpp
 *  @brief Adapts a pre-loaded MnistDataset onto the generic Dataset interface.
 *  @ingroup data_pipeline
 */
#pragma once

#include <stdexcept>
#include <utility>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/mnist_loader.hpp"

namespace pulsatrix {

/**
 * @brief Adapts a pre-loaded MnistDataset (MnistIdxLoader::Load's output) onto the generic
 *        Dataset interface -- minimal-diff retrofit (campaign_exai_dl_library_data_pipeline,
 *        Mission 4): MnistIdxLoader/MnistDataset themselves are unchanged, still exercised
 *        directly by mnist_loader_test.cpp; this adapter is purely additive, fulfilling
 *        mnist_loader.hpp's own note that a second real dataset is the moment to
 *        generalize.
 */
class MnistDatasetAdapter : public Dataset {
public:
    MnistDatasetAdapter(MnistDataset dataset, DeviceBackend* backend)
        : dataset_(std::move(dataset)), backend_(backend) {}

    [[nodiscard]] int64_t size() const override { return static_cast<int64_t>(dataset_.images.size()); }

    /**
     * @throws std::out_of_range if index is out of bounds.
     * @return {image (1,1,28,28), label as a (1,) float Tensor} -- the label conversion is
     *         the only new work here; the image Tensor is already batch-of-one shaped,
     *         matching this codebase's existing convention, so it is returned unchanged
     *         (deep-copied via Tensor's copy constructor, since Sample owns its fields).
     */
    [[nodiscard]] Sample get(int64_t index) const override {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("MnistDatasetAdapter::get: index out of range");
        }
        size_t i = static_cast<size_t>(index);
        Tensor label(Shape({1}), backend_, {static_cast<float>(dataset_.labels[i])});
        return Sample{{dataset_.images[i], std::move(label)}};
    }

private:
    MnistDataset dataset_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
