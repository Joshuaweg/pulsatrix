/** @file dataset.hpp
 *  @brief Random-access dataset abstraction -- Sample, Dataset (size()/get()).
 *  @ingroup dl_modules
 */
#pragma once

#include <stdexcept>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief One dataset sample: an ordered list of Tensor fields (e.g. {features, label} or
 *        {image, label}). Field order/count is a contract between a Dataset implementation
 *        and whatever CollateFn (collate.hpp) later assembles samples into a Batch.
 */
struct Sample {
    std::vector<Tensor> fields;
};

/**
 * @brief Random-access dataset abstraction -- pulsatrix's analogue of PyTorch's
 *        torch.utils.data.Dataset / LibTorch's torch::data::Dataset (__len__/__getitem__).
 *        Every concrete modality (CsvDataset, MnistDatasetAdapter, future image/text/audio
 *        readers) subclasses this; DataLoader depends only on this interface.
 * @note NOT a Module -- a Dataset produces Tensors, it does not compute over them, so it has
 *       no forward()/backward()/propagate_relevance() (charter non-negotiable #5's LRP
 *       requirement applies to Module subclasses only, per
 *       campaign_exai_dl_library_data_pipeline's explicit scope note).
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    /** @brief Number of samples in this dataset. */
    [[nodiscard]] virtual int64_t size() const = 0;

    /**
     * @brief Loads one sample by index.
     * @param index Sample index, must be in [0, size()).
     * @return The sample at index.
     * @throws std::out_of_range if index is out of bounds -- external boundary: the index
     *         is chosen by a Sampler/DataLoader from possibly-external configuration
     *         (shuffle seed, resumed epoch offset), not a compile-time-known loop bound.
     */
    [[nodiscard]] virtual Sample get(int64_t index) const = 0;
};

}  // namespace pulsatrix
