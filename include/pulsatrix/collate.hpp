/** @file collate.hpp
 *  @brief Batch assembly -- Batch, CollateFn, DefaultCollate.
 *  @ingroup data_pipeline
 */
#pragma once

#include <functional>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One collated batch: one stacked Tensor per Sample field position. */
struct Batch {
    std::vector<Tensor> fields;

    /** @brief Number of samples in this batch -- fields[0]'s leading dimension. */
    [[nodiscard]] int64_t size() const { return fields.empty() ? 0 : fields[0].shape().dim(0); }
};

/**
 * @brief A function assembling a list of Samples into one Batch -- pulsatrix's analogue of
 *        PyTorch's collate_fn. The standard extension point for ragged/variable-length
 *        modalities (text padding, variable-length audio): a caller-supplied CollateFn
 *        replacing DefaultCollate, not a subclass hierarchy.
 */
using CollateFn = std::function<Batch(std::vector<Sample>, DeviceBackend*)>;

/**
 * @brief Stacks a list of samples into one Batch, field-by-field, via Tensor::Stack --
 *        pulsatrix's default CollateFn (PyTorch's default_collate analogue).
 * @param samples Non-empty list of samples, each with the same field count.
 * @param backend Backend to allocate stacked field tensors through.
 * @return A Batch with one stacked Tensor per field position.
 * @throws std::invalid_argument if samples is empty or samples have differing field
 *         counts -- external boundary: the list of samples is assembled by a DataLoader
 *         from independently constructed Dataset::get() results, not a compile-time-known
 *         invariant. Per-field shape/device mismatches are reported by Tensor::Stack.
 */
[[nodiscard]] Batch DefaultCollate(std::vector<Sample> samples, DeviceBackend* backend);

}  // namespace pulsatrix
