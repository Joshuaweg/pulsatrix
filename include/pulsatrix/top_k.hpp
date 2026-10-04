/** @file top_k.hpp
 *  @brief Top-k selection along a tensor's last dimension, on any device (roadmap FND-3).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief top_k()'s result: the selected values and their positions along the last dimension. */
struct TopKResult {
    /** @brief Shape `(..., k)`: each row's k best values, best first. */
    Tensor values;
    /** @brief Shape `(..., k)`: each value's index within its row, as a whole-number float. */
    Tensor indices;
};

/**
 * @brief Selects the k largest (or smallest) entries of every row along the last dimension.
 * @param input Tensor of any rank >= 1. Every slice along its last dimension is one row.
 * @param k Entries kept per row; 1 <= k <= the last dimension's size.
 * @param largest true for the k largest (default), false for the k smallest.
 * @return Values and indices of shape `(..., k)`, in rank order, on input's device.
 * @note Order is fully deterministic and identical on every backend: NaN ranks above every
 *       number (first when largest, last when smallest), and equal values keep the lower index
 *       first. Indices are whole-number floats, like every other index tensor in this library,
 *       and so are exact only below 2^24; longer rows are rejected.
 * @note Selection only -- no gradient. A module that routes gradients through a top-k (a TopK
 *       sparse autoencoder, a mixture-of-experts router) scatters through `indices` itself.
 * @throws std::invalid_argument if input is empty or rank 0, k is outside [1, last dimension],
 *         or the last dimension exceeds 2^24 -- external boundary.
 */
[[nodiscard]] TopKResult top_k(const Tensor& input, int64_t k, bool largest = true);

}  // namespace pulsatrix
