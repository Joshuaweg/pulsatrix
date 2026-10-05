/** @file sensitivity.hpp
 *  @brief Local sensitivity: one-at-a-time input changes and occlusion (CFS-3).
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief The low and high value to try for every feature of an input (flat order). */
struct SensitivityBounds {
    std::vector<float> low;
    std::vector<float> high;
};

/** @brief x - delta and x + delta for every feature. @throws std::invalid_argument if delta <= 0. */
[[nodiscard]] SensitivityBounds DeltaBounds(const Tensor& input, float delta);

/**
 * @brief x -/+ num_std standard deviations of each feature over @p background, so every feature
 *        moves by a comparable amount whatever its units. A feature constant over the background
 *        doesn't move.
 * @throws std::invalid_argument if num_std <= 0, background is empty, or a background instance's
 *         shape differs from input's.
 */
[[nodiscard]] SensitivityBounds ScaledDeltaBounds(const Tensor& input, const std::vector<Tensor>& background,
                                                  float num_std = 1.0f);

/**
 * @brief Each feature's lower and upper percentile over @p background (linear interpolation, as
 *        numpy.percentile): the classic tornado-diagram range, a typical low and high value.
 * @throws std::invalid_argument if background is empty, its shapes differ, or the percentiles
 *         aren't 0 <= lower < upper <= 1.
 */
[[nodiscard]] SensitivityBounds RangeBounds(const std::vector<Tensor>& background, float lower_percentile = 0.05f,
                                            float upper_percentile = 0.95f);

/** @brief How one feature's low and high value move the output, everything else held fixed. */
struct FeatureSensitivity {
    int64_t feature_index = 0;
    float value = 0.0f;  ///< The input's own value
    float low = 0.0f;
    float high = 0.0f;
    float output_low = 0.0f;   ///< The target output with the feature at `low`
    float output_high = 0.0f;  ///< ... and at `high`

    /** @brief |output_high - output_low|: the bar length in a tornado chart. */
    [[nodiscard]] float swing() const;
    /** @brief (output_high - output_low) / (high - low): a finite-difference slope; 0 if high == low. */
    [[nodiscard]] float slope() const;
};

/** @brief One-at-a-time sensitivity of one output around one input. */
struct LocalSensitivityResult {
    int64_t target_index = 0;
    float output = 0.0f;  ///< The target output at the unchanged input
    std::vector<FeatureSensitivity> features;
};

/**
 * @brief Sets each feature in turn to its low and then its high bound, leaving the others at the
 *        input's values, and records the target output each time.
 * @param features Flat feature indices to vary; empty means all of them.
 * @note 1 + 2 * features model calls. Unlike a gradient, which describes an infinitesimal
 *       neighborhood, this measures a finite one: the two disagree exactly where the model
 *       saturates or has a kink.
 * @throws std::invalid_argument if bounds don't have one entry per feature, a feature index or
 *         target index is out of range.
 */
[[nodiscard]] LocalSensitivityResult ComputeLocalSensitivity(const std::function<Tensor(const Tensor&)>& predict,
                                                             const Tensor& input, int64_t target_index,
                                                             const SensitivityBounds& bounds,
                                                             const std::vector<int64_t>& features = {});

/**
 * @brief Occlusion (Zeiler and Fergus 2014) with Captum 0.9's semantics: a window slides over the
 *        input with the given strides, everything under it is set to @p baseline, and each element's
 *        attribution is the drop in the target output, f(x) - f(occluded), averaged over the windows
 *        that covered it.
 * @param window One size per input dimension (for an instance shaped (C, H, W), for example
 *        (C, 3, 3) occludes 3x3 patches across all channels; for a (L, d) sequence, (k, d) occludes
 *        spans of k tokens).
 * @param strides One per dimension, each >= 1 and no larger than the window where the window is
 *        smaller than the input. The last window in a dimension is cropped at the edge, as in Captum.
 * @return Attribution "occlusion", shaped like the input, beside it (same backend and device).
 * @throws std::invalid_argument if window or strides have the wrong length or invalid sizes, or the
 *         target index is out of range.
 */
[[nodiscard]] Attribution Occlusion(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                    int64_t target_index, const std::vector<int64_t>& window,
                                    const std::vector<int64_t>& strides, float baseline = 0.0f);

}  // namespace pulsatrix
