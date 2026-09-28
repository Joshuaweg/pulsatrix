/** @file explainer_stability.hpp
 *  @brief Production API for an explainer's repeated-run stability audit metric (charter Phase 4).
 *  @ingroup interpretability_dl
 */
#pragma once

#include <vector>

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief The result of measuring an explainer's variance across repeated runs on the same
 *        input.
 */
struct StabilityResult {
    /** @brief Per-element variance across runs, averaged over every element. 0 iff is_deterministic. */
    float mean_variance;

    /** @brief True iff every run was bit-for-bit identical -- checked by exact equality, not
     *         by mean_variance == 0.0f, since floating-point summation order can introduce
     *         ~1e-15 noise into a variance computation even over exactly-identical inputs. */
    bool is_deterministic;
};

/**
 * @brief Measures an explainer's stability across repeated runs on the same input (charter's
 *        "repeated-run variance measured and documented" audit category). Known-unstable
 *        methods (LIME, KernelSHAP) are expected to show is_deterministic == false; every
 *        other native/surrogate explainer in this codebase is deterministic by construction.
 * @param repeated_runs Two or more Attribution results from independent calls to the same
 *        explainer on the same input (varying only, e.g., a stochastic method's seed).
 * @throws std::invalid_argument if repeated_runs is empty, or if the runs' Attribution
 *         values do not all share the same element count -- external boundary: the caller
 *         assembles this list from independent explainer invocations, not an internal
 *         invariant.
 */
[[nodiscard]] StabilityResult ComputeAttributionStability(const std::vector<Attribution>& repeated_runs);

}  // namespace pulsatrix
