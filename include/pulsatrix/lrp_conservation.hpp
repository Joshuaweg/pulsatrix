/** @file lrp_conservation.hpp
 *  @brief Production API for LRP's conservation-delta audit metric (charter Phase 4).
 *  @ingroup interpretability_lrp
 */
#pragma once

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The result of comparing a relevance-propagation step's input and output totals.
 * @note LRP's defining property is that relevance is conserved layer-to-layer -- the sum of
 *       relevance at any layer should equal the sum at the next (up to the small absorption
 *       epsilon-rule variants introduce). Before this type existed, that check lived only as
 *       test-only local lambdas in lrp_conservation_test.cpp/stability_test.cpp; this
 *       promotes it to a real, reusable API so non-test code (the ExplanationScoreCard's
 *       "trustworthiness" tile) can compute the same number without duplicating the math.
 */
struct ConservationResult {
    float relevance_in_sum;
    float relevance_out_sum;

    /** @brief |relevance_in_sum - relevance_out_sum| -- 0 is perfect conservation. */
    [[nodiscard]] float delta() const;
};

/**
 * @brief Sums relevance_in and relevance_out independently and reports their gap.
 * @param relevance_in A module's propagate_relevance() input-side relevance tensor.
 * @param relevance_out The same call's output-side relevance tensor.
 * @note Deliberately does not call Module::propagate_relevance itself -- the caller has
 *       already produced both tensors (from whichever module/config it used); this function
 *       is a pure summation over already-computed results, reusable regardless of which
 *       module type or LRP rule produced them.
 */
[[nodiscard]] ConservationResult ComputeConservation(const Tensor& relevance_in, const Tensor& relevance_out);

}  // namespace pulsatrix
