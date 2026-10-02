/** @file saliency.hpp
 *  @brief Saliency maps -- raw gradient of a target output w.r.t. the input (charter
 *         Part 1, Phase 2).
 *  @ingroup interpretability_dl
 */
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Raw-gradient saliency: d(output[target_index])/d(input), computed by seeding
 *        ExplainerContext::backward_pass with a one-hot vector at target_index.
 * @note A pure graph walker built entirely against ExplainerContext's public interface --
 *       no core (Tensor/ComputationGraph/Autograd/Module) changes needed, per the
 *       charter's own red-flag check for explainer additions.
 * @note Device-generic: forward/backward run on the network's device; only the one-hot seed is
 *       built on the host and uploaded beside the output (explainer_detail host boundary).
 */
class Saliency {
public:
    /**
     * @brief Computes the saliency map for one output index.
     * @param ctx Context to run the forward/backward pass through.
     * @param input Input to explain.
     * @param target_index Which output element's gradient to compute (0-based, flat index
     *        into the network's output).
     * @param backend Backend to allocate the one-hot seed tensor through (the output's own
     *        backend is used instead when this one serves a different device).
     * @return An Attribution with method "saliency", values = the input gradient, and
     *         metadata recording the target index used.
     * @note Assumes a rank-2 (N, num_classes) network output -- migrated by
     *       campaign_exai_dl_library_batch_dimension_support from the original rank-1
     *       assumption. target_index selects the same class for every example in the
     *       batch (seeding a one-hot at [n, target_index] for every row n), not a
     *       per-example target vector -- a deliberate scope boundary (this migration fixes
     *       the shape contract, it doesn't add per-example target selection as a new
     *       capability).
     */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, int64_t target_index,
                                       DeviceBackend* backend) const {
        Tensor output = ctx.forward_pass(input);
        // External boundary (Mission 2, finding 15 systemic sweep; rank check added by
        // campaign_exai_dl_library_batch_dimension_support) -- escalated from
        // PULSATRIX_ASSERT-only.
        if (output.rank() != 2) {
            throw std::invalid_argument("Saliency::explain: network output must be rank-2 (N, num_classes)");
        }
        if (target_index < 0 || target_index >= output.shape().dim(1)) {
            throw std::invalid_argument("Saliency::explain: target_index out of range");
        }
        int64_t N = output.shape().dim(0);
        int64_t C = output.shape().dim(1);

        std::vector<float> seed_values(static_cast<size_t>(output.numel()), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            seed_values[static_cast<size_t>(n * C + target_index)] = 1.0f;
        }
        Tensor seed(output.shape(), explainer_detail::backend_beside(output, backend), seed_values,
                    output.device());

        Tensor grad = ctx.backward_pass(seed);

        // Attribution has no default constructor -- Tensor (its values member) doesn't
        // either, by design (Phase 0). Aggregate-initialize directly instead.
        return Attribution{"saliency", std::move(grad), {{"target_index", std::to_string(target_index)}}};
    }
};

}  // namespace pulsatrix
