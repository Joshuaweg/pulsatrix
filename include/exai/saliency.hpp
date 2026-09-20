/** @file saliency.hpp
 *  @brief Saliency maps -- raw gradient of a target output w.r.t. the input (charter
 *         Part 1, Phase 2).
 */
#pragma once

#include <stdexcept>
#include <string>

#include "exai/assert.hpp"
#include "exai/attribution.hpp"
#include "exai/device_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief Raw-gradient saliency: d(output[target_index])/d(input), computed by seeding
 *        ExplainerContext::backward_pass with a one-hot vector at target_index.
 * @note A pure graph walker built entirely against ExplainerContext's public interface --
 *       no core (Tensor/ComputationGraph/Autograd/Module) changes needed, per the
 *       charter's own red-flag check for explainer additions.
 */
class Saliency {
public:
    /**
     * @brief Computes the saliency map for one output index.
     * @param ctx Context to run the forward/backward pass through.
     * @param input Input to explain.
     * @param target_index Which output element's gradient to compute (0-based, flat index
     *        into the network's output).
     * @param backend Backend to allocate the one-hot seed tensor through.
     * @return An Attribution with method "saliency", values = the input gradient, and
     *         metadata recording the target index used.
     * @note Assumes a rank-1 network output (this codebase's current unbatched-output
     *       scope everywhere) -- target_index is used as a single-dimension Tensor::at()
     *       index. A rank>1 output (e.g. a Conv-only classifier with no flattening module)
     *       would need a real multi-dimension target selector; not built until a real
     *       network needs one.
     */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, int64_t target_index,
                                       DeviceBackend* backend) const {
        Tensor output = ctx.forward_pass(input);
        // External boundary (Mission 2, finding 15 systemic sweep) -- escalated from
        // EXAI_ASSERT-only.
        if (target_index < 0 || target_index >= output.numel()) {
            throw std::invalid_argument("Saliency::explain: target_index out of range");
        }

        Tensor seed(output.shape(), backend);
        seed.at({target_index}) = 1.0f;

        Tensor grad = ctx.backward_pass(seed);

        // Attribution has no default constructor -- Tensor (its values member) doesn't
        // either, by design (Phase 0). Aggregate-initialize directly instead.
        return Attribution{"saliency", std::move(grad), {{"target_index", std::to_string(target_index)}}};
    }
};

}  // namespace exai
