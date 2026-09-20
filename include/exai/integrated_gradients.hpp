/** @file integrated_gradients.hpp
 *  @brief Integrated Gradients -- baseline-interpolated gradient integral (charter Part 1,
 *         Phase 2; theory: xai_context.aDNA's vision_integrated_gradients.md).
 */
#pragma once

#include <stdexcept>
#include <string>

#include "exai/assert.hpp"
#include "exai/attribution.hpp"
#include "exai/device_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/saliency.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief IG_i(x) = (x_i - baseline_i) * (1/steps) * sum_{k=1}^{steps}
 *        d(F(baseline + (k/steps)(x - baseline)))/dx_i -- a Riemann-sum approximation of
 *        the straight-line path integral from baseline to input.
 * @note Reuses Saliency::explain per interpolation step (each step's gradient IS exactly
 *       a raw-gradient saliency computation at one interpolated point) rather than
 *       reimplementing the forward/backward mechanics.
 * @note Correctness is verified via the completeness axiom (sum(IG(x)) == F(x) -
 *       F(baseline)), not just "it runs" -- see vision_integrated_gradients.md.
 * @note A pure graph walker built entirely against ExplainerContext/Saliency's public
 *       interfaces -- no core (Tensor/ComputationGraph/Autograd/Module) changes needed.
 * @note The final interpolation/scaling arithmetic is a raw host loop over
 *       input/baseline/accumulated-gradient buffers -- DeviceBackend has no
 *       tensor-tensor subtract or scalar-multiply primitive, and this campaign's Phase
 *       1.5-style device guards weren't extended to Phase 2's new explainer code (no CUDA
 *       explainer work exists yet to require it). CPU-only until that need arises.
 */
class IntegratedGradients {
public:
    /**
     * @brief Computes the Integrated Gradients attribution for one output index.
     * @param ctx Context to run each interpolation step's forward/backward pass through.
     * @param input Input to explain.
     * @param baseline Reference "uninformative" input. Must match input's shape.
     * @param target_index Which output element to attribute (0-based, flat index).
     * @param steps Number of Riemann-sum interpolation steps (50-300 typical; more steps
     *        tightens the completeness-axiom approximation error).
     * @param backend Backend to allocate intermediate tensors through.
     * @return An Attribution with method "integrated_gradients", values = the IG map, and
     *         metadata recording steps and target index used.
     */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, const Tensor& baseline,
                                       int64_t target_index, int64_t steps, DeviceBackend* backend) const {
        // External boundary (Mission 2, finding 15 systemic sweep) -- escalated from
        // EXAI_ASSERT-only.
        if (steps <= 0) {
            throw std::invalid_argument("IntegratedGradients::explain: steps must be positive");
        }
        if (input.numel() != baseline.numel()) {
            throw std::invalid_argument("IntegratedGradients::explain: input and baseline must have the same numel");
        }

        Saliency saliency;
        Tensor accumulated_grad(input.shape(), backend);  // zero-initialized

        for (int64_t k = 1; k <= steps; ++k) {
            float alpha = static_cast<float>(k) / static_cast<float>(steps);
            Tensor interpolated(input.shape(), backend);
            for (int64_t i = 0; i < input.numel(); ++i) {
                interpolated.data()[i] = baseline.data()[i] + alpha * (input.data()[i] - baseline.data()[i]);
            }

            Attribution step = saliency.explain(ctx, interpolated, target_index, backend);
            accumulated_grad.accumulate(step.values);
        }

        Tensor ig_values(input.shape(), backend);
        for (int64_t i = 0; i < input.numel(); ++i) {
            float avg_grad = accumulated_grad.data()[i] / static_cast<float>(steps);
            ig_values.data()[i] = (input.data()[i] - baseline.data()[i]) * avg_grad;
        }

        return Attribution{"integrated_gradients", std::move(ig_values),
                            {{"steps", std::to_string(steps)}, {"target_index", std::to_string(target_index)}}};
    }
};

}  // namespace exai
