/** @file integrated_gradients.hpp
 *  @brief Integrated Gradients -- baseline-interpolated gradient integral (charter Part 1,
 *         Phase 2; theory: xai_context.aDNA's vision_integrated_gradients.md).
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
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

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
 * @note Device-generic host boundary: input/baseline are read to the host once, each
 *       interpolated point is uploaded beside the input, gradients accumulate on the device,
 *       and the final (x - baseline) * avg_grad product runs on one host read of the sum.
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
        // PULSATRIX_ASSERT-only.
        if (steps <= 0) {
            throw std::invalid_argument("IntegratedGradients::explain: steps must be positive");
        }
        if (input.numel() != baseline.numel()) {
            throw std::invalid_argument("IntegratedGradients::explain: input and baseline must have the same numel");
        }

        Saliency saliency;
        DeviceBackend* input_backend = explainer_detail::backend_beside(input, backend);
        Tensor accumulated_grad(input.shape(), input_backend, input.device());  // zero-initialized

        const std::vector<float> input_values = input.to_host_vector();
        const std::vector<float> baseline_values = baseline.to_host_vector();
        const auto numel = static_cast<size_t>(input.numel());

        std::vector<float> interpolated_values(numel);
        for (int64_t k = 1; k <= steps; ++k) {
            float alpha = static_cast<float>(k) / static_cast<float>(steps);
            for (size_t i = 0; i < numel; ++i) {
                interpolated_values[i] = baseline_values[i] + alpha * (input_values[i] - baseline_values[i]);
            }
            Tensor interpolated(input.shape(), input_backend, interpolated_values, input.device());

            Attribution step = saliency.explain(ctx, interpolated, target_index, backend);
            accumulated_grad.accumulate(step.values);
        }

        const std::vector<float> accumulated = accumulated_grad.to_host_vector();
        std::vector<float> ig(numel);
        for (size_t i = 0; i < numel; ++i) {
            float avg_grad = accumulated[i] / static_cast<float>(steps);
            ig[i] = (input_values[i] - baseline_values[i]) * avg_grad;
        }
        Tensor ig_values(input.shape(), input_backend, ig, input.device());

        return Attribution{"integrated_gradients", std::move(ig_values),
                            {{"steps", std::to_string(steps)}, {"target_index", std::to_string(target_index)}}};
    }
};

}  // namespace pulsatrix
