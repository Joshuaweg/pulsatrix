/** @file lime.hpp
 *  @brief LIME -- local interpretable model-agnostic explanations (charter Part 1, Phase
 *         3; theory: xai_context.aDNA's technique_lime.md).
 */
#pragma once

#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/attribution.hpp"
#include "exai/device_backend.hpp"
#include "exai/tensor.hpp"
#include "exai/weighted_linear_regression.hpp"

namespace exai {

/**
 * @brief Fits a locality-weighted linear surrogate around one input: perturb x with
 *        Gaussian noise, weight each perturbed sample by an exponential locality kernel
 *        pi(z) = exp(-||z-x||^2 / (2*sigma^2)), fit w* = argmin_w sum_i pi_i*(f(z_i) -
 *        f(x) - w^T(z_i-x))^2 + l2_lambda*||w||^2 via fit_weighted_linear_regression.
 * @note Takes a forward-pass callable, not ExplainerContext& -- makes model-agnosticism
 *       true by construction (there is nothing else in the type signature to call), per
 *       campaign_exai_dl_library_phase3_surrogate_explainers.md's Context section. Works
 *       with any callable a caller provides, including but not limited to an
 *       ExplainerContext::forward_pass wrapped in a lambda.
 * @note Targets are centered on f(x) (not fit with a separate intercept term) -- the
 *       local linear model is g(z) = f(x) + w^T(z-x), so g(x) = f(x) trivially; only w
 *       (the local sensitivity, this method's actual Attribution) needs fitting.
 * @note A pure graph-free explainer -- no core (Tensor/ComputationGraph/Autograd/Module)
 *       or ExplainerContext changes needed.
 */
class LIME {
public:
    /**
     * @brief Computes the LIME local surrogate explanation for one output index.
     * @param predict Forward-pass callable: input Tensor in, output Tensor out.
     * @param input Input to explain.
     * @param target_index Which output element to explain (0-based, flat index).
     * @param num_samples How many perturbed samples to draw.
     * @param sigma Perturbation std. dev. and locality-kernel width (charter/LIME
     *        convention: a single sigma controls both, per technique_lime.md's own
     *        default-heuristic framing).
     * @param l2_lambda Ridge regularization strength for the surrogate fit.
     * @param seed RNG seed, for reproducibility.
     * @param backend Backend to allocate intermediate tensors through.
     * @return An Attribution with method "lime", values = the local linear coefficients
     *         (same shape as input), and metadata recording the sampling parameters used.
     */
    [[nodiscard]] Attribution explain(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                       int64_t target_index, int64_t num_samples, float sigma, float l2_lambda,
                                       unsigned seed, DeviceBackend* backend) const {
        EXAI_ASSERT(num_samples > 0);
        EXAI_ASSERT(sigma > 0.0f);

        Tensor base_output = predict(input);
        EXAI_ASSERT(target_index >= 0 && target_index < base_output.numel());
        float base_value = base_output.data()[target_index];

        int64_t n_features = input.numel();
        std::mt19937 rng(seed);
        std::normal_distribution<float> noise(0.0f, sigma);

        std::vector<std::vector<float>> samples;
        std::vector<float> targets;
        std::vector<float> weights;
        samples.reserve(static_cast<size_t>(num_samples));
        targets.reserve(static_cast<size_t>(num_samples));
        weights.reserve(static_cast<size_t>(num_samples));

        for (int64_t s = 0; s < num_samples; ++s) {
            Tensor perturbed(input.shape(), backend);
            std::vector<float> delta(static_cast<size_t>(n_features));
            float squared_distance = 0.0f;
            for (int64_t i = 0; i < n_features; ++i) {
                float d = noise(rng);
                delta[static_cast<size_t>(i)] = d;
                perturbed.data()[i] = input.data()[i] + d;
                squared_distance += d * d;
            }

            Tensor output = predict(perturbed);
            float centered_target = output.data()[target_index] - base_value;
            float weight = std::exp(-squared_distance / (2.0f * sigma * sigma));

            samples.push_back(std::move(delta));
            targets.push_back(centered_target);
            weights.push_back(weight);
        }

        std::vector<float> coefficients = fit_weighted_linear_regression(samples, targets, weights, l2_lambda);

        Tensor values(input.shape(), backend);
        for (int64_t i = 0; i < n_features; ++i) {
            values.data()[i] = coefficients[static_cast<size_t>(i)];
        }

        return Attribution{"lime", std::move(values),
                            {{"target_index", std::to_string(target_index)},
                             {"num_samples", std::to_string(num_samples)},
                             {"sigma", std::to_string(sigma)}}};
    }
};

}  // namespace exai
