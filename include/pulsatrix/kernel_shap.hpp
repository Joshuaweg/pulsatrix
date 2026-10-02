/** @file kernel_shap.hpp
 *  @brief KernelSHAP -- model-agnostic Shapley value approximation via weighted linear
 *         regression (charter Part 1, Phase 3; theory: xai_context.aDNA's
 *         technique_shap.md).
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"
#include "pulsatrix/weighted_linear_regression.hpp"

namespace pulsatrix {

namespace detail {

/** @brief Binomial coefficient C(n,k), computed iteratively to avoid factorial overflow. */
inline double BinomialCoefficient(int64_t n, int64_t k) {
    double result = 1.0;
    for (int64_t i = 1; i <= k; ++i) {
        result = result * static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return result;
}

/**
 * @brief The SHAP kernel pi(z') = (n-1) / [C(n,|z'|) * |z'| * (n-|z'|)], for 1 <= |z'| <= n-1.
 * @note Undefined (division by zero) at |z'|=0 and |z'|=n -- those coalitions are handled
 *       separately (they anchor f(baseline) and f(x) exactly), never passed here.
 */
inline float ShapKernelWeight(int64_t n, int64_t coalition_size) {
    double denominator =
        BinomialCoefficient(n, coalition_size) * static_cast<double>(coalition_size) * static_cast<double>(n - coalition_size);
    return static_cast<float>(static_cast<double>(n - 1) / denominator);
}

}  // namespace detail

/**
 * @brief Approximates Shapley values via full coalition enumeration + SHAP-kernel-weighted
 *        linear regression, reusing fit_weighted_linear_regression for the reduced
 *        (n-1)-dimensional problem the efficiency-axiom substitution produces.
 * @note The empty/full coalitions have undefined SHAP kernel weight (division by zero) --
 *       phi_0 is fixed to f(baseline) exogenously, and the efficiency axiom
 *       (sum(phi_i) = f(x) - f(baseline)) is enforced by substitution (eliminating the
 *       last feature's coefficient), not by including those two coalitions in the
 *       regression. See mission_kernel_shap.md's Recon for the hand-derived n=2 case this
 *       was verified against before implementation.
 * @note Full 2^n enumeration, not sampling -- exact at this codebase's small feature
 *       counts (mathematically equivalent to the definitional Shapley formula at full
 *       enumeration), not the charter's excluded O(2^n)/O(n!) brute-force-at-scale case.
 * @note Takes a forward-pass callable, not ExplainerContext& -- model-agnosticism by
 *       construction, same pattern as LIME.
 * @note Device-generic host boundary: coalitions are built on the host from one read of
 *       input/baseline, uploaded beside the input, and only f(z)[target] is read back.
 */
class KernelSHAP {
public:
    /**
     * @brief Computes Shapley value approximations for one output index.
     * @param predict Forward-pass callable: input Tensor in, output Tensor out.
     * @param input Input to explain.
     * @param baseline Reference "feature absent" input. Must match input's shape.
     * @param target_index Which output element to attribute (0-based, flat index).
     * @param backend Backend to allocate intermediate tensors through.
     * @return An Attribution with method "kernel_shap", values = the per-feature Shapley
     *         values (same shape as input), and metadata recording the target index used.
     */
    [[nodiscard]] Attribution explain(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                       const Tensor& baseline, int64_t target_index, DeviceBackend* backend) const {
        // External boundary (Mission 2, finding 15 systemic sweep) -- escalated from
        // PULSATRIX_ASSERT-only.
        if (input.numel() != baseline.numel()) {
            throw std::invalid_argument("KernelSHAP::explain: input and baseline must have the same numel");
        }
        const int64_t n = input.numel();
        if (n < 1 || n > 20) {  // full 2^n enumeration -- this campaign's small-n scope
            throw std::invalid_argument("KernelSHAP::explain: input feature count must be in [1, 20]");
        }

        const std::vector<float> input_values = input.to_host_vector();
        const std::vector<float> baseline_values = baseline.to_host_vector();
        DeviceBackend* input_backend = explainer_detail::backend_beside(input, backend);
        std::vector<float> z_values(static_cast<size_t>(n));

        auto coalition_value = [&](uint64_t mask) {
            for (int64_t i = 0; i < n; ++i) {
                bool active = ((mask >> i) & 1u) != 0;
                const auto idx = static_cast<size_t>(i);
                z_values[idx] = active ? input_values[idx] : baseline_values[idx];
            }
            Tensor z(input.shape(), input_backend, z_values, input.device());
            Tensor out = predict(z);
            return out.read_element(target_index);
        };

        const float f_baseline = coalition_value(0);
        const uint64_t full_mask = (n == 64) ? ~0ULL : ((1ULL << n) - 1);
        const float total_diff = coalition_value(full_mask) - f_baseline;

        std::vector<float> phi(static_cast<size_t>(n), 0.0f);

        if (n == 1) {
            phi[0] = total_diff;
        } else {
            std::vector<std::vector<float>> reduced_samples;
            std::vector<float> reduced_targets;
            std::vector<float> weights;

            for (uint64_t mask = 1; mask < full_mask; ++mask) {
                int64_t coalition_size = 0;
                for (int64_t i = 0; i < n; ++i) {
                    if ((mask >> i) & 1u) {
                        ++coalition_size;
                    }
                }
                bool z_last = ((mask >> (n - 1)) & 1u) != 0;
                float value = coalition_value(mask) - f_baseline;
                float reduced_target = value - (z_last ? total_diff : 0.0f);

                std::vector<float> reduced_row(static_cast<size_t>(n - 1));
                for (int64_t i = 0; i < n - 1; ++i) {
                    bool z_i = ((mask >> i) & 1u) != 0;
                    reduced_row[static_cast<size_t>(i)] = (z_i ? 1.0f : 0.0f) - (z_last ? 1.0f : 0.0f);
                }

                reduced_samples.push_back(std::move(reduced_row));
                reduced_targets.push_back(reduced_target);
                weights.push_back(detail::ShapKernelWeight(n, coalition_size));
            }

            std::vector<float> reduced_phi =
                fit_weighted_linear_regression(reduced_samples, reduced_targets, weights, 0.0f);

            float sum_reduced = 0.0f;
            for (int64_t i = 0; i < n - 1; ++i) {
                phi[static_cast<size_t>(i)] = reduced_phi[static_cast<size_t>(i)];
                sum_reduced += reduced_phi[static_cast<size_t>(i)];
            }
            phi[static_cast<size_t>(n - 1)] = total_diff - sum_reduced;
        }

        Tensor values(input.shape(), input_backend, phi, input.device());

        return Attribution{"kernel_shap", std::move(values), {{"target_index", std::to_string(target_index)}}};
    }
};

}  // namespace pulsatrix
