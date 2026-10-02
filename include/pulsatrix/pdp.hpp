/** @file pdp.hpp
 *  @brief Partial Dependence Plot -- a global marginal-effect technique, sweeping one
 *         feature's value while averaging the model's response over a background set
 *         (charter addition 2026-09-19, see charter Decisions Log).
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief PDP_j(v) = (1/|B|) * sum_{b in B} f(x_j=v, x_{-j}=b_{-j}) -- for each grid value
 *        v, replace every background instance's feature j with v (keeping its other
 *        features), average the model's output over the whole background set.
 * @note Unlike every other explainer in this project, PDP is genuinely global, not
 *       per-instance -- it describes the model's average response to one feature, not an
 *       explanation of a single prediction.
 * @note Takes a caller-supplied background set, not a dataset loader -- no dataset
 *       infrastructure exists in this codebase (same gap real-MNIST-training discussions
 *       surfaced), same "caller supplies reference data" pattern IntegratedGradients'
 *       baseline and LIME's perturbation-around-x already established.
 * @note Takes a forward-pass callable, not ExplainerContext& -- model-agnosticism by
 *       construction, same pattern as LIME/KernelSHAP.
 * @note Device-generic host boundary: each background instance is read to the host once; the
 *       swept copies are uploaded beside it and only f(z)[target] is read back.
 */
class PDP {
public:
    /**
     * @brief Computes the PDP curve for one feature.
     * @param predict Forward-pass callable: input Tensor in, output Tensor out.
     * @param background Reference instances to average over. Must be non-empty; every
     *        instance must have the same shape.
     * @param feature_index Which feature to sweep (0-based, flat index).
     * @param target_index Which output element to read (0-based, flat index).
     * @param grid_min First grid value.
     * @param grid_max Last grid value (inclusive). Ignored (only grid_min used) when
     *        grid_size == 1.
     * @param grid_size Number of evenly-spaced grid points, >= 1.
     * @param backend Backend to allocate intermediate tensors through.
     * @return An Attribution with method "pdp", values = the grid_size-length PDP curve,
     *         and metadata recording the sweep parameters used.
     * @throws std::invalid_argument if background is empty -- external boundary
     *         (campaign_exai_dl_library_adversarial_hardening.md, Mission 2, finding 10):
     *         escalated from PULSATRIX_ASSERT-only, which left this a raw out-of-bounds
     *         background[0] access in Release builds.
     */
    [[nodiscard]] Attribution explain(const std::function<Tensor(const Tensor&)>& predict,
                                       const std::vector<Tensor>& background, int64_t feature_index,
                                       int64_t target_index, float grid_min, float grid_max, int64_t grid_size,
                                       DeviceBackend* backend) const {
        if (background.empty()) {
            throw std::invalid_argument("PDP::explain: background must not be empty");
        }
        PULSATRIX_ASSERT(grid_size >= 1);
        PULSATRIX_ASSERT(feature_index >= 0 && feature_index < background[0].numel());

        std::vector<std::vector<float>> background_values;
        background_values.reserve(background.size());
        for (const Tensor& instance : background) {
            background_values.push_back(instance.to_host_vector());
        }
        std::vector<float> curve(static_cast<size_t>(grid_size));

        for (int64_t k = 0; k < grid_size; ++k) {
            float v = (grid_size == 1) ? grid_min : grid_min + static_cast<float>(k) * (grid_max - grid_min) /
                                                                    static_cast<float>(grid_size - 1);

            float sum = 0.0f;
            for (size_t b = 0; b < background.size(); ++b) {
                const Tensor& instance = background[b];
                std::vector<float> perturbed_values = background_values[b];
                perturbed_values[static_cast<size_t>(feature_index)] = v;
                Tensor perturbed(instance.shape(), instance.backend(), perturbed_values, instance.device());
                Tensor output = predict(perturbed);
                sum += output.read_element(target_index);
            }
            curve[static_cast<size_t>(k)] = sum / static_cast<float>(background.size());
        }

        Tensor curve_tensor(Shape({grid_size}), explainer_detail::backend_beside(background[0], backend), curve,
                            background[0].device());
        return Attribution{"pdp", std::move(curve_tensor),
                            {{"feature_index", std::to_string(feature_index)},
                             {"target_index", std::to_string(target_index)},
                             {"grid_size", std::to_string(grid_size)}}};
    }
};

}  // namespace pulsatrix
