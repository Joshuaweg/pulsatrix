/** @file null_model_baseline.hpp
 *  @brief The baseline rule as one call (roadmap XAI-6): run any explainer, probe or featurizer
 *         on the trained model and on a re-initialized ("null") model, and compare.
 *  @ingroup interpretability_agnostic
 *  @note Why: attribution maps, probes, sparse autoencoders and causal analyses all produce
 *        plausible-looking output on randomly initialized networks (Méloux et al., "The Dead
 *        Salmons of AI Interpretability", 2025; Adebayo et al. 2018; Heap et al. 2025). A result
 *        that looks the same on the null model says nothing about the trained one.
 */
#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/** @brief Saves every parameter of a module and restores them exactly when it goes out of scope. */
class ParameterSnapshot {
public:
    explicit ParameterSnapshot(Module& model) {
        for (ParamRef p : model.parameters()) {
            saved_.emplace_back(p.value, *p.value);
        }
    }
    ~ParameterSnapshot() { restore(); }
    ParameterSnapshot(const ParameterSnapshot&) = delete;
    ParameterSnapshot& operator=(const ParameterSnapshot&) = delete;

    /** @brief Puts the saved values back now (the destructor does it again, harmlessly). */
    void restore() {
        for (auto& [target, value] : saved_) {
            *target = value;  // keeps the target's device and requires_grad flag
        }
    }

private:
    std::vector<std::pair<Tensor*, Tensor>> saved_;
};

/**
 * @brief Replaces each parameter tensor with Gaussian noise of mean 0 and that tensor's own
 *        standard deviation, from a portable generator seeded by `seed`.
 * @param prefix Only parameters named `prefix` or under it (dot boundaries); empty means all.
 * @note Keeps each tensor's scale, so activations stay in a realistic range. A tensor whose values
 *       are all equal (a zero bias, say) has standard deviation 0 and becomes all zeros. Buffers,
 *       such as BatchNorm's running statistics, are left alone.
 */
void ReinitializeParameters(Module& model, uint64_t seed, const std::string& prefix = "");

/** @brief NullModelBaseline()'s result: the analysis on the trained model and on the null model. */
template <typename Result>
struct NullModelComparison {
    Result trained;
    Result null_model;
};

/**
 * @brief Runs `analysis()` on `model`, then again after ReinitializeParameters(model, seed),
 *        then restores the model exactly (also if `analysis` throws).
 * @param analysis Anything callable with no arguments that reads the model: an explainer run, a
 *        probe's accuracy, a featurizer's statistics. Its return type is the result type.
 */
template <typename Analysis>
[[nodiscard]] auto NullModelBaseline(Module& model, Analysis&& analysis, uint64_t seed)
    -> NullModelComparison<std::decay_t<std::invoke_result_t<Analysis&>>> {
    using Result = std::decay_t<std::invoke_result_t<Analysis&>>;
    Result trained = analysis();
    ParameterSnapshot saved(model);
    ReinitializeParameters(model, seed);
    Result null_model = analysis();
    return {std::move(trained), std::move(null_model)};
}

/** @brief NullModelBaseline() for an explanation of one input. */
struct AttributionNullReport {
    Attribution trained;
    Attribution null_model;
    /** @brief Spearman rank correlation of the two attributions' magnitudes. Close to 1 means the
     *         explanation barely depends on what the model learned. */
    float rank_similarity;
};

/** @brief Explains `input` with the trained and the null model and compares the two. */
[[nodiscard]] AttributionNullReport NullModelBaseline(Module& model, const ExplainFn& explain, const Tensor& input,
                                                      uint64_t seed);

}  // namespace pulsatrix
