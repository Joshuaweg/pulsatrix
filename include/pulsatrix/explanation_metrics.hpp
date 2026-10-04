/** @file explanation_metrics.hpp
 *  @brief Explanation-quality metrics in the Quantus families (roadmap XAI-5): faithfulness
 *         (deletion and insertion curves, with ROAD's imputation), randomization (the
 *         model-parameter randomization test) and complexity (sparseness and complexity).
 *  @ingroup interpretability_agnostic
 *  @note Explainer-agnostic: the faithfulness metrics take a prediction function and a finished
 *        Attribution; the randomization test takes a function that explains an input with the
 *        model as it currently is.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief The model as a function: input (with a leading batch dimension of 1) to outputs. */
using PredictFn = std::function<Tensor(const Tensor& input)>;

/** @brief Explains `input` using the model in its current state. */
using ExplainFn = std::function<Attribution(const Tensor& input)>;

// --- Complexity family -------------------------------------------------------------------------

/**
 * @brief Gini index of the attribution magnitudes (Chalasani et al. 2020; Quantus Sparseness):
 *        0 when relevance is spread evenly, approaching 1 when one feature holds it all.
 * @return 0 for an all-zero attribution (Quantus would return NaN).
 */
[[nodiscard]] float Sparseness(const Attribution& attribution);

/**
 * @brief Entropy of the fractional contributions |a_i| / sum |a| (Bhatt et al. 2020; Quantus
 *        Complexity), natural log: 0 when one feature holds all relevance, ln n when spread evenly.
 * @return 0 for an all-zero attribution.
 */
[[nodiscard]] float Complexity(const Attribution& attribution);

/**
 * @brief Spearman rank correlation, with tied values given their average rank.
 * @return 0 when either input is constant. @throws std::invalid_argument if the sizes differ or
 *         are below 2.
 */
[[nodiscard]] float SpearmanRankCorrelation(const std::vector<float>& x, const std::vector<float>& y);

// --- Faithfulness family -----------------------------------------------------------------------

/** @brief How a removed feature is filled in when perturbing an input. */
struct Imputation {
    enum class Kind { Constant, NoisyLinear };
    Kind kind = Kind::Constant;
    float value = 0.0f;       ///< Constant
    float noise_std = 0.01f;  ///< NoisyLinear
    uint64_t seed = 0;        ///< NoisyLinear

    /** @brief Every removed element becomes `value`. */
    [[nodiscard]] static Imputation Constant(float value = 0.0f) { return {Kind::Constant, value, 0.0f, 0}; }

    /**
     * @brief ROAD (Rong et al., "A Consistent and Efficient Evaluation Strategy for Attribution
     *        Methods", ICML 2022): each removed pixel becomes the weighted average of its
     *        neighbors in its (H, W) plane -- direct neighbors 1/6, diagonal ones 1/12, solved
     *        jointly for all removed pixels -- plus Gaussian noise. A constant fill leaks which
     *        pixels were removed through the shape of the hole; this doesn't.
     * @note Meant for images: the last two dimensions are treated as (H, W).
     */
    [[nodiscard]] static Imputation NoisyLinear(float noise_std = 0.01f, uint64_t seed = 0) {
        return {Kind::NoisyLinear, 0.0f, noise_std, seed};
    }
};

/**
 * @brief Returns `input` with every element whose `removed` flag is set filled by `imputation`.
 * @throws std::invalid_argument if `removed` doesn't have one flag per element, or NoisyLinear is
 *         used on an input of rank below 2.
 */
[[nodiscard]] Tensor Impute(const Tensor& input, const std::vector<bool>& removed, const Imputation& imputation);

/** @brief Options for DeletionCurve() and InsertionCurve(). */
struct PerturbationOptions {
    int64_t steps = 20;          ///< points on the curve after the first
    int64_t target = 0;          ///< which output (flat index into the prediction) is scored
    Imputation imputation = Imputation::Constant(0.0f);
};

/** @brief A perturbation curve: the target score after each fraction of features was changed. */
struct PerturbationCurve {
    std::vector<float> fractions;  ///< 0, 1/steps, ..., 1
    std::vector<float> scores;
    float auc;  ///< trapezoid-rule area under scores over fractions
};

/**
 * @brief Deletion (MoRF): removes features from most to least relevant, imputing them, and
 *        records the target score. A faithful attribution makes the score drop fast: lower AUC
 *        is better.
 * @note Features are the input's elements, ordered by attribution value (ties by index).
 * @throws std::invalid_argument if the attribution's shape differs from the input's, steps < 1,
 *         or target is outside the prediction.
 */
[[nodiscard]] PerturbationCurve DeletionCurve(const PredictFn& predict, const Tensor& input,
                                              const Attribution& attribution, const PerturbationOptions& options = {});

/**
 * @brief Insertion: starts from the fully imputed input and restores features from most to least
 *        relevant. A faithful attribution makes the score rise fast: higher AUC is better.
 * @throws std::invalid_argument as DeletionCurve().
 */
[[nodiscard]] PerturbationCurve InsertionCurve(const PredictFn& predict, const Tensor& input,
                                               const Attribution& attribution, const PerturbationOptions& options = {});

// --- Randomization family ----------------------------------------------------------------------

/** @brief The model-parameter randomization test's result. */
struct RandomizationResult {
    /** @brief Layers randomized, in order: top-level name prefixes from the output layer down. */
    std::vector<std::string> layers;
    /** @brief After randomizing layers[0..i] (cascading), the Spearman rank correlation of the
     *         new attribution magnitudes with the original ones. */
    std::vector<float> similarity;
};

/**
 * @brief The model-parameter randomization test (Adebayo et al., "Sanity Checks for Saliency
 *        Maps", NeurIPS 2018), cascading: randomizes the model's layers from the output down,
 *        re-explains `input` after each, and measures how similar the explanation stays.
 * @note An explanation that stays similar after its model was randomized can't be explaining the
 *       model. Each parameter tensor is replaced by Gaussian noise with that tensor's own standard
 *       deviation, drawn from a portable generator seeded by `seed`. Layers are the first segment
 *       of named_parameters() names (a SequentialModule's indices, for example).
 * @note The model's parameters are restored exactly before returning, even if `explain` throws.
 */
[[nodiscard]] RandomizationResult ModelParameterRandomizationTest(Module& model, const ExplainFn& explain,
                                                                  const Tensor& input, uint64_t seed);

}  // namespace pulsatrix
