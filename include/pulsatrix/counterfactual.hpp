/** @file counterfactual.hpp
 *  @brief Gradient-based counterfactual explanations (Wachter et al. 2017), CFS-5.
 *  @ingroup interpretability_dl
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief What the counterfactual must achieve. */
struct CounterfactualTarget {
    enum class Kind {
        /** Output `index` must exceed every other output by at least `margin` (a class). */
        Class,
        /** Output `index` must land in [low, high] (a regression value). */
        Range,
    };
    Kind kind = Kind::Class;
    int64_t index = 0;
    float margin = 0.0f;
    float low = 0.0f;
    float high = 0.0f;

    [[nodiscard]] static CounterfactualTarget ToClass(int64_t class_index, float margin = 0.0f) {
        return {Kind::Class, class_index, margin, 0.0f, 0.0f};
    }
    [[nodiscard]] static CounterfactualTarget ToRange(int64_t output_index, float low, float high) {
        return {Kind::Range, output_index, 0.0f, low, high};
    }
};

/** @brief Which changes are allowed, and how changes are measured. */
struct CounterfactualConstraints {
    /** @brief Per-feature distance scale (flat order); a change of `scale[j]` costs 1. Usually
     *         MedianAbsoluteDeviation(background). Empty means 1 for every feature. */
    std::vector<float> scale;
    /** @brief Features that must not change, for example age or a protected attribute. */
    std::vector<int64_t> immutable;
    /** @brief Per-feature [lower, upper] limits (flat order), or empty for none. */
    std::vector<float> lower;
    std::vector<float> upper;
    /** @brief Features rounded to whole numbers at the end (counts, ordinal codes). */
    std::vector<int64_t> integer;
    /** @brief Groups of features that one-hot encode one categorical feature: at the end, each
     *         group is set to 1 at its largest value and 0 elsewhere. If that loses validity,
     *         every category of every group is tried and the nearest valid one kept. */
    std::vector<std::vector<int64_t>> one_hot_groups;
};

/** @brief Search settings. Defaults suit standardized tabular inputs. */
struct CounterfactualOptions {
    /** @brief Weight of the prediction loss against the distance at the start of the search. */
    float lambda = 0.1f;
    /** @brief lambda is multiplied by this whenever a round ends without a valid counterfactual. */
    float lambda_growth = 10.0f;
    int max_rounds = 5;
    int steps_per_round = 500;
    float learning_rate = 0.05f;
    /** @brief A feature counts as changed when it moves by more than this times its scale. */
    float change_tolerance = 1e-3f;
};

/** @brief A counterfactual and how good it is (the DiCE metrics, for one counterfactual). */
struct CounterfactualResult {
    Tensor counterfactual;
    /** @brief True when the counterfactual reaches the target, after rounding and one-hot
     *         projection. */
    bool valid = false;
    /** @brief The target output (CounterfactualTarget::index) before and after. */
    float output_before = 0.0f;
    float output_after = 0.0f;
    /** @brief Sum over features of |change| / scale: the objective's distance (proximity). */
    float distance = 0.0f;
    /** @brief Euclidean length of the raw change. */
    float l2 = 0.0f;
    /** @brief Number of features that changed (sparsity). */
    int64_t num_changed = 0;
    int rounds = 0;
    float final_lambda = 0.0f;
};

/**
 * @brief Each feature's median absolute deviation over @p background (flat order), the distance
 *        scale Wachter et al. recommend for tabular data. A feature with zero MAD gets 1, as in
 *        DiCE, so it can still move.
 * @throws std::invalid_argument if background is empty or its shapes differ.
 */
[[nodiscard]] std::vector<float> MedianAbsoluteDeviation(const std::vector<Tensor>& background);

/**
 * @brief Finds a nearby input that reaches @p target (Wachter, Mittelstadt and Russell 2017):
 *        minimizes `lambda * loss(f(x'), target) + sum_j |x'_j - x_j| / scale_j` by proximal
 *        gradient descent, so features the prediction doesn't need stay exactly unchanged.
 *        If a round ends without reaching the target, lambda grows and the search continues
 *        from where it was.
 *
 * The prediction loss is a hinge, zero once the target is reached: for a class,
 * `max(0, max_{k != t} z_k - z_t + margin)` on the model's outputs (logits); for a range, the
 * distance from the output to [low, high].
 *
 * @param ctx The model, run through forward_pass and backward_pass. Parameter gradients
 *        accumulate in the model, as with Saliency.
 * @param input One instance with its batch dimension, for example (1, F); every element is a
 *        feature.
 * @note Gradient-based, so the model must be differentiable in its input. On images and other
 *       high-dimensional inputs a counterfactual found this way is usually an adversarial
 *       example (Freiesleben 2022): a change the model reacts to, not one that means anything.
 * @throws std::invalid_argument if the target index is out of range, a constraint vector has the
 *         wrong length or an index out of range, or an option is invalid.
 */
[[nodiscard]] CounterfactualResult FindCounterfactual(ExplainerContext& ctx, const Tensor& input,
                                                      const CounterfactualTarget& target,
                                                      const CounterfactualConstraints& constraints = {},
                                                      const CounterfactualOptions& options = {});

}  // namespace pulsatrix
