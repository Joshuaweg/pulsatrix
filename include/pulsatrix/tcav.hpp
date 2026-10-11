/** @file tcav.hpp
 *  @brief Testing with Concept Activation Vectors (INT-6): how much a concept, given only by
 *         examples, pushes a model towards a class, and whether that's more than chance.
 *  @ingroup mech_interp
 *
 * TCAV (Kim et al., ICML 2018) works in three steps:
 *
 * 1. **A concept activation vector (CAV).** A linear classifier separates a layer's activations on
 *    examples of a concept (striped images, say) from its activations on random examples. Its
 *    normal, pointing towards the concept, is the CAV.
 * 2. **Conceptual sensitivity.** For an input, the directional derivative of a class logit along
 *    the CAV, at that layer: `grad_layer(logit) . cav`. Positive means moving the activation
 *    towards the concept raises the class.
 * 3. **The TCAV score.** The fraction of a class's inputs whose sensitivity is positive.
 *
 * One score means little. A CAV trained against one random set can point anywhere that separates
 * the two sets, and a meaningless direction scores anywhere between 0 and 1. So RunTcav() trains
 * CAVs against several random sets, and does the same with random sets in place of the concept.
 * A two-sided t-test between the two lists of scores says whether the concept differs from
 * chance.
 *
 * The models are ExplainerContext chains, whose gradients are available at every module's
 * output. Language models aren't covered: they don't expose gradients at an intermediate layer.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How TrainCav() finds the concept direction. */
enum class CavMethod {
    /** @brief The weights of an L2-regularized logistic regression, as Kim et al. train them.
     *         The features are standardized while fitting and the direction mapped back. */
    LogisticRegression,
    /** @brief The difference of the two sets' mean activations. It needs no fitting, and doesn't
     *         rotate towards dimensions that only separate the two samples by chance. */
    DifferenceOfMeans,
};

struct CavOptions {
    CavMethod method = CavMethod::LogisticRegression;
    /** @brief The L2 penalty on the standardized weights, per example. */
    double l2 = 0.01;
    int64_t iterations = 500;
    /** @brief Every this-many-th example of each set is held out to measure accuracy. 0 holds
     *         none out, and the accuracy is then measured on the training examples. */
    int64_t holdout_every = 3;
};

/** @brief A concept activation vector. */
struct Cav {
    /** @brief A unit vector in the layer's activation space, pointing towards the concept. */
    std::vector<double> direction;
    /** @brief How well the direction's linear classifier separates concept from random examples
     *         that it wasn't fitted on (see CavOptions::holdout_every). Near 0.5, the layer
     *         doesn't separate them and the direction means nothing. */
    double accuracy = 0;
};

/**
 * @brief Trains a CAV from a layer's activations on concept examples and on random examples.
 * @param concept_activations, random_activations shape (n, ...), a row per example, the same
 *        width. Each needs at least 2 rows (3 with the default holdout).
 * @throws std::invalid_argument if the widths differ, a set is too small, or a value isn't finite.
 */
[[nodiscard]] Cav TrainCav(const Tensor& concept_activations, const Tensor& random_activations, const CavOptions& options = {});

/**
 * @brief Each input's directional derivative along the CAV: the row of @p gradients dotted with
 *        cav.direction.
 * @param gradients shape (n, ...): the gradient of the class logit with respect to the layer, a
 *        row per input.
 * @throws std::invalid_argument if the rows aren't the CAV's width.
 */
[[nodiscard]] std::vector<double> ConceptualSensitivities(const Tensor& gradients, const Cav& cav);

/** @brief The fraction of sensitivities above zero. @throws std::invalid_argument if empty. */
[[nodiscard]] double TcavScore(const std::vector<double>& sensitivities);

/** @brief A two-sample t-test's result. */
struct TTestResult {
    double t = 0;
    double degrees_of_freedom = 0;
    /** @brief Two-sided. */
    double p_value = 1;
};

/**
 * @brief Student's two-sample t-test with pooled variance, two-sided, as scipy.stats.ttest_ind
 *        computes by default and the TCAV library uses.
 * @note When both samples have no variance, t is 0 and p is 1 if their means are equal, and t is
 *       infinite and p is 0 if not.
 * @throws std::invalid_argument if either sample has fewer than 2 values, or a value isn't finite.
 */
[[nodiscard]] TTestResult StudentTTest(const std::vector<double>& a, const std::vector<double>& b);

/** @brief The layer's activations for @p inputs, from one forward pass: shape (n, width). */
[[nodiscard]] Tensor LayerActivations(ExplainerContext& ctx, const Tensor& inputs, NodeId layer);

/** @brief The gradient of logit @p target with respect to the layer, for each of @p inputs:
 *         shape (n, width). */
[[nodiscard]] Tensor LayerGradients(ExplainerContext& ctx, const Tensor& inputs, NodeId layer, int64_t target, DeviceBackend* backend);

struct TcavOptions {
    CavOptions cav;
    /** @brief The significance level for RunTcav()'s t-test. With several concepts, layers or
     *         classes, divide it by their number (Bonferroni), as Kim et al. do. */
    double alpha = 0.05;
};

/** @brief RunTcav()'s result. */
struct TcavResult {
    /** @brief The concept's TCAV score against each random set. */
    std::vector<double> scores;
    /** @brief Each random set's TCAV score, as a concept, against the next random set: what
     *         chance looks like. */
    std::vector<double> random_scores;
    /** @brief The CAVs' accuracies, concept against each random set. */
    std::vector<double> cav_accuracies;
    double mean_score = 0;
    double mean_random_score = 0;
    TTestResult test;
    /** @brief Whether test.p_value is below the alpha asked for. */
    bool significant = false;
};

/**
 * @brief TCAV with Kim et al.'s significance test, for one concept, layer and class.
 * @param ctx The model, as an ExplainerContext; its output is (n, classes).
 * @param layer The node whose output is the layer: module i's output is node i + 1.
 * @param target The class whose logit is explained.
 * @param class_inputs Inputs of that class, the model's input shape with a leading batch.
 * @param concept_inputs Examples of the concept.
 * @param random_inputs At least 3 sets of random examples. CAV i is the concept against set i;
 *        random CAV i is set i against set i + 1 (the last against the first).
 * @throws std::invalid_argument if layer isn't a node of the model, target is out of range,
 *         there are fewer than 3 random sets, or for what TrainCav() rejects.
 */
[[nodiscard]] TcavResult RunTcav(ExplainerContext& ctx, NodeId layer, int64_t target, const Tensor& class_inputs, const Tensor& concept_inputs,
                                 const std::vector<Tensor>& random_inputs, DeviceBackend* backend, const TcavOptions& options = {});

}  // namespace pulsatrix
