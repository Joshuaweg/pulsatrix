/** @file prototypes.hpp
 *  @brief Explaining by example (XAI-4): the background inputs nearest one input that the model
 *         treats the same way and differently, and ProtoDash's weighted prototypes that summarize a
 *         dataset (Gurumoorthy, Dhurandhar, Cecchi and Aggarwal, ICDM 2019).
 *  @ingroup interpretability_agnostic
 *
 * "This input is classed like these, and unlike those" is often easier to check than a heatmap. The
 * nearest same-class examples are the input's prototypes; the nearest examples of another class are
 * its contrasts, the closest cases where the model decided differently.
 *
 * Distances are Euclidean, either in the input space with a per-feature scale (pass
 * MedianAbsoluteDeviation() for tabular data in mixed units) or in a representation the caller
 * computes, such as a layer's activations.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Maps an input to the vector distances are measured in. */
using EmbedFn = std::function<std::vector<float>(const Tensor& input)>;

/** @brief FindPrototypes()' settings. */
struct PrototypeOptions {
    /** @brief How many prototypes, and how many contrasts. */
    int64_t k = 3;
    /** @brief Per-feature distance scale in the input space (a change of scale[j] counts 1); empty
     *         for 1. Ignored when embed is set. */
    std::vector<float> scale;
    /** @brief Measure distances in this representation instead of the input space. */
    EmbedFn embed;
};

/** @brief One background example. */
struct Example {
    /** @brief Its index in the background. */
    int64_t index = 0;
    double distance = 0.0;
    /** @brief The model's class for it. */
    int64_t predicted_class = 0;
};

/** @brief FindPrototypes()' result. */
struct PrototypeResult {
    /** @brief The model's class for the input. */
    int64_t predicted_class = 0;
    /** @brief The nearest examples the model gives the same class, nearest first. */
    std::vector<Example> prototypes;
    /** @brief The nearest examples the model gives another class, nearest first. */
    std::vector<Example> contrasts;
};

/**
 * @brief The background examples nearest @p input that the model classes the same way, and those it
 *        classes differently. Classes are the index of the model's largest output.
 * @note An example identical to the input counts, at distance 0; leave the input out of the
 *       background to exclude it.
 * @throws std::invalid_argument if background is empty, its inputs aren't the input's size, k < 1,
 *         or scale has the wrong length or a value that isn't positive.
 */
[[nodiscard]] PrototypeResult FindPrototypes(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                             const std::vector<Tensor>& background, const PrototypeOptions& options = {});

/** @brief ProtoDash()'s settings. */
struct ProtoDashOptions {
    /** @brief How many prototypes to choose, at most. */
    int64_t count = 5;
    /** @brief The Gaussian kernel's width, exp(-|a - b|^2 / (2 sigma^2)); 0 for the median distance
     *         between the source points (the median heuristic). */
    double sigma = 0.0;
    /** @brief Distances in this representation instead of the input space. */
    EmbedFn embed;
};

/** @brief ProtoDash()'s result. */
struct ProtoDashResult {
    /** @brief The chosen source points, in the order chosen. */
    std::vector<int64_t> indices;
    /** @brief Their non-negative weights, which sum to 1: how much of the target each stands for. */
    std::vector<double> weights;
    /** @brief The objective w'mu - w'Kw/2 (unnormalized weights) after each choice; it never falls. */
    std::vector<double> objective;
    double sigma = 0.0;
};

/**
 * @brief ProtoDash: chooses up to options.count points of @p source, with weights, whose weighted
 *        kernel mean best matches the kernel mean of @p target (an MMD criterion).
 *
 * Greedy, as in the paper: each step adds the source point with the largest gradient of
 * l(w) = w'mu - w'Kw/2, where mu_j is point j's mean kernel similarity to the target and K the
 * kernel between source points, then re-fits the weights of the chosen points under w >= 0. It
 * stops early when no point would raise the objective. Use the same set for both to summarize a
 * dataset; a target of one class and a source of all data to find what represents that class.
 * @throws std::invalid_argument if either set is empty, the sizes differ, count < 1 or sigma < 0.
 */
[[nodiscard]] ProtoDashResult ProtoDash(const std::vector<Tensor>& source, const std::vector<Tensor>& target,
                                        const ProtoDashOptions& options = {});

/** @brief The result as a `pulsatrix.prototypes.v1` document. */
[[nodiscard]] JsonValue ToJson(const PrototypeResult& result);

}  // namespace pulsatrix
