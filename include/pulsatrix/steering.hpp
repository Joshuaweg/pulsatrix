/** @file steering.hpp
 *  @brief Steering a model by adding a direction to its residual stream (FEAT-7), with
 *         difference-of-means as the default direction, a featurizer's decoder direction as an
 *         option, and a report on how reliably steering works.
 *  @ingroup mech_interp
 *  @note Why the report: steering vectors often look effective on average while failing on many
 *        inputs, some moving the wrong way, and they misgeneralize out of distribution (Tan et al.,
 *        "Analyzing the Generalization and Reliability of Steering Vectors", NeurIPS 2024). On
 *        AxBench (Wu et al., arXiv 2501.17148) SAE features steer worse than simple baselines such
 *        as difference-of-means. So every steering result here comes with its per-input spread, the
 *        share of inputs it moves the wrong way, and random directions of the same size.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/causal_lm.hpp"  // HiddenStateHook
#include "pulsatrix/featurizer.hpp"

namespace pulsatrix {

/**
 * @brief The difference of means: the mean of @p positive minus the mean of @p negative, both
 *        `(N, d)` row-major. The simple baseline that usually wins.
 * @throws std::invalid_argument for empty sets or rows of different sizes.
 */
[[nodiscard]] std::vector<float> DifferenceOfMeans(const std::vector<float>& positive, const std::vector<float>& negative, int64_t d);

/** @brief Feature @p feature's decoder direction, as a steering direction (unit length for a
 *         featurizer with unit-norm decoders). */
[[nodiscard]] std::vector<float> FeaturizerDirection(Featurizer& featurizer, int64_t feature);

/**
 * @brief A hook that adds `coefficient * direction` to the hidden states at @p position (0 is the
 *        embeddings, i layer i's output), for the rows in @p rows (every row when empty). Install
 *        it with CausalLM::set_hidden_state_hook() or EncoderLM::set_hidden_state_hook().
 * @note The hook throws std::invalid_argument if the hidden size isn't the direction's, or @p rows
 *       has the wrong length.
 */
[[nodiscard]] HiddenStateHook SteeringHook(int64_t position, std::vector<float> direction, float coefficient, std::vector<bool> rows = {});

struct SteeringOptions {
    /** @brief The coefficients the direction is scaled by, 0 included as the unsteered baseline.
     *         Small by default: the slopes assume a linear response, and on ESM-2 8M a range of
     *         ±2 disrupted the model in any direction, random ones included. */
    std::vector<double> coefficients = {-0.5, -0.25, 0.0, 0.25, 0.5};
    /** @brief Random directions of the same norm, the control. */
    int64_t random_directions = 3;
    uint64_t seed = 0;
};

/** @brief MeasureSteering()'s result. */
struct SteeringReport {
    std::vector<double> coefficients;
    /** @brief The behavior averaged over inputs, per coefficient. */
    std::vector<double> mean_behavior;
    /** @brief The same, averaged over the random directions too. */
    std::vector<double> random_mean_behavior;
    /** @brief Each input's steerability: the least-squares slope of its behavior against the
     *         coefficient (Tan et al.). */
    std::vector<double> steerability;
    double mean_steerability = 0;
    double steerability_sd = 0;
    /** @brief The share of inputs with negative steerability: steering moves them the wrong way. */
    double anti_steerable_fraction = 0;
    /** @brief The random directions' mean steerability, and the largest of their magnitudes. */
    double random_mean_steerability = 0;
    double random_max_abs_steerability = 0;
    /** @brief mean_steerability over the largest random magnitude: how far the direction stands
     *         above directions with no meaning. Below about 1, it doesn't. */
    double over_random = 0;
};

/**
 * @brief Measures steering along @p direction, and along random directions of the same norm.
 * @param behavior Returns input @p i's behavior, a number that steering should raise (a logit
 *        difference, a log-probability, a probe's score), with the hook it is given installed;
 *        it must remove the hook afterwards. It is called with each coefficient for each input.
 * @param num_inputs How many inputs @p behavior knows.
 * @throws std::invalid_argument for no inputs, fewer than two distinct coefficients, an empty or
 *         all-zero direction, or random_directions < 0.
 */
[[nodiscard]] SteeringReport MeasureSteering(const std::function<double(int64_t i, const HiddenStateHook& hook)>& behavior,
                                             int64_t num_inputs, int64_t position, const std::vector<float>& direction,
                                             const SteeringOptions& options = {});

}  // namespace pulsatrix
