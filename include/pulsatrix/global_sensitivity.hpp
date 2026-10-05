/** @file global_sensitivity.hpp
 *  @brief Global sensitivity analysis: Morris elementary effects and Sobol indices (CFS-4).
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The input space to explore: which features vary, over which ranges. Every other
 *        feature keeps its value from @p base.
 */
struct GlobalSensitivityProblem {
    /** @brief An instance shaped like the model's input. */
    Tensor base;
    /** @brief Flat indices of the features to vary; non-empty, no repeats. */
    std::vector<int64_t> features;
    /** @brief Each varied feature's range, uniform between lower[i] and upper[i] (lower < upper). */
    std::vector<float> lower;
    std::vector<float> upper;
};

/** @brief Morris screening results, one entry per varied feature (SALib's `morris.analyze`). */
struct MorrisResult {
    std::vector<int64_t> features;
    /** @brief Mean elementary effect: the average direction of the feature's effect. */
    std::vector<float> mu;
    /** @brief Mean absolute elementary effect: overall importance (Campolongo et al. 2007). */
    std::vector<float> mu_star;
    /** @brief Standard deviation of the elementary effects (ddof 1): large when the effect depends
     *         on where you are, i.e. nonlinearity or interaction. */
    std::vector<float> sigma;
    /** @brief Half-width of the bootstrap confidence interval of mu_star. */
    std::vector<float> mu_star_conf;
    int64_t num_trajectories = 0;
};

/** @brief Morris settings. */
struct MorrisOptions {
    int64_t num_trajectories = 20;
    /** @brief Grid levels per feature, even and >= 2; the step is levels / (2 (levels - 1)) of
     *         the range. */
    int num_levels = 4;
    int num_resamples = 1000;
    float confidence = 0.95f;
    uint64_t seed = 0;
};

/**
 * @brief Morris trajectories (Morris 1991): each starts at a random grid point and moves one
 *        feature at a time, in a random order, by the Morris step. Rows are feature values in the
 *        problem's ranges, (num_features + 1) per trajectory.
 * @note Deterministic for a seed on every platform (no std:: distributions).
 * @throws std::invalid_argument for an invalid problem or options.
 */
[[nodiscard]] std::vector<std::vector<float>> MorrisSample(const GlobalSensitivityProblem& problem,
                                                           const MorrisOptions& options = {});

/**
 * @brief Morris statistics from samples laid out as MorrisSample's (or SALib's) trajectories and
 *        one output per row: the same elementary effects and mu, mu*, sigma as SALib 1.6
 *        `morris.analyze` (unscaled). The confidence interval is a bootstrap with this library's
 *        own generator, so it agrees with SALib's in distribution, not digit for digit.
 * @throws std::invalid_argument if the rows don't form whole trajectories of num_features + 1 rows,
 *         a step changes other than exactly one feature, or an option is invalid.
 */
[[nodiscard]] MorrisResult AnalyzeMorris(const std::vector<std::vector<float>>& samples,
                                         const std::vector<float>& outputs, const MorrisOptions& options = {});

/** @brief Samples, runs the model and analyzes: MorrisSample, then AnalyzeMorris. */
[[nodiscard]] MorrisResult ComputeMorris(const std::function<Tensor(const Tensor&)>& predict,
                                         const GlobalSensitivityProblem& problem, int64_t target_index,
                                         const MorrisOptions& options = {});

/** @brief Sobol indices, one entry per varied feature (SALib's `sobol.analyze`, first and total
 *         order). */
struct SobolResult {
    std::vector<int64_t> features;
    /** @brief The share of the output's variance the feature explains alone. */
    std::vector<float> first_order;
    /** @brief The share it explains including every interaction it takes part in. total - first
     *         is the feature's interaction share. */
    std::vector<float> total_order;
    std::vector<float> first_order_conf;
    std::vector<float> total_order_conf;
    int64_t num_samples = 0;
};

/** @brief Sobol settings. */
struct SobolOptions {
    /** @brief Base samples N; the model runs N * (num_features + 2) times. */
    int64_t num_samples = 1024;
    int num_resamples = 100;
    float confidence = 0.95f;
    uint64_t seed = 0;
};

/**
 * @brief Sobol indices from outputs in SALib's layout without second order: N blocks of
 *        (num_features + 2) outputs, f(A), f(AB_1) ... f(AB_D), f(B). Outputs are standardized
 *        first, then the first order uses Saltelli et al. (2010)'s estimator and the total order
 *        Jansen's, exactly as SALib 1.6 does. The bootstrap uses this library's generator.
 * @throws std::invalid_argument if the outputs don't form whole blocks or an option is invalid.
 */
[[nodiscard]] SobolResult AnalyzeSobol(const std::vector<float>& outputs, int64_t num_features,
                                       const SobolOptions& options = {});

/**
 * @brief Samples two independent uniform matrices A and B over the problem's ranges, builds the
 *        Saltelli design (AB_j is A with column j from B), runs the model on every row and calls
 *        AnalyzeSobol.
 * @note Sobol indices assume the varied features are independent.
 * @throws std::invalid_argument for an invalid problem or options, or a target out of range.
 */
[[nodiscard]] SobolResult ComputeSobol(const std::function<Tensor(const Tensor&)>& predict,
                                       const GlobalSensitivityProblem& problem, int64_t target_index,
                                       const SobolOptions& options = {});

}  // namespace pulsatrix
