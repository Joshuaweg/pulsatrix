/** @file acquisition_functions.hpp
 *  @brief Bayesian-Optimization acquisition functions over a GP posterior: Expected
 *         Improvement, Probability of Improvement, GP-Upper-Confidence-Bound (Snoek,
 *         Larochelle, Adams, "Practical Bayesian Optimization of Machine Learning
 *         Algorithms," NeurIPS 2012).
 *  @ingroup hyperparameter_optimization
 *  @note Maximization convention throughout, matching this campaign's own established
 *        convention (Evolutionary DL's selection/crossover operators, this campaign's
 *        SearchSpace-driven samplers) -- a caller minimizing a loss negates it first.
 */
#pragma once

#include <cmath>
#include <stdexcept>

namespace pulsatrix {

/** @brief Standard normal PDF, phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2). */
inline double StandardNormalPdf(double z) {
    constexpr double kInvSqrt2Pi = 0.3989422804014327;  // 1 / sqrt(2*pi)
    return kInvSqrt2Pi * std::exp(-0.5 * z * z);
}

/** @brief Standard normal CDF, Phi(z) = 0.5 * (1 + erf(z / sqrt(2))). */
inline double StandardNormalCdf(double z) { return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0))); }

/**
 * @brief Expected Improvement: the expected amount by which a candidate exceeds
 *        best_value + xi, under the posterior N(mean, variance).
 * @param variance Posterior variance (not standard deviation) -- must be non-negative.
 * @param xi Small exploration margin (default 0.01, a common practical default -- without it,
 *        EI can collapse to pure exploitation once the posterior mean exceeds best_value by
 *        even a negligible amount).
 * @throws std::invalid_argument if variance < 0.
 * @note If sigma (sqrt(variance)) is ~0 (a point the GP is fully confident about, e.g. exactly
 *       an already-observed point), EI is defined as exactly 0 -- no uncertainty means no
 *       potential for improvement beyond what standard credit already reflects.
 */
inline double ExpectedImprovement(double mean, double variance, double best_value, double xi = 0.01) {
    if (variance < 0.0) {
        throw std::invalid_argument("ExpectedImprovement: variance must be non-negative");
    }
    double sigma = std::sqrt(variance);
    if (sigma <= 1e-12) {
        return 0.0;
    }
    double z = (mean - best_value - xi) / sigma;
    return (mean - best_value - xi) * StandardNormalCdf(z) + sigma * StandardNormalPdf(z);
}

/**
 * @brief Probability of Improvement: P(candidate's value > best_value + xi) under the
 *        posterior N(mean, variance).
 * @throws std::invalid_argument if variance < 0.
 * @note If sigma is ~0, PI is defined as the degenerate limit: 1 if mean already exceeds
 *       best_value + xi, else 0.
 */
inline double ProbabilityOfImprovement(double mean, double variance, double best_value, double xi = 0.01) {
    if (variance < 0.0) {
        throw std::invalid_argument("ProbabilityOfImprovement: variance must be non-negative");
    }
    double sigma = std::sqrt(variance);
    if (sigma <= 1e-12) {
        return (mean > best_value + xi) ? 1.0 : 0.0;
    }
    double z = (mean - best_value - xi) / sigma;
    return StandardNormalCdf(z);
}

/**
 * @brief GP-Upper-Confidence-Bound: mean + kappa * sqrt(variance).
 * @param kappa Exploration weight (default 2.0, a common practical default -- larger values
 *        favor exploring high-uncertainty regions over exploiting the current best mean).
 * @throws std::invalid_argument if variance < 0.
 */
inline double UpperConfidenceBound(double mean, double variance, double kappa = 2.0) {
    if (variance < 0.0) {
        throw std::invalid_argument("UpperConfidenceBound: variance must be non-negative");
    }
    return mean + kappa * std::sqrt(variance);
}

}  // namespace pulsatrix
