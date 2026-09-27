/** @file cma_es.hpp
 *  @brief A CMA-ES-*style* ask-tell evolutionary strategy (Hansen & Ostermeier's Covariance
 *         Matrix Adaptation Evolution Strategy) for continuous hyperparameter search: an
 *         adaptive mean, a per-dimension adaptive scale (separable/diagonal covariance, Ros &
 *         Hansen, "A Simple Modification in CMA-ES Achieving Linear Time and Space
 *         Complexity," PPSN 2008), and an adaptive global step size.
 *  @ingroup evolutionary
 *  @note Deliberate simplifications from full CMA-ES, matching this campaign's own scope
 *        wording ("CMA-ES-*style*", not "CMA-ES"): (1) diagonal-only covariance (sep-CMA-ES),
 *        avoiding any eigendecomposition/Cholesky factorization -- a real, published,
 *        citable reduction, not an invented shortcut; (2) equal-weight ("intermediate")
 *        recombination over the top mu offspring, the classic (mu/mu_I, lambda)-ES scheme
 *        that predates and still coexists with log-rank weighting in the literature, rather
 *        than the refined log-rank weights modern CMA-ES defaults to; (3) fixed learning
 *        rates for both the step-size and per-dimension-scale updates, rather than the full
 *        algorithm's mu_eff-dependent formulas and multi-generation evolution-path memory
 *        (cumulative step-size adaptation, the rank-one update). What is preserved exactly:
 *        the ask-tell interface, adaptive mean, per-dimension variance *matching* the
 *        empirical spread of successful steps (the essential idea evolution paths refine, not
 *        replace), and global step-size adaptation via the classic "successful step norm
 *        versus its expected norm under N(0,I)" comparison.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsatrix {

/** @brief This algorithm's full adaptive state: the search mean, the global step size, and
 *         each dimension's own variance (the diagonal of the covariance matrix). */
struct CMAESState {
    std::vector<double> mean;
    double sigma;
    std::vector<double> variances;
};

/**
 * @brief Pure core: decodes an explicit set of standard-normal sample vectors into offspring
 *        points, x_i = mean + sigma * sqrt(variances) (elementwise) * z_i.
 * @throws std::invalid_argument if any z_sample's dimension doesn't match state.mean's.
 */
inline std::vector<std::vector<double>> AskGivenSamples(const CMAESState& state,
                                                         const std::vector<std::vector<double>>& z_samples) {
    const size_t n = state.mean.size();
    std::vector<std::vector<double>> offspring;
    offspring.reserve(z_samples.size());
    for (const auto& z : z_samples) {
        if (z.size() != n) {
            throw std::invalid_argument("AskGivenSamples: every z_sample must match state.mean's dimension");
        }
        std::vector<double> x(n);
        for (size_t j = 0; j < n; ++j) {
            x[j] = state.mean[j] + state.sigma * std::sqrt(state.variances[j]) * z[j];
        }
        offspring.push_back(std::move(x));
    }
    return offspring;
}

/**
 * @brief Pure core: given the offspring AskGivenSamples produced (same order), their
 *        maximization-convention fitness values, and the z_samples that produced them,
 *        returns the next generation's state.
 * @param step_size_learning_rate,scale_learning_rate Fixed adaptation rates (this variant's
 *        own simplification -- full CMA-ES derives these from mu_eff/n instead).
 * @throws std::invalid_argument if z_samples/offspring/fitness sizes disagree, or fewer than
 *         2 samples are given (mu = size/2 must be >= 1).
 */
inline CMAESState TellGivenSamples(const CMAESState& state, const std::vector<std::vector<double>>& z_samples,
                                    const std::vector<std::vector<double>>& offspring,
                                    const std::vector<double>& fitness, double step_size_learning_rate,
                                    double scale_learning_rate) {
    if (z_samples.size() != offspring.size() || z_samples.size() != fitness.size()) {
        throw std::invalid_argument("TellGivenSamples: z_samples, offspring, and fitness must be the same size");
    }
    if (z_samples.size() < 2) {
        throw std::invalid_argument("TellGivenSamples: need at least 2 samples");
    }

    const size_t n = state.mean.size();
    const size_t lambda = z_samples.size();
    const size_t mu = std::max<size_t>(1, lambda / 2);

    std::vector<size_t> order(lambda);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return fitness[a] > fitness[b]; });

    std::vector<double> y(n, 0.0);          // mean of the top-mu displacements y_i = sqrt(var)*z_i
    std::vector<double> y_squared(n, 0.0);  // mean of the top-mu displacements' squares
    for (size_t rank = 0; rank < mu; ++rank) {
        const auto& z = z_samples[order[rank]];
        for (size_t j = 0; j < n; ++j) {
            double y_ij = std::sqrt(state.variances[j]) * z[j];
            y[j] += y_ij;
            y_squared[j] += y_ij * y_ij;
        }
    }
    for (size_t j = 0; j < n; ++j) {
        y[j] /= static_cast<double>(mu);
        y_squared[j] /= static_cast<double>(mu);
    }

    CMAESState next;
    next.mean.resize(n);
    for (size_t j = 0; j < n; ++j) {
        next.mean[j] = state.mean[j] + state.sigma * y[j];
    }

    double y_norm = 0.0;
    for (size_t j = 0; j < n; ++j) {
        y_norm += y[j] * y[j];
    }
    y_norm = std::sqrt(y_norm);
    double expected_norm = std::sqrt(static_cast<double>(n));
    next.sigma = state.sigma * std::exp(step_size_learning_rate * (y_norm / expected_norm - 1.0));

    next.variances.resize(n);
    for (size_t j = 0; j < n; ++j) {
        next.variances[j] = (1.0 - scale_learning_rate) * state.variances[j] + scale_learning_rate * y_squared[j];
    }

    return next;
}

/**
 * @brief RNG-driven ask-tell wrapper: caches the z-samples an Ask() call draws so a matching
 *        Tell() call can reuse them without the caller needing to track them.
 */
class CMAES {
public:
    /**
     * @throws std::invalid_argument if initial_mean is empty, initial_sigma <= 0, or
     *         lambda < 2.
     */
    CMAES(std::vector<double> initial_mean, double initial_sigma, size_t lambda,
          double step_size_learning_rate = 0.3, double scale_learning_rate = 0.3)
        : lambda_(lambda),
          step_size_learning_rate_(step_size_learning_rate),
          scale_learning_rate_(scale_learning_rate) {
        if (initial_mean.empty()) {
            throw std::invalid_argument("CMAES: initial_mean must not be empty");
        }
        if (initial_sigma <= 0.0) {
            throw std::invalid_argument("CMAES: initial_sigma must be positive");
        }
        if (lambda < 2) {
            throw std::invalid_argument("CMAES: lambda must be >= 2");
        }
        state_.mean = std::move(initial_mean);
        state_.sigma = initial_sigma;
        state_.variances.assign(state_.mean.size(), 1.0);
    }

    /** @brief Draws lambda fresh offspring from the current state, caching the standard-
     *         normal samples used for the next Tell() call. */
    template <typename RNG>
    std::vector<std::vector<double>> Ask(RNG& rng) {
        std::normal_distribution<double> dist(0.0, 1.0);
        last_z_samples_.assign(lambda_, std::vector<double>(state_.mean.size()));
        for (auto& z : last_z_samples_) {
            for (double& v : z) {
                v = dist(rng);
            }
        }
        return AskGivenSamples(state_, last_z_samples_);
    }

    /**
     * @brief Updates the internal state from the most recent Ask() call's offspring and their
     *        fitness values (maximization convention).
     * @throws std::invalid_argument if Ask() was never called, or offspring/fitness don't
     *         match the cached sample count -- see TellGivenSamples.
     */
    void Tell(const std::vector<std::vector<double>>& offspring, const std::vector<double>& fitness) {
        if (last_z_samples_.empty()) {
            throw std::invalid_argument("CMAES::Tell: called before any Ask()");
        }
        state_ = TellGivenSamples(state_, last_z_samples_, offspring, fitness, step_size_learning_rate_,
                                   scale_learning_rate_);
    }

    [[nodiscard]] const std::vector<double>& mean() const { return state_.mean; }
    [[nodiscard]] double sigma() const { return state_.sigma; }
    [[nodiscard]] const std::vector<double>& variances() const { return state_.variances; }

private:
    CMAESState state_;
    size_t lambda_;
    double step_size_learning_rate_;
    double scale_learning_rate_;
    std::vector<std::vector<double>> last_z_samples_;
};

}  // namespace pulsatrix
