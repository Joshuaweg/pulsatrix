/** @file evolution_strategies.hpp
 *  @brief Evolution Strategies (Salimans et al. 2017, "Evolution Strategies as a Scalable
 *         Alternative to Reinforcement Learning"): a black-box, gradient-free optimizer over an
 *         arbitrary flat parameter vector, driven entirely by a scalar fitness function --
 *         zero inherent RL dependency, applicable to any fixed-topology network's flattened
 *         weight vector or any other real-valued parameterization.
 *  @ingroup evolutionary
 *  @note Uses mirrored (antithetic) sampling -- every sampled perturbation `epsilon` is scored
 *        alongside its negation `-epsilon` -- matching Salimans et al.'s own standard
 *        variance-reduction technique, not an invented addition. This is why population_size
 *        must be even.
 */
#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsatrix {

/**
 * @brief Pure core: the exact ES parameter update given already-sampled perturbations and
 *        their fitness scores -- `theta' = theta + (alpha / (N*sigma)) * sum_i(F_i * epsilon_i)`.
 * @throws std::invalid_argument if epsilons is empty, epsilons.size() != fitnesses.size(), any
 *         epsilon's dimension doesn't match theta's, or sigma is not positive.
 */
inline std::vector<double> ESUpdateGivenPerturbations(const std::vector<double>& theta,
                                                       const std::vector<std::vector<double>>& epsilons,
                                                       const std::vector<double>& fitnesses, double alpha,
                                                       double sigma) {
    if (epsilons.empty()) {
        throw std::invalid_argument("ESUpdateGivenPerturbations: epsilons must not be empty");
    }
    if (epsilons.size() != fitnesses.size()) {
        throw std::invalid_argument("ESUpdateGivenPerturbations: epsilons and fitnesses must have the same size");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("ESUpdateGivenPerturbations: sigma must be positive");
    }
    size_t dim = theta.size();
    for (const auto& eps : epsilons) {
        if (eps.size() != dim) {
            throw std::invalid_argument("ESUpdateGivenPerturbations: every epsilon must match theta's dimension");
        }
    }

    std::vector<double> gradient_estimate(dim, 0.0);
    for (size_t i = 0; i < epsilons.size(); ++i) {
        for (size_t d = 0; d < dim; ++d) {
            gradient_estimate[d] += fitnesses[i] * epsilons[i][d];
        }
    }

    double scale = alpha / (static_cast<double>(epsilons.size()) * sigma);
    std::vector<double> updated(dim);
    for (size_t d = 0; d < dim; ++d) {
        updated[d] = theta[d] + scale * gradient_estimate[d];
    }
    return updated;
}

/**
 * @brief RNG-driven wrapper: samples population_size/2 standard-normal perturbation vectors,
 *        scores theta+sigma*epsilon and theta-sigma*epsilon for each (mirrored sampling), and
 *        returns the updated theta via ESUpdateGivenPerturbations.
 * @throws std::invalid_argument if population_size is not a positive even number, or sigma is
 *         not positive.
 */
template <typename FitnessFn, typename RNG>
std::vector<double> ESStep(const std::vector<double>& theta, FitnessFn fitness_fn, int population_size, double sigma,
                            double alpha, RNG& rng) {
    if (population_size <= 0 || population_size % 2 != 0) {
        throw std::invalid_argument("ESStep: population_size must be a positive even number (mirrored sampling)");
    }
    if (sigma <= 0.0) {
        throw std::invalid_argument("ESStep: sigma must be positive");
    }

    size_t dim = theta.size();
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<std::vector<double>> epsilons;
    std::vector<double> fitnesses;
    epsilons.reserve(static_cast<size_t>(population_size));
    fitnesses.reserve(static_cast<size_t>(population_size));

    for (int i = 0; i < population_size / 2; ++i) {
        std::vector<double> eps(dim);
        for (size_t d = 0; d < dim; ++d) {
            eps[d] = noise(rng);
        }
        std::vector<double> neg_eps(dim);
        for (size_t d = 0; d < dim; ++d) {
            neg_eps[d] = -eps[d];
        }

        std::vector<double> theta_plus(dim);
        std::vector<double> theta_minus(dim);
        for (size_t d = 0; d < dim; ++d) {
            theta_plus[d] = theta[d] + sigma * eps[d];
            theta_minus[d] = theta[d] + sigma * neg_eps[d];
        }

        epsilons.push_back(eps);
        fitnesses.push_back(fitness_fn(theta_plus));
        epsilons.push_back(std::move(neg_eps));
        fitnesses.push_back(fitness_fn(theta_minus));
    }

    return ESUpdateGivenPerturbations(theta, epsilons, fitnesses, alpha, sigma);
}

/** @brief Result of a full Evolution Strategies run. */
struct ESResult {
    std::vector<double> best_theta;
    double best_fitness;
    int iterations_run;
};

/**
 * @brief Runs num_iterations of Evolution Strategies starting from theta, tracking the best
 *        (theta, fitness) pair seen across every evaluated center point (global elitism, the
 *        same precedent this campaign's NEAT evolutionary loop already established) -- vanilla
 *        ES itself has no such tracking, but a returnable "best point found" is genuinely
 *        necessary for this to be usable as an optimizer, so it is added here as a small,
 *        logged addition beyond the paper's own bare update rule.
 * @throws std::invalid_argument if theta is empty or num_iterations is not positive (ESStep's
 *         own guards apply to population_size/sigma).
 */
template <typename FitnessFn, typename RNG>
ESResult RunEvolutionStrategies(std::vector<double> theta, FitnessFn fitness_fn, int num_iterations,
                                 int population_size, double sigma, double alpha, RNG& rng) {
    if (theta.empty()) {
        throw std::invalid_argument("RunEvolutionStrategies: initial theta must not be empty");
    }
    if (num_iterations <= 0) {
        throw std::invalid_argument("RunEvolutionStrategies: num_iterations must be positive");
    }

    std::vector<double> best_theta = theta;
    double best_fitness = fitness_fn(theta);

    for (int iter = 0; iter < num_iterations; ++iter) {
        double current_fitness = fitness_fn(theta);
        if (current_fitness > best_fitness) {
            best_fitness = current_fitness;
            best_theta = theta;
        }
        theta = ESStep(theta, fitness_fn, population_size, sigma, alpha, rng);
    }

    double final_fitness = fitness_fn(theta);
    if (final_fitness > best_fitness) {
        best_fitness = final_fitness;
        best_theta = std::move(theta);
    }

    return ESResult{std::move(best_theta), best_fitness, num_iterations};
}

}  // namespace pulsatrix
