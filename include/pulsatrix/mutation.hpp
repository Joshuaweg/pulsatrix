/** @file mutation.hpp
 *  @brief Genetic-algorithm mutation operators: bit-flip (generic boolean genotypes) and
 *         Gaussian, polynomial (real-valued genotypes).
 *  @ingroup evolutionary
 *  @note Same pure-core/RNG-wrapper split as selection.hpp and crossover.hpp: every
 *        stochastic operator's pure core takes an explicit mask/noise/draw, making its
 *        correctness hand-derivable independent of any particular RNG's output stream.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsatrix {

// --- Bit-flip mutation ---

/**
 * @brief Flips each gene where flip_mask is true, leaves the rest unchanged.
 * @throws std::invalid_argument if genotype and flip_mask sizes differ.
 */
inline std::vector<bool> BitFlipMutationByMask(const std::vector<bool>& genotype,
                                                const std::vector<bool>& flip_mask) {
    if (genotype.size() != flip_mask.size()) {
        throw std::invalid_argument("BitFlipMutationByMask: sizes must match");
    }
    std::vector<bool> result = genotype;
    for (size_t i = 0; i < flip_mask.size(); ++i) {
        if (flip_mask[i]) {
            result[i] = !result[i];
        }
    }
    return result;
}

/**
 * @brief RNG-driven wrapper: each gene flips independently with probability mutation_probability.
 * @throws std::invalid_argument if mutation_probability is outside [0, 1].
 * @note mutation_probability == 0.0 or 1.0 make std::bernoulli_distribution deterministic
 *       (never / always flips) -- exact, RNG-independent correctness cases.
 */
template <typename RNG>
std::vector<bool> BitFlipMutation(const std::vector<bool>& genotype,
                                   double mutation_probability, RNG& rng) {
    if (mutation_probability < 0.0 || mutation_probability > 1.0) {
        throw std::invalid_argument("BitFlipMutation: mutation_probability must be in [0, 1]");
    }
    std::bernoulli_distribution dist(mutation_probability);
    std::vector<bool> mask(genotype.size());
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = dist(rng);
    }
    return BitFlipMutationByMask(genotype, mask);
}

// --- Gaussian mutation ---

/**
 * @brief Adds noise[i] to genotype[i] wherever apply_mask[i] is true, leaves the rest
 *        unchanged.
 * @throws std::invalid_argument if genotype/noise/apply_mask sizes are not all equal.
 */
inline std::vector<double> GaussianMutationByNoise(const std::vector<double>& genotype,
                                                    const std::vector<double>& noise,
                                                    const std::vector<bool>& apply_mask) {
    if (genotype.size() != noise.size() || genotype.size() != apply_mask.size()) {
        throw std::invalid_argument("GaussianMutationByNoise: sizes must all match");
    }
    std::vector<double> result(genotype.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = apply_mask[i] ? genotype[i] + noise[i] : genotype[i];
    }
    return result;
}

/**
 * @brief RNG-driven wrapper: each gene independently receives N(0, sigma^2) noise with
 *        probability mutation_probability.
 * @throws std::invalid_argument if sigma is negative or mutation_probability is outside
 *         [0, 1].
 * @note mutation_probability == 0.0 leaves the genotype unchanged exactly (mask always
 *       false); sigma == 0.0 with mutation_probability == 1.0 also leaves it unchanged
 *       exactly (every draw from N(0, 0) is exactly 0.0) -- two independent, RNG-stream-
 *       -independent correctness cases.
 */
template <typename RNG>
std::vector<double> GaussianMutation(const std::vector<double>& genotype, double sigma,
                                      double mutation_probability, RNG& rng) {
    if (sigma < 0.0) {
        throw std::invalid_argument("GaussianMutation: sigma must be non-negative");
    }
    if (mutation_probability < 0.0 || mutation_probability > 1.0) {
        throw std::invalid_argument("GaussianMutation: mutation_probability must be in [0, 1]");
    }
    std::bernoulli_distribution mask_dist(mutation_probability);
    // Standard normal draws scaled by sigma: std::normal_distribution requires a positive
    // stddev, and z * sigma is the same value it would draw for any sigma > 0.
    std::normal_distribution<double> noise_dist(0.0, 1.0);
    std::vector<bool> mask(genotype.size());
    std::vector<double> noise(genotype.size());
    for (size_t i = 0; i < genotype.size(); ++i) {
        mask[i] = mask_dist(rng);
        noise[i] = noise_dist(rng) * sigma;
    }
    return GaussianMutationByNoise(genotype, noise, mask);
}

// --- Polynomial mutation (Deb & Goyal 1996; DEAP's mutPolynomialBounded) ---

/**
 * @brief Polynomial-mutates a single bounded gene given an explicit draw.
 * @param x Current gene value, must be in [lower, upper].
 * @param lower Lower bound (must be strictly less than upper).
 * @param upper Upper bound.
 * @param eta Distribution index (non-negative; larger values concentrate mutations closer
 *        to x).
 * @param u Draw in [0, 1].
 * @return The mutated, bound-clipped gene value.
 * @throws std::invalid_argument if lower >= upper, x is outside [lower, upper], eta is
 *         negative, or u is outside [0, 1].
 * @note Three draw values give exact, eta-independent results, all algebraically derivable
 *       from the formula: u == 0 -> exactly lower; u == 0.5 -> exactly x (unchanged);
 *       u == 1 -> exactly upper. Unlike SimulatedBinaryCrossoverByDraw, this formula has no
 *       1/(1-u) term, so u == 1 is a valid, safe draw here (not excluded).
 */
inline double PolynomialMutationByDraw(double x, double lower, double upper, double eta,
                                        double u) {
    if (!(lower < upper)) {
        throw std::invalid_argument("PolynomialMutationByDraw: lower must be < upper");
    }
    if (x < lower || x > upper) {
        throw std::invalid_argument("PolynomialMutationByDraw: x must be in [lower, upper]");
    }
    if (eta < 0.0) {
        throw std::invalid_argument("PolynomialMutationByDraw: eta must be non-negative");
    }
    if (u < 0.0 || u > 1.0) {
        throw std::invalid_argument("PolynomialMutationByDraw: u must be in [0, 1]");
    }

    const double range = upper - lower;
    const double delta1 = (x - lower) / range;
    const double delta2 = (upper - x) / range;
    const double mut_pow = 1.0 / (eta + 1.0);
    double delta_q;
    if (u <= 0.5) {
        double xy = 1.0 - delta1;
        double val = 2.0 * u + (1.0 - 2.0 * u) * std::pow(xy, eta + 1.0);
        delta_q = std::pow(val, mut_pow) - 1.0;
    } else {
        double xy = 1.0 - delta2;
        double val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * std::pow(xy, eta + 1.0);
        delta_q = 1.0 - std::pow(val, mut_pow);
    }
    double result = x + delta_q * range;
    return std::min(std::max(result, lower), upper);
}

/**
 * @brief RNG-driven wrapper: each gene independently mutates (via PolynomialMutationByDraw)
 *        with probability mutation_probability.
 * @throws std::invalid_argument if genotype/lower_bounds/upper_bounds sizes are not all
 *         equal, or mutation_probability is outside [0, 1] -- per-gene bound/eta validity is
 *         enforced by PolynomialMutationByDraw itself.
 */
template <typename RNG>
std::vector<double> PolynomialMutation(const std::vector<double>& genotype,
                                        const std::vector<double>& lower_bounds,
                                        const std::vector<double>& upper_bounds, double eta,
                                        double mutation_probability, RNG& rng) {
    if (genotype.size() != lower_bounds.size() || genotype.size() != upper_bounds.size()) {
        throw std::invalid_argument("PolynomialMutation: sizes must all match");
    }
    if (mutation_probability < 0.0 || mutation_probability > 1.0) {
        throw std::invalid_argument("PolynomialMutation: mutation_probability must be in [0, 1]");
    }
    std::bernoulli_distribution mask_dist(mutation_probability);
    std::uniform_real_distribution<double> draw_dist(0.0, 1.0);
    std::vector<double> result(genotype.size());
    for (size_t i = 0; i < genotype.size(); ++i) {
        if (mask_dist(rng)) {
            result[i] = PolynomialMutationByDraw(genotype[i], lower_bounds[i], upper_bounds[i],
                                                  eta, draw_dist(rng));
        } else {
            result[i] = genotype[i];
        }
    }
    return result;
}

}  // namespace pulsatrix
