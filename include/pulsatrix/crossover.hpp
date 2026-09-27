/** @file crossover.hpp
 *  @brief Genetic-algorithm crossover operators: one-point, two-point, uniform (generic
 *         sequence genotypes), and blend/BLX-alpha, simulated binary (SBX) (real-valued
 *         genotypes).
 *  @ingroup evolutionary
 *  @note Every stochastic operator splits into a pure, deterministic core taking an explicit
 *        crossover point/mask/gamma/draw, plus a thin RNG-driven wrapper -- the same pattern
 *        selection.hpp established, for the same reason: hand-derivable/closed-form
 *        correctness tests independent of any particular RNG's output stream.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsatrix {

// --- One-point crossover ---

/**
 * @brief Splits both parents at point and swaps tails.
 * @throws std::invalid_argument if parent sizes differ or point > parent1.size().
 * @note point == 0 or point == size() are valid (degenerate, fully-swapped/no-swap) cases --
 *       this pure core does not restrict to "interior" points; OnePointCrossover (the RNG
 *       wrapper, below) does, since a degenerate point is never a useful recombination.
 */
template <typename T>
std::pair<std::vector<T>, std::vector<T>> OnePointCrossoverAtPoint(
    const std::vector<T>& parent1, const std::vector<T>& parent2, size_t point) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("OnePointCrossoverAtPoint: parent sizes must match");
    }
    if (point > parent1.size()) {
        throw std::invalid_argument("OnePointCrossoverAtPoint: point must be <= parent size");
    }
    std::vector<T> child1(parent1.begin(), parent1.begin() + static_cast<std::ptrdiff_t>(point));
    child1.insert(child1.end(), parent2.begin() + static_cast<std::ptrdiff_t>(point), parent2.end());
    std::vector<T> child2(parent2.begin(), parent2.begin() + static_cast<std::ptrdiff_t>(point));
    child2.insert(child2.end(), parent1.begin() + static_cast<std::ptrdiff_t>(point), parent1.end());
    return {child1, child2};
}

/**
 * @brief RNG-driven wrapper: draws an interior point in [1, size-1] uniformly.
 * @throws std::invalid_argument if parent sizes differ or parent size < 2 (no interior point
 *         exists to draw).
 */
template <typename T, typename RNG>
std::pair<std::vector<T>, std::vector<T>> OnePointCrossover(const std::vector<T>& parent1,
                                                             const std::vector<T>& parent2,
                                                             RNG& rng) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("OnePointCrossover: parent sizes must match");
    }
    if (parent1.size() < 2) {
        throw std::invalid_argument("OnePointCrossover: parent size must be >= 2");
    }
    std::uniform_int_distribution<size_t> dist(1, parent1.size() - 1);
    return OnePointCrossoverAtPoint(parent1, parent2, dist(rng));
}

// --- Two-point crossover ---

/**
 * @brief Swaps the [point1, point2) segment between both parents.
 * @throws std::invalid_argument if parent sizes differ, point1 > point2, or point2 exceeds
 *         parent size.
 */
template <typename T>
std::pair<std::vector<T>, std::vector<T>> TwoPointCrossoverAtPoints(
    const std::vector<T>& parent1, const std::vector<T>& parent2, size_t point1, size_t point2) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("TwoPointCrossoverAtPoints: parent sizes must match");
    }
    if (point1 > point2 || point2 > parent1.size()) {
        throw std::invalid_argument(
            "TwoPointCrossoverAtPoints: require point1 <= point2 <= parent size");
    }
    std::vector<T> child1 = parent1;
    std::vector<T> child2 = parent2;
    for (size_t i = point1; i < point2; ++i) {
        std::swap(child1[i], child2[i]);
    }
    return {child1, child2};
}

/**
 * @brief RNG-driven wrapper: draws two points in [0, size], sorted ascending.
 * @throws std::invalid_argument if parent sizes differ or parent size < 2.
 */
template <typename T, typename RNG>
std::pair<std::vector<T>, std::vector<T>> TwoPointCrossover(const std::vector<T>& parent1,
                                                             const std::vector<T>& parent2,
                                                             RNG& rng) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("TwoPointCrossover: parent sizes must match");
    }
    if (parent1.size() < 2) {
        throw std::invalid_argument("TwoPointCrossover: parent size must be >= 2");
    }
    std::uniform_int_distribution<size_t> dist(0, parent1.size());
    size_t a = dist(rng);
    size_t b = dist(rng);
    if (a > b) {
        std::swap(a, b);
    }
    return TwoPointCrossoverAtPoints(parent1, parent2, a, b);
}

// --- Uniform crossover ---

/**
 * @brief Swaps each gene independently wherever swap_mask is true.
 * @throws std::invalid_argument if parent1/parent2/swap_mask sizes are not all equal.
 */
template <typename T>
std::pair<std::vector<T>, std::vector<T>> UniformCrossoverByMask(
    const std::vector<T>& parent1, const std::vector<T>& parent2,
    const std::vector<bool>& swap_mask) {
    if (parent1.size() != parent2.size() || parent1.size() != swap_mask.size()) {
        throw std::invalid_argument("UniformCrossoverByMask: sizes must all match");
    }
    std::vector<T> child1 = parent1;
    std::vector<T> child2 = parent2;
    for (size_t i = 0; i < swap_mask.size(); ++i) {
        if (swap_mask[i]) {
            std::swap(child1[i], child2[i]);
        }
    }
    return {child1, child2};
}

/**
 * @brief RNG-driven wrapper: each gene swaps independently with probability swap_probability.
 * @throws std::invalid_argument if parent sizes differ or swap_probability is outside [0, 1].
 * @note swap_probability == 0.0 or 1.0 make std::bernoulli_distribution deterministic (always
 *       false / always true respectively) -- exact, RNG-independent correctness cases.
 */
template <typename T, typename RNG>
std::pair<std::vector<T>, std::vector<T>> UniformCrossover(const std::vector<T>& parent1,
                                                            const std::vector<T>& parent2,
                                                            double swap_probability, RNG& rng) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("UniformCrossover: parent sizes must match");
    }
    if (swap_probability < 0.0 || swap_probability > 1.0) {
        throw std::invalid_argument("UniformCrossover: swap_probability must be in [0, 1]");
    }
    std::bernoulli_distribution dist(swap_probability);
    std::vector<bool> mask(parent1.size());
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = dist(rng);
    }
    return UniformCrossoverByMask(parent1, parent2, mask);
}

// --- Blend crossover (BLX-alpha) ---

/**
 * @brief Blends each gene pair via an explicit per-gene gamma: child1_i = (1-gamma_i)*x1_i +
 *        gamma_i*x2_i, child2_i = gamma_i*x1_i + (1-gamma_i)*x2_i.
 * @throws std::invalid_argument if parent1/parent2/gamma sizes are not all equal.
 * @note child1_i + child2_i == parent1_i + parent2_i for any gamma_i -- an algebraic
 *       invariant, not just a property at specific gamma values, useful for testing the
 *       RNG-driven wrapper without controlling its draws directly.
 */
inline std::pair<std::vector<double>, std::vector<double>> BlendCrossoverByGamma(
    const std::vector<double>& parent1, const std::vector<double>& parent2,
    const std::vector<double>& gamma) {
    if (parent1.size() != parent2.size() || parent1.size() != gamma.size()) {
        throw std::invalid_argument("BlendCrossoverByGamma: sizes must all match");
    }
    std::vector<double> child1(parent1.size());
    std::vector<double> child2(parent1.size());
    for (size_t i = 0; i < parent1.size(); ++i) {
        child1[i] = (1.0 - gamma[i]) * parent1[i] + gamma[i] * parent2[i];
        child2[i] = gamma[i] * parent1[i] + (1.0 - gamma[i]) * parent2[i];
    }
    return {child1, child2};
}

/**
 * @brief RNG-driven wrapper (DEAP's cxBlend): draws gamma_i = (1+2*alpha)*u_i - alpha per
 *        gene, u_i ~ Uniform(0, 1).
 * @throws std::invalid_argument if parent sizes differ or alpha < 0.
 */
template <typename RNG>
std::pair<std::vector<double>, std::vector<double>> BlendCrossover(
    const std::vector<double>& parent1, const std::vector<double>& parent2, double alpha,
    RNG& rng) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("BlendCrossover: parent sizes must match");
    }
    if (alpha < 0.0) {
        throw std::invalid_argument("BlendCrossover: alpha must be non-negative");
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> gamma(parent1.size());
    for (size_t i = 0; i < gamma.size(); ++i) {
        gamma[i] = (1.0 + 2.0 * alpha) * dist(rng) - alpha;
    }
    return BlendCrossoverByGamma(parent1, parent2, gamma);
}

// --- Simulated binary crossover (SBX) ---

/**
 * @brief Simulated binary crossover (Deb & Agrawal 1995; DEAP's cxSimulatedBinary), given an
 *        explicit per-gene draw in [0, 1).
 * @throws std::invalid_argument if parent1/parent2/draws sizes are not all equal, eta is
 *         negative, or any draw is outside [0, 1).
 * @note draws must exclude 1.0: the u > 0.5 branch divides by (1 - u), which is exactly zero
 *       at u == 1 -- excluded by validation rather than silently producing inf/nan.
 * @note child1_i + child2_i == parent1_i + parent2_i for any beta_q (and therefore any eta,
 *       draw) -- an algebraic invariant independent of the specific draw, useful for testing
 *       the RNG-driven wrapper.
 */
inline std::pair<std::vector<double>, std::vector<double>> SimulatedBinaryCrossoverByDraw(
    const std::vector<double>& parent1, const std::vector<double>& parent2, double eta,
    const std::vector<double>& draws) {
    if (parent1.size() != parent2.size() || parent1.size() != draws.size()) {
        throw std::invalid_argument("SimulatedBinaryCrossoverByDraw: sizes must all match");
    }
    if (eta < 0.0) {
        throw std::invalid_argument("SimulatedBinaryCrossoverByDraw: eta must be non-negative");
    }
    for (double u : draws) {
        if (u < 0.0 || u >= 1.0) {
            throw std::invalid_argument(
                "SimulatedBinaryCrossoverByDraw: every draw must be in [0, 1)");
        }
    }

    std::vector<double> child1(parent1.size());
    std::vector<double> child2(parent1.size());
    const double exponent = 1.0 / (eta + 1.0);
    for (size_t i = 0; i < parent1.size(); ++i) {
        double u = draws[i];
        double beta_q = (u <= 0.5) ? std::pow(2.0 * u, exponent)
                                    : std::pow(1.0 / (2.0 * (1.0 - u)), exponent);
        child1[i] = 0.5 * ((1.0 + beta_q) * parent1[i] + (1.0 - beta_q) * parent2[i]);
        child2[i] = 0.5 * ((1.0 - beta_q) * parent1[i] + (1.0 + beta_q) * parent2[i]);
    }
    return {child1, child2};
}

/**
 * @brief RNG-driven wrapper: draws u_i ~ Uniform(0, 1) per gene (std::uniform_real_distribution
 *        is documented to produce values in [0, 1), matching the pure core's requirement).
 * @throws std::invalid_argument -- see SimulatedBinaryCrossoverByDraw.
 */
template <typename RNG>
std::pair<std::vector<double>, std::vector<double>> SimulatedBinaryCrossover(
    const std::vector<double>& parent1, const std::vector<double>& parent2, double eta,
    RNG& rng) {
    if (parent1.size() != parent2.size()) {
        throw std::invalid_argument("SimulatedBinaryCrossover: parent sizes must match");
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> draws(parent1.size());
    for (size_t i = 0; i < draws.size(); ++i) {
        draws[i] = dist(rng);
    }
    return SimulatedBinaryCrossoverByDraw(parent1, parent2, eta, draws);
}

}  // namespace pulsatrix
