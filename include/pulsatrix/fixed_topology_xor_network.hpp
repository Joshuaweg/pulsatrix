/** @file fixed_topology_xor_network.hpp
 *  @brief A small, fixed-topology (2 input -> 2 hidden -> 1 output, both layers sigmoid) MLP
 *         whose weights are a flat parameter vector, plus a plain analytic fitness function
 *         scoring it against the four XOR patterns -- the "fixed-topology network" and "plain
 *         analytic fitness function" this campaign's Evolution Strategies mission needs,
 *         deliberately using the same XOR benchmark and the same fitness convention (4.0 minus
 *         sum of squared error) as Mission 2's NEAT proof, for a direct same-problem
 *         cross-check between the two techniques.
 *  @ingroup evolutionary
 *  @note Not a Module -- ES needs no gradient/backward pass at all (it is itself a
 *        gradient-free, black-box optimizer), so there is nothing here for
 *        ComputationGraph/Autograd/LRP to attach to; this mirrors NEAT's own phenotype's
 *        already-logged Risk Register scope cut for the same underlying reason.
 */
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pulsatrix {

/** @brief Total flat-parameter count: 2*2 (input->hidden weights) + 2 (hidden biases) + 2
 *         (hidden->output weights) + 1 (output bias) = 9. */
inline constexpr size_t kFixedTopologyXORNumParams = 9;

namespace detail {
inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace detail

/**
 * @brief Forward pass: theta layout is
 *        [w1_00, w1_01, b1_0, w1_10, w1_11, b1_1, w2_0, w2_1, b2] -- `h_j =
 *        sigmoid(w1_j0*x0 + w1_j1*x1 + b1_j)` for j in {0,1}, `y = sigmoid(w2_0*h0 + w2_1*h1 +
 *        b2)`.
 * @throws std::invalid_argument if theta.size() != kFixedTopologyXORNumParams.
 */
inline double FixedTopologyXORForward(const std::vector<double>& theta, const std::array<double, 2>& inputs) {
    if (theta.size() != kFixedTopologyXORNumParams) {
        throw std::invalid_argument("FixedTopologyXORForward: theta must have exactly kFixedTopologyXORNumParams entries");
    }
    double h0 = detail::Sigmoid(theta[0] * inputs[0] + theta[1] * inputs[1] + theta[2]);
    double h1 = detail::Sigmoid(theta[3] * inputs[0] + theta[4] * inputs[1] + theta[5]);
    return detail::Sigmoid(theta[6] * h0 + theta[7] * h1 + theta[8]);
}

/**
 * @brief Scores theta against all four XOR patterns as 4.0 minus the sum of squared errors --
 *        identical convention to XORFitness (neat_xor_fitness.hpp), so an all-zero theta
 *        scores exactly 3.0 (every pattern outputs sigmoid(0)=0.5), the same fixed point NEAT's
 *        own fresh, all-zero-weight genome scores.
 */
inline double FixedTopologyXORFitness(const std::vector<double>& theta) {
    static const std::array<std::array<double, 2>, 4> kInputs{
        {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}};
    static const std::array<double, 4> kExpected{0.0, 1.0, 1.0, 0.0};

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < kInputs.size(); ++i) {
        double actual = FixedTopologyXORForward(theta, kInputs[i]);
        double error = kExpected[i] - actual;
        sum_squared_error += error * error;
    }
    return 4.0 - sum_squared_error;
}

}  // namespace pulsatrix
