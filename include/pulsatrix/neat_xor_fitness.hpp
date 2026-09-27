/** @file neat_xor_fitness.hpp
 *  @brief XOR fitness function for NEAT -- Stanley & Miikkulainen 2002's own validation task,
 *         chosen here for the same reason it was chosen there: XOR is the canonical
 *         not-linearly-separable problem, so a genome that solves it starting from a fully
 *         connected (no-hidden-node) minimal topology has genuinely grown new structure to do
 *         so, not merely tuned weights on an already-sufficient network shape.
 *  @ingroup evolutionary
 */
#pragma once

#include <array>
#include <vector>

#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_phenotype.hpp"

namespace pulsatrix {

/**
 * @brief Evaluates genome's phenotype on all four XOR patterns ((0,0)->0, (0,1)->1, (1,0)->1,
 *        (1,1)->0, in that order) and scores it as 4.0 minus the sum of squared errors -- a
 *        perfect fit scores 4.0; a genome producing exactly 0.5 for every pattern (e.g. a
 *        fresh, all-zero-weight genome, before any weight differentiation has emerged) scores
 *        exactly 3.0 (4.0 - 4*0.25).
 * @throws Whatever EvaluateNEATPhenotype throws if genome's Input node count isn't 2.
 */
inline double XORFitness(const NEATGenome& genome) {
    static const std::array<std::array<double, 2>, 4> kInputs{
        {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}};
    static const std::array<double, 4> kExpected{0.0, 1.0, 1.0, 0.0};

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < kInputs.size(); ++i) {
        std::vector<double> inputs(kInputs[i].begin(), kInputs[i].end());
        double actual = EvaluateNEATPhenotype(genome, inputs)[0];
        double error = kExpected[i] - actual;
        sum_squared_error += error * error;
    }
    return 4.0 - sum_squared_error;
}

}  // namespace pulsatrix
