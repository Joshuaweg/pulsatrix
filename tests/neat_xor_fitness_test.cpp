#include <gtest/gtest.h>

#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_xor_fitness.hpp"

namespace pulsatrix {
namespace {

TEST(NEATXORFitnessTest, FreshAllZeroWeightGenomeScoresExactlyThree) {
    InnovationTracker tracker(4);
    NEATGenome genome(2, 1, /*has_bias=*/true, tracker);
    // Every connection weight is 0.0 (default), so every pattern's output is sigmoid(0) = 0.5.
    // Sum of squared errors = (0-0.5)^2 + (1-0.5)^2 + (1-0.5)^2 + (0-0.5)^2 = 4*0.25 = 1.0.

    double fitness = XORFitness(genome);

    EXPECT_NEAR(fitness, 3.0, 1e-12);
}

TEST(NEATXORFitnessTest, ChangingWeightsChangesFitness) {
    InnovationTracker tracker(4);
    NEATGenome genome(2, 1, /*has_bias=*/true, tracker);
    genome.SetConnectionWeight(0, 5.0);
    genome.SetConnectionWeight(1, 5.0);
    genome.SetConnectionWeight(2, -2.5);

    double fitness = XORFitness(genome);

    EXPECT_NE(fitness, 3.0);
}

}  // namespace
}  // namespace pulsatrix
