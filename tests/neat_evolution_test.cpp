#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/neat_evolution.hpp"
#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_phenotype.hpp"
#include "pulsatrix/neat_xor_fitness.hpp"

namespace pulsatrix {
namespace {

TEST(AllocateOffspringCountsTest, DistributesProportionallyWithLargestRemainderTieBreakByIndex) {
    std::vector<double> sums{10.0, 5.0, 5.0};

    std::vector<int> counts = AllocateOffspringCounts(sums, 10);

    // exact = [5.0, 2.5, 2.5]; floors = [5, 2, 2], allocated=9, remaining=1; tie broken toward
    // the earliest-index species among the equal .5 fractional remainders (index 1).
    EXPECT_EQ(counts, (std::vector<int>{5, 3, 2}));
}

TEST(AllocateOffspringCountsTest, FallsBackToEqualSplitWhenAllSumsNonPositive) {
    std::vector<double> sums{0.0, 0.0, 0.0};

    std::vector<int> counts = AllocateOffspringCounts(sums, 10);

    // base = 10/3 = 3, remainder = 1, given to the earliest species (index 0).
    EXPECT_EQ(counts, (std::vector<int>{4, 3, 3}));
}

TEST(AllocateOffspringCountsTest, ThrowsOnEmptySums) {
    EXPECT_THROW((void)AllocateOffspringCounts({}, 10), std::invalid_argument);
}

TEST(AllocateOffspringCountsTest, ThrowsOnNonPositivePopulationSize) {
    EXPECT_THROW((void)AllocateOffspringCounts({1.0}, 0), std::invalid_argument);
}

TEST(ReproduceOffspringTest, WeightMutationWithProbabilityOneChangesAtLeastOneWeight) {
    InnovationTracker tracker(4);
    NEATGenome parent(2, 1, /*has_bias=*/true, tracker);
    std::mt19937 rng(42);

    NEATGenome child = ReproduceOffspring(parent, tracker, rng, /*weight_mutation_sigma=*/0.5,
                                           /*weight_mutation_probability=*/1.0, /*add_connection_probability=*/0.0,
                                           /*add_node_probability=*/0.0);

    bool any_weight_changed = false;
    for (const auto& c : child.connections()) {
        if (c.weight != 0.0) {
            any_weight_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_weight_changed);
    EXPECT_EQ(child.connections().size(), parent.connections().size());
    EXPECT_EQ(child.nodes().size(), parent.nodes().size());
}

TEST(ReproduceOffspringTest, ZeroProbabilitiesLeaveChildIdenticalToParent) {
    InnovationTracker tracker(4);
    NEATGenome parent(2, 1, /*has_bias=*/true, tracker);
    std::mt19937 rng(42);

    NEATGenome child = ReproduceOffspring(parent, tracker, rng, /*weight_mutation_sigma=*/0.5,
                                           /*weight_mutation_probability=*/0.0, /*add_connection_probability=*/0.0,
                                           /*add_node_probability=*/0.0);

    ASSERT_EQ(child.connections().size(), parent.connections().size());
    for (size_t i = 0; i < child.connections().size(); ++i) {
        EXPECT_EQ(child.connections()[i].weight, parent.connections()[i].weight);
    }
}

TEST(RunNEATEvolutionTest, ThrowsOnEmptyPopulation) {
    InnovationTracker tracker(4);
    std::mt19937 rng(1);
    EXPECT_THROW((void)RunNEATEvolution(std::vector<NEATGenome>{}, XORFitness, 5, 3.0, 1.0, 1.0, 0.4, 0.5, 0.8, 0.05,
                                         0.03, tracker, rng),
                 std::invalid_argument);
}

TEST(RunNEATEvolutionTest, ThrowsOnNonPositiveGenerations) {
    InnovationTracker tracker(4);
    NEATGenome genome(2, 1, /*has_bias=*/true, tracker);
    std::mt19937 rng(1);
    EXPECT_THROW((void)RunNEATEvolution(std::vector<NEATGenome>{genome}, XORFitness, 0, 3.0, 1.0, 1.0, 0.4, 0.5, 0.8,
                                         0.05, 0.03, tracker, rng),
                 std::invalid_argument);
}

TEST(RunNEATEvolutionTest, BestFitnessNeverDecreasesAcrossGenerations) {
    InnovationTracker tracker(4);
    std::vector<NEATGenome> population;
    for (int i = 0; i < 20; ++i) {
        population.emplace_back(2, 1, /*has_bias=*/true, tracker);
    }
    std::mt19937 rng(7);

    NEATEvolutionResult result =
        RunNEATEvolution(population, XORFitness, 10, /*compatibility_threshold=*/3.0, /*c1=*/1.0, /*c2=*/1.0,
                          /*c3=*/0.4, /*weight_mutation_sigma=*/0.5, /*weight_mutation_probability=*/0.8,
                          /*add_connection_probability=*/0.05, /*add_node_probability=*/0.03, tracker, rng);

    EXPECT_GE(result.best_fitness, 3.0);
    EXPECT_EQ(result.generations_run, 10);
}

// Phase 3's own exit gate: NEAT evolves a working network from minimal topology on XOR,
// matching the original 2002 paper's own validation task. Fixed seed for reproducibility, per
// this project's standing precedent for stochastic training-convergence integration tests.
TEST(NEATEvolutionXORIntegrationTest, EvolvesANetworkThatClassifiesAllFourXORPatternsCorrectly) {
    InnovationTracker tracker(4);
    std::vector<NEATGenome> population;
    for (int i = 0; i < 150; ++i) {
        population.emplace_back(2, 1, /*has_bias=*/true, tracker);
    }
    std::mt19937 rng(42);

    NEATEvolutionResult result =
        RunNEATEvolution(population, XORFitness, /*num_generations=*/200, /*compatibility_threshold=*/3.0,
                          /*c1=*/1.0, /*c2=*/1.0, /*c3=*/0.4, /*weight_mutation_sigma=*/0.5,
                          /*weight_mutation_probability=*/0.8, /*add_connection_probability=*/0.05,
                          /*add_node_probability=*/0.03, tracker, rng);

    const std::vector<std::pair<std::vector<double>, double>> patterns{
        {{0.0, 0.0}, 0.0}, {{0.0, 1.0}, 1.0}, {{1.0, 0.0}, 1.0}, {{1.0, 1.0}, 0.0}};
    for (const auto& [inputs, expected] : patterns) {
        double actual = EvaluateNEATPhenotype(result.best_genome, inputs)[0];
        if (expected > 0.5) {
            EXPECT_GT(actual, 0.5) << "expected class 1 for input (" << inputs[0] << "," << inputs[1] << ")";
        } else {
            EXPECT_LT(actual, 0.5) << "expected class 0 for input (" << inputs[0] << "," << inputs[1] << ")";
        }
    }
}

}  // namespace
}  // namespace pulsatrix
