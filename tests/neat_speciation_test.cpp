#include <gtest/gtest.h>

#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_speciation.hpp"

namespace pulsatrix {
namespace {

constexpr double kC1 = 1.0;
constexpr double kC2 = 1.0;
constexpr double kC3 = 0.4;

TEST(NEATSpeciationTest, CompatibilityDistanceMatchingGenesOnly) {
    InnovationTracker tracker(2);
    NEATGenome a(1, 1, /*has_bias=*/false, tracker);
    NEATGenome b(1, 1, /*has_bias=*/false, tracker);
    a.SetConnectionWeight(0, 1.0);
    b.SetConnectionWeight(0, 4.0);

    double distance = CompatibilityDistance(a, b, kC1, kC2, kC3);

    EXPECT_NEAR(distance, 1.2, 1e-12);
}

TEST(NEATSpeciationTest, CompatibilityDistanceWithExcessGenes) {
    InnovationTracker tracker(2);
    NEATGenome a(1, 1, /*has_bias=*/false, tracker);
    NEATGenome b(1, 1, /*has_bias=*/false, tracker);
    a.SetConnectionWeight(0, 1.0);
    b.SetConnectionWeight(0, 4.0);
    // Splitting b's innov0 creates hidden node id=2, innov1 (0->2), innov2 (2->1, weight
    // preserved as 4.0 from the just-set original connection).
    b.AddNodeSplitting(0, tracker);

    double distance = CompatibilityDistance(a, b, kC1, kC2, kC3);

    // Matching: innov0 only (a=1.0, b=4.0 but disabled in b -- still counted per the doc's own
    // "regardless of enabled status" rule), diff=3.0. Excess: innov1, innov2 (both > overlap
    // bound of 0, the smaller genome's own max innovation). Disjoint: 0. N=1 (both genomes have
    // fewer than 20 connection genes).
    EXPECT_NEAR(distance, 3.2, 1e-12);
}

TEST(NEATSpeciationTest, SpeciatePopulationGroupsByCompatibilityThreshold) {
    InnovationTracker tracker(2);
    NEATGenome genome_a(1, 1, /*has_bias=*/false, tracker);
    genome_a.SetConnectionWeight(0, 1.0);
    NEATGenome genome_b(1, 1, /*has_bias=*/false, tracker);
    genome_b.SetConnectionWeight(0, 1.0);
    NEATGenome genome_c(1, 1, /*has_bias=*/false, tracker);
    genome_c.SetConnectionWeight(0, 4.0);
    genome_c.AddNodeSplitting(0, tracker);

    std::vector<NEATGenome> population;
    population.push_back(genome_a);
    population.push_back(genome_b);
    population.push_back(genome_c);

    SpeciesAssignment assignment = SpeciatePopulation(population, /*compatibility_threshold=*/1.0, kC1, kC2, kC3);

    ASSERT_EQ(assignment.species.size(), 2u);
    EXPECT_EQ(assignment.species[0], (std::vector<size_t>{0, 1}));
    EXPECT_EQ(assignment.species[1], (std::vector<size_t>{2}));
}

TEST(NEATSpeciationTest, SpeciatePopulationThrowsOnEmptyPopulation) {
    EXPECT_THROW((void)SpeciatePopulation({}, 1.0, kC1, kC2, kC3), std::invalid_argument);
}

TEST(NEATSpeciationTest, SpeciatePopulationThrowsOnNonPositiveThreshold) {
    InnovationTracker tracker(2);
    std::vector<NEATGenome> population;
    population.emplace_back(1, 1, /*has_bias=*/false, tracker);

    EXPECT_THROW((void)SpeciatePopulation(population, 0.0, kC1, kC2, kC3), std::invalid_argument);
}

TEST(NEATSpeciationTest, ComputeAdjustedFitnessDividesBySpeciesSize) {
    std::vector<double> raw_fitness{10.0, 20.0, 30.0};
    std::vector<std::vector<size_t>> species{{0, 1}, {2}};

    std::vector<double> adjusted = ComputeAdjustedFitness(raw_fitness, species);

    ASSERT_EQ(adjusted.size(), 3u);
    EXPECT_NEAR(adjusted[0], 5.0, 1e-12);
    EXPECT_NEAR(adjusted[1], 10.0, 1e-12);
    EXPECT_NEAR(adjusted[2], 30.0, 1e-12);
}

TEST(NEATSpeciationTest, ComputeAdjustedFitnessThrowsOnSizeMismatch) {
    std::vector<double> raw_fitness{10.0, 20.0};
    std::vector<std::vector<size_t>> species{{0, 1}, {2}};

    EXPECT_THROW((void)ComputeAdjustedFitness(raw_fitness, species), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
