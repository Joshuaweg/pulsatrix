#include <gtest/gtest.h>

#include <array>
#include <random>
#include <vector>

#include "pulsatrix/individual.hpp"
#include "pulsatrix/selection.hpp"

namespace pulsatrix {
namespace {

using IntInd = Individual<int, double>;

std::vector<IntInd> MakePopulation(std::initializer_list<double> fitnesses) {
    std::vector<IntInd> population;
    int genes = 0;
    for (double f : fitnesses) {
        population.push_back(IntInd{genes++, f});
    }
    return population;
}

// --- TournamentSelect ---

TEST(TournamentSelectTest, FullPopulationSizeAlwaysReturnsGlobalBestRegardlessOfSeed) {
    // Fitness values chosen distinct so "the global best" is unambiguous: index 2 (fitness 9.0).
    auto population = MakePopulation({1.0, 5.0, 9.0, 3.0, 2.0});

    for (unsigned seed : {0u, 1u, 42u, 12345u, 999999u}) {
        std::mt19937 rng(seed);
        size_t selected = TournamentSelect(population, population.size(), rng);
        EXPECT_EQ(selected, 2u) << "seed=" << seed;
    }
}

TEST(TournamentSelectTest, SizeOneIsApproximatelyUniformOverManyTrials) {
    auto population = MakePopulation({1.0, 1.0, 1.0, 1.0});
    std::mt19937 rng(7);
    std::array<int, 4> counts{0, 0, 0, 0};
    constexpr int kTrials = 20000;
    for (int i = 0; i < kTrials; ++i) {
        counts[TournamentSelect(population, 1, rng)]++;
    }
    // Expected count per index: kTrials / 4 = 5000. Generous +/-15% tolerance -- this is a
    // statistical sanity check on genuine randomness, not a closed-form correctness test
    // (which FullPopulationSizeAlwaysReturnsGlobalBest, above, already provides).
    for (int count : counts) {
        EXPECT_NEAR(count, kTrials / 4, kTrials / 4 * 0.15);
    }
}

TEST(TournamentSelectTest, ThrowsOnEmptyPopulation) {
    std::vector<IntInd> population;
    std::mt19937 rng(0);
    EXPECT_THROW(TournamentSelect(population, 1, rng), std::invalid_argument);
}

TEST(TournamentSelectTest, ThrowsOnZeroTournamentSize) {
    auto population = MakePopulation({1.0, 2.0});
    std::mt19937 rng(0);
    EXPECT_THROW(TournamentSelect(population, 0, rng), std::invalid_argument);
}

TEST(TournamentSelectTest, ThrowsWhenTournamentSizeExceedsPopulation) {
    auto population = MakePopulation({1.0, 2.0});
    std::mt19937 rng(0);
    EXPECT_THROW(TournamentSelect(population, 3, rng), std::invalid_argument);
}

// --- RouletteSelectByDraw ---

TEST(RouletteSelectByDrawTest, ExactBoundaries) {
    // Fitness {1.0, 3.0, 2.0}, total = 6.0. Cumulative buckets: [0,1) -> 0, [1,4) -> 1, [4,6) -> 2.
    auto population = MakePopulation({1.0, 3.0, 2.0});

    EXPECT_EQ(RouletteSelectByDraw(population, 0.0), 0u);
    EXPECT_EQ(RouletteSelectByDraw(population, 0.999999), 0u);
    EXPECT_EQ(RouletteSelectByDraw(population, 1.0), 1u);
    EXPECT_EQ(RouletteSelectByDraw(population, 3.999999), 1u);
    EXPECT_EQ(RouletteSelectByDraw(population, 4.0), 2u);
    EXPECT_EQ(RouletteSelectByDraw(population, 5.999999), 2u);
}

TEST(RouletteSelectByDrawTest, ThrowsOnEmptyPopulation) {
    std::vector<IntInd> population;
    EXPECT_THROW(RouletteSelectByDraw(population, 0.0), std::invalid_argument);
}

TEST(RouletteSelectByDrawTest, ThrowsOnNegativeFitness) {
    auto population = MakePopulation({1.0, -2.0});
    EXPECT_THROW(RouletteSelectByDraw(population, 0.0), std::invalid_argument);
}

TEST(RouletteSelectByDrawTest, ThrowsOnAllZeroFitness) {
    auto population = MakePopulation({0.0, 0.0, 0.0});
    EXPECT_THROW(RouletteSelectByDraw(population, 0.0), std::invalid_argument);
}

TEST(RouletteSelectByDrawTest, ThrowsOnDrawOutOfRange) {
    auto population = MakePopulation({1.0, 3.0, 2.0});  // total = 6.0
    EXPECT_THROW(RouletteSelectByDraw(population, -0.001), std::invalid_argument);
    EXPECT_THROW(RouletteSelectByDraw(population, 6.0), std::invalid_argument);
    EXPECT_THROW(RouletteSelectByDraw(population, 100.0), std::invalid_argument);
}

TEST(RouletteSelectTest, WrapperNeverThrowsAndReturnsValidIndexWithSeededRng) {
    auto population = MakePopulation({1.0, 3.0, 2.0});
    std::mt19937 rng(2026);
    for (int i = 0; i < 1000; ++i) {
        size_t selected = RouletteSelect(population, rng);
        EXPECT_LT(selected, population.size());
    }
}

// --- RankSelectByDraw ---

TEST(RankSelectByDrawTest, ExactBoundaries) {
    // Fitness {5.0, 1.0, 9.0} at indices {0, 1, 2}. Ascending order by fitness: index 1
    // (1.0, rank 1, weight 1), index 0 (5.0, rank 2, weight 2), index 2 (9.0, rank 3, weight
    // 3). Total weight = 1+2+3 = 6. Cumulative buckets: [0,1) -> idx1, [1,3) -> idx0, [3,6) -> idx2.
    auto population = MakePopulation({5.0, 1.0, 9.0});

    EXPECT_EQ(RankSelectByDraw(population, 0.0), 1u);
    EXPECT_EQ(RankSelectByDraw(population, 0.999999), 1u);
    EXPECT_EQ(RankSelectByDraw(population, 1.0), 0u);
    EXPECT_EQ(RankSelectByDraw(population, 2.999999), 0u);
    EXPECT_EQ(RankSelectByDraw(population, 3.0), 2u);
    EXPECT_EQ(RankSelectByDraw(population, 5.999999), 2u);
}

TEST(RankSelectByDrawTest, ThrowsOnEmptyPopulation) {
    std::vector<IntInd> population;
    EXPECT_THROW(RankSelectByDraw(population, 0.0), std::invalid_argument);
}

TEST(RankSelectByDrawTest, ThrowsOnDrawOutOfRange) {
    auto population = MakePopulation({5.0, 1.0, 9.0});  // total_weight = 6.0
    EXPECT_THROW(RankSelectByDraw(population, -0.001), std::invalid_argument);
    EXPECT_THROW(RankSelectByDraw(population, 6.0), std::invalid_argument);
}

TEST(RankSelectTest, WrapperNeverThrowsAndReturnsValidIndexWithSeededRng) {
    auto population = MakePopulation({5.0, 1.0, 9.0});
    std::mt19937 rng(2026);
    for (int i = 0; i < 1000; ++i) {
        size_t selected = RankSelect(population, rng);
        EXPECT_LT(selected, population.size());
    }
}

}  // namespace
}  // namespace pulsatrix
