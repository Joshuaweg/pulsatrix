#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "pulsatrix/crossover.hpp"
#include "pulsatrix/evolutionary_loop.hpp"
#include "pulsatrix/individual.hpp"
#include "pulsatrix/mutation.hpp"
#include "pulsatrix/nsga2.hpp"

namespace pulsatrix {
namespace {

using IntInd = Individual<int, Objectives>;

std::vector<int> Genes(const std::vector<IntInd>& pop) {
    std::vector<int> genes;
    for (const auto& ind : pop) {
        genes.push_back(ind.genes);
    }
    std::sort(genes.begin(), genes.end());
    return genes;
}

// --- Dominates ---

TEST(DominatesTest, StrictlyBetterInBothObjectivesDominates) {
    EXPECT_TRUE(Dominates({2.0, 2.0}, {1.0, 1.0}));
}

TEST(DominatesTest, MixedTradeOffNeitherDominates) {
    EXPECT_FALSE(Dominates({2.0, 1.0}, {1.0, 2.0}));
    EXPECT_FALSE(Dominates({1.0, 2.0}, {2.0, 1.0}));
}

TEST(DominatesTest, IdenticalObjectivesNeitherDominates) {
    EXPECT_FALSE(Dominates({2.0, 2.0}, {2.0, 2.0}));
}

TEST(DominatesTest, EqualInOneStrictlyBetterInOtherDominates) {
    EXPECT_TRUE(Dominates({4.0, 1.0}, {1.0, 1.0}));
}

TEST(DominatesTest, ThrowsOnMismatchedSizes) {
    EXPECT_THROW(Dominates({1.0, 2.0}, {1.0}), std::invalid_argument);
}

// --- FastNonDominatedSort ---
//
// P0=(1,4), P1=(2,3), P2=(3,2), P3=(4,1): a mutually non-dominated trade-off curve (front 0).
// P4=(1,1): dominated by all four (each has >= in both objectives, strictly greater in at
// least one) -- front 1, alone.

TEST(FastNonDominatedSortTest, SeparatesTradeOffCurveFromDominatedPoint) {
    std::vector<Objectives> objectives = {
        {1.0, 4.0}, {2.0, 3.0}, {3.0, 2.0}, {4.0, 1.0}, {1.0, 1.0},
    };
    auto fronts = FastNonDominatedSort(objectives);
    ASSERT_EQ(fronts.size(), 2u);

    auto front0 = fronts[0];
    std::sort(front0.begin(), front0.end());
    EXPECT_EQ(front0, (std::vector<size_t>{0, 1, 2, 3}));

    EXPECT_EQ(fronts[1], (std::vector<size_t>{4}));
}

// --- CrowdingDistance ---

TEST(CrowdingDistanceTest, FrontOfOneOrTwoIsAllInfinite) {
    auto one = CrowdingDistance({{1.0, 2.0}});
    EXPECT_TRUE(std::isinf(one[0]));

    auto two = CrowdingDistance({{1.0, 2.0}, {3.0, 4.0}});
    EXPECT_TRUE(std::isinf(two[0]));
    EXPECT_TRUE(std::isinf(two[1]));
}

TEST(CrowdingDistanceTest, DegenerateObjectiveContributesZeroNotDivideByZero) {
    // Objective 0 is uniform (5.0 everywhere) across this front -- must be skipped, not
    // divide-by-zero. Objective 1 varies (1,2,3): interior point 2 gets (3-1)/(3-1) = 1.0.
    std::vector<Objectives> front = {{5.0, 1.0}, {5.0, 2.0}, {5.0, 3.0}};
    auto distances = CrowdingDistance(front);
    EXPECT_TRUE(std::isinf(distances[0]));
    EXPECT_NEAR(distances[1], 1.0, 1e-9);
    EXPECT_TRUE(std::isinf(distances[2]));
}

TEST(CrowdingDistanceTest, ExactHandDerivedDistancesOnAsymmetricAntiChain) {
    // I1=(1,20), I2=(2,8), I3=(3,5), I4=(4,2), I5=(5,1) -- see mission notes for the full
    // hand derivation. Expected: I1, I5 infinite; I2 ~= 1.289474; I3 ~= 0.815789;
    // I4 ~= 0.710526.
    std::vector<Objectives> front = {
        {1.0, 20.0}, {2.0, 8.0}, {3.0, 5.0}, {4.0, 2.0}, {5.0, 1.0},
    };
    auto d = CrowdingDistance(front);
    EXPECT_TRUE(std::isinf(d[0]));  // I1
    EXPECT_NEAR(d[1], 0.5 + 15.0 / 19.0, 1e-9);  // I2
    EXPECT_NEAR(d[2], 0.5 + 6.0 / 19.0, 1e-9);   // I3
    EXPECT_NEAR(d[3], 0.5 + 4.0 / 19.0, 1e-9);   // I4
    EXPECT_TRUE(std::isinf(d[4]));  // I5
}

TEST(CrowdingDistanceTest, ThrowsOnMismatchedObjectiveVectorSizes) {
    std::vector<Objectives> front = {{1.0, 2.0}, {1.0}, {3.0, 4.0}};
    EXPECT_THROW(CrowdingDistance(front), std::invalid_argument);
}

// --- NSGA2Replacement ---

TEST(NSGA2ReplacementTest, WholeFrontIncludedWhenItFitsExactly) {
    // Same P0..P4 fixture as FastNonDominatedSortTest: front0 has exactly 4 members, mu=4.
    std::vector<IntInd> population = {
        IntInd{0, {1.0, 4.0}}, IntInd{1, {2.0, 3.0}}, IntInd{2, {3.0, 2.0}},
    };
    std::vector<IntInd> offspring = {
        IntInd{3, {4.0, 1.0}},
        IntInd{4, {1.0, 1.0}},  // dominated by every front-0 member
    };
    auto next_gen = NSGA2Replacement(population, offspring, 4);
    EXPECT_EQ(Genes(next_gen), (std::vector<int>{0, 1, 2, 3}));
}

TEST(NSGA2ReplacementTest, AllIndividualsSurviveWhenMuEqualsCombinedSize) {
    std::vector<IntInd> population = {IntInd{0, {1.0, 4.0}}, IntInd{1, {2.0, 3.0}}};
    std::vector<IntInd> offspring = {
        IntInd{2, {3.0, 2.0}}, IntInd{3, {4.0, 1.0}}, IntInd{4, {1.0, 1.0}}};
    auto next_gen = NSGA2Replacement(population, offspring, 5);
    EXPECT_EQ(Genes(next_gen), (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(NSGA2ReplacementTest, PartialFrontTruncatedByCrowdingDistance) {
    // Same asymmetric anti-chain as CrowdingDistanceTest's hand-derived case, all one front:
    // I1(inf), I5(inf), I2(1.289), I3(0.816), I4(0.711) by crowding distance. mu=3 must keep
    // the two boundary points (I1, I5) plus the single highest-crowding-distance interior
    // point (I2), excluding I3 and I4.
    std::vector<IntInd> population = {
        IntInd{1, {1.0, 20.0}}, IntInd{2, {2.0, 8.0}}, IntInd{3, {3.0, 5.0}},
    };
    std::vector<IntInd> offspring = {IntInd{4, {4.0, 2.0}}, IntInd{5, {5.0, 1.0}}};
    auto next_gen = NSGA2Replacement(population, offspring, 3);
    EXPECT_EQ(Genes(next_gen), (std::vector<int>{1, 2, 5}));
}

TEST(NSGA2ReplacementTest, ThrowsWhenCombinedPoolSmallerThanMu) {
    std::vector<IntInd> population = {IntInd{0, {1.0, 1.0}}};
    std::vector<IntInd> offspring = {IntInd{1, {2.0, 2.0}}};
    EXPECT_THROW(NSGA2Replacement(population, offspring, 3), std::invalid_argument);
}

// --- Genuinely multi-objective integration benchmark: Schaffer's function N.1 ---
//
// f1(x) = x^2, f2(x) = (x-2)^2 (both minimized; this codebase maximizes, so objectives are
// negated). The Pareto-optimal set is the closed-form-known interval x in [0, 2]: for any
// x < 0, x'=0 has strictly smaller f1 (0 < x^2) AND strictly smaller f2 (4 < (x-2)^2 for
// x < 0), so x'=0 dominates it; symmetrically x'=2 dominates any x > 2. This is Phase 1's
// exit-gate "second, genuinely multi-objective benchmark via NSGA-II... verified against a
// known-optimal... expected result," distinct from FastNonDominatedSortTest/
// CrowdingDistanceTest's small hand-derived unit fixtures above.

TEST(NSGA2IntegrationTest, ConvergesToKnownParetoOptimalIntervalOnSchafferFunction) {
    constexpr size_t kPopulationSize = 40;
    constexpr size_t kGenerations = 100;

    auto schaffer_objectives = [](const std::vector<double>& genes) -> Objectives {
        double x = genes[0];
        return {-(x * x), -((x - 2.0) * (x - 2.0))};
    };

    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> init_dist(-5.0, 5.0);
    std::uniform_int_distribution<size_t> parent_dist(0, kPopulationSize - 1);

    std::vector<Individual<std::vector<double>, Objectives>> population;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        population.push_back(Individual<std::vector<double>, Objectives>{{init_dist(rng)}, {}});
    }

    auto produce_offspring =
        [&](const std::vector<Individual<std::vector<double>, Objectives>>& pop) {
            size_t i1 = parent_dist(rng);
            size_t i2 = parent_dist(rng);
            auto [child1, child2] = BlendCrossover(pop[i1].genes, pop[i2].genes, 0.5, rng);
            (void)child2;
            return GaussianMutation(child1, 0.3, 0.5, rng);
        };

    auto result = RunEvolutionaryLoop(population, kGenerations, kPopulationSize,
                                       schaffer_objectives, produce_offspring,
                                       NSGA2Replacement<std::vector<double>>);

    std::vector<Objectives> objectives;
    for (const auto& ind : result) {
        objectives.push_back(ind.fitness);
    }
    auto fronts = FastNonDominatedSort(objectives);
    ASSERT_FALSE(fronts.empty());

    std::vector<double> front0_x;
    for (size_t idx : fronts[0]) {
        front0_x.push_back(result[idx].genes[0]);
    }
    ASSERT_FALSE(front0_x.empty());

    // Every surviving front-0 solution lies within a generous tolerance band around the true
    // Pareto-optimal interval [0, 2] -- not a tight bound, allowing for residual mutation
    // noise, but tight enough that a solution far outside [0, 2] (which is provably
    // dominated) would fail it.
    for (double x : front0_x) {
        EXPECT_GE(x, -0.5);
        EXPECT_LE(x, 2.5);
    }

    // Crowding distance must have preserved diversity across the front, not collapsed it to
    // a single point -- both ends of the trade-off should be represented.
    double min_x = *std::min_element(front0_x.begin(), front0_x.end());
    double max_x = *std::max_element(front0_x.begin(), front0_x.end());
    EXPECT_LT(min_x, 0.7);
    EXPECT_GT(max_x, 1.3);
}

}  // namespace
}  // namespace pulsatrix
