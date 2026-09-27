#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/evolution_strategies.hpp"
#include "pulsatrix/fixed_topology_xor_network.hpp"

namespace pulsatrix {
namespace {

TEST(ESUpdateGivenPerturbationsTest, ComputesExactHandDerivedUpdate) {
    std::vector<double> theta{0.0, 0.0};
    std::vector<std::vector<double>> epsilons{{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}};
    std::vector<double> fitnesses{2.0, -2.0, 1.0, -1.0};

    // sum_i F_i*epsilon_i = [4.0, 2.0]; scale = alpha/(N*sigma) = 1/(4*1) = 0.25.
    std::vector<double> updated = ESUpdateGivenPerturbations(theta, epsilons, fitnesses, /*alpha=*/1.0, /*sigma=*/1.0);

    ASSERT_EQ(updated.size(), 2u);
    EXPECT_NEAR(updated[0], 1.0, 1e-12);
    EXPECT_NEAR(updated[1], 0.5, 1e-12);
}

TEST(ESUpdateGivenPerturbationsTest, ThrowsOnEmptyEpsilons) {
    EXPECT_THROW((void)ESUpdateGivenPerturbations({0.0}, {}, {}, 1.0, 1.0), std::invalid_argument);
}

TEST(ESUpdateGivenPerturbationsTest, ThrowsOnMismatchedEpsilonsAndFitnessesSize) {
    EXPECT_THROW((void)ESUpdateGivenPerturbations({0.0}, {{1.0}}, {1.0, 2.0}, 1.0, 1.0), std::invalid_argument);
}

TEST(ESUpdateGivenPerturbationsTest, ThrowsOnEpsilonDimensionMismatch) {
    EXPECT_THROW((void)ESUpdateGivenPerturbations({0.0, 0.0}, {{1.0}}, {1.0}, 1.0, 1.0), std::invalid_argument);
}

TEST(ESUpdateGivenPerturbationsTest, ThrowsOnNonPositiveSigma) {
    EXPECT_THROW((void)ESUpdateGivenPerturbations({0.0}, {{1.0}}, {1.0}, 1.0, 0.0), std::invalid_argument);
}

TEST(ESStepTest, ThrowsOnOddPopulationSize) {
    std::mt19937 rng(1);
    auto fitness_fn = [](const std::vector<double>& theta) { return -theta[0] * theta[0]; };
    EXPECT_THROW((void)ESStep(std::vector<double>{0.0}, fitness_fn, 3, 1.0, 1.0, rng), std::invalid_argument);
}

TEST(ESStepTest, MovesThetaTowardHigherFitnessOnASimpleQuadratic) {
    // fitness(theta) = -(theta[0] - 3)^2 -- maximized at theta[0]=3. Starting at 0, a single
    // ES step (fixed seed) should move theta measurably toward 3, not away from it.
    std::mt19937 rng(42);
    auto fitness_fn = [](const std::vector<double>& theta) { return -(theta[0] - 3.0) * (theta[0] - 3.0); };

    std::vector<double> updated = ESStep(std::vector<double>{0.0}, fitness_fn, 50, /*sigma=*/1.0, /*alpha=*/0.5, rng);

    ASSERT_EQ(updated.size(), 1u);
    EXPECT_GT(updated[0], 0.0);
}

TEST(RunEvolutionStrategiesTest, ThrowsOnEmptyTheta) {
    std::mt19937 rng(1);
    auto fitness_fn = [](const std::vector<double>&) { return 0.0; };
    EXPECT_THROW((void)RunEvolutionStrategies(std::vector<double>{}, fitness_fn, 5, 10, 1.0, 1.0, rng),
                 std::invalid_argument);
}

TEST(RunEvolutionStrategiesTest, ThrowsOnNonPositiveIterations) {
    std::mt19937 rng(1);
    auto fitness_fn = [](const std::vector<double>&) { return 0.0; };
    EXPECT_THROW((void)RunEvolutionStrategies(std::vector<double>{0.0}, fitness_fn, 0, 10, 1.0, 1.0, rng),
                 std::invalid_argument);
}

TEST(RunEvolutionStrategiesTest, BestFitnessNeverDecreasesAcrossIterations) {
    std::mt19937 rng(7);
    auto fitness_fn = [](const std::vector<double>& theta) { return -(theta[0] - 3.0) * (theta[0] - 3.0); };

    ESResult result =
        RunEvolutionStrategies(std::vector<double>{0.0}, fitness_fn, /*num_iterations=*/20, /*population_size=*/20,
                                /*sigma=*/1.0, /*alpha=*/0.3, rng);

    EXPECT_GE(result.best_fitness, fitness_fn({0.0}));
    EXPECT_EQ(result.iterations_run, 20);
}

// Phase 3's other, independent exit-gate requirement: ES trains a fixed-topology network to a
// measurable improvement on a plain analytic fitness function, zero RL dependency. Same XOR
// benchmark and fitness convention as Mission 2's NEAT proof, for a direct cross-check.
TEST(EvolutionStrategiesXORIntegrationTest, EvolvesAFixedTopologyNetworkThatSolvesXOR) {
    std::mt19937 rng(42);
    // A small, fixed, sign-varied initial theta -- NOT all-zero. This mirrors a real,
    // already-documented gotcha in this codebase's own gradient-based XOR example
    // (xor_training_example.hpp): an all-zero starting point is a symmetric saddle (both
    // hidden units start identical, so nothing differentiates them), and ES's gradient
    // estimate near that saddle is dominated by sampling noise rather than real signal --
    // confirmed empirically during this mission's design (500-2000 iterations from an
    // all-zero start plateaued around fitness 3.0-3.2, never solving XOR).
    std::vector<double> initial_theta{0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.0};
    double initial_fitness = FixedTopologyXORFitness(initial_theta);

    ESResult result = RunEvolutionStrategies(initial_theta, FixedTopologyXORFitness, /*num_iterations=*/1000,
                                              /*population_size=*/100, /*sigma=*/0.5, /*alpha=*/0.3, rng);

    // Exit-gate requirement: a measurable improvement over the untrained starting point.
    EXPECT_GT(result.best_fitness, initial_fitness);
    // Stronger, hand-verifiable claim: the evolved network classifies every XOR pattern
    // correctly, not just "some numeric improvement happened."
    const std::array<std::array<double, 2>, 4> inputs{{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}};
    const std::array<double, 4> expected{0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < inputs.size(); ++i) {
        double actual = FixedTopologyXORForward(result.best_theta, inputs[i]);
        if (expected[i] > 0.5) {
            EXPECT_GT(actual, 0.5) << "expected class 1 for input (" << inputs[i][0] << "," << inputs[i][1] << ")";
        } else {
            EXPECT_LT(actual, 0.5) << "expected class 0 for input (" << inputs[i][0] << "," << inputs[i][1] << ")";
        }
    }
}

}  // namespace
}  // namespace pulsatrix
