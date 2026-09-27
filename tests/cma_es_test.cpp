#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cma_es.hpp"

namespace pulsatrix {
namespace {

// --- AskGivenSamples ---

TEST(AskGivenSamplesTest, ExactArithmeticWithNonTrivialMeanSigmaAndVariance) {
    // x = mean + sigma * sqrt(variance) * z = 1.0 + 0.5 * sqrt(4.0) * 3.0 = 1.0 + 3.0 = 4.0.
    CMAESState state{{1.0}, 0.5, {4.0}};
    auto offspring = AskGivenSamples(state, {{3.0}});
    ASSERT_EQ(offspring.size(), 1u);
    EXPECT_DOUBLE_EQ(offspring[0][0], 4.0);
}

TEST(AskGivenSamplesTest, ZeroZGivesExactlyTheMean) {
    CMAESState state{{2.0, -3.0}, 1.5, {9.0, 4.0}};
    auto offspring = AskGivenSamples(state, {{0.0, 0.0}});
    EXPECT_DOUBLE_EQ(offspring[0][0], 2.0);
    EXPECT_DOUBLE_EQ(offspring[0][1], -3.0);
}

TEST(AskGivenSamplesTest, ThrowsOnDimensionMismatch) {
    CMAESState state{{0.0}, 1.0, {1.0}};
    EXPECT_THROW(AskGivenSamples(state, {{1.0, 2.0}}), std::invalid_argument);
}

// --- TellGivenSamples ---

// Hand-derived (see cma_es.hpp's own header comment for the update formulas): n=1,
// mean=[0.0], sigma=1.0, variance=[1.0]. z_samples = [-2,-1,1,2] (as 1D vectors); since
// variance=1, y_i = z_i exactly, so x_i = y_i = [-2,-1,1,2]. fitness = x (maximize) picks the
// top mu=2 as x=2 (z=2) and x=1 (z=1).
//   mean_y = (2+1)/2 = 1.5           -> mean_new = 0 + 1.0*1.5 = 1.5
//   mean_y_squared = (4+1)/2 = 2.5   -> variance_new = 0.7*1.0 + 0.3*2.5 = 1.45
//   y_norm = |1.5| = 1.5, expected_norm = sqrt(1) = 1.0
//   sigma_new = 1.0 * exp(0.3*(1.5/1.0 - 1)) = exp(0.15)
TEST(TellGivenSamplesTest, ExactHandDerivedUpdateOnASingleDimension) {
    CMAESState state{{0.0}, 1.0, {1.0}};
    std::vector<std::vector<double>> z_samples = {{-2.0}, {-1.0}, {1.0}, {2.0}};
    auto offspring = AskGivenSamples(state, z_samples);
    std::vector<double> fitness = {-2.0, -1.0, 1.0, 2.0};  // fitness == x here (variance=1)

    auto next = TellGivenSamples(state, z_samples, offspring, fitness, /*step_size_lr=*/0.3,
                                  /*scale_lr=*/0.3);

    EXPECT_NEAR(next.mean[0], 1.5, 1e-9);
    EXPECT_NEAR(next.variances[0], 1.45, 1e-9);
    EXPECT_NEAR(next.sigma, std::exp(0.15), 1e-9);
}

TEST(TellGivenSamplesTest, ThrowsOnMismatchedSizes) {
    CMAESState state{{0.0}, 1.0, {1.0}};
    std::vector<std::vector<double>> z = {{1.0}, {2.0}};
    std::vector<std::vector<double>> offspring = {{1.0}, {2.0}};
    std::vector<double> fitness = {1.0};  // wrong size
    EXPECT_THROW(TellGivenSamples(state, z, offspring, fitness, 0.3, 0.3), std::invalid_argument);
}

TEST(TellGivenSamplesTest, ThrowsOnFewerThanTwoSamples) {
    CMAESState state{{0.0}, 1.0, {1.0}};
    std::vector<std::vector<double>> z = {{1.0}};
    std::vector<std::vector<double>> offspring = {{1.0}};
    std::vector<double> fitness = {1.0};
    EXPECT_THROW(TellGivenSamples(state, z, offspring, fitness, 0.3, 0.3), std::invalid_argument);
}

// --- CMAES (ask-tell wrapper) ---

TEST(CMAESTest, ConstructorThrowsOnInvalidParameters) {
    EXPECT_THROW(CMAES({}, 1.0, 4), std::invalid_argument);
    EXPECT_THROW(CMAES({0.0}, 0.0, 4), std::invalid_argument);
    EXPECT_THROW(CMAES({0.0}, 1.0, 1), std::invalid_argument);
}

TEST(CMAESTest, TellThrowsIfCalledBeforeAsk) {
    CMAES cmaes({0.0}, 1.0, 4);
    EXPECT_THROW(cmaes.Tell({{0.0}, {1.0}}, {0.0, 1.0}), std::invalid_argument);
}

TEST(CMAESTest, AskReturnsExactlyLambdaOffspringOfTheRightDimension) {
    CMAES cmaes({0.0, 0.0}, 1.0, 6);
    std::mt19937 rng(0);
    auto offspring = cmaes.Ask(rng);
    ASSERT_EQ(offspring.size(), 6u);
    for (const auto& x : offspring) {
        EXPECT_EQ(x.size(), 2u);
    }
}

TEST(CMAESTest, ConvergesTowardTheOptimumOnASimpleUnimodalObjective) {
    // Maximize -(x-3)^2 -- a clean, single-peaked objective with a known optimum at x=3.
    CMAES cmaes({0.0}, 1.0, 10);
    std::mt19937 rng(2026);

    for (int generation = 0; generation < 40; ++generation) {
        auto offspring = cmaes.Ask(rng);
        std::vector<double> fitness(offspring.size());
        for (size_t i = 0; i < offspring.size(); ++i) {
            double x = offspring[i][0];
            fitness[i] = -(x - 3.0) * (x - 3.0);
        }
        cmaes.Tell(offspring, fitness);
    }

    EXPECT_NEAR(cmaes.mean()[0], 3.0, 0.3);
}

}  // namespace
}  // namespace pulsatrix
