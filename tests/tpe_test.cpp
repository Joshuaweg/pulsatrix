#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/search_space.hpp"
#include "pulsatrix/tpe.hpp"

namespace pulsatrix {
namespace {

// --- GaussianKdeDensity ---

TEST(GaussianKdeDensityTest, SingleObservationReducesToPlainGaussian) {
    // At the observation itself, N(x;x,bandwidth^2) peaks at 1/(bandwidth*sqrt(2*pi)).
    double density = GaussianKdeDensity({0.0}, 1.0, 0.0);
    EXPECT_NEAR(density, 0.3989422804014327, 1e-9);
}

TEST(GaussianKdeDensityTest, TwoSymmetricObservationsAverageToTheSameValueAtTheirMidpoint) {
    // Observations at -1 and 1, evaluated at their midpoint x=0: both terms are identical by
    // symmetry (each is 1 bandwidth away), so the average equals either term alone.
    double density = GaussianKdeDensity({-1.0, 1.0}, 1.0, 0.0);
    double expected_single_term = 0.3989422804014327 * std::exp(-0.5);
    EXPECT_NEAR(density, expected_single_term, 1e-9);
}

TEST(GaussianKdeDensityTest, ThrowsOnEmptyObservationsOrNonPositiveBandwidth) {
    EXPECT_THROW(GaussianKdeDensity({}, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianKdeDensity({0.0}, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(GaussianKdeDensity({0.0}, -1.0, 0.0), std::invalid_argument);
}

// --- CategoricalDensity ---

TEST(CategoricalDensityTest, LaplaceSmoothedFrequenciesSumToOne) {
    std::vector<std::string> observations = {"a", "a", "b"};
    double p_a = CategoricalDensity(observations, 2, "a");
    double p_b = CategoricalDensity(observations, 2, "b");
    EXPECT_NEAR(p_a, 3.0 / 5.0, 1e-12);
    EXPECT_NEAR(p_b, 2.0 / 5.0, 1e-12);
    EXPECT_NEAR(p_a + p_b, 1.0, 1e-12);
}

TEST(CategoricalDensityTest, UnobservedCategoryGetsUniformPriorMass) {
    std::vector<std::string> observations = {"a", "a", "b"};
    EXPECT_NEAR(CategoricalDensity(observations, 3, "c"), 1.0 / 6.0, 1e-12);
}

TEST(CategoricalDensityTest, EmptyObservationsReducesToUniformPrior) {
    EXPECT_NEAR(CategoricalDensity({}, 4, "x"), 0.25, 1e-12);
}

TEST(CategoricalDensityTest, ThrowsOnZeroCategories) {
    EXPECT_THROW(CategoricalDensity({"a"}, 0, "a"), std::invalid_argument);
}

// --- LogDensityRatio ---

TEST(LogDensityRatioTest, ExactHandDerivedValueOnASingleContinuousParameter) {
    // space range [-10,10] -> bandwidth = 0.2*20 = 4.0. good obs at 0.0, bad obs at 8.0
    // (distance 8 = 2 bandwidths). Candidate exactly at the good observation (0.0):
    // log(density_good) - log(density_bad)
    //   = [-0.5*(0/4)^2] - [-0.5*((0-8)/4)^2]   (the shared 1/(bandwidth*sqrt(2*pi)) prefactor
    //     cancels exactly in the subtraction, since both densities use the same bandwidth)
    //   = 0 - (-2.0) = 2.0 exactly.
    SearchSpace space;
    space.AddContinuous("x", -10.0, 10.0);
    Configuration candidate{{"x", 0.0}};
    std::vector<Configuration> good = {{{"x", 0.0}}};
    std::vector<Configuration> bad = {{{"x", 8.0}}};

    double log_ratio = LogDensityRatio(space, candidate, good, bad);
    EXPECT_NEAR(log_ratio, 2.0, 1e-6);
}

TEST(LogDensityRatioTest, CandidateNearGoodClusterScoresHigherThanNearBadCluster) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 100.0);
    std::vector<Configuration> good = {{{"x", 10.0}}, {{"x", 12.0}}, {{"x", 11.0}}};
    std::vector<Configuration> bad = {{{"x", 90.0}}, {{"x", 88.0}}, {{"x", 91.0}}};

    double score_near_good = LogDensityRatio(space, Configuration{{"x", 11.0}}, good, bad);
    double score_near_bad = LogDensityRatio(space, Configuration{{"x", 89.0}}, good, bad);
    EXPECT_GT(score_near_good, score_near_bad);
}

TEST(LogDensityRatioTest, HandlesMixedContinuousAndCategoricalParametersTogether) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 100.0);
    space.AddCategorical("kind", {"fast", "slow"});

    std::vector<Configuration> good = {
        {{"x", 10.0}, {"kind", std::string("fast")}},
        {{"x", 12.0}, {"kind", std::string("fast")}},
    };
    std::vector<Configuration> bad = {
        {{"x", 90.0}, {"kind", std::string("slow")}},
        {{"x", 88.0}, {"kind", std::string("slow")}},
    };

    Configuration candidate_like_good{{"x", 11.0}, {"kind", std::string("fast")}};
    Configuration candidate_like_bad{{"x", 89.0}, {"kind", std::string("slow")}};
    EXPECT_GT(LogDensityRatio(space, candidate_like_good, good, bad),
              LogDensityRatio(space, candidate_like_bad, good, bad));
}

TEST(LogDensityRatioTest, ThrowsWhenEitherGroupIsEmpty) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::vector<Configuration> good = {{{"x", 0.5}}};
    std::vector<Configuration> empty;
    EXPECT_THROW(LogDensityRatio(space, Configuration{{"x", 0.5}}, good, empty), std::invalid_argument);
    EXPECT_THROW(LogDensityRatio(space, Configuration{{"x", 0.5}}, empty, good), std::invalid_argument);
}

// --- RunTPELoop ---

TEST(RunTPELoopTest, ThrowsOnTooFewInitialRandomTrials) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    auto objective = [](const Configuration&) { return 0.0; };
    EXPECT_THROW(RunTPELoop(space, objective, 1, 3, 0.2, 10, rng), std::invalid_argument);
}

TEST(RunTPELoopTest, ThrowsOnGammaOutOfRange) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    auto objective = [](const Configuration&) { return 0.0; };
    EXPECT_THROW(RunTPELoop(space, objective, 3, 3, 0.0, 10, rng), std::invalid_argument);
    EXPECT_THROW(RunTPELoop(space, objective, 3, 3, 1.0, 10, rng), std::invalid_argument);
}

TEST(RunTPELoopTest, ReturnsExactlyInitialPlusIterationTrials) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    auto objective = [](const Configuration& config) { return -std::get<double>(config.at("x")); };
    auto trials = RunTPELoop(space, objective, 3, 4, 0.2, 20, rng);
    EXPECT_EQ(trials.size(), 7u);
}

TEST(RunTPELoopTest, FindsNearOptimalPointOnASimpleUnimodalObjective) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(2026);
    auto objective = [](const Configuration& config) {
        double x = std::get<double>(config.at("x"));
        return -(x - 0.3) * (x - 0.3);
    };
    auto trials = RunTPELoop(space, objective, 4, 10, 0.25, 100, rng);

    double best_x = 0.0, best_value = -1e18;
    for (const auto& trial : trials) {
        double value = *trial.LatestMetric("objective");
        if (value > best_value) {
            best_value = value;
            best_x = std::get<double>(trial.configuration().at("x"));
        }
    }
    EXPECT_NEAR(best_x, 0.3, 0.15);
}

TEST(RunTPELoopTest, HandlesAMixedCategoricalAndContinuousSearchSpace) {
    // The kind of search space GP-BO cannot handle at all (UnitCubeToConfiguration throws on
    // Categorical) -- this is TPE's own differentiator, exercised end-to-end, not just at the
    // LogDensityRatio unit level.
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    space.AddCategorical("mode", {"a", "b", "c"});
    std::mt19937 rng(7);
    auto objective = [](const Configuration& config) {
        double x = std::get<double>(config.at("x"));
        std::string mode = std::get<std::string>(config.at("mode"));
        double mode_bonus = (mode == "b") ? 1.0 : 0.0;
        return mode_bonus - (x - 0.5) * (x - 0.5);
    };
    auto trials = RunTPELoop(space, objective, 4, 8, 0.25, 50, rng);
    EXPECT_EQ(trials.size(), 12u);
    for (const auto& trial : trials) {
        EXPECT_TRUE(trial.configuration().count("x"));
        EXPECT_TRUE(trial.configuration().count("mode"));
    }
}

}  // namespace
}  // namespace pulsatrix
