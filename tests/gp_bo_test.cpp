#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/gp_bo.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {
namespace {

// --- UnitCubeToConfiguration ---

TEST(UnitCubeToConfigurationTest, MapsContinuousLinearly) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 10.0);
    auto config = UnitCubeToConfiguration(space, {0.5f});
    EXPECT_DOUBLE_EQ(std::get<double>(config.at("x")), 5.0);
}

TEST(UnitCubeToConfigurationTest, MapsLogUniformGeometrically) {
    // t=0.5 is the log-space midpoint of [1, 100] -> sqrt(1*100) = 10 exactly.
    SearchSpace space;
    space.AddLogUniform("lr", 1.0, 100.0);
    auto config = UnitCubeToConfiguration(space, {0.5f});
    EXPECT_NEAR(std::get<double>(config.at("lr")), 10.0, 1e-6);
}

TEST(UnitCubeToConfigurationTest, BoundaryValuesMapToBounds) {
    SearchSpace space;
    space.AddContinuous("x", 2.0, 8.0);
    EXPECT_DOUBLE_EQ(std::get<double>(UnitCubeToConfiguration(space, {0.0f}).at("x")), 2.0);
    EXPECT_DOUBLE_EQ(std::get<double>(UnitCubeToConfiguration(space, {1.0f}).at("x")), 8.0);
}

TEST(UnitCubeToConfigurationTest, ThrowsOnSizeMismatch) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    EXPECT_THROW(UnitCubeToConfiguration(space, {0.5f, 0.5f}), std::invalid_argument);
}

TEST(UnitCubeToConfigurationTest, ThrowsOnOutOfRangeT) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    EXPECT_THROW(UnitCubeToConfiguration(space, {1.5f}), std::invalid_argument);
    EXPECT_THROW(UnitCubeToConfiguration(space, {-0.1f}), std::invalid_argument);
}

TEST(UnitCubeToConfigurationTest, ThrowsOnIntegerParameter) {
    SearchSpace space;
    space.AddInteger("n", 1, 10);
    EXPECT_THROW(UnitCubeToConfiguration(space, {0.5f}), std::invalid_argument);
}

TEST(UnitCubeToConfigurationTest, ThrowsOnCategoricalParameter) {
    SearchSpace space;
    space.AddCategorical("c", {"a", "b"});
    EXPECT_THROW(UnitCubeToConfiguration(space, {0.5f}), std::invalid_argument);
}

// --- RunGPBOLoop ---

TEST(RunGPBOLoopTest, ThrowsOnZeroInitialRandomTrials) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    auto objective = [](const Configuration&) { return 0.0; };
    EXPECT_THROW(RunGPBOLoop(space, objective, 0, 5, AcquisitionKind::ExpectedImprovement, 10, rng),
                 std::invalid_argument);
}

TEST(RunGPBOLoopTest, ReturnsExactlyInitialPlusIterationTrials) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    auto objective = [](const Configuration& config) { return -std::get<double>(config.at("x")); };
    auto trials = RunGPBOLoop(space, objective, 3, 4, AcquisitionKind::ExpectedImprovement, 20, rng);
    EXPECT_EQ(trials.size(), 7u);
    for (const auto& trial : trials) {
        EXPECT_TRUE(trial.LatestMetric("objective").has_value());
    }
}

TEST(RunGPBOLoopTest, FindsNearOptimalPointOnASimpleUnimodalObjective) {
    // Maximize -(x-0.7)^2 -- a clean, single-peaked objective with a known optimum at x=0.7.
    // GP-BO with a real (if modest) iteration budget should get close.
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(2026);
    auto objective = [](const Configuration& config) {
        double x = std::get<double>(config.at("x"));
        return -(x - 0.7) * (x - 0.7);
    };
    auto trials = RunGPBOLoop(space, objective, 3, 10, AcquisitionKind::ExpectedImprovement, 100, rng);

    double best_x = 0.0;
    double best_value = -1e18;
    for (const auto& trial : trials) {
        double value = *trial.LatestMetric("objective");
        if (value > best_value) {
            best_value = value;
            best_x = std::get<double>(trial.configuration().at("x"));
        }
    }
    EXPECT_NEAR(best_x, 0.7, 0.15);
}

}  // namespace
}  // namespace pulsatrix
