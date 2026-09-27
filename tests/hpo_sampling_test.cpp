#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <string>

#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {
namespace {

// --- RandomSample ---

TEST(RandomSampleTest, EmptySearchSpaceProducesEmptyConfiguration) {
    SearchSpace space;
    std::mt19937 rng(0);
    EXPECT_TRUE(RandomSample(space, rng).empty());
}

TEST(RandomSampleTest, EveryParameterKindStaysWithinBoundsAcrossManyDraws) {
    SearchSpace space;
    space.AddContinuous("cont", 1.0, 2.0);
    space.AddLogUniform("log", 1.0, 100.0);
    space.AddInteger("integ", 5, 10);
    space.AddCategorical("cat", {"a", "b", "c"});
    std::mt19937 rng(42);

    for (int i = 0; i < 500; ++i) {
        auto config = RandomSample(space, rng);
        ASSERT_EQ(config.size(), 4u);

        double cont = std::get<double>(config.at("cont"));
        EXPECT_GE(cont, 1.0);
        EXPECT_LE(cont, 2.0);

        double log_val = std::get<double>(config.at("log"));
        EXPECT_GE(log_val, 1.0);
        EXPECT_LE(log_val, 100.0);

        int64_t integ = std::get<int64_t>(config.at("integ"));
        EXPECT_GE(integ, 5);
        EXPECT_LE(integ, 10);

        std::string cat = std::get<std::string>(config.at("cat"));
        EXPECT_TRUE(cat == "a" || cat == "b" || cat == "c");
    }
}

TEST(RandomSampleTest, LogUniformDrawsSpanMultipleOrdersOfMagnitudeNotJustTheTopDecade) {
    // If log-uniform sampling were (incorrectly) implemented as plain linear uniform sampling
    // over [1, 1000], ~99.9% of draws would fall in [100, 1000] (the top decade) and almost
    // none below 100. True log-uniform sampling should distribute roughly evenly across the
    // three decades [1,10), [10,100), [100,1000].
    SearchSpace space;
    space.AddLogUniform("lr", 1.0, 1000.0);
    std::mt19937 rng(7);

    int low_decade = 0, mid_decade = 0, high_decade = 0;
    constexpr int kTrials = 3000;
    for (int i = 0; i < kTrials; ++i) {
        double v = std::get<double>(RandomSample(space, rng).at("lr"));
        if (v < 10.0) {
            low_decade++;
        } else if (v < 100.0) {
            mid_decade++;
        } else {
            high_decade++;
        }
    }
    // Each decade should get roughly a third; a generous tolerance, not a tight one.
    EXPECT_GT(low_decade, kTrials / 10);
    EXPECT_GT(mid_decade, kTrials / 10);
    EXPECT_GT(high_decade, kTrials / 10);
}

// --- GridSample ---

TEST(GridSampleTest, ExactGridSizeAndValuesOverMixedParameterKinds) {
    SearchSpace space;
    space.AddContinuous("cont", 0.0, 10.0);   // 3 points -> {0, 5, 10}
    space.AddInteger("integ", 1, 2);          // always {1, 2}
    space.AddCategorical("cat", {"a", "b"});  // always {"a", "b"}

    auto grid = GridSample(space, 3);
    EXPECT_EQ(grid.size(), 3u * 2u * 2u);

    std::set<double> cont_values;
    for (const auto& config : grid) {
        cont_values.insert(std::get<double>(config.at("cont")));
    }
    ASSERT_EQ(cont_values.size(), 3u);
    auto it = cont_values.begin();
    EXPECT_DOUBLE_EQ(*it++, 0.0);
    EXPECT_DOUBLE_EQ(*it++, 5.0);
    EXPECT_DOUBLE_EQ(*it++, 10.0);
}

TEST(GridSampleTest, LogUniformGridIsGeometricallySpaced) {
    // log(1)=0, log(100)=ln(100); the midpoint in log-space is exp(ln(100)/2) = sqrt(100) =
    // 10 exactly -- a clean, hand-derivable geometric-mean check.
    SearchSpace space;
    space.AddLogUniform("lr", 1.0, 100.0);
    auto grid = GridSample(space, 3);
    ASSERT_EQ(grid.size(), 3u);

    std::vector<double> values;
    for (const auto& config : grid) {
        values.push_back(std::get<double>(config.at("lr")));
    }
    std::sort(values.begin(), values.end());
    EXPECT_NEAR(values[0], 1.0, 1e-9);
    EXPECT_NEAR(values[1], 10.0, 1e-9);
    EXPECT_NEAR(values[2], 100.0, 1e-9);
}

TEST(GridSampleTest, ThrowsWhenPointsPerDimensionBelowTwo) {
    SearchSpace space;
    space.AddContinuous("a", 0.0, 1.0);
    EXPECT_THROW(GridSample(space, 1), std::invalid_argument);
    EXPECT_THROW(GridSample(space, 0), std::invalid_argument);
}

TEST(GridSampleTest, EmptySearchSpaceProducesSingleEmptyConfiguration) {
    SearchSpace space;
    auto grid = GridSample(space, 5);
    ASSERT_EQ(grid.size(), 1u);
    EXPECT_TRUE(grid[0].empty());
}

}  // namespace
}  // namespace pulsatrix
