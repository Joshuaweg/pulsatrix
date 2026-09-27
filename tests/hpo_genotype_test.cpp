#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/hpo_genotype.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {
namespace {

TEST(DecodeGenotypeTest, ContinuousMapsLinearly) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 10.0);
    auto config = DecodeGenotype(space, {0.5});
    EXPECT_DOUBLE_EQ(std::get<double>(config.at("x")), 5.0);
}

TEST(DecodeGenotypeTest, ContinuousBoundaryValuesMapToBounds) {
    SearchSpace space;
    space.AddContinuous("x", 2.0, 8.0);
    EXPECT_DOUBLE_EQ(std::get<double>(DecodeGenotype(space, {0.0}).at("x")), 2.0);
    EXPECT_DOUBLE_EQ(std::get<double>(DecodeGenotype(space, {1.0}).at("x")), 8.0);
}

TEST(DecodeGenotypeTest, LogUniformMapsGeometrically) {
    // gene=0.5 is the log-space midpoint of [1, 100] -> sqrt(1*100) = 10 exactly.
    SearchSpace space;
    space.AddLogUniform("lr", 1.0, 100.0);
    auto config = DecodeGenotype(space, {0.5});
    EXPECT_NEAR(std::get<double>(config.at("lr")), 10.0, 1e-9);
}

TEST(DecodeGenotypeTest, IntegerRoundsToNearestAndHitsBothBounds) {
    SearchSpace space;
    space.AddInteger("n", 0, 10);
    EXPECT_EQ(std::get<int64_t>(DecodeGenotype(space, {0.0}).at("n")), 0);
    EXPECT_EQ(std::get<int64_t>(DecodeGenotype(space, {1.0}).at("n")), 10);
    // gene=0.25 -> interpolated 2.5 -> std::round ties away from zero -> 3 (portable, not
    // round-half-to-even).
    EXPECT_EQ(std::get<int64_t>(DecodeGenotype(space, {0.25}).at("n")), 3);
}

TEST(DecodeGenotypeTest, CategoricalFloorsIntoBucketsAndClampsAtOne) {
    SearchSpace space;
    space.AddCategorical("mode", {"a", "b", "c", "d"});  // 4 buckets: [0,.25),[.25,.5),[.5,.75),[.75,1]
    EXPECT_EQ(std::get<std::string>(DecodeGenotype(space, {0.0}).at("mode")), "a");
    EXPECT_EQ(std::get<std::string>(DecodeGenotype(space, {0.25}).at("mode")), "b");
    EXPECT_EQ(std::get<std::string>(DecodeGenotype(space, {0.99}).at("mode")), "d");
    // gene == 1.0 exactly: floor(1.0 * 4) = 4, one past the last index -- must clamp to "d",
    // not throw or read out of bounds.
    EXPECT_EQ(std::get<std::string>(DecodeGenotype(space, {1.0}).at("mode")), "d");
}

TEST(DecodeGenotypeTest, HandlesAllFourParameterKindsTogether) {
    SearchSpace space;
    space.AddContinuous("cont", 0.0, 10.0);
    space.AddLogUniform("log", 1.0, 100.0);
    space.AddInteger("integ", 0, 10);
    space.AddCategorical("cat", {"a", "b"});

    auto config = DecodeGenotype(space, {0.5, 0.5, 0.0, 1.0});
    EXPECT_DOUBLE_EQ(std::get<double>(config.at("cont")), 5.0);
    EXPECT_NEAR(std::get<double>(config.at("log")), 10.0, 1e-9);
    EXPECT_EQ(std::get<int64_t>(config.at("integ")), 0);
    EXPECT_EQ(std::get<std::string>(config.at("cat")), "b");
}

TEST(DecodeGenotypeTest, ThrowsOnSizeMismatch) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    EXPECT_THROW(DecodeGenotype(space, {0.5, 0.5}), std::invalid_argument);
}

TEST(DecodeGenotypeTest, ThrowsOnOutOfRangeGene) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    EXPECT_THROW(DecodeGenotype(space, {1.5}), std::invalid_argument);
    EXPECT_THROW(DecodeGenotype(space, {-0.1}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
