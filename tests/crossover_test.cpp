#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "pulsatrix/crossover.hpp"

namespace pulsatrix {
namespace {

// --- One-point crossover ---

TEST(OnePointCrossoverAtPointTest, ExactSwapAtInteriorPoint) {
    std::vector<int> p1{1, 2, 3, 4, 5};
    std::vector<int> p2{10, 20, 30, 40, 50};
    auto [c1, c2] = OnePointCrossoverAtPoint(p1, p2, 2);
    EXPECT_EQ(c1, (std::vector<int>{1, 2, 30, 40, 50}));
    EXPECT_EQ(c2, (std::vector<int>{10, 20, 3, 4, 5}));
}

TEST(OnePointCrossoverAtPointTest, PointZeroFullySwaps) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    auto [c1, c2] = OnePointCrossoverAtPoint(p1, p2, 0);
    EXPECT_EQ(c1, p2);
    EXPECT_EQ(c2, p1);
}

TEST(OnePointCrossoverAtPointTest, PointAtSizeNeverSwaps) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    auto [c1, c2] = OnePointCrossoverAtPoint(p1, p2, 3);
    EXPECT_EQ(c1, p1);
    EXPECT_EQ(c2, p2);
}

TEST(OnePointCrossoverAtPointTest, ThrowsOnMismatchedSizes) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20};
    EXPECT_THROW(OnePointCrossoverAtPoint(p1, p2, 1), std::invalid_argument);
}

TEST(OnePointCrossoverAtPointTest, ThrowsWhenPointExceedsSize) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    EXPECT_THROW(OnePointCrossoverAtPoint(p1, p2, 4), std::invalid_argument);
}

TEST(OnePointCrossoverTest, WrapperProducesFullLengthChildrenAcrossSeeds) {
    std::vector<int> p1{1, 2, 3, 4};
    std::vector<int> p2{10, 20, 30, 40};
    for (unsigned seed : {0u, 1u, 42u}) {
        std::mt19937 rng(seed);
        auto [c1, c2] = OnePointCrossover(p1, p2, rng);
        EXPECT_EQ(c1.size(), p1.size());
        EXPECT_EQ(c2.size(), p2.size());
    }
}

TEST(OnePointCrossoverTest, ThrowsWhenParentSizeBelowTwo) {
    std::vector<int> p1{1};
    std::vector<int> p2{10};
    std::mt19937 rng(0);
    EXPECT_THROW(OnePointCrossover(p1, p2, rng), std::invalid_argument);
}

// --- Two-point crossover ---

TEST(TwoPointCrossoverAtPointsTest, ExactSwapOfMiddleSegment) {
    std::vector<int> p1{1, 2, 3, 4, 5};
    std::vector<int> p2{10, 20, 30, 40, 50};
    auto [c1, c2] = TwoPointCrossoverAtPoints(p1, p2, 1, 3);
    EXPECT_EQ(c1, (std::vector<int>{1, 20, 30, 4, 5}));
    EXPECT_EQ(c2, (std::vector<int>{10, 2, 3, 40, 50}));
}

TEST(TwoPointCrossoverAtPointsTest, EqualPointsIsANoOp) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    auto [c1, c2] = TwoPointCrossoverAtPoints(p1, p2, 2, 2);
    EXPECT_EQ(c1, p1);
    EXPECT_EQ(c2, p2);
}

TEST(TwoPointCrossoverAtPointsTest, ThrowsWhenPoint1ExceedsPoint2) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    EXPECT_THROW(TwoPointCrossoverAtPoints(p1, p2, 2, 1), std::invalid_argument);
}

TEST(TwoPointCrossoverAtPointsTest, ThrowsWhenPoint2ExceedsSize) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    EXPECT_THROW(TwoPointCrossoverAtPoints(p1, p2, 0, 4), std::invalid_argument);
}

TEST(TwoPointCrossoverTest, WrapperProducesFullLengthChildrenAcrossSeeds) {
    std::vector<int> p1{1, 2, 3, 4, 5};
    std::vector<int> p2{10, 20, 30, 40, 50};
    for (unsigned seed : {0u, 1u, 42u}) {
        std::mt19937 rng(seed);
        auto [c1, c2] = TwoPointCrossover(p1, p2, rng);
        EXPECT_EQ(c1.size(), p1.size());
        EXPECT_EQ(c2.size(), p2.size());
    }
}

// --- Uniform crossover ---

TEST(UniformCrossoverByMaskTest, ExactSwapAtMaskedPositions) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    auto [c1, c2] = UniformCrossoverByMask(p1, p2, {true, false, true});
    EXPECT_EQ(c1, (std::vector<int>{10, 2, 30}));
    EXPECT_EQ(c2, (std::vector<int>{1, 20, 3}));
}

TEST(UniformCrossoverByMaskTest, ThrowsOnMismatchedSizes) {
    std::vector<int> p1{1, 2, 3};
    std::vector<int> p2{10, 20, 30};
    EXPECT_THROW(UniformCrossoverByMask(p1, p2, {true, false}), std::invalid_argument);
}

TEST(UniformCrossoverTest, ProbabilityZeroNeverSwaps) {
    std::vector<int> p1{1, 2, 3, 4};
    std::vector<int> p2{10, 20, 30, 40};
    std::mt19937 rng(123);
    auto [c1, c2] = UniformCrossover(p1, p2, 0.0, rng);
    EXPECT_EQ(c1, p1);
    EXPECT_EQ(c2, p2);
}

TEST(UniformCrossoverTest, ProbabilityOneAlwaysSwaps) {
    std::vector<int> p1{1, 2, 3, 4};
    std::vector<int> p2{10, 20, 30, 40};
    std::mt19937 rng(123);
    auto [c1, c2] = UniformCrossover(p1, p2, 1.0, rng);
    EXPECT_EQ(c1, p2);
    EXPECT_EQ(c2, p1);
}

TEST(UniformCrossoverTest, ThrowsOnInvalidProbability) {
    std::vector<int> p1{1, 2};
    std::vector<int> p2{10, 20};
    std::mt19937 rng(0);
    EXPECT_THROW(UniformCrossover(p1, p2, -0.1, rng), std::invalid_argument);
    EXPECT_THROW(UniformCrossover(p1, p2, 1.1, rng), std::invalid_argument);
}

// --- Blend crossover (BLX-alpha) ---

TEST(BlendCrossoverByGammaTest, GammaZeroIsIdentity) {
    std::vector<double> p1{2.0, 4.0};
    std::vector<double> p2{8.0, 16.0};
    auto [c1, c2] = BlendCrossoverByGamma(p1, p2, {0.0, 0.0});
    EXPECT_DOUBLE_EQ(c1[0], 2.0);
    EXPECT_DOUBLE_EQ(c1[1], 4.0);
    EXPECT_DOUBLE_EQ(c2[0], 8.0);
    EXPECT_DOUBLE_EQ(c2[1], 16.0);
}

TEST(BlendCrossoverByGammaTest, GammaOneIsFullSwap) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    auto [c1, c2] = BlendCrossoverByGamma(p1, p2, {1.0});
    EXPECT_DOUBLE_EQ(c1[0], 8.0);
    EXPECT_DOUBLE_EQ(c2[0], 2.0);
}

TEST(BlendCrossoverByGammaTest, GammaHalfIsMidpointForBothChildren) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    auto [c1, c2] = BlendCrossoverByGamma(p1, p2, {0.5});
    EXPECT_DOUBLE_EQ(c1[0], 5.0);
    EXPECT_DOUBLE_EQ(c2[0], 5.0);
}

TEST(BlendCrossoverByGammaTest, ThrowsOnMismatchedSizes) {
    std::vector<double> p1{2.0, 4.0};
    std::vector<double> p2{8.0};
    EXPECT_THROW(BlendCrossoverByGamma(p1, p2, {0.5, 0.5}), std::invalid_argument);
}

TEST(BlendCrossoverTest, ChildrenSumEqualsParentSumForAnySeedOrAlpha) {
    std::vector<double> p1{2.0, -3.5, 10.0};
    std::vector<double> p2{8.0, 6.5, -4.0};
    for (double alpha : {0.0, 0.5, 1.5}) {
        for (unsigned seed : {0u, 1u, 42u}) {
            std::mt19937 rng(seed);
            auto [c1, c2] = BlendCrossover(p1, p2, alpha, rng);
            for (size_t i = 0; i < p1.size(); ++i) {
                EXPECT_NEAR(c1[i] + c2[i], p1[i] + p2[i], 1e-9);
            }
        }
    }
}

TEST(BlendCrossoverTest, ThrowsOnNegativeAlpha) {
    std::vector<double> p1{1.0};
    std::vector<double> p2{2.0};
    std::mt19937 rng(0);
    EXPECT_THROW(BlendCrossover(p1, p2, -0.1, rng), std::invalid_argument);
}

// --- Simulated binary crossover (SBX) ---

TEST(SimulatedBinaryCrossoverByDrawTest, DrawZeroAveragesBothGenesEtaIndependent) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    for (double eta : {0.0, 2.0, 5.0}) {
        auto [c1, c2] = SimulatedBinaryCrossoverByDraw(p1, p2, eta, {0.0});
        EXPECT_NEAR(c1[0], 5.0, 1e-9);
        EXPECT_NEAR(c2[0], 5.0, 1e-9);
    }
}

TEST(SimulatedBinaryCrossoverByDrawTest, DrawHalfIsIdentityEtaIndependent) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    for (double eta : {0.0, 2.0, 5.0}) {
        auto [c1, c2] = SimulatedBinaryCrossoverByDraw(p1, p2, eta, {0.5});
        EXPECT_NEAR(c1[0], 2.0, 1e-9);
        EXPECT_NEAR(c2[0], 8.0, 1e-9);
    }
}

TEST(SimulatedBinaryCrossoverByDrawTest, ThrowsOnDrawEqualToOne) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    EXPECT_THROW(SimulatedBinaryCrossoverByDraw(p1, p2, 2.0, {1.0}), std::invalid_argument);
}

TEST(SimulatedBinaryCrossoverByDrawTest, ThrowsOnNegativeDraw) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    EXPECT_THROW(SimulatedBinaryCrossoverByDraw(p1, p2, 2.0, {-0.1}), std::invalid_argument);
}

TEST(SimulatedBinaryCrossoverByDrawTest, ThrowsOnNegativeEta) {
    std::vector<double> p1{2.0};
    std::vector<double> p2{8.0};
    EXPECT_THROW(SimulatedBinaryCrossoverByDraw(p1, p2, -1.0, {0.3}), std::invalid_argument);
}

TEST(SimulatedBinaryCrossoverTest, ChildrenSumEqualsParentSumForAnySeedOrEta) {
    std::vector<double> p1{2.0, -3.5, 10.0};
    std::vector<double> p2{8.0, 6.5, -4.0};
    for (double eta : {0.0, 2.0, 20.0}) {
        for (unsigned seed : {0u, 1u, 42u}) {
            std::mt19937 rng(seed);
            auto [c1, c2] = SimulatedBinaryCrossover(p1, p2, eta, rng);
            for (size_t i = 0; i < p1.size(); ++i) {
                EXPECT_NEAR(c1[i] + c2[i], p1[i] + p2[i], 1e-9);
            }
        }
    }
}

}  // namespace
}  // namespace pulsatrix
