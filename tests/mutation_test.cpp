#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "pulsatrix/mutation.hpp"

namespace pulsatrix {
namespace {

// --- Bit-flip mutation ---

TEST(BitFlipMutationByMaskTest, FlipsOnlyMaskedPositions) {
    std::vector<bool> genotype{true, false, true, false};
    auto result = BitFlipMutationByMask(genotype, {true, false, false, true});
    EXPECT_EQ(result, (std::vector<bool>{false, false, true, true}));
}

TEST(BitFlipMutationByMaskTest, ThrowsOnMismatchedSizes) {
    std::vector<bool> genotype{true, false};
    EXPECT_THROW(BitFlipMutationByMask(genotype, {true}), std::invalid_argument);
}

TEST(BitFlipMutationTest, ProbabilityZeroNeverFlips) {
    std::vector<bool> genotype{true, false, true, false};
    std::mt19937 rng(7);
    EXPECT_EQ(BitFlipMutation(genotype, 0.0, rng), genotype);
}

TEST(BitFlipMutationTest, ProbabilityOneAlwaysFlips) {
    std::vector<bool> genotype{true, false, true, false};
    std::mt19937 rng(7);
    auto result = BitFlipMutation(genotype, 1.0, rng);
    EXPECT_EQ(result, (std::vector<bool>{false, true, false, true}));
}

TEST(BitFlipMutationTest, ThrowsOnInvalidProbability) {
    std::vector<bool> genotype{true, false};
    std::mt19937 rng(0);
    EXPECT_THROW(BitFlipMutation(genotype, -0.1, rng), std::invalid_argument);
    EXPECT_THROW(BitFlipMutation(genotype, 1.1, rng), std::invalid_argument);
}

// --- Gaussian mutation ---

TEST(GaussianMutationByNoiseTest, AddsNoiseOnlyAtMaskedPositions) {
    std::vector<double> genotype{1.0, 2.0, 3.0};
    auto result = GaussianMutationByNoise(genotype, {0.5, -1.0, 2.0}, {true, false, true});
    EXPECT_DOUBLE_EQ(result[0], 1.5);
    EXPECT_DOUBLE_EQ(result[1], 2.0);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
}

TEST(GaussianMutationByNoiseTest, ThrowsOnMismatchedSizes) {
    std::vector<double> genotype{1.0, 2.0};
    EXPECT_THROW(GaussianMutationByNoise(genotype, {0.5}, {true, false}), std::invalid_argument);
}

TEST(GaussianMutationTest, ProbabilityZeroLeavesGenotypeUnchanged) {
    std::vector<double> genotype{1.0, 2.0, 3.0};
    std::mt19937 rng(7);
    auto result = GaussianMutation(genotype, 5.0, 0.0, rng);
    EXPECT_EQ(result, genotype);
}

TEST(GaussianMutationTest, ZeroSigmaWithProbabilityOneLeavesGenotypeUnchanged) {
    std::vector<double> genotype{1.0, 2.0, 3.0};
    std::mt19937 rng(7);
    auto result = GaussianMutation(genotype, 0.0, 1.0, rng);
    for (size_t i = 0; i < genotype.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], genotype[i]);
    }
}

TEST(GaussianMutationTest, ThrowsOnNegativeSigmaOrInvalidProbability) {
    std::vector<double> genotype{1.0};
    std::mt19937 rng(0);
    EXPECT_THROW(GaussianMutation(genotype, -1.0, 0.5, rng), std::invalid_argument);
    EXPECT_THROW(GaussianMutation(genotype, 1.0, 1.5, rng), std::invalid_argument);
}

// --- Polynomial mutation ---

TEST(PolynomialMutationByDrawTest, DrawZeroReturnsLowerBoundEtaIndependent) {
    for (double eta : {0.0, 2.0, 5.0}) {
        EXPECT_NEAR(PolynomialMutationByDraw(5.0, 0.0, 10.0, eta, 0.0), 0.0, 1e-9);
    }
}

TEST(PolynomialMutationByDrawTest, DrawHalfLeavesValueUnchangedEtaIndependent) {
    for (double eta : {0.0, 2.0, 5.0}) {
        EXPECT_NEAR(PolynomialMutationByDraw(5.0, 0.0, 10.0, eta, 0.5), 5.0, 1e-9);
    }
}

TEST(PolynomialMutationByDrawTest, DrawOneReturnsUpperBoundEtaIndependent) {
    for (double eta : {0.0, 2.0, 5.0}) {
        EXPECT_NEAR(PolynomialMutationByDraw(5.0, 0.0, 10.0, eta, 1.0), 10.0, 1e-9);
    }
}

TEST(PolynomialMutationByDrawTest, ThrowsOnInvertedBounds) {
    EXPECT_THROW(PolynomialMutationByDraw(5.0, 10.0, 0.0, 2.0, 0.5), std::invalid_argument);
}

TEST(PolynomialMutationByDrawTest, ThrowsWhenXOutsideBounds) {
    EXPECT_THROW(PolynomialMutationByDraw(11.0, 0.0, 10.0, 2.0, 0.5), std::invalid_argument);
}

TEST(PolynomialMutationByDrawTest, ThrowsOnNegativeEta) {
    EXPECT_THROW(PolynomialMutationByDraw(5.0, 0.0, 10.0, -1.0, 0.5), std::invalid_argument);
}

TEST(PolynomialMutationByDrawTest, ThrowsOnDrawOutsideZeroOneRange) {
    EXPECT_THROW(PolynomialMutationByDraw(5.0, 0.0, 10.0, 2.0, -0.1), std::invalid_argument);
    EXPECT_THROW(PolynomialMutationByDraw(5.0, 0.0, 10.0, 2.0, 1.1), std::invalid_argument);
}

TEST(PolynomialMutationTest, ProbabilityZeroLeavesGenotypeUnchanged) {
    std::vector<double> genotype{1.0, 5.0, 9.0};
    std::vector<double> lower{0.0, 0.0, 0.0};
    std::vector<double> upper{10.0, 10.0, 10.0};
    std::mt19937 rng(7);
    auto result = PolynomialMutation(genotype, lower, upper, 2.0, 0.0, rng);
    EXPECT_EQ(result, genotype);
}

TEST(PolynomialMutationTest, ProbabilityOneStaysWithinBounds) {
    std::vector<double> genotype{1.0, 5.0, 9.0};
    std::vector<double> lower{0.0, 0.0, 0.0};
    std::vector<double> upper{10.0, 10.0, 10.0};
    std::mt19937 rng(7);
    auto result = PolynomialMutation(genotype, lower, upper, 2.0, 1.0, rng);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_GE(result[i], lower[i]);
        EXPECT_LE(result[i], upper[i]);
    }
}

TEST(PolynomialMutationTest, ThrowsOnMismatchedSizes) {
    std::vector<double> genotype{1.0, 2.0};
    std::vector<double> lower{0.0};
    std::vector<double> upper{10.0, 10.0};
    std::mt19937 rng(0);
    EXPECT_THROW(PolynomialMutation(genotype, lower, upper, 2.0, 0.5, rng), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
