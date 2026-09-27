#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "pulsatrix/acquisition_functions.hpp"

namespace pulsatrix {
namespace {

// --- StandardNormalPdf / StandardNormalCdf ---

TEST(StandardNormalTest, PdfAtZeroIsInverseSqrtTwoPi) {
    EXPECT_NEAR(StandardNormalPdf(0.0), 0.3989422804014327, 1e-12);
}

TEST(StandardNormalTest, CdfAtZeroIsOneHalf) {
    EXPECT_NEAR(StandardNormalCdf(0.0), 0.5, 1e-12);
}

TEST(StandardNormalTest, CdfSaturatesAtExtremes) {
    EXPECT_NEAR(StandardNormalCdf(-10.0), 0.0, 1e-9);
    EXPECT_NEAR(StandardNormalCdf(10.0), 1.0, 1e-9);
}

// --- ExpectedImprovement ---

TEST(ExpectedImprovementTest, AtExactMarginEqualsSigmaTimesPdfZero) {
    // z = (mean - best_value - xi) / sigma = 0 when mean == best_value + xi exactly.
    // EI = 0 * Phi(0) + sigma * phi(0) = sigma / sqrt(2*pi).
    double best_value = 5.0, xi = 0.01, sigma = 2.0;
    double mean = best_value + xi;
    double ei = ExpectedImprovement(mean, sigma * sigma, best_value, xi);
    EXPECT_NEAR(ei, sigma * 0.3989422804014327, 1e-9);
}

TEST(ExpectedImprovementTest, ZeroVarianceIsAlwaysZero) {
    EXPECT_DOUBLE_EQ(ExpectedImprovement(100.0, 0.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(ExpectedImprovement(-100.0, 0.0, 5.0), 0.0);
}

TEST(ExpectedImprovementTest, ThrowsOnNegativeVariance) {
    EXPECT_THROW(ExpectedImprovement(1.0, -0.1, 0.0), std::invalid_argument);
}

// --- ProbabilityOfImprovement ---

TEST(ProbabilityOfImprovementTest, AtExactMarginIsOneHalf) {
    double best_value = 5.0, xi = 0.01;
    double mean = best_value + xi;
    EXPECT_NEAR(ProbabilityOfImprovement(mean, 4.0, best_value, xi), 0.5, 1e-9);
}

TEST(ProbabilityOfImprovementTest, ZeroVarianceIsDegenerateStepFunction) {
    EXPECT_DOUBLE_EQ(ProbabilityOfImprovement(10.0, 0.0, 5.0), 1.0);
    EXPECT_DOUBLE_EQ(ProbabilityOfImprovement(1.0, 0.0, 5.0), 0.0);
}

TEST(ProbabilityOfImprovementTest, ThrowsOnNegativeVariance) {
    EXPECT_THROW(ProbabilityOfImprovement(1.0, -0.1, 0.0), std::invalid_argument);
}

// --- UpperConfidenceBound ---

TEST(UpperConfidenceBoundTest, ExactArithmetic) {
    EXPECT_DOUBLE_EQ(UpperConfidenceBound(5.0, 4.0, 1.5), 8.0);  // 5 + 1.5*sqrt(4) = 5+3
}

TEST(UpperConfidenceBoundTest, ZeroVarianceReturnsJustMean) {
    EXPECT_DOUBLE_EQ(UpperConfidenceBound(5.0, 0.0, 2.0), 5.0);
}

TEST(UpperConfidenceBoundTest, ThrowsOnNegativeVariance) {
    EXPECT_THROW(UpperConfidenceBound(1.0, -0.1), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
