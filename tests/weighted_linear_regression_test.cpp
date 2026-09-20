#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "exai/weighted_linear_regression.hpp"

// fit_weighted_linear_regression is the shared utility Phase 3's explainers (LIME,
// KernelSHAP) both need -- no linear-algebra solve exists anywhere else in this codebase.
// Solves w* = argmin_w sum_i weight_i*(y_i - w^T x_i)^2 + lambda*||w||^2 via the normal
// equations, small-n Gaussian elimination.
namespace exai {
namespace {

TEST(WeightedLinearRegressionTest, RecoversExactCoefficientForNoiseFreeOneFeatureData) {
    // y = 2x exactly -- noise-free, so least squares (any weighting) recovers w=2 exactly.
    std::vector<std::vector<float>> samples = {{1.0f}, {2.0f}, {3.0f}};
    std::vector<float> targets = {2.0f, 4.0f, 6.0f};
    std::vector<float> weights = {1.0f, 1.0f, 1.0f};

    std::vector<float> w = fit_weighted_linear_regression(samples, targets, weights, /*l2_lambda=*/0.0f);

    ASSERT_EQ(w.size(), 1u);
    EXPECT_NEAR(w[0], 2.0f, 1e-4f);
}

// Hand-solved 2x2 system. y = 2*x0 + 3*x1 exactly.
// samples=[[1,0],[0,1],[1,1]], y=[2,3,5], uniform weights.
// A = X^T X = [[2,1],[1,2]], b = X^T y = [7,8] -> w = [2,3] (worked by hand in the mission
// doc's Recon section before this test was written).
TEST(WeightedLinearRegressionTest, RecoversExactCoefficientsForNoiseFreeTwoFeatureData) {
    std::vector<std::vector<float>> samples = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    std::vector<float> targets = {2.0f, 3.0f, 5.0f};
    std::vector<float> weights = {1.0f, 1.0f, 1.0f};

    std::vector<float> w = fit_weighted_linear_regression(samples, targets, weights, /*l2_lambda=*/0.0f);

    ASSERT_EQ(w.size(), 2u);
    EXPECT_NEAR(w[0], 2.0f, 1e-4f);
    EXPECT_NEAR(w[1], 3.0f, 1e-4f);
}

// Two inconsistent single-feature "votes" (y=1 at x=1, y=100 at x=2 -- not a real line).
// With near-zero weight on the second sample, the fit should land close to the
// first sample's implied slope (w=1) rather than being pulled toward the second's (w=50).
TEST(WeightedLinearRegressionTest, PerSampleWeightingVisiblyChangesTheFit) {
    std::vector<std::vector<float>> samples = {{1.0f}, {2.0f}};
    std::vector<float> targets = {1.0f, 100.0f};

    std::vector<float> heavily_weighted_first = {1000.0f, 0.001f};
    std::vector<float> w_first = fit_weighted_linear_regression(samples, targets, heavily_weighted_first, 0.0f);

    std::vector<float> heavily_weighted_second = {0.001f, 1000.0f};
    std::vector<float> w_second = fit_weighted_linear_regression(samples, targets, heavily_weighted_second, 0.0f);

    EXPECT_NEAR(w_first[0], 1.0f, 0.5f);       // pulled toward the first sample's slope
    EXPECT_NEAR(w_second[0], 50.0f, 5.0f);     // pulled toward the second sample's slope
    EXPECT_GT(w_second[0], w_first[0]);        // and they're clearly different
}

TEST(WeightedLinearRegressionTest, ThrowsOnNearSingularSystem) {
    // Two identical samples -> the normal-equations matrix is singular (rank-deficient).
    std::vector<std::vector<float>> samples = {{1.0f, 2.0f}, {1.0f, 2.0f}};
    std::vector<float> targets = {1.0f, 1.0f};
    std::vector<float> weights = {1.0f, 1.0f};

    EXPECT_THROW((void)fit_weighted_linear_regression(samples, targets, weights, 0.0f), std::runtime_error);
}

}  // namespace
}  // namespace exai
