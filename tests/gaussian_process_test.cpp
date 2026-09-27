#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/gaussian_process.hpp"

namespace pulsatrix {
namespace {

TEST(GaussianProcessRegressorTest, ThrowsOnNonPositiveHyperparameters) {
    EXPECT_THROW(GaussianProcessRegressor(0.0f, 1.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(GaussianProcessRegressor(1.0f, 0.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(GaussianProcessRegressor(1.0f, 1.0f, -0.1f), std::invalid_argument);
}

TEST(GaussianProcessRegressorTest, PredictThrowsBeforeFit) {
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.0f);
    EXPECT_THROW((void)gp.Predict({0.0f}), std::runtime_error);
}

TEST(GaussianProcessRegressorTest, FitThrowsOnMismatchedSizes) {
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.0f);
    EXPECT_THROW(gp.Fit({{0.0f}, {1.0f}}, {1.0f}), std::invalid_argument);
}

TEST(GaussianProcessRegressorTest, FitThrowsOnEmptyInputs) {
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.0f);
    EXPECT_THROW(gp.Fit({}, {}), std::invalid_argument);
}

TEST(GaussianProcessRegressorTest, PredictThrowsOnDimensionMismatch) {
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.0f);
    gp.Fit({{0.0f, 0.0f}}, {1.0f});
    EXPECT_THROW((void)gp.Predict({0.0f}), std::invalid_argument);
}

// A single noiseless observation is an exact, kernel-and-hyperparameter-independent
// invariant: with n=1, K=[[k11]], alpha=y1/k11, mean = k11*alpha = y1 exactly, and
// variance = k11 - k11*(1/k11)*k11 = 0 exactly -- true for any sigma_f/length_scale/y1 as
// long as noise_variance == 0. This is the campaign's own named "1-2 observation GP
// posterior computed by hand" exit-gate test case.
TEST(GaussianProcessRegressorTest, SingleNoiselessObservationPosteriorIsExactRegardlessOfKernel) {
    for (float sigma_f : {0.5f, 1.0f, 3.0f}) {
        for (float length_scale : {0.5f, 2.0f}) {
            GaussianProcessRegressor gp(sigma_f, length_scale, 0.0f);
            gp.Fit({{2.0f}}, {7.0f});
            auto posterior = gp.Predict({2.0f});
            EXPECT_NEAR(posterior.mean, 7.0, 1e-4) << "sigma_f=" << sigma_f << " length_scale=" << length_scale;
            EXPECT_NEAR(posterior.variance, 0.0, 1e-4) << "sigma_f=" << sigma_f << " length_scale=" << length_scale;
        }
    }
}

// Two-observation posterior, hand-derived closed form (see mission notes for the full
// algebraic simplification): sigma_f=1, length_scale=1, noise=0; x1=0, x2=1, y1=2, y2=4;
// query at the midpoint x*=0.5. k12 = exp(-0.5) (squared distance 1, /(2*1^2)); k* = exp(-0.125)
// for both training points (0.5 is equidistant from both). By symmetry the algebra reduces
// to: mean = k* * (y1+y2) / (1+k12); variance = k(x*,x*) - 2*k*^2/(1+k12).
TEST(GaussianProcessRegressorTest, TwoObservationPosteriorMatchesHandDerivedClosedForm) {
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.0f);
    gp.Fit({{0.0f}, {1.0f}}, {2.0f, 4.0f});

    double k12 = std::exp(-0.5);
    double k_star = std::exp(-0.125);
    double expected_mean = k_star * 6.0 / (1.0 + k12);
    double expected_variance = 1.0 - 2.0 * k_star * k_star / (1.0 + k12);

    auto posterior = gp.Predict({0.5f});
    EXPECT_NEAR(posterior.mean, expected_mean, 1e-3);
    EXPECT_NEAR(posterior.variance, expected_variance, 1e-3);
}

TEST(GaussianProcessRegressorTest, VarianceApproachesPriorFarFromAnyTrainingPoint) {
    // Far from every training point, the kernel value between the query and each training
    // point vanishes (squared-exponential decays to 0), so the posterior should approach the
    // zero-mean, sigma_f^2-variance prior.
    GaussianProcessRegressor gp(2.0f, 1.0f, 0.0f);
    gp.Fit({{0.0f}, {1.0f}}, {5.0f, -3.0f});
    auto posterior = gp.Predict({1000.0f});
    EXPECT_NEAR(posterior.mean, 0.0, 1e-3);
    EXPECT_NEAR(posterior.variance, 4.0, 1e-3);  // sigma_f^2 = 2^2 = 4
}

TEST(GaussianProcessRegressorTest, NoiseVarianceStrictlyReducesConfidenceAtTrainingPoint) {
    // With noise_variance > 0, even at a training point the posterior variance must be
    // strictly between 0 (the noiseless case) and sigma_f^2 (the prior) -- the GP is not
    // certain its own observation is exact.
    GaussianProcessRegressor gp(1.0f, 1.0f, 0.1f);
    gp.Fit({{0.0f}}, {5.0f});
    auto posterior = gp.Predict({0.0f});
    EXPECT_GT(posterior.variance, 0.0);
    EXPECT_LT(posterior.variance, 1.0);
}

}  // namespace
}  // namespace pulsatrix
