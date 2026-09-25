/** @file noise_schedule_test.cpp
 *  @brief Unit tests for the DDPM linear noise schedule and its forward/reverse steps.
 *
 *  There is no finite-difference check anywhere in this file, by design: NoiseSchedule has no
 *  learned parameters and no backward pass of its own (mission_diffusion_module.md). Every
 *  quantity it produces is deterministic host-side math, so correctness is established by
 *  direct comparison against independently computed reference values instead -- the schedule
 *  numbers below were computed outside this codebase (double-precision, from the published
 *  formulas) rather than by re-running the implementation and recording what it printed.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/noise_schedule.hpp"

namespace pulsatrix {
namespace {

class NoiseScheduleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // A deliberately coarse 5-step schedule: beta spans 0.1 .. 0.5, so alpha_bar falls fast
    // enough that a wrong cumulative product (e.g. a per-step product instead of a running
    // one) cannot hide inside float noise, unlike the near-1 alphas of the default schedule.
    NoiseSchedule coarse{5, 0.1f, 0.5f};
};

// ---------------------------------------------------------------------------------------
// Schedule values vs. independently computed references.
// ---------------------------------------------------------------------------------------

TEST_F(NoiseScheduleTest, BetaIsLinearlySpacedAcrossBothEndpoints) {
    // beta_t = 0.1 + 0.4*(t-1)/4 -> 0.1, 0.2, 0.3, 0.4, 0.5.
    EXPECT_NEAR(coarse.beta(1), 0.1f, 1e-6f);
    EXPECT_NEAR(coarse.beta(2), 0.2f, 1e-6f);
    EXPECT_NEAR(coarse.beta(3), 0.3f, 1e-6f);
    EXPECT_NEAR(coarse.beta(4), 0.4f, 1e-6f);
    EXPECT_NEAR(coarse.beta(5), 0.5f, 1e-6f);
}

TEST_F(NoiseScheduleTest, AlphaIsOneMinusBeta) {
    EXPECT_NEAR(coarse.alpha(1), 0.9f, 1e-6f);
    EXPECT_NEAR(coarse.alpha(2), 0.8f, 1e-6f);
    EXPECT_NEAR(coarse.alpha(3), 0.7f, 1e-6f);
    EXPECT_NEAR(coarse.alpha(4), 0.6f, 1e-6f);
    EXPECT_NEAR(coarse.alpha(5), 0.5f, 1e-6f);
}

TEST_F(NoiseScheduleTest, AlphaBarIsTheCumulativeProduct) {
    // 0.9, 0.9*0.8 = 0.72, *0.7 = 0.504, *0.6 = 0.3024, *0.5 = 0.1512 -- computed
    // independently, not read back from the implementation.
    EXPECT_NEAR(coarse.alpha_bar(1), 0.9f, 1e-6f);
    EXPECT_NEAR(coarse.alpha_bar(2), 0.72f, 1e-6f);
    EXPECT_NEAR(coarse.alpha_bar(3), 0.504f, 1e-6f);
    EXPECT_NEAR(coarse.alpha_bar(4), 0.3024f, 1e-6f);
    EXPECT_NEAR(coarse.alpha_bar(5), 0.1512f, 1e-6f);
    // ...and strictly decreasing, which a per-step (non-cumulative) alpha would not be here.
    for (int64_t t = 2; t <= 5; ++t) {
        EXPECT_LT(coarse.alpha_bar(t), coarse.alpha_bar(t - 1)) << "t = " << t;
    }
}

TEST_F(NoiseScheduleTest, DefaultPublishedScheduleMatchesReferenceValues) {
    // The DDPM paper's own T = 1000, beta in [1e-4, 0.02] schedule. References computed in
    // double precision outside this codebase; the float tolerance on alpha_bar(1000) is
    // relative, since it is a product of 1000 floats and accumulates rounding.
    NoiseSchedule schedule(1000);

    EXPECT_NEAR(schedule.beta(1), 1e-4f, 1e-9f);
    EXPECT_NEAR(schedule.beta(1000), 0.02f, 1e-9f);
    EXPECT_NEAR(schedule.beta(2), 0.0001199199f, 1e-8f);
    EXPECT_NEAR(schedule.beta(500), 0.0100400400f, 1e-7f);

    EXPECT_NEAR(schedule.alpha(1), 0.9999f, 1e-7f);
    EXPECT_NEAR(schedule.alpha(500), 0.9899599600f, 1e-6f);
    EXPECT_NEAR(schedule.alpha(1000), 0.98f, 1e-7f);

    EXPECT_NEAR(schedule.alpha_bar(1), 0.9999f, 1e-6f);
    EXPECT_NEAR(schedule.alpha_bar(2), 0.9997800921f, 1e-6f);
    EXPECT_NEAR(schedule.alpha_bar(500), 0.0785872429f, 1e-4f);
    EXPECT_NEAR(schedule.alpha_bar(1000), 4.03583e-5f, 4e-7f);  // ~1% relative
}

TEST_F(NoiseScheduleTest, SingleTimestepScheduleUsesBetaStart) {
    // T == 1 has no interval to interpolate across; the documented degenerate case.
    NoiseSchedule single(1, 0.25f, 0.75f);
    EXPECT_EQ(single.num_timesteps(), 1);
    EXPECT_NEAR(single.beta(1), 0.25f, 1e-6f);
    EXPECT_NEAR(single.alpha(1), 0.75f, 1e-6f);
    EXPECT_NEAR(single.alpha_bar(1), 0.75f, 1e-6f);
}

// ---------------------------------------------------------------------------------------
// Constructor / accessor boundaries.
// ---------------------------------------------------------------------------------------

TEST_F(NoiseScheduleTest, ConstructorThrowsOnNonPositiveTimesteps) {
    EXPECT_THROW({ NoiseSchedule s(0); }, std::invalid_argument);
    EXPECT_THROW({ NoiseSchedule s(-4); }, std::invalid_argument);
}

TEST_F(NoiseScheduleTest, ConstructorThrowsWhenBetaStartIsNotBelowBetaEnd) {
    EXPECT_THROW({ NoiseSchedule s(10, 0.2f, 0.2f); }, std::invalid_argument);
    EXPECT_THROW({ NoiseSchedule s(10, 0.5f, 0.1f); }, std::invalid_argument);
}

TEST_F(NoiseScheduleTest, AccessorsThrowOutOfRangeOutsideOneToT) {
    EXPECT_THROW({ (void)coarse.beta(0); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.alpha(0); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.alpha_bar(0); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.beta(6); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.alpha(-1); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.alpha_bar(6); }, std::out_of_range);
}

// ---------------------------------------------------------------------------------------
// add_noise -- forward (noising) process.
// ---------------------------------------------------------------------------------------

TEST_F(NoiseScheduleTest, AddNoiseMatchesClosedFormByHand) {
    // t = 3 -> alpha_bar = 0.504; signal scale sqrt(0.504), noise scale sqrt(0.496).
    const float signal = std::sqrt(0.504f);
    const float noise = std::sqrt(0.496f);

    Tensor x0(Shape({1, 3}), &backend, {1.0f, -2.0f, 0.5f});
    Tensor eps(Shape({1, 3}), &backend, {0.25f, 3.0f, -1.5f});

    Tensor x_t = coarse.add_noise(x0, eps, 3);

    ASSERT_EQ(x_t.numel(), 3);
    EXPECT_NEAR(x_t.data()[0], signal * 1.0f + noise * 0.25f, 1e-5f);
    EXPECT_NEAR(x_t.data()[1], signal * -2.0f + noise * 3.0f, 1e-5f);
    EXPECT_NEAR(x_t.data()[2], signal * 0.5f + noise * -1.5f, 1e-5f);
}

TEST_F(NoiseScheduleTest, AddNoiseWithZeroEpsilonScalesX0Only) {
    Tensor x0(Shape({2, 2}), &backend, {1.0f, -2.0f, 0.5f, 7.0f});
    Tensor eps(Shape({2, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f});

    Tensor x_t = coarse.add_noise(x0, eps, 1);

    const float signal = std::sqrt(0.9f);
    for (int64_t i = 0; i < x_t.numel(); ++i) {
        EXPECT_NEAR(x_t.data()[i], signal * x0.data()[i], 1e-5f) << "index " << i;
    }
}

TEST_F(NoiseScheduleTest, AddNoisePreservesShape) {
    Tensor x0(Shape({4, 5}), &backend);
    Tensor eps(Shape({4, 5}), &backend);

    Tensor x_t = coarse.add_noise(x0, eps, 2);

    EXPECT_EQ(x_t.shape().rank(), 2);
    EXPECT_EQ(x_t.shape().dim(0), 4);
    EXPECT_EQ(x_t.shape().dim(1), 5);
}

TEST_F(NoiseScheduleTest, AddNoiseThrowsOnShapeMismatch) {
    Tensor x0(Shape({2, 3}), &backend);
    Tensor eps(Shape({2, 4}), &backend);
    EXPECT_THROW({ (void)coarse.add_noise(x0, eps, 1); }, std::invalid_argument);
}

TEST_F(NoiseScheduleTest, AddNoiseThrowsOnOutOfRangeTimestep) {
    Tensor x0(Shape({1, 2}), &backend);
    Tensor eps(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)coarse.add_noise(x0, eps, 0); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.add_noise(x0, eps, 6); }, std::out_of_range);
}

// Late timesteps must destroy more signal than early ones -- the entire point of a schedule.
TEST_F(NoiseScheduleTest, LaterTimestepsCarryLessOfTheOriginalSignal) {
    Tensor x0(Shape({1, 1}), &backend, {1.0f});
    Tensor eps(Shape({1, 1}), &backend, {0.0f});

    float previous = 2.0f;
    for (int64_t t = 1; t <= 5; ++t) {
        const float value = coarse.add_noise(x0, eps, t).data()[0];
        EXPECT_LT(value, previous) << "t = " << t;
        previous = value;
    }
}

// ---------------------------------------------------------------------------------------
// denoise_step -- reverse (sampling) process.
// ---------------------------------------------------------------------------------------

TEST_F(NoiseScheduleTest, DenoiseStepMatchesPublishedFormulaByHand) {
    // t = 3 -> alpha = 0.7, beta = 0.3, alpha_bar = 0.504.
    const float inv_sqrt_alpha = 1.0f / std::sqrt(0.7f);
    const float eps_coefficient = 0.3f / std::sqrt(0.496f);
    const float z_scale = std::sqrt(0.3f);

    Tensor x_t(Shape({1, 3}), &backend, {0.8f, -1.2f, 2.0f});
    Tensor eps_theta(Shape({1, 3}), &backend, {0.3f, 0.9f, -0.4f});
    Tensor z(Shape({1, 3}), &backend, {1.0f, -0.5f, 0.25f});

    Tensor x_prev = coarse.denoise_step(x_t, eps_theta, z, 3);

    ASSERT_EQ(x_prev.numel(), 3);
    for (int64_t i = 0; i < 3; ++i) {
        const float expected =
            inv_sqrt_alpha * (x_t.data()[i] - eps_coefficient * eps_theta.data()[i]) + z_scale * z.data()[i];
        EXPECT_NEAR(x_prev.data()[i], expected, 1e-5f) << "index " << i;
    }
    // Non-vacuity: the step genuinely moved every element.
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_GT(std::fabs(x_prev.data()[i] - x_t.data()[i]), 1e-3f) << "index " << i;
    }
}

TEST_F(NoiseScheduleTest, DenoiseStepWithZeroZIsTheDeterministicMean) {
    // The t == 1 final-step case: z = 0 leaves only the scaled mean term.
    Tensor x_t(Shape({1, 2}), &backend, {0.6f, -0.4f});
    Tensor eps_theta(Shape({1, 2}), &backend, {0.2f, 0.1f});
    Tensor zeros(Shape({1, 2}), &backend, {0.0f, 0.0f});

    Tensor x_prev = coarse.denoise_step(x_t, eps_theta, zeros, 1);

    const float inv_sqrt_alpha = 1.0f / std::sqrt(0.9f);
    const float eps_coefficient = 0.1f / std::sqrt(1.0f - 0.9f);
    EXPECT_NEAR(x_prev.data()[0], inv_sqrt_alpha * (0.6f - eps_coefficient * 0.2f), 1e-5f);
    EXPECT_NEAR(x_prev.data()[1], inv_sqrt_alpha * (-0.4f - eps_coefficient * 0.1f), 1e-5f);
}

// A perfect noise prediction must invert add_noise's mean term: feeding the *true* epsilon
// back in recovers the posterior mean, which is the closest thing this class has to a
// round-trip identity.
TEST_F(NoiseScheduleTest, DenoiseStepWithTrueEpsilonRecoversPosteriorMean) {
    Tensor x0(Shape({1, 3}), &backend, {0.5f, -0.75f, 1.25f});
    Tensor eps(Shape({1, 3}), &backend, {0.4f, -1.1f, 0.6f});
    Tensor zeros(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});

    Tensor x_t = coarse.add_noise(x0, eps, 2);
    Tensor x_prev = coarse.denoise_step(x_t, eps, zeros, 2);

    // With the true epsilon and no added noise, x_{t-1} equals the closed-form posterior mean
    // (1/sqrt(alpha_2)) * (x_2 - (beta_2/sqrt(1-alpha_bar_2)) * eps), computed here from the
    // schedule constants directly.
    const float inv_sqrt_alpha = 1.0f / std::sqrt(0.8f);
    const float eps_coefficient = 0.2f / std::sqrt(1.0f - 0.72f);
    for (int64_t i = 0; i < 3; ++i) {
        const float expected = inv_sqrt_alpha * (x_t.data()[i] - eps_coefficient * eps.data()[i]);
        EXPECT_NEAR(x_prev.data()[i], expected, 1e-5f) << "index " << i;
        EXPECT_TRUE(std::isfinite(x_prev.data()[i]));
    }
}

TEST_F(NoiseScheduleTest, DenoiseStepThrowsOnMismatchedPredictedEpsilonShape) {
    Tensor x_t(Shape({2, 3}), &backend);
    Tensor eps_theta(Shape({2, 4}), &backend);
    Tensor z(Shape({2, 3}), &backend);
    EXPECT_THROW({ (void)coarse.denoise_step(x_t, eps_theta, z, 1); }, std::invalid_argument);
}

TEST_F(NoiseScheduleTest, DenoiseStepThrowsOnMismatchedZShape) {
    Tensor x_t(Shape({2, 3}), &backend);
    Tensor eps_theta(Shape({2, 3}), &backend);
    Tensor z(Shape({3, 3}), &backend);
    EXPECT_THROW({ (void)coarse.denoise_step(x_t, eps_theta, z, 1); }, std::invalid_argument);
}

TEST_F(NoiseScheduleTest, DenoiseStepThrowsOnOutOfRangeTimestep) {
    Tensor x_t(Shape({1, 2}), &backend);
    Tensor eps_theta(Shape({1, 2}), &backend);
    Tensor z(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)coarse.denoise_step(x_t, eps_theta, z, 0); }, std::out_of_range);
    EXPECT_THROW({ (void)coarse.denoise_step(x_t, eps_theta, z, 6); }, std::out_of_range);
}

// ---------------------------------------------------------------------------------------
// Device guards. Both methods dereference Tensor::data() directly in raw host loops --
// undefined behavior on a CUDA-backed Tensor, so both are PULSATRIX_ASSERT-guarded
// (mission_host_loop_guards.md). No real GPU needed: this reuses LinearModuleDeathTest's
// mislabeled-Tensor pattern.
// ---------------------------------------------------------------------------------------

using NoiseScheduleDeathTest = NoiseScheduleTest;

TEST_F(NoiseScheduleDeathTest, AddNoiseAbortsOnNonCpuTensor) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x0(Shape({1, 2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    Tensor eps(Shape({1, 2}), &backend, {0.0f, 0.0f});
    EXPECT_DEATH({ (void)coarse.add_noise(x0, eps, 1); }, "PULSATRIX_ASSERT failed");
}

TEST_F(NoiseScheduleDeathTest, DenoiseStepAbortsOnNonCpuTensor) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x_t(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor eps_theta(Shape({1, 2}), &backend, {0.0f, 0.0f});
    Tensor z(Shape({1, 2}), &backend, {0.0f, 0.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)coarse.denoise_step(x_t, eps_theta, z, 1); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
