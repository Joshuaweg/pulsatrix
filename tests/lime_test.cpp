#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/linear_module.hpp"

// LIME (theory: xai_context.aDNA's technique_lime.md). Correctness oracle: run against a
// network that is already exactly linear (a bare LinearModule, no nonlinearity) --
// weighted least squares on noise-free linear data recovers the true coefficients exactly
// regardless of sample count or kernel width, giving the same hand-verifiable rigor
// Saliency's own test used (the same weight-column oracle), and cross-validating two
// independently-designed explainers against the same ground truth.
namespace pulsatrix {
namespace {

class LIMETest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from PULSATRIX_ASSERT-only to a real throw -- num_samples originates from
// caller-supplied data. In Release, num_samples<=0 previously cast to a huge size_t for
// vector::reserve, throwing an unhelpful std::length_error instead of a clear error.
TEST_F(LIMETest, ExplainThrowsOnNonPositiveNumSamples) {
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };
    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});

    LIME lime;
    EXPECT_THROW({ (void)lime.explain(predict, input, 0, /*num_samples=*/0, 1.0f, 0.0f, 42, &backend); },
                 std::invalid_argument);
}

TEST_F(LIMETest, ExplainThrowsOnNonPositiveSigma) {
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };
    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});

    LIME lime;
    EXPECT_THROW({ (void)lime.explain(predict, input, 0, /*num_samples=*/50, /*sigma=*/0.0f, 0.0f, 42, &backend); },
                 std::invalid_argument);
}

TEST_F(LIMETest, RecoversExactWeightColumnForLinearOnlyNetwork) {
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    linear.set_bias({100.0f, 100.0f});  // deliberately large/irrelevant -- must not affect the result

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

    LIME lime;
    Attribution attr =
        lime.explain(predict, input, /*target_index=*/1, /*num_samples=*/300, /*sigma=*/1.0f,
                     /*l2_lambda=*/0.0f, /*seed=*/42, &backend);

    EXPECT_EQ(attr.method, "lime");
    EXPECT_NEAR(attr.values.data()[0], 2.0f, 1e-2f);
    EXPECT_NEAR(attr.values.data()[1], 4.0f, 1e-2f);
    EXPECT_NEAR(attr.values.data()[2], 6.0f, 1e-2f);
    EXPECT_EQ(attr.metadata.at("target_index"), "1");
    // A linear model is fitted exactly.
    EXPECT_NEAR(std::stod(attr.metadata.at("local_fidelity")), 1.0, 1e-4);
}

// XAI-4: the local fidelity says when the coefficients don't describe the model. At the bottom of a
// bowl, f(z) = |z|^2, every direction is flat to first order: the fitted slopes are noise and the
// fit explains almost none of the variation. Moved off-center, the slope dominates.
TEST_F(LIMETest, LocalFidelityIsLowWhereNoLinearModelFits) {
    auto bowl = [this](const Tensor& x) {
        float sum = 0.0f;
        for (float v : x.to_host_vector()) sum += v * v;
        return Tensor(Shape({1, 1}), &backend, {sum});
    };
    LIME lime;
    const Attribution center = lime.explain(bowl, Tensor(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f}), 0, 500, 0.5f,
                                            0.0f, 7, &backend);
    const Attribution side = lime.explain(bowl, Tensor(Shape({1, 3}), &backend, {4.0f, 0.0f, 0.0f}), 0, 500, 0.5f,
                                          0.0f, 7, &backend);
    EXPECT_LT(std::stod(center.metadata.at("local_fidelity")), 0.1);  // negative: no intercept
    EXPECT_GT(std::stod(side.metadata.at("local_fidelity")), 0.9);
}

TEST(LIMELocalFidelity, IsTheWeightedRSquared) {
    // Targets 1, 2, 3 at weights 1, 1, 2 (weighted mean 2.25); the surrogate predicts 1, 2, 2.
    const std::vector<std::vector<float>> samples = {{1.0f}, {2.0f}, {2.0f}};
    const std::vector<float> targets = {1.0f, 2.0f, 3.0f}, weights = {1.0f, 1.0f, 2.0f};
    // residual = 2 * 1^2 = 2; total = 1.5625 + 0.0625 + 2 * 0.5625 = 2.75
    EXPECT_NEAR(LIME::local_fidelity(samples, targets, weights, {1.0f}), 1.0 - 2.0 / 2.75, 1e-12);
    EXPECT_EQ(LIME::local_fidelity(samples, {2.0f, 2.0f, 2.0f}, weights, {0.0f}), 0.0);  // constant, missed
    EXPECT_EQ(LIME::local_fidelity({{0.0f}}, {0.0f}, {1.0f}, {5.0f}), 1.0);              // constant, hit
}

using LIMEDeathTest = LIMETest;

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from PULSATRIX_ASSERT (this death test) to a real throw -- see
// LIMETest.ExplainThrowsOnNonPositiveNumSamples below.

}  // namespace
}  // namespace pulsatrix
