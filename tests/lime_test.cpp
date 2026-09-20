#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/lime.hpp"
#include "exai/linear_module.hpp"

// LIME (theory: xai_context.aDNA's technique_lime.md). Correctness oracle: run against a
// network that is already exactly linear (a bare LinearModule, no nonlinearity) --
// weighted least squares on noise-free linear data recovers the true coefficients exactly
// regardless of sample count or kernel width, giving the same hand-verifiable rigor
// Saliency's own test used (the same weight-column oracle), and cross-validating two
// independently-designed explainers against the same ground truth.
namespace exai {
namespace {

class LIMETest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT-only to a real throw -- num_samples originates from
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
}

using LIMEDeathTest = LIMETest;

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT (this death test) to a real throw -- see
// LIMETest.ExplainThrowsOnNonPositiveNumSamples below.

}  // namespace
}  // namespace exai
