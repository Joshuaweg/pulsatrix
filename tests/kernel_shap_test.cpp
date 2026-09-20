#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/kernel_shap.hpp"
#include "exai/linear_module.hpp"

// KernelSHAP (theory: xai_context.aDNA's technique_shap.md). Correctness oracle: for a
// linear-only network, Shapley values have a known, exact closed form
// phi_i = w_i * (x_i - baseline_i) -- a stronger, exact-not-just-axiom-approximate check
// than Integrated Gradients' own tolerance-bounded completeness axiom.
namespace exai {
namespace {

class KernelSHAPTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// n=1 base case -- no coalition exists between empty and full, phi_1 = f(x) - f(baseline)
// exactly, no regression involved at all.
TEST_F(KernelSHAPTest, SingleFeatureEqualsFullDifference) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({3.0f});
    linear.set_bias({0.0f});

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({1}), &backend, {5.0f});
    Tensor baseline(Shape({1}), &backend, {0.0f});

    KernelSHAP shap;
    Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);

    EXPECT_EQ(attr.method, "kernel_shap");
    EXPECT_NEAR(attr.values.data()[0], 15.0f, 1e-3f);  // 3*5 - 3*0
}

// n=2 case, matching this mission's Recon hand-derivation exactly.
TEST_F(KernelSHAPTest, TwoFeaturesMatchClosedFormShapleyValues) {
    LinearModule linear(2, 1, &backend);
    linear.set_weight({2.0f, -3.0f});
    linear.set_bias({0.0f});

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor baseline(Shape({2}), &backend, {0.0f, 0.0f});

    KernelSHAP shap;
    Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);

    EXPECT_NEAR(attr.values.data()[0], 2.0f * 1.0f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[1], -3.0f * 2.0f, 1e-3f);
}

// n=3 case -- closed-form values plus an independent efficiency-axiom check.
TEST_F(KernelSHAPTest, ThreeFeaturesMatchClosedFormAndSatisfyEfficiency) {
    LinearModule linear(3, 1, &backend);
    linear.set_weight({2.0f, -3.0f, 5.0f});
    linear.set_bias({100.0f});  // bias cancels in every f(S)-f(baseline) difference

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});

    KernelSHAP shap;
    Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);

    EXPECT_NEAR(attr.values.data()[0], 2.0f * 1.0f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[1], -3.0f * 2.0f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[2], 5.0f * 3.0f, 1e-3f);

    float sum_phi = attr.values.data()[0] + attr.values.data()[1] + attr.values.data()[2];
    Tensor out_x = predict(input);
    Tensor out_baseline = predict(baseline);
    EXPECT_NEAR(sum_phi, out_x.data()[0] - out_baseline.data()[0], 1e-3f);
}

using KernelSHAPDeathTest = KernelSHAPTest;

TEST_F(KernelSHAPDeathTest, AbortsOnMismatchedInputAndBaselineShapes) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});

    KernelSHAP shap;
    EXPECT_DEATH({ (void)shap.explain(predict, input, baseline, 0, &backend); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
