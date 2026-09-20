#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/linear_module.hpp"
#include "exai/pdp.hpp"

// PDP -- the first genuinely global (not per-instance) technique this project has built.
// Correctness oracle: for a linear-only network, the PDP curve is exactly linear in the
// swept feature's value, with slope equal to that feature's weight, regardless of
// background content -- a third independent cross-validation of the same weight-column
// ground truth Saliency (Phase 2) and LIME (this campaign's Mission 0) already established.
namespace exai {
namespace {

class PDPTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT-only to a real throw -- external boundary (background
// originates from caller-supplied data).
TEST_F(PDPTest, ExplainThrowsOnEmptyBackground) {
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    std::vector<Tensor> empty_background;
    PDP pdp;
    EXPECT_THROW({ (void)pdp.explain(predict, empty_background, 0, 0, 0.0f, 1.0f, 5, &backend); },
                 std::invalid_argument);
}

TEST_F(PDPTest, CurveIsExactlyLinearWithTrueFeatureWeightAsSlope) {
    LinearModule linear(3, 1, &backend);
    linear.set_weight({2.0f, -3.0f, 5.0f});
    linear.set_bias({100.0f});

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    std::vector<Tensor> background;
    background.emplace_back(Shape({3}), &backend, std::initializer_list<float>{1.0f, 1.0f, 1.0f});
    background.emplace_back(Shape({3}), &backend, std::initializer_list<float>{5.0f, -2.0f, 0.0f});
    background.emplace_back(Shape({3}), &backend, std::initializer_list<float>{-3.0f, 4.0f, 2.0f});

    PDP pdp;
    Attribution attr = pdp.explain(predict, background, /*feature_index=*/1, /*target_index=*/0, /*grid_min=*/-2.0f,
                                    /*grid_max=*/4.0f, /*grid_size=*/7, &backend);

    EXPECT_EQ(attr.method, "pdp");
    ASSERT_EQ(attr.values.numel(), 7);

    // Empirical slope between the first and last grid points must equal the true weight
    // (-3.0) exactly, regardless of the constant offset from the other features/bias.
    float v0 = attr.values.data()[0];
    float v6 = attr.values.data()[6];
    float grid_step = (4.0f - (-2.0f)) / (7 - 1);
    float empirical_slope = (v6 - v0) / (6 * grid_step);
    EXPECT_NEAR(empirical_slope, -3.0f, 1e-3f);
}

TEST_F(PDPTest, SingleBackgroundInstanceDegeneratesToPerInstanceLine) {
    LinearModule linear(2, 1, &backend);
    linear.set_weight({2.0f, 7.0f});
    linear.set_bias({0.0f});

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    std::vector<Tensor> background;
    background.emplace_back(Shape({2}), &backend, std::initializer_list<float>{10.0f, 10.0f});

    PDP pdp;
    Attribution attr =
        pdp.explain(predict, background, /*feature_index=*/0, /*target_index=*/0, 0.0f, 5.0f, 6, &backend);

    // background[0] fixed feature 1 = 10 -> contributes 7*10=70 constant. Sweeping feature 0
    // from 0 to 5: PDP(v) = 2*v + 70.
    for (int64_t k = 0; k < 6; ++k) {
        float v = static_cast<float>(k);  // grid step is 1.0 here (0..5 over 6 points)
        EXPECT_NEAR(attr.values.data()[k], 2.0f * v + 70.0f, 1e-3f) << "mismatch at grid index " << k;
    }
}

TEST_F(PDPTest, GridSizeOneEvaluatesSinglePointCorrectly) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({4.0f});
    linear.set_bias({1.0f});

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    std::vector<Tensor> background;
    background.emplace_back(Shape({1}), &backend, std::initializer_list<float>{0.0f});

    PDP pdp;
    Attribution attr =
        pdp.explain(predict, background, /*feature_index=*/0, /*target_index=*/0, /*grid_min=*/3.0f,
                    /*grid_max=*/3.0f, /*grid_size=*/1, &backend);

    ASSERT_EQ(attr.values.numel(), 1);
    EXPECT_NEAR(attr.values.data()[0], 4.0f * 3.0f + 1.0f, 1e-3f);
}

using PDPDeathTest = PDPTest;

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT (this death test) to a real throw -- see
// PDPTest.ExplainThrowsOnEmptyBackground below.

}  // namespace
}  // namespace exai
