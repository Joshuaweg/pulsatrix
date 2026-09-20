#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/integrated_gradients.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// Integrated Gradients (charter: "graph traversal + baseline interpolation" -- the real
// test of whether Phase 0's graph design was done right). Theory:
// what/context/xai/domain/modalities/xai_vision_models/vision_integrated_gradients.md
// (xai_context.aDNA). Correctness is the completeness axiom:
// sum(IG(x)) == F(x) - F(baseline), not just "it runs" -- per this vault's TDD discipline
// and the charter's Explanation Fidelity audit row.
namespace exai {
namespace {

class IntegratedGradientsTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(IntegratedGradientsTest, CompletenessAxiomHoldsWithinTolerance) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});

    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});
    constexpr int64_t target_index = 0;

    Tensor output_x = ctx.forward_pass(input);
    float f_x = output_x.data()[target_index];
    Tensor output_baseline = ctx.forward_pass(baseline);
    float f_baseline = output_baseline.data()[target_index];

    IntegratedGradients ig;
    Attribution attr = ig.explain(ctx, input, baseline, target_index, /*steps=*/200, &backend);

    EXPECT_EQ(attr.method, "integrated_gradients");
    EXPECT_EQ(attr.metadata.at("steps"), "200");

    float sum_ig = 0.0f;
    for (int64_t i = 0; i < attr.values.numel(); ++i) {
        sum_ig += attr.values.data()[i];
    }
    EXPECT_NEAR(sum_ig, f_x - f_baseline, 1e-3f)
        << "completeness axiom violated: sum(IG) should equal F(x) - F(baseline)";
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 2):
// escalated from EXAI_ASSERT (was a death test) to a real throw -- external boundary.
TEST_F(IntegratedGradientsTest, ExplainThrowsOnZeroSteps) {
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor baseline(Shape({2}), &backend, {0.0f, 0.0f});

    IntegratedGradients ig;
    EXPECT_THROW({ (void)ig.explain(ctx, input, baseline, 0, /*steps=*/0, &backend); }, std::invalid_argument);
}

TEST_F(IntegratedGradientsTest, ExplainThrowsOnMismatchedInputAndBaselineShapes) {
    LinearModule linear(2, 1, &backend);
    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor mismatched_baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});

    IntegratedGradients ig;
    EXPECT_THROW({ (void)ig.explain(ctx, input, mismatched_baseline, 0, /*steps=*/10, &backend); },
                 std::invalid_argument);
}

}  // namespace
}  // namespace exai
