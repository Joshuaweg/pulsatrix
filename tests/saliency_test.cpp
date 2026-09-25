#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/saliency.hpp"

// Saliency (charter Part 1: "raw gradient") is the campaign's first explainer that
// actually produces an Attribution. It wraps ExplainerContext::forward_pass + a
// one-hot-seeded backward_pass -- no new theory beyond what Autograd/ExplainerContext
// already implement.
namespace pulsatrix {
namespace {

class SaliencyTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Single LinearModule, no nonlinearity in the way: d(output[j])/d(input[i]) = weight[i][j]
// exactly -- bias doesn't participate in the gradient at all, deliberately set to a large
// value to prove it has zero effect.
TEST_F(SaliencyTest, GradientMatchesWeightColumnForLinearOnlyNetwork) {
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // (in=3, out=2), row-major
    linear.set_bias({100.0f, 100.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

    Saliency saliency;
    Attribution attr = saliency.explain(ctx, input, /*target_index=*/1, &backend);

    EXPECT_EQ(attr.method, "saliency");
    EXPECT_FLOAT_EQ(attr.values.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[1], 4.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[2], 6.0f);
    EXPECT_EQ(attr.metadata.at("target_index"), "1");
}

TEST_F(SaliencyTest, DifferentTargetIndexSelectsDifferentWeightColumn) {
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

    Saliency saliency;
    Attribution attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);

    EXPECT_FLOAT_EQ(attr.values.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[1], 3.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[2], 5.0f);
    EXPECT_EQ(attr.metadata.at("target_index"), "0");
}

}  // namespace
}  // namespace pulsatrix
