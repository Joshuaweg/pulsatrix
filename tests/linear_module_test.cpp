#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/lrp_rule_config.hpp"

namespace exai {
namespace {

class LinearModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(LinearModuleTest, ConstructionSetsWeightAndBiasShapes) {
    LinearModule linear(3, 2, &backend);
    EXPECT_EQ(linear.weight().shape(), Shape({3, 2}));
    EXPECT_EQ(linear.bias().shape(), Shape({2}));
}

TEST_F(LinearModuleTest, WeightAndBiasGradientsStartAtZero) {
    LinearModule linear(3, 2, &backend);
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 0.0f);
}

TEST_F(LinearModuleTest, ForwardComputesHandVerifiedOutput) {
    // in_features=2, out_features=2. W = [[1,2],[3,4]] (in x out, row-major), b = [0.5, -0.5].
    // x = [1, 1]. y = x @ W + b = [1*1+1*3, 1*2+1*4] + [0.5,-0.5] = [4,6] + [0.5,-0.5] = [4.5, 5.5]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.5f, -0.5f});

    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor y = linear.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 4.5f);
    EXPECT_FLOAT_EQ(y.data()[1], 5.5f);
}

TEST_F(LinearModuleTest, BackwardComputesHandVerifiedInputGradient) {
    // Same W as above. grad_y = [1, 1]. grad_x = grad_y @ W^T.
    // W^T = [[1,3],[2,4]]. grad_x = [1*1+1*2, 1*3+1*4] = [3, 7]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.0f, 0.0f});

    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);  // must forward first -- backward needs the cached input

    Tensor grad_y(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor grad_x = linear.backward(grad_y);

    EXPECT_FLOAT_EQ(grad_x.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[1], 7.0f);
}

TEST_F(LinearModuleTest, BackwardAccumulatesWeightGradientAsOuterProduct) {
    // grad_W = outer(x, grad_y). x=[1,2], grad_y=[3,4] -> grad_W = [[1*3,1*4],[2*3,2*4]] = [[3,4],[6,8]]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
    linear.set_bias({0.0f, 0.0f});

    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {3.0f, 4.0f});
    (void)linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 3.0f);  // W[0][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[1], 4.0f);  // W[0][1]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[2], 6.0f);  // W[1][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[3], 8.0f);  // W[1][1]
}

TEST_F(LinearModuleTest, BackwardAccumulatesBiasGradientAsGradOutput) {
    LinearModule linear(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {2.5f, -1.5f});
    (void)linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.5f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[1], -1.5f);
}

TEST_F(LinearModuleTest, GradientsAccumulateAcrossTwoBackwardCalls) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({1.0f});
    linear.set_bias({0.0f});

    Tensor x(Shape({1}), &backend, {2.0f});
    Tensor grad_y(Shape({1}), &backend, {1.0f});

    (void)linear.forward(x);
    (void)linear.backward(grad_y);  // grad_W += 2*1 = 2
    (void)linear.forward(x);
    (void)linear.backward(grad_y);  // grad_W += 2*1 = 2 again -> total 4

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 4.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.0f);
}

// Bias is deliberately excluded from the LRP rule's z_j (the pre-bias linear output is
// used, not the forward() return value) -- this is what makes exact conservation possible;
// see this mission's Notes for the full rationale (bias has no associated input feature to
// redistribute relevance to, a well-known LRP simplification).
TEST_F(LinearModuleTest, PropagateRelevanceMatchesHandDerivedValues) {
    // W = [[2],[3]] (2 in, 1 out). x = [1, 2]. z_0 (pre-bias) = 1*2 + 2*3 = 8.
    LinearModule linear(2, 1, &backend);
    linear.set_weight({2.0f, 3.0f});
    linear.set_bias({100.0f});  // deliberately large/irrelevant -- must not affect the result

    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)linear.forward(x);

    Tensor relevance_out(Shape({1}), &backend, {10.0f});
    LRPRuleConfig config;
    config.epsilon = 0.0f;  // exact -- no absorption term, isolates the rule's core correctness

    Tensor relevance_in = linear.propagate_relevance(relevance_out, config);

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 2.5f);  // (1*2/8)*10
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 7.5f);  // (2*3/8)*10
}

TEST_F(LinearModuleTest, PropagateRelevanceConservesTotalRelevance) {
    // The mission's real acceptance criterion for the LRP half -- see mission_module_linear.md.
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, -2.0f, 3.0f, 0.5f, 2.5f, -1.0f});
    linear.set_bias({0.1f, -0.2f});

    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 0.5f});
    (void)linear.forward(x);

    Tensor relevance_out(Shape({2}), &backend, {4.0f, 6.0f});
    LRPRuleConfig config;  // default epsilon (1e-6) -- proves conservation holds in normal use, not just the epsilon=0 special case

    Tensor relevance_in = linear.propagate_relevance(relevance_out, config);

    float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
    float sum_out = relevance_out.data()[0] + relevance_out.data()[1];
    EXPECT_NEAR(sum_in, sum_out, 1e-3f);
}

}  // namespace
}  // namespace exai
