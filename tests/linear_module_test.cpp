#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"

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
    linear.forward(x);  // must forward first -- backward needs the cached input

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
    linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {3.0f, 4.0f});
    linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 3.0f);  // W[0][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[1], 4.0f);  // W[0][1]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[2], 6.0f);  // W[1][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[3], 8.0f);  // W[1][1]
}

TEST_F(LinearModuleTest, BackwardAccumulatesBiasGradientAsGradOutput) {
    LinearModule linear(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {2.5f, -1.5f});
    linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.5f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[1], -1.5f);
}

TEST_F(LinearModuleTest, GradientsAccumulateAcrossTwoBackwardCalls) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({1.0f});
    linear.set_bias({0.0f});

    Tensor x(Shape({1}), &backend, {2.0f});
    Tensor grad_y(Shape({1}), &backend, {1.0f});

    linear.forward(x);
    linear.backward(grad_y);  // grad_W += 2*1 = 2
    linear.forward(x);
    linear.backward(grad_y);  // grad_W += 2*1 = 2 again -> total 4

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 4.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.0f);
}

}  // namespace
}  // namespace exai
