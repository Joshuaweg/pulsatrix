#include <gtest/gtest.h>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"

namespace exai {
namespace {

class Conv2DModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(Conv2DModuleTest, ConstructionSetsKernelAndBiasShapes) {
    Conv2DModule conv(1, 2, 2, 2, &backend);  // in=1, out=2, 2x2 kernel
    EXPECT_EQ(conv.kernel().shape(), Shape({2, 1, 2, 2}));
    EXPECT_EQ(conv.bias().shape(), Shape({2}));
}

TEST_F(Conv2DModuleTest, ForwardComputesHandVerifiedOutput) {
    // 1 in-channel, 1 out-channel, 3x3 input, 2x2 kernel, stride 1 -> 2x2 output.
    // input = [[1,2,3],[4,5,6],[7,8,9]], kernel = [[1,0],[0,1]] (picks top-left + bottom-right).
    // patch(0,0)=[[1,2],[4,5]] -> 1*1+2*0+4*0+5*1=6
    // patch(0,1)=[[2,3],[5,6]] -> 2*1+3*0+5*0+6*1=8
    // patch(1,0)=[[4,5],[7,8]] -> 4*1+5*0+7*0+8*1=12
    // patch(1,1)=[[5,6],[8,9]] -> 5*1+6*0+8*0+9*1=14
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor output = conv.forward(input);

    EXPECT_EQ(output.shape(), Shape({1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 1}), 14.0f);
}

TEST_F(Conv2DModuleTest, ForwardAddsBiasPerOutputChannel) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({100.0f});

    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor output = conv.forward(input);

    EXPECT_FLOAT_EQ(output.at({0, 0, 0}), 106.0f);  // 6 + 100
    EXPECT_FLOAT_EQ(output.at({0, 1, 1}), 114.0f);  // 14 + 100
}

}  // namespace
}  // namespace exai
