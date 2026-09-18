#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/relu_module.hpp"

namespace exai {
namespace {

class ReluModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    ReluModule relu{&backend};
};

TEST_F(ReluModuleTest, ForwardClampsNegativeValuesToZero) {
    Tensor x(Shape({5}), &backend, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    Tensor y = relu.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[3], 0.5f);
    EXPECT_FLOAT_EQ(y.data()[4], 2.0f);
}

TEST_F(ReluModuleTest, BackwardPassesGradientThroughPositiveInputsOnly) {
    Tensor x(Shape({4}), &backend, {-1.0f, 0.0f, 1.0f, 2.0f});
    (void)relu.forward(x);  // must forward first -- backward needs the cached input

    Tensor grad_y(Shape({4}), &backend, {10.0f, 10.0f, 10.0f, 10.0f});
    Tensor grad_x = relu.backward(grad_y);

    EXPECT_FLOAT_EQ(grad_x.data()[0], 0.0f);   // x=-1: blocked
    EXPECT_FLOAT_EQ(grad_x.data()[1], 0.0f);   // x=0: project convention -- treated as blocked, not passed
    EXPECT_FLOAT_EQ(grad_x.data()[2], 10.0f);  // x=1: passed through
    EXPECT_FLOAT_EQ(grad_x.data()[3], 10.0f);  // x=2: passed through
}

}  // namespace
}  // namespace exai
