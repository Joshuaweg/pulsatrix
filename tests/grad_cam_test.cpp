#include <gtest/gtest.h>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/flatten_module.hpp"
#include "exai/grad_cam.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// Grad-CAM (charter: "Grad-CAM analog for conv layers"; theory:
// xai_context.aDNA's vision_gradcam.md). Baseline algorithm: find the last OpType::Conv
// node, global-average-pool its gradient per channel to get alpha_k, weighted-sum its
// activation channels, ReLU the result. Correctness verified by hand-derivation, per this
// vault's TDD discipline -- not just "it runs".
namespace exai {
namespace {

class GradCAMTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Hand-derived case. Conv2DModule(1,2,2,2) on a 3x3 input -> ReluModule (identity here,
// all conv outputs are positive by construction) -> FlattenModule -> LinearModule(8,2).
//
// input = [[1,2,3],[4,0,5],[6,7,8]] (asymmetric, so the two conv channels differ).
// kernel channel0 = [[1,0],[0,1]] (top-left + bottom-right), channel1 = [[0,1],[1,0]]
// (top-right + bottom-left).
//   A^0 = [[1,7],[11,8]]   (patch tl+br per 2x2 window)
//   A^1 = [[6,3],[6,12]]   (patch tr+bl per 2x2 window)
//
// LinearModule weight's column 0 (target_index=0) is [1,1,1,1,2,2,2,2] over the flattened
// (channel-major) order -- so seeding backward with a one-hot at index 0 gives a gradient
// at the conv node of channel0=[[1,1],[1,1]], channel1=[[2,2],[2,2]] exactly (Linear's
// backward: grad_input[i] = W[i][target_index], and ReLU/Flatten are identity here).
//   alpha_0 = mean(1,1,1,1) = 1
//   alpha_1 = mean(2,2,2,2) = 2
//   L = ReLU(1*A^0 + 2*A^1):
//     L[0][0] = 1*1 + 2*6  = 13
//     L[0][1] = 1*7 + 2*3  = 13
//     L[1][0] = 1*11 + 2*6 = 23
//     L[1][1] = 1*8 + 2*12 = 32
TEST_F(GradCAMTest, ComputesHandDerivedCAMForSimpleNetwork) {
    Conv2DModule conv(1, 2, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    conv.set_bias({0.0f, 0.0f});

    ReluModule relu(&backend);
    FlattenModule flatten(&backend);

    LinearModule linear(8, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&conv, &relu, &flatten, &linear});
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    GradCAM gradcam;
    Attribution attr = gradcam.explain(ctx, input, /*target_index=*/0, &backend);

    EXPECT_EQ(attr.method, "grad_cam");
    EXPECT_EQ(attr.values.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(attr.values.data()[0], 13.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[1], 13.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[2], 23.0f);
    EXPECT_FLOAT_EQ(attr.values.data()[3], 32.0f);
    EXPECT_EQ(attr.metadata.at("target_index"), "0");
}

using GradCAMDeathTest = GradCAMTest;

TEST_F(GradCAMDeathTest, AbortsWhenGraphHasNoConvLayer) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LinearModule linear(2, 2, &backend);
    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});

    GradCAM gradcam;
    EXPECT_DEATH({ (void)gradcam.explain(ctx, input, 0, &backend); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
