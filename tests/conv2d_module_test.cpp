#include <gtest/gtest.h>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"

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

TEST_F(Conv2DModuleTest, BackwardComputesHandVerifiedGradients) {
    // Same 1x1-channel, 3x3 input, 2x2 kernel setup as the forward test. grad_output = all
    // ones -- deliberately chosen so grad_kernel[p] = sum over q of im2col[p][q], and
    // grad_input becomes a pure "how many times was each input pixel touched, weighted by
    // kernel value" map. The center input pixel (1,1) is touched by all 4 output positions
    // (genuine overlap -- this is what col2im's summing exists to get right).
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor grad_output(Shape({1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_input = conv.backward(grad_output);

    // grad_kernel = sum over the 4 patches of each kernel position's contributing pixel.
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[0], 12.0f);  // k00: 1+2+4+5
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[1], 16.0f);  // k01: 2+3+5+6
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[2], 24.0f);  // k10: 4+5+7+8
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[3], 28.0f);  // k11: 5+6+8+9

    EXPECT_FLOAT_EQ(conv.bias_grad().data()[0], 4.0f);  // sum of 4 output positions, all grad 1

    EXPECT_EQ(grad_input.shape(), Shape({1, 3, 3}));
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 2}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 1, 1}), 2.0f);  // overlap: touched by all 4 patches
    EXPECT_FLOAT_EQ(grad_input.at({0, 1, 2}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 2, 0}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 2, 1}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 2, 2}), 1.0f);
}

TEST_F(Conv2DModuleTest, BackwardAccumulatesGradientsAcrossTwoCalls) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor grad_output(Shape({1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});

    (void)conv.forward(input);
    (void)conv.backward(grad_output);
    (void)conv.forward(input);
    (void)conv.backward(grad_output);

    EXPECT_FLOAT_EQ(conv.bias_grad().data()[0], 8.0f);  // 4 + 4
}

// Conservation is this mission's real acceptance criterion for the LRP half (see
// mission_conv2d_losses.md's exit gate) -- the rule structurally reuses LinearModule's
// epsilon rule per output position via the im2col representation, then col2im's the
// per-position relevance contributions back to input space (summing overlaps, same
// mechanism backward()'s col2im already proved correct in Objective 3).
TEST_F(Conv2DModuleTest, PropagateRelevanceConservesTotalRelevance) {
    Conv2DModule conv(1, 2, 2, 2, &backend);
    conv.set_kernel({1.0f, -0.5f, 0.5f, 2.0f, -1.0f, 1.5f, 0.5f, -0.5f});
    conv.set_bias({0.1f, -0.2f});

    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor relevance_out(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f});
    LRPRuleConfig config;  // default epsilon -- proves conservation holds in normal use

    Tensor relevance_in = conv.propagate_relevance(relevance_out, config);

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];

    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

TEST_F(Conv2DModuleTest, ParametersExposesKernelAndBiasByPointer) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    auto params = conv.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].value, &conv.kernel());
    EXPECT_EQ(params[0].grad, &conv.kernel_grad());
    EXPECT_EQ(params[1].value, &conv.bias());
    EXPECT_EQ(params[1].grad, &conv.bias_grad());
}

using Conv2DModuleDeathTest = Conv2DModuleTest;

// Unlike LinearModule/ReluModule, ALL THREE of Conv2DModule's methods dereference
// Tensor::data() in raw host loops (forward_impl's bias-add loop and its im2col() helper;
// backward's transpose2d()/col2im(); propagate_relevance's own loop and col2im()) --
// undefined behavior on a CUDA-backed Tensor. Phase 1.5 Mission 2
// (mission_host_loop_guards.md) guards all three with EXAI_ASSERT. No real GPU needed: see
// LinearModuleDeathTest for the mislabeled-Tensor testing pattern this reuses.
TEST_F(Conv2DModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                 DeviceType::Cuda);
    EXPECT_DEATH({ (void)conv.forward(input); }, "EXAI_ASSERT failed");
}

TEST_F(Conv2DModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor grad_output(Shape({1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)conv.backward(grad_output); }, "EXAI_ASSERT failed");
}

TEST_F(Conv2DModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor relevance_out(Shape({1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f}, DeviceType::Cuda);
    LRPRuleConfig config;
    EXPECT_DEATH({ (void)conv.propagate_relevance(relevance_out, config); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
