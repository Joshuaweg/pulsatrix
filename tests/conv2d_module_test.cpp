#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

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

// std::vector overloads exist alongside the initializer_list ones for runtime-sized
// callers (Phase 5 Python bindings, or any C++ caller loading a kernel from a file).
TEST_F(Conv2DModuleTest, SetKernelAcceptsVector) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f});
    EXPECT_FLOAT_EQ(conv.kernel().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(conv.kernel().data()[3], 1.0f);
}

TEST_F(Conv2DModuleTest, SetBiasAcceptsVector) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_bias(std::vector<float>{3.0f});
    EXPECT_FLOAT_EQ(conv.bias().data()[0], 3.0f);
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 1):
// forward_impl previously had no rank/channel-count validation at all before calling
// input.shape().dim(1)/dim(2) -- external boundary per Mission 0's classification table.
// Shape generalized to rank-4 (N, in_channels, H, W) by
// campaign_exai_dl_library_batch_dimension_support -- this now also confirms the old
// unbatched rank-3 shape is correctly rejected, not silently accepted.
TEST_F(Conv2DModuleTest, ForwardThrowsOnWrongRank) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor wrong_rank(Shape({1, 3, 3}), &backend);  // old-style unbatched shape, now rejected
    EXPECT_THROW({ (void)conv.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(Conv2DModuleTest, ForwardThrowsOnWrongChannelCount) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor wrong_channels(Shape({1, 2, 3, 3}), &backend);
    EXPECT_THROW({ (void)conv.forward(wrong_channels); }, std::invalid_argument);
}

// Finding 6: a kernel larger than the input previously drove out_h/out_w negative with
// no check -- silently producing a wrong-but-valid-looking positive Q when both go
// negative (e.g. out_h=-2, out_w=-3 -> Q=6), rather than erroring.
TEST_F(Conv2DModuleTest, ForwardThrowsWhenKernelTallerThanInput) {
    Conv2DModule conv(1, 1, 5, 2, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)conv.forward(small_input); }, std::invalid_argument);
}

TEST_F(Conv2DModuleTest, ForwardThrowsWhenKernelWiderThanInput) {
    Conv2DModule conv(1, 1, 2, 5, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)conv.forward(small_input); }, std::invalid_argument);
}

// Finding 12: backward()/propagate_relevance() silently computed a meaningless answer
// from zero-initialized cached state if called before any forward() -- now a real error.
TEST_F(Conv2DModuleTest, BackwardThrowsIfCalledBeforeForward) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor grad(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)conv.backward(grad); }, std::logic_error);
}

TEST_F(Conv2DModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor relevance(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)conv.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(Conv2DModuleTest, ForwardComputesHandVerifiedOutputForSingleExampleBatch) {
    // 1 in-channel, 1 out-channel, 3x3 input, 2x2 kernel, stride 1 -> 2x2 output.
    // input = [[1,2,3],[4,5,6],[7,8,9]], kernel = [[1,0],[0,1]] (picks top-left + bottom-right).
    // patch(0,0)=[[1,2],[4,5]] -> 1*1+2*0+4*0+5*1=6
    // patch(0,1)=[[2,3],[5,6]] -> 2*1+3*0+5*0+6*1=8
    // patch(1,0)=[[4,5],[7,8]] -> 4*1+5*0+7*0+8*1=12
    // patch(1,1)=[[5,6],[8,9]] -> 5*1+6*0+8*0+9*1=14
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor output = conv.forward(input);

    EXPECT_EQ(output.shape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 14.0f);
}

// The real acceptance criterion for the batch migration's forward pass -- N=2, a second
// example with different pixel values, proving each row of the batch is convolved
// independently rather than mixed together.
TEST_F(Conv2DModuleTest, ForwardComputesHandVerifiedOutputAcrossTwoExampleBatch) {
    // Row 0: as above -> [6,8,12,14]. Row 1: input=[[0,1,2],[3,4,5],[6,7,8]] (each pixel
    // one less than row 0's). patch(0,0)=0*1+1*0+3*0+4*1=4; patch(0,1)=1+5=6;
    // patch(1,0)=3+7=10; patch(1,1)=4+8=12.
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({2, 1, 3, 3}), &backend,
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                  8.0f});
    Tensor output = conv.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 12.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 14.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 1, 0}), 10.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 1, 1}), 12.0f);
}

TEST_F(Conv2DModuleTest, ForwardAddsBiasPerOutputChannel) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({100.0f});

    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor output = conv.forward(input);

    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 106.0f);  // 6 + 100
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 114.0f);  // 14 + 100
}

// New adversarial case introduced by the batch migration
// (campaign_exai_dl_library_batch_dimension_support): grad_output's batch size must match
// what forward() actually cached, not just have the right channel/spatial shape.
TEST_F(Conv2DModuleTest, BackwardThrowsOnBatchSizeMismatch) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);
    Tensor wrong_batch_grad(Shape({2, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)conv.backward(wrong_batch_grad); }, std::invalid_argument);
}

TEST_F(Conv2DModuleTest, BackwardComputesHandVerifiedGradientsForSingleExampleBatch) {
    // Same 1x1-channel, 3x3 input, 2x2 kernel setup as the forward test. grad_output = all
    // ones -- deliberately chosen so grad_kernel[p] = sum over q of im2col[p][q], and
    // grad_input becomes a pure "how many times was each input pixel touched, weighted by
    // kernel value" map. The center input pixel (1,1) is touched by all 4 output positions
    // (genuine overlap -- this is what col2im's summing exists to get right).
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_input = conv.backward(grad_output);

    // grad_kernel = sum over the 4 patches of each kernel position's contributing pixel.
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[0], 12.0f);  // k00: 1+2+4+5
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[1], 16.0f);  // k01: 2+3+5+6
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[2], 24.0f);  // k10: 4+5+7+8
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[3], 28.0f);  // k11: 5+6+8+9

    EXPECT_FLOAT_EQ(conv.bias_grad().data()[0], 4.0f);  // sum of 4 output positions, all grad 1

    EXPECT_EQ(grad_input.shape(), Shape({1, 1, 3, 3}));
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 2}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 1}), 2.0f);  // overlap: touched by all 4 patches
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 2}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 2, 0}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 2, 1}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 2, 2}), 1.0f);
}

// The real acceptance criterion for the batch migration's gradient reduction --
// kernel_grad/bias_grad must SUM contributions from both examples in the batch, and
// grad_input's two rows must be computed independently (row 1's gradient map matches row
// 0's touch-count pattern exactly, since both examples use the identical picking kernel).
TEST_F(Conv2DModuleTest, BackwardSumsKernelAndBiasGradientAcrossTwoExampleBatch) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});

    Tensor input(Shape({2, 1, 3, 3}), &backend,
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                  8.0f});
    (void)conv.forward(input);

    Tensor grad_output(Shape({2, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_input = conv.backward(grad_output);

    // Row 0 contributes kernel_grad [12,16,24,28] (as in the single-example test); row 1
    // (each pixel one less) contributes [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] shifted down
    // by 1 per pixel: [8,12,20,24]. Sum: [20,28,44,52].
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[0], 20.0f);
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[1], 28.0f);
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[2], 44.0f);
    EXPECT_FLOAT_EQ(conv.kernel_grad().data()[3], 52.0f);

    EXPECT_FLOAT_EQ(conv.bias_grad().data()[0], 8.0f);  // 4 (row 0) + 4 (row 1)

    EXPECT_EQ(grad_input.shape(), Shape({2, 1, 3, 3}));
    // Row 1's touch-count map is identical to row 0's (same kernel, same picking pattern,
    // only the pixel VALUES differ, not which pixels get touched how many times).
    EXPECT_FLOAT_EQ(grad_input.at({1, 0, 1, 1}), 2.0f);  // overlap center, same as row 0
    EXPECT_FLOAT_EQ(grad_input.at({1, 0, 0, 2}), 0.0f);
}

TEST_F(Conv2DModuleTest, BackwardAccumulatesGradientsAcrossTwoCalls) {
    Conv2DModule conv(1, 1, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f});
    conv.set_bias({0.0f});
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});

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
TEST_F(Conv2DModuleTest, PropagateRelevanceConservesTotalRelevancePerExample) {
    Conv2DModule conv(1, 2, 2, 2, &backend);
    conv.set_kernel({1.0f, -0.5f, 0.5f, 2.0f, -1.0f, 1.5f, 0.5f, -0.5f});
    conv.set_bias({0.1f, -0.2f});

    Tensor input(Shape({2, 1, 3, 3}), &backend,
                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                  8.0f});
    (void)conv.forward(input);

    Tensor relevance_out(Shape({2, 2, 2, 2}), &backend,
                          {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f,
                           0.5f});
    LRPRuleConfig config;  // default epsilon -- proves conservation holds in normal use

    Tensor relevance_in = conv.propagate_relevance(relevance_out, config);

    // Checked per-example: conservation is a per-row property, not a whole-batch sum.
    float sum_in_row0 = 0.0f;
    for (int64_t i = 0; i < 9; ++i) sum_in_row0 += relevance_in.data()[i];
    float sum_out_row0 = 0.0f;
    for (int64_t i = 0; i < 8; ++i) sum_out_row0 += relevance_out.data()[i];
    EXPECT_NEAR(sum_in_row0, sum_out_row0, 1e-2f);

    float sum_in_row1 = 0.0f;
    for (int64_t i = 9; i < 18; ++i) sum_in_row1 += relevance_in.data()[i];
    float sum_out_row1 = 0.0f;
    for (int64_t i = 8; i < 16; ++i) sum_out_row1 += relevance_out.data()[i];
    EXPECT_NEAR(sum_in_row1, sum_out_row1, 1e-2f);
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
// backward's transpose2d()/col2im_into(); propagate_relevance's own loop and
// col2im_into()) -- undefined behavior on a CUDA-backed Tensor. Phase 1.5 Mission 2
// (mission_host_loop_guards.md) guards all three with EXAI_ASSERT. No real GPU needed: see
// LinearModuleDeathTest for the mislabeled-Tensor testing pattern this reuses.
TEST_F(Conv2DModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                 DeviceType::Cuda);
    EXPECT_DEATH({ (void)conv.forward(input); }, "EXAI_ASSERT failed");
}

TEST_F(Conv2DModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)conv.backward(grad_output); }, "EXAI_ASSERT failed");
}

TEST_F(Conv2DModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Conv2DModule conv(1, 1, 2, 2, &backend);
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    (void)conv.forward(input);

    Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f}, DeviceType::Cuda);
    LRPRuleConfig config;
    EXPECT_DEATH({ (void)conv.propagate_relevance(relevance_out, config); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
