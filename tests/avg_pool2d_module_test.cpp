#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class AvgPool2DModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(AvgPool2DModuleTest, ConstructionThrowsOnZeroKernelHeight) {
    EXPECT_THROW({ AvgPool2DModule pool(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, ConstructionThrowsOnZeroKernelWidth) {
    EXPECT_THROW({ AvgPool2DModule pool(2, 0, &backend); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, ConstructionThrowsOnNegativeKernelWidth) {
    EXPECT_THROW({ AvgPool2DModule pool(2, -1, &backend); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, ForwardThrowsOnWrongRank) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 4, 4}), &backend);
    EXPECT_THROW({ (void)pool.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, ForwardThrowsWhenKernelTallerThanInput) {
    AvgPool2DModule pool(5, 2, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.forward(small_input); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, ForwardThrowsWhenKernelWiderThanInput) {
    AvgPool2DModule pool(2, 5, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.forward(small_input); }, std::invalid_argument);
}

TEST_F(AvgPool2DModuleTest, BackwardThrowsIfCalledBeforeForward) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor grad(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)pool.backward(grad); }, std::logic_error);
}

TEST_F(AvgPool2DModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor relevance(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)pool.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(AvgPool2DModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);
    Tensor wrong_shape_grad(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.backward(wrong_shape_grad); }, std::invalid_argument);
}

// 1x1x4x4 input, kernel=2 -> 1x1x2x2 output. Window(0,0)=[[1,2],[5,6]] mean=3.5.
// Window(0,1)=[[3,4],[7,8]] mean=5.5. Window(1,0)=[[9,10],[13,14]] mean=11.5.
// Window(1,1)=[[11,12],[15,16]] mean=13.5.
TEST_F(AvgPool2DModuleTest, ForwardComputesHandVerifiedOutput) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor output = pool.forward(input);

    EXPECT_EQ(output.shape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 3.5f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 5.5f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 11.5f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 13.5f);
}

// N=2, C=2 -- proves per-(batch,channel) independence.
TEST_F(AvgPool2DModuleTest, ForwardHandlesMultiBatchMultiChannel) {
    AvgPool2DModule pool(2, 2, &backend);
    std::vector<float> row0_chan0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> row0_chan1(16, 4.0f);  // constant -> every window mean = 4
    std::vector<float> row1_chan0(16, 2.0f);
    std::vector<float> row1_chan1 = {0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4};

    std::vector<float> values;
    values.insert(values.end(), row0_chan0.begin(), row0_chan0.end());
    values.insert(values.end(), row0_chan1.begin(), row0_chan1.end());
    values.insert(values.end(), row1_chan0.begin(), row1_chan0.end());
    values.insert(values.end(), row1_chan1.begin(), row1_chan1.end());

    Tensor input(Shape({2, 2, 4, 4}), &backend, values);
    Tensor output = pool.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 2, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 3.5f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(output.at({1, 1, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({1, 1, 0, 1}), 4.0f);
}

// backward(): uniform 1/K gradient to every position in a window -- the true derivative
// of a mean, independent of activation value.
TEST_F(AvgPool2DModuleTest, BackwardDistributesGradientUniformly) {
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {4.0f, 8.0f, 12.0f, 16.0f});
    Tensor grad_input = pool.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), Shape({1, 1, 4, 4}));
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 0}), 1.0f);  // 4/4
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 1}), 1.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 2}), 2.0f);  // 8/4
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 3}), 2.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 3, 3}), 4.0f);  // 16/4
}

// propagate_relevance(): epsilon/z-rule, weight = 1/K, eps=0 -> exactly proportional to
// each input's own activation within its window (R_i = a_i / sum(a_window) * R_out).
// Window(0,0) = [1,2,5,6], sum=14. relevance_out=14.0 -> R_1=1, R_2=2, R_5=5, R_6=6.
TEST_F(AvgPool2DModuleTest, PropagateRelevanceIsProportionalToActivationAndConserves) {
    AvgPool2DModule pool(2, 2, &backend, /*eps=*/0.0f);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {14.0f, 0.0f, 0.0f, 0.0f});
    Tensor relevance_in = pool.propagate_relevance(relevance_out, LRPRuleConfig{0.0f});

    EXPECT_NEAR(relevance_in.at({0, 0, 0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(relevance_in.at({0, 0, 0, 1}), 2.0f, 1e-4f);
    EXPECT_NEAR(relevance_in.at({0, 0, 1, 0}), 5.0f, 1e-4f);
    EXPECT_NEAR(relevance_in.at({0, 0, 1, 1}), 6.0f, 1e-4f);

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    EXPECT_NEAR(sum_in, 14.0f, 1e-3f);
}

using AvgPool2DModuleDeathTest = AvgPool2DModuleTest;

// forward_impl/backward/propagate_relevance all dereference Tensor::data() in raw host
// loops -- undefined behavior on a CUDA-backed Tensor. See LinearModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses. Logged as a coverage gap by the Phase 1
// close-out review (campaign_exai_dl_library_phase6_modern_architectures.md, 2026-09-22)
// -- remediated here.
TEST_F(AvgPool2DModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.forward(input); }, "PULSATRIX_ASSERT failed");
}

TEST_F(AvgPool2DModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.backward(grad_output); }, "PULSATRIX_ASSERT failed");
}

TEST_F(AvgPool2DModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    AvgPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
