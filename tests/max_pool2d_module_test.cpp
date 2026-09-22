#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/max_pool2d_module.hpp"

namespace exai {
namespace {

class MaxPool2DModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(MaxPool2DModuleTest, ConstructionThrowsOnZeroKernelHeight) {
    EXPECT_THROW({ MaxPool2DModule pool(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, ConstructionThrowsOnZeroKernelWidth) {
    EXPECT_THROW({ MaxPool2DModule pool(2, 0, &backend); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, ConstructionThrowsOnNegativeKernelHeight) {
    EXPECT_THROW({ MaxPool2DModule pool(-1, 2, &backend); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, ForwardThrowsOnWrongRank) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 4, 4}), &backend);
    EXPECT_THROW({ (void)pool.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, ForwardThrowsWhenKernelTallerThanInput) {
    MaxPool2DModule pool(5, 2, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.forward(small_input); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, ForwardThrowsWhenKernelWiderThanInput) {
    MaxPool2DModule pool(2, 5, &backend);
    Tensor small_input(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.forward(small_input); }, std::invalid_argument);
}

TEST_F(MaxPool2DModuleTest, BackwardThrowsIfCalledBeforeForward) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor grad(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)pool.backward(grad); }, std::logic_error);
}

TEST_F(MaxPool2DModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor relevance(Shape({1, 1, 2, 2}), &backend);
    EXPECT_THROW({ (void)pool.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(MaxPool2DModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);
    Tensor wrong_shape_grad(Shape({1, 1, 3, 3}), &backend);
    EXPECT_THROW({ (void)pool.backward(wrong_shape_grad); }, std::invalid_argument);
}

// 1x1x4x4 input, kernel=2 (non-overlapping) -> 1x1x2x2 output.
// Window(0,0) = [[1,2],[5,6]] -> max=6 at (1,1). Window(0,1) = [[3,4],[7,8]] -> max=8 at (0,1).
// Window(1,0) = [[9,10],[13,14]] -> max=14 at (1,1). Window(1,1) = [[11,12],[15,16]] -> max=16.
TEST_F(MaxPool2DModuleTest, ForwardComputesHandVerifiedOutput) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    Tensor output = pool.forward(input);

    EXPECT_EQ(output.shape(), Shape({1, 1, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 0}), 14.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1, 1}), 16.0f);
}

// N=2, C=2 -- proves each (batch, channel) plane is pooled independently.
TEST_F(MaxPool2DModuleTest, ForwardHandlesMultiBatchMultiChannel) {
    MaxPool2DModule pool(2, 2, &backend);
    // Row0/Chan0: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] -> max window values as above.
    // Row0/Chan1: all zeros except one high value at (0,0) -> max=100 in window(0,0), 0 elsewhere.
    // Row1/Chan0: all 5s -> every window max = 5.
    // Row1/Chan1: descending values, window(0,0) max at position (0,0).
    std::vector<float> row0_chan0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> row0_chan1 = {100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> row1_chan0(16, 5.0f);
    std::vector<float> row1_chan1 = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    std::vector<float> values;
    values.insert(values.end(), row0_chan0.begin(), row0_chan0.end());
    values.insert(values.end(), row0_chan1.begin(), row0_chan1.end());
    values.insert(values.end(), row1_chan0.begin(), row1_chan0.end());
    values.insert(values.end(), row1_chan1.begin(), row1_chan1.end());

    Tensor input(Shape({2, 2, 4, 4}), &backend, values);
    Tensor output = pool.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 2, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0, 0}), 100.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(output.at({1, 1, 0, 0}), 16.0f);
}

// Winner-take-all: backward() gradient must land exactly at the argmax position within
// each window and nowhere else -- verified against a central-finite-difference style
// direct check (the argmax positions are known exactly from the hand-derived forward case
// above, so this checks the routed gradient value, not a numerical approximation).
TEST_F(MaxPool2DModuleTest, BackwardRoutesGradientOnlyToArgmaxPosition) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {10.0f, 20.0f, 30.0f, 40.0f});
    Tensor grad_input = pool.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), Shape({1, 1, 4, 4}));
    // Window(0,0) argmax was value 6 at (1,1) -> flat index 1*4+1=5.
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 1}), 10.0f);
    // Every non-argmax position in that window is 0.
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 0}), 0.0f);
    // Window(0,1) argmax was value 8 at (1,3).
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 1, 3}), 20.0f);
    // Window(1,1) argmax was value 16 at (3,3).
    EXPECT_FLOAT_EQ(grad_input.at({0, 0, 3, 3}), 40.0f);
}

// propagate_relevance: winner-take-all -- non-argmax positions get exactly 0, argmax gets
// exactly relevance_out's value. Conserves by construction (sum(relevance_in) ==
// sum(relevance_out) trivially, since every output value is copied to exactly one input
// position and every other input position is 0).
TEST_F(MaxPool2DModuleTest, PropagateRelevanceRoutesOnlyToArgmaxAndConserves) {
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor relevance_in = pool.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_FLOAT_EQ(relevance_in.at({0, 0, 1, 1}), 1.0f);
    EXPECT_FLOAT_EQ(relevance_in.at({0, 0, 1, 3}), 2.0f);
    EXPECT_FLOAT_EQ(relevance_in.at({0, 0, 3, 1}), 3.0f);
    EXPECT_FLOAT_EQ(relevance_in.at({0, 0, 3, 3}), 4.0f);

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 1.0f + 2.0f + 3.0f + 4.0f;
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

using MaxPool2DModuleDeathTest = MaxPool2DModuleTest;

// forward_impl/backward/propagate_relevance all dereference Tensor::data() in raw host
// loops -- undefined behavior on a CUDA-backed Tensor. See LinearModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses: Tensor::device() is metadata decoupled
// from which DeviceBackend* actually allocated its buffer, so a Tensor tagged
// DeviceType::Cuda over real CPUBackend memory triggers the guard just like a genuine
// CUDA tensor would. Logged as a coverage gap by the Phase 1 close-out review
// (campaign_exai_dl_library_phase6_modern_architectures.md, 2026-09-22) -- remediated here.
TEST_F(MaxPool2DModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.forward(input); }, "EXAI_ASSERT failed");
}

TEST_F(MaxPool2DModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor grad_output(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.backward(grad_output); }, "EXAI_ASSERT failed");
}

TEST_F(MaxPool2DModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MaxPool2DModule pool(2, 2, &backend);
    Tensor input(Shape({1, 1, 4, 4}), &backend,
                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    (void)pool.forward(input);

    Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)pool.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
