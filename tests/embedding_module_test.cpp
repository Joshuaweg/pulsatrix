#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class EmbeddingModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(EmbeddingModuleTest, ConstructionThrowsOnZeroNumEmbeddings) {
    EXPECT_THROW({ EmbeddingModule emb(0, 4, &backend); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, ConstructionThrowsOnNegativeNumEmbeddings) {
    EXPECT_THROW({ EmbeddingModule emb(-1, 4, &backend); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, ConstructionThrowsOnZeroEmbeddingDim) {
    EXPECT_THROW({ EmbeddingModule emb(4, 0, &backend); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, ForwardThrowsOnWrongRank) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor wrong_rank(Shape({2}), &backend, {0.0f, 1.0f});
    EXPECT_THROW({ (void)emb.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, ForwardThrowsOnNegativeIndex) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor input(Shape({1, 1}), &backend, {-1.0f});
    EXPECT_THROW({ (void)emb.forward(input); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, ForwardThrowsOnIndexAtOrAboveNumEmbeddings) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor input(Shape({1, 1}), &backend, {4.0f});
    EXPECT_THROW({ (void)emb.forward(input); }, std::invalid_argument);
}

TEST_F(EmbeddingModuleTest, BackwardThrowsIfCalledBeforeForward) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor grad(Shape({1, 1, 3}), &backend);
    EXPECT_THROW({ (void)emb.backward(grad); }, std::logic_error);
}

TEST_F(EmbeddingModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor relevance(Shape({1, 1, 3}), &backend);
    EXPECT_THROW({ (void)emb.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(EmbeddingModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    EmbeddingModule emb(4, 3, &backend);
    Tensor input(Shape({1, 1}), &backend, {0.0f});
    (void)emb.forward(input);
    Tensor wrong_shape_grad(Shape({1, 1, 2}), &backend);
    EXPECT_THROW({ (void)emb.backward(wrong_shape_grad); }, std::invalid_argument);
}

// 4-row, 3-wide table; look up rows 0 and 2 for a (1,2) input.
TEST_F(EmbeddingModuleTest, ForwardComputesHandVerifiedOutput) {
    EmbeddingModule emb(4, 3, &backend);
    emb.set_weight({1.0f, 2.0f, 3.0f,      // row 0
                     4.0f, 5.0f, 6.0f,      // row 1
                     7.0f, 8.0f, 9.0f,      // row 2
                     10.0f, 11.0f, 12.0f}); // row 3

    Tensor input(Shape({1, 2}), &backend, {0.0f, 2.0f});
    Tensor output = emb.forward(input);

    EXPECT_EQ(output.shape(), Shape({1, 2, 3}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0}), 7.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 1}), 8.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 2}), 9.0f);
}

// The float-indices design decision: round-to-nearest, not truncation. 1.9999999f must
// resolve to index 2, not 1.
TEST_F(EmbeddingModuleTest, ForwardRoundsNearFloatIndexToNearest) {
    EmbeddingModule emb(4, 2, &backend);
    emb.set_weight({0.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 0.0f, 0.0f});  // row 2 = [30,40]

    Tensor input(Shape({1, 1}), &backend, {1.9999999f});
    Tensor output = emb.forward(input);

    EXPECT_FLOAT_EQ(output.at({0, 0, 0}), 30.0f);
    EXPECT_FLOAT_EQ(output.at({0, 0, 1}), 40.0f);
}

// N=2, L=2 -- proves per-position independence across the batch.
TEST_F(EmbeddingModuleTest, ForwardHandlesMultiBatchMultiPosition) {
    EmbeddingModule emb(3, 2, &backend);
    emb.set_weight({1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f});  // row i = [i+1, i+1]

    Tensor input(Shape({2, 2}), &backend, {0.0f, 1.0f, 2.0f, 0.0f});
    Tensor output = emb.forward(input);

    EXPECT_EQ(output.shape(), Shape({2, 2, 2}));
    EXPECT_FLOAT_EQ(output.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(output.at({0, 1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(output.at({1, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(output.at({1, 1, 0}), 1.0f);
}

// Scatter-add: two positions referencing the same row (index 0 appears twice) must
// accumulate, not overwrite.
TEST_F(EmbeddingModuleTest, BackwardScatterAddsGradientForRepeatedIndex) {
    EmbeddingModule emb(3, 2, &backend);
    Tensor input(Shape({1, 3}), &backend, {0.0f, 1.0f, 0.0f});  // index 0 used twice
    (void)emb.forward(input);

    Tensor grad_output(Shape({1, 3, 2}), &backend, {1.0f, 1.0f, 5.0f, 5.0f, 2.0f, 2.0f});
    (void)emb.backward(grad_output);

    // row 0 grad = position0's [1,1] + position2's [2,2] = [3,3]
    EXPECT_FLOAT_EQ(emb.weight_grad().at({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(emb.weight_grad().at({0, 1}), 3.0f);
    // row 1 grad = position1's [5,5]
    EXPECT_FLOAT_EQ(emb.weight_grad().at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(emb.weight_grad().at({1, 1}), 5.0f);
    // row 2 untouched
    EXPECT_FLOAT_EQ(emb.weight_grad().at({2, 0}), 0.0f);
}

// Gradient w.r.t. discrete indices is always zero.
TEST_F(EmbeddingModuleTest, BackwardReturnsZeroGradientForInputIndices) {
    EmbeddingModule emb(3, 2, &backend);
    Tensor input(Shape({1, 2}), &backend, {0.0f, 1.0f});
    (void)emb.forward(input);

    Tensor grad_output(Shape({1, 2, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_input = emb.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), Shape({1, 2}));
    EXPECT_FLOAT_EQ(grad_input.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(grad_input.data()[1], 0.0f);
}

// propagate_relevance: sum over the embedding dimension per position, conserves exactly.
TEST_F(EmbeddingModuleTest, PropagateRelevanceSumsOverEmbeddingDimensionAndConserves) {
    EmbeddingModule emb(3, 3, &backend);
    Tensor input(Shape({1, 2}), &backend, {0.0f, 1.0f});
    (void)emb.forward(input);

    Tensor relevance_out(Shape({1, 2, 3}), &backend, {1.0f, 2.0f, 3.0f, 0.5f, 0.5f, 1.0f});
    Tensor relevance_in = emb.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_EQ(relevance_in.shape(), Shape({1, 2}));
    EXPECT_FLOAT_EQ(relevance_in.at({0, 0}), 6.0f);  // 1+2+3
    EXPECT_FLOAT_EQ(relevance_in.at({0, 1}), 2.0f);  // 0.5+0.5+1.0

    float sum_in = relevance_in.at({0, 0}) + relevance_in.at({0, 1});
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

using EmbeddingModuleDeathTest = EmbeddingModuleTest;

}  // namespace
}  // namespace pulsatrix
