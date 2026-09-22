#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/dropout_module.hpp"
#include "exai/linear_module.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/module.hpp"
#include "exai/op_type.hpp"
#include "exai/relu_module.hpp"
#include "exai/sequential_module.hpp"

namespace exai {
namespace {

class SequentialModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(SequentialModuleTest, ConstructionThrowsOnEmptyLayers) {
    EXPECT_THROW({ SequentialModule seq(std::vector<Module*>{}); }, std::invalid_argument);
}

TEST_F(SequentialModuleTest, ConstructionThrowsOnNullEntry) {
    LinearModule linear(2, 2, &backend);
    EXPECT_THROW({ SequentialModule seq(std::vector<Module*>{&linear, nullptr}); }, std::invalid_argument);
}

TEST_F(SequentialModuleTest, OpTypeIsComposite) {
    ReluModule relu(&backend);
    SequentialModule seq({&relu});
    EXPECT_EQ(seq.op_type(), OpType::Composite);
}

TEST_F(SequentialModuleTest, BackwardThrowsIfCalledBeforeForward) {
    ReluModule relu(&backend);
    SequentialModule seq({&relu});
    Tensor grad(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)seq.backward(grad); }, std::logic_error);
}

TEST_F(SequentialModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    ReluModule relu(&backend);
    SequentialModule seq({&relu});
    Tensor relevance(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)seq.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

// Linear -> Relu -> Linear, output matches manually chaining the same three calls
// directly -- proves delegation, not new math (same network shape
// LRPConservationEndToEndTest already uses).
TEST_F(SequentialModuleTest, ForwardMatchesManualChaining) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    // Manual chain (separate instances with the same weights, to avoid cached-state
    // interference with the SequentialModule instances used below).
    LinearModule ml1(3, 4, &backend);
    ml1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    ml1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule mrelu(&backend);
    LinearModule ml2(4, 2, &backend);
    ml2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    ml2.set_bias({0.05f, -0.05f});
    Tensor expected = ml2.forward(mrelu.forward(ml1.forward(input)));

    SequentialModule seq({&linear1, &relu, &linear2});
    Tensor output = seq.forward(input);

    EXPECT_EQ(output.shape(), expected.shape());
    for (int64_t i = 0; i < output.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], expected.data()[i]);
    }
}

TEST_F(SequentialModuleTest, BackwardMatchesManualReverseChaining) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    // Manual reverse chain, same weights, separate instances.
    LinearModule ml1(3, 4, &backend);
    ml1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    ml1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule mrelu(&backend);
    LinearModule ml2(4, 2, &backend);
    ml2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    ml2.set_bias({0.05f, -0.05f});
    (void)ml2.forward(mrelu.forward(ml1.forward(input)));

    Tensor grad_seed(Shape({1, 2}), &backend, {1.0f, -1.0f});
    Tensor expected_grad = ml1.backward(mrelu.backward(ml2.backward(grad_seed)));

    SequentialModule seq({&linear1, &relu, &linear2});
    (void)seq.forward(input);
    Tensor grad_input = seq.backward(grad_seed);

    EXPECT_EQ(grad_input.shape(), expected_grad.shape());
    for (int64_t i = 0; i < grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_input.data()[i], expected_grad.data()[i]);
    }
}

TEST_F(SequentialModuleTest, PropagateRelevanceMatchesManualReverseChaining) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    LinearModule ml1(3, 4, &backend);
    ml1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    ml1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule mrelu(&backend);
    LinearModule ml2(4, 2, &backend);
    ml2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    ml2.set_bias({0.05f, -0.05f});
    (void)ml2.forward(mrelu.forward(ml1.forward(input)));

    Tensor relevance_seed(Shape({1, 2}), &backend, {4.0f, 6.0f});
    LRPRuleConfig config;
    Tensor expected_relevance =
        ml1.propagate_relevance(mrelu.propagate_relevance(ml2.propagate_relevance(relevance_seed, config), config), config);

    SequentialModule seq({&linear1, &relu, &linear2});
    (void)seq.forward(input);
    Tensor relevance_input = seq.propagate_relevance(relevance_seed, config);

    EXPECT_EQ(relevance_input.shape(), expected_relevance.shape());
    for (int64_t i = 0; i < relevance_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(relevance_input.data()[i], expected_relevance.data()[i]);
    }
}

TEST_F(SequentialModuleTest, ParametersAggregatesAllContainedLayers) {
    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    SequentialModule seq({&linear1, &relu, &linear2});

    size_t expected = linear1.parameters().size() + relu.parameters().size() + linear2.parameters().size();
    EXPECT_EQ(seq.parameters().size(), expected);
    EXPECT_EQ(seq.parameters().size(), 4u);  // linear1's {weight,bias} + linear2's {weight,bias}
}

// The specific case the virtual-dispatch upgrade exists to fix: cascading via a Module*
// base pointer, not just SequentialModule's own concrete type.
TEST_F(SequentialModuleTest, SetTrainingCascadesToContainedLayersThroughBasePointer) {
    DropoutModule dropout(0.5f, &backend);
    LinearModule linear(2, 2, &backend);
    SequentialModule seq({&linear, &dropout});

    Module* base = &seq;
    base->set_training(false);

    EXPECT_FALSE(seq.is_training());
    EXPECT_FALSE(dropout.is_training());

    base->set_training(true);
    EXPECT_TRUE(dropout.is_training());
}

}  // namespace
}  // namespace exai
