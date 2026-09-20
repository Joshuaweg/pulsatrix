#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// ExplainerContext (charter Part 2 SS2) is what every explainer gets, regardless of type --
// graph-native explainers (Missions 2-3) use graph()/backward_pass(); this mission ships
// the interface itself, built directly on Module::forward_traced (Mission 0).
namespace exai {
namespace {

class ExplainerContextTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(ExplainerContextTest, ForwardPassSingleModuleMatchesDirectForward) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.5f, -0.5f});

    LinearModule reference(2, 2, &backend);
    reference.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    reference.set_bias({0.5f, -0.5f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});

    Tensor traced_output = ctx.forward_pass(input);
    Tensor direct_output = reference.forward(input);

    for (int64_t i = 0; i < direct_output.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_output.data()[i], direct_output.data()[i]);
    }
}

TEST_F(ExplainerContextTest, ForwardPassBuildsCorrectGraphStructureForMultiModuleChain) {
    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    // input node + 3 module nodes.
    EXPECT_EQ(ctx.graph().node_count(), 4u);

    std::vector<NodeId> linear_nodes = ctx.graph().nodes_by_op_type(OpType::Linear);
    EXPECT_EQ(linear_nodes.size(), 2u);
    std::vector<NodeId> activation_nodes = ctx.graph().nodes_by_op_type(OpType::Activation);
    EXPECT_EQ(activation_nodes.size(), 1u);
}

TEST_F(ExplainerContextTest, ActivationReturnsCachedValuePerNode) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});  // identity-ish
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {7.0f, 9.0f});
    Tensor output = ctx.forward_pass(input);

    std::vector<NodeId> order = ctx.graph().topological_order();
    ASSERT_EQ(order.size(), 2u);
    NodeId input_node = order[0];
    NodeId output_node = order[1];

    const Tensor& cached_input = ctx.activation(input_node);
    EXPECT_FLOAT_EQ(cached_input.data()[0], 7.0f);
    EXPECT_FLOAT_EQ(cached_input.data()[1], 9.0f);

    const Tensor& cached_output = ctx.activation(output_node);
    EXPECT_FLOAT_EQ(cached_output.data()[0], output.data()[0]);
    EXPECT_FLOAT_EQ(cached_output.data()[1], output.data()[1]);
}

TEST_F(ExplainerContextTest, LayerLabelPassesThroughInputNodeLabel) {
    LinearModule linear(2, 2, &backend);
    ExplainerContext ctx({&linear});
    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});
    (void)ctx.forward_pass(input);

    std::vector<NodeId> order = ctx.graph().topological_order();
    NodeId input_node = order[0];

    ASSERT_TRUE(ctx.layer_label(input_node).has_value());
    EXPECT_EQ(ctx.layer_label(input_node).value(), "input");
}

TEST_F(ExplainerContextTest, BackwardPassMatchesDirectChainedBackward) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    Tensor grad_output(Shape({2}), &backend, {1.0f, -1.0f});
    Tensor traced_grad_input = ctx.backward_pass(grad_output);

    // Reference: the existing, already-proven-correct way to compute this gradient --
    // direct chained backward() calls, the same oracle pattern as
    // mission_module_graph_wiring.md's own integration test.
    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);
    Tensor direct_grad_input = linear1.backward(direct_grad_h1);

    for (int64_t i = 0; i < direct_grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_input.data()[i], direct_grad_input.data()[i]) << "mismatch at index " << i;
    }
}

TEST_F(ExplainerContextTest, GradientReturnsCachedValuePerIntermediateNode) {
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    (void)ctx.forward_pass(input);

    Tensor grad_output(Shape({2}), &backend, {1.0f, -1.0f});
    (void)ctx.backward_pass(grad_output);

    // Reference: the same oracle Objective 3 of mission_module_graph_wiring.md and
    // Objective 3 of mission_explainer_context.md already used -- direct chained
    // backward() calls.
    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);

    std::vector<NodeId> order = ctx.graph().topological_order();
    ASSERT_EQ(order.size(), 4u);
    NodeId h1_node = order[1];
    NodeId h2_node = order[2];

    const Tensor& traced_grad_h1 = ctx.gradient(h1_node);
    for (int64_t i = 0; i < direct_grad_h1.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h1.data()[i], direct_grad_h1.data()[i]) << "h1 mismatch at index " << i;
    }

    const Tensor& traced_grad_h2 = ctx.gradient(h2_node);
    for (int64_t i = 0; i < direct_grad_h2.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h2.data()[i], direct_grad_h2.data()[i]) << "h2 mismatch at index " << i;
    }
}

TEST_F(ExplainerContextTest, RepeatedForwardPassDoesNotLeakStateFromPriorCall) {
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    linear.set_bias({0.0f, 0.0f});

    ExplainerContext ctx({&linear});

    Tensor first_input(Shape({2}), &backend, {1.0f, 2.0f});
    (void)ctx.forward_pass(first_input);
    EXPECT_EQ(ctx.graph().node_count(), 2u);

    Tensor second_input(Shape({2}), &backend, {10.0f, 20.0f});
    Tensor second_output = ctx.forward_pass(second_input);

    // Still exactly 2 nodes -- a fresh graph each call, not an ever-growing one.
    EXPECT_EQ(ctx.graph().node_count(), 2u);
    EXPECT_FLOAT_EQ(second_output.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(second_output.data()[1], 20.0f);

    std::vector<NodeId> order = ctx.graph().topological_order();
    const Tensor& cached_input = ctx.activation(order[0]);
    EXPECT_FLOAT_EQ(cached_input.data()[0], 10.0f);  // reflects the second call, not the first
    EXPECT_FLOAT_EQ(cached_input.data()[1], 20.0f);
}

}  // namespace
}  // namespace exai
