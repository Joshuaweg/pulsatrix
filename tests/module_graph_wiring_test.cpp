#include <gtest/gtest.h>

#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// Phase 2 Mission 0's actual exit-gate proof: a real multi-module network built via
// Module::forward_traced produces (a) a ComputationGraph whose structure matches
// hand-derived expectations, and (b) gradients via Autograd::backward() that match the
// same network's gradients computed the existing way (direct chained backward() calls, as
// XorNetwork's training loop already does) -- proving the traced path is a real
// alternative route to the same math, not a divergent one.
namespace exai {
namespace {

TEST(ModuleGraphWiringTest, ThreeModuleChainProducesExpectedGraphStructure) {
    CPUBackend backend;
    ComputationGraph graph;
    Autograd autograd;

    LinearModule linear1(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);

    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    NodeId input_node = graph.add_node(OpType::Elementwise, input.shape(), "input");

    auto [h1, h1_node] = linear1.forward_traced(input, input_node, graph, autograd);
    auto [h2, h2_node] = relu.forward_traced(h1, h1_node, graph, autograd);
    auto [out, out_node] = linear2.forward_traced(h2, h2_node, graph, autograd);
    (void)out;

    EXPECT_EQ(graph.node_count(), 4u);

    EXPECT_EQ(graph.node(h1_node).op_type(), OpType::Linear);
    ASSERT_EQ(graph.node(h1_node).parents().size(), 1u);
    EXPECT_EQ(graph.node(h1_node).parents()[0]->id(), input_node);

    EXPECT_EQ(graph.node(h2_node).op_type(), OpType::Activation);
    ASSERT_EQ(graph.node(h2_node).parents().size(), 1u);
    EXPECT_EQ(graph.node(h2_node).parents()[0]->id(), h1_node);

    EXPECT_EQ(graph.node(out_node).op_type(), OpType::Linear);
    ASSERT_EQ(graph.node(out_node).parents().size(), 1u);
    EXPECT_EQ(graph.node(out_node).parents()[0]->id(), h2_node);

    // The op-type-tagging query mechanism (charter Part 2 SS3) -- finds both Linear layers
    // by tag, not by name.
    std::vector<NodeId> linear_nodes = graph.nodes_by_op_type(OpType::Linear);
    ASSERT_EQ(linear_nodes.size(), 2u);
    EXPECT_EQ(linear_nodes[0], h1_node);
    EXPECT_EQ(linear_nodes[1], out_node);

    std::vector<NodeId> order = graph.topological_order();
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], input_node);
    EXPECT_EQ(order[1], h1_node);
    EXPECT_EQ(order[2], h2_node);
    EXPECT_EQ(order[3], out_node);
}

TEST(ModuleGraphWiringTest, TracedBackwardMatchesDirectChainedBackwardAtEveryNode) {
    CPUBackend backend;
    ComputationGraph graph;
    Autograd autograd;

    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    NodeId input_node = graph.add_node(OpType::Elementwise, input.shape(), "input");

    auto [h1, h1_node] = linear1.forward_traced(input, input_node, graph, autograd);
    auto [h2, h2_node] = relu.forward_traced(h1, h1_node, graph, autograd);
    auto [out, out_node] = linear2.forward_traced(h2, h2_node, graph, autograd);
    (void)out;

    Tensor grad_output(Shape({2}), &backend, {1.0f, -1.0f});
    autograd.backward(graph, out_node, grad_output);

    // Reference: the existing, already-proven-correct way to compute these gradients --
    // direct chained backward() calls, the same pattern XorNetwork's training loop uses.
    // Calling backward() again here is safe: it only recomputes and returns the input
    // gradient (a pure function of cached forward-pass state), the weight/bias gradient
    // ACCUMULATION side effect is irrelevant to this comparison.
    Tensor direct_grad_h2 = linear2.backward(grad_output);
    Tensor direct_grad_h1 = relu.backward(direct_grad_h2);
    Tensor direct_grad_input = linear1.backward(direct_grad_h1);

    ASSERT_TRUE(autograd.has_gradient(h2_node));
    const Tensor& traced_grad_h2 = autograd.gradient(h2_node);
    for (int64_t i = 0; i < direct_grad_h2.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h2.data()[i], direct_grad_h2.data()[i]) << "h2 mismatch at index " << i;
    }

    ASSERT_TRUE(autograd.has_gradient(h1_node));
    const Tensor& traced_grad_h1 = autograd.gradient(h1_node);
    for (int64_t i = 0; i < direct_grad_h1.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_h1.data()[i], direct_grad_h1.data()[i]) << "h1 mismatch at index " << i;
    }

    ASSERT_TRUE(autograd.has_gradient(input_node));
    const Tensor& traced_grad_input = autograd.gradient(input_node);
    for (int64_t i = 0; i < direct_grad_input.numel(); ++i) {
        EXPECT_FLOAT_EQ(traced_grad_input.data()[i], direct_grad_input.data()[i]) << "input mismatch at index " << i;
    }
}

}  // namespace
}  // namespace exai
