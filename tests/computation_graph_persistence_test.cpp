#include <gtest/gtest.h>

#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/cpu_backend.hpp"

// This is the mission's real acceptance criterion, deliberately kept in its own file so
// it's immediately findable rather than folded quietly into computation_graph_test.cpp's
// construction tests -- see
// how/campaigns/campaign_exai_dl_library_phase0_tensor_autograd_core/missions/mission_computation_graph.md.
//
// Charter, verbatim: "Most autograd implementations discard the graph after backward() to
// save memory. This design keeps it... This is the single most important divergence from
// a 'normal' autograd implementation." Mission 3 (reverse-mode autodiff) will exercise this
// against a *real* backward() call; this test proves the guarantee holds against the graph
// structure alone, before autograd exists to complicate the picture.

namespace exai {
namespace {

TEST(ComputationGraphPersistenceTest, GraphSurvivesReadHeavyTraversalUnchanged) {
    ComputationGraph graph;
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    NodeId b = graph.add_node(OpType::Activation, Shape({4}), std::nullopt, {a});
    NodeId c = graph.add_node(OpType::Elementwise, Shape({4}), std::nullopt, {b});

    const size_t node_count_before = graph.node_count();
    const std::vector<NodeId> order_before = graph.topological_order();
    const size_t a_children_before = graph.node(a).children().size();
    const size_t b_parents_before = graph.node(b).parents().size();
    const size_t c_parents_before = graph.node(c).parents().size();

    // Simulate a backward-pass-like traversal: walk the graph in reverse topological
    // order, reading every node's op type, shape, and edges -- structurally the same
    // access pattern Mission 3's real backward() will perform. Nothing here mutates
    // the graph; ComputationGraph has no method that could.
    for (auto it = order_before.rbegin(); it != order_before.rend(); ++it) {
        const Node& n = graph.node(*it);
        (void)n.op_type();
        (void)n.shape();
        (void)n.parents();
        (void)n.children();
    }
    (void)graph.nodes_by_op_type(OpType::Linear);  // exercise the other read-only traversal shape

    EXPECT_EQ(graph.node_count(), node_count_before)
        << "node count changed after a read-only traversal -- graph structure was not persisted";
    EXPECT_EQ(graph.topological_order(), order_before)
        << "topological order changed -- an edge was mutated during traversal";
    EXPECT_EQ(graph.node(a).children().size(), a_children_before);
    EXPECT_EQ(graph.node(b).parents().size(), b_parents_before);
    EXPECT_EQ(graph.node(c).parents().size(), c_parents_before);
}

// The definitive version of the Mission 2 test: exercises persistence against a genuine
// Autograd::backward() call rather than a synthetic read-only traversal. This is the
// charter's actual requirement -- "the graph structure survives past the backward pass" --
// proven against the real mechanism, not a stand-in for it.
TEST(ComputationGraphPersistenceTest, GraphSurvivesRealBackwardPassUnchanged) {
    CPUBackend backend;
    ComputationGraph graph;
    Autograd autograd;

    NodeId x_id = graph.add_node(OpType::Elementwise, Shape({2}), "x");
    NodeId y_id = graph.add_node(OpType::Elementwise, Shape({2}), "neg", {x_id});

    autograd.register_backward(y_id, [&backend](const Tensor& grad_output) {
        Tensor grad_input(grad_output.shape(), &backend);
        backend.elementwise(ElementwiseOp::Neg, grad_output.data(), grad_input.data(), grad_output.numel());
        return grad_input;
    });

    const size_t node_count_before = graph.node_count();
    const std::vector<NodeId> order_before = graph.topological_order();
    const size_t x_children_before = graph.node(x_id).children().size();
    const size_t y_parents_before = graph.node(y_id).parents().size();

    Tensor seed(Shape({2}), &backend, {1.0f, 1.0f});
    autograd.backward(graph, y_id, seed);

    EXPECT_EQ(graph.node_count(), node_count_before);
    EXPECT_EQ(graph.topological_order(), order_before);
    EXPECT_EQ(graph.node(x_id).children().size(), x_children_before);
    EXPECT_EQ(graph.node(y_id).parents().size(), y_parents_before);

    // And the graph is still fully usable afterward -- walkable by a second explainer-style
    // pass, which is the entire point of not discarding it.
    EXPECT_EQ(graph.nodes_by_op_type(OpType::Elementwise).size(), 2u);
}

}  // namespace
}  // namespace exai
