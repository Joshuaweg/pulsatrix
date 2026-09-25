#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "exai/circuit_graph.hpp"

// CircuitGraph (campaign_exai_dl_library_mechanistic_interpretability, Phase 5 Mission 3)
// is the campaign's final output artifact: a plain, copyable value type holding a list of
// scored nodes (id, op_type, label, ablation_effect) and a list of weighted edges
// (from, to, weight). Raw data only -- rendering it is explicitly deferred (see the
// header's own note). This file covers the value type itself; the production path that
// computes the scores, ExplainerContext::build_circuit_graph(), lives in
// explainer_context_test.cpp -- mirroring how activation_snapshot_test.cpp covers
// ActivationSnapshot while its capture path is tested against the real context.
namespace exai {
namespace {

class CircuitGraphTest : public ::testing::Test {
protected:
    // A hand-built three-node/two-edge circuit standing in for what
    // ExplainerContext::build_circuit_graph() produces -- construction is that method's
    // job in production, so this fixture is the only place circuits are assembled by hand.
    static CircuitGraph make_circuit() {
        std::vector<CircuitNode> nodes{
            CircuitNode{0u, OpType::Elementwise, std::string("input"), 2.5f},
            CircuitNode{1u, OpType::Linear, std::nullopt, 7.25f},
            CircuitNode{2u, OpType::Activation, std::string("relu"), 0.0f},
        };
        std::vector<CircuitEdge> edges{
            CircuitEdge{0u, 1u, 2.5f},
            CircuitEdge{1u, 2u, 7.25f},
        };
        return CircuitGraph(std::move(nodes), std::move(edges));
    }
};

TEST_F(CircuitGraphTest, NodesReturnsConstructedNodesInConstructedOrder) {
    const CircuitGraph circuit = make_circuit();

    const std::vector<CircuitNode>& nodes = circuit.nodes();
    ASSERT_EQ(nodes.size(), 3u);

    EXPECT_EQ(nodes[0].id, 0u);
    EXPECT_EQ(nodes[0].op_type, OpType::Elementwise);
    ASSERT_TRUE(nodes[0].label.has_value());
    EXPECT_EQ(*nodes[0].label, "input");
    EXPECT_FLOAT_EQ(nodes[0].ablation_effect, 2.5f);

    EXPECT_EQ(nodes[1].id, 1u);
    EXPECT_EQ(nodes[1].op_type, OpType::Linear);
    EXPECT_FALSE(nodes[1].label.has_value());
    EXPECT_FLOAT_EQ(nodes[1].ablation_effect, 7.25f);

    EXPECT_EQ(nodes[2].id, 2u);
    EXPECT_EQ(nodes[2].op_type, OpType::Activation);
    EXPECT_FLOAT_EQ(nodes[2].ablation_effect, 0.0f);
}

TEST_F(CircuitGraphTest, EdgesReturnsConstructedEdgesInConstructedOrder) {
    const CircuitGraph circuit = make_circuit();

    const std::vector<CircuitEdge>& edges = circuit.edges();
    ASSERT_EQ(edges.size(), 2u);

    EXPECT_EQ(edges[0].from, 0u);
    EXPECT_EQ(edges[0].to, 1u);
    EXPECT_FLOAT_EQ(edges[0].weight, 2.5f);

    EXPECT_EQ(edges[1].from, 1u);
    EXPECT_EQ(edges[1].to, 2u);
    EXPECT_FLOAT_EQ(edges[1].weight, 7.25f);
}

// A CircuitGraph is a value: copying one must not alias the original's storage, since
// the point of shipping raw data (rather than a view onto a live ExplainerContext) is
// that the artifact outlives the context that produced it.
TEST_F(CircuitGraphTest, CopyIsIndependentOfTheOriginal) {
    CircuitGraph original = make_circuit();
    CircuitGraph copy = original;

    ASSERT_EQ(copy.nodes().size(), original.nodes().size());
    EXPECT_NE(copy.nodes().data(), original.nodes().data());
    EXPECT_NE(copy.edges().data(), original.edges().data());
    EXPECT_FLOAT_EQ(copy.nodes()[1].ablation_effect, 7.25f);
}

// Adversarial (degenerate construction): an empty circuit is a well-formed value, not an
// error -- it is what a caller assembling one incrementally starts from, and the
// zero-edge case is reachable for real from a graph with a single node.
TEST_F(CircuitGraphTest, EmptyCircuitIsWellFormedAndReportsEmptyNodesAndEdges) {
    const CircuitGraph circuit(std::vector<CircuitNode>{}, std::vector<CircuitEdge>{});

    EXPECT_TRUE(circuit.nodes().empty());
    EXPECT_TRUE(circuit.edges().empty());
}

// Adversarial (degenerate construction): one node, zero edges -- structurally valid, and
// exactly the shape a single-node chain produces. No edge invariant is cross-checked
// against the node list by the constructor, deliberately (documented in the header).
TEST_F(CircuitGraphTest, SingleNodeZeroEdgeCircuitIsWellFormed) {
    std::vector<CircuitNode> nodes{CircuitNode{0u, OpType::Elementwise, std::string("input"), 0.0f}};
    const CircuitGraph circuit(std::move(nodes), std::vector<CircuitEdge>{});

    ASSERT_EQ(circuit.nodes().size(), 1u);
    EXPECT_EQ(circuit.nodes()[0].id, 0u);
    EXPECT_TRUE(circuit.edges().empty());
}

}  // namespace
}  // namespace exai
