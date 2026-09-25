#include <gtest/gtest.h>

#include <algorithm>

#include "pulsatrix/computation_graph.hpp"

namespace pulsatrix {
namespace {

class ComputationGraphTest : public ::testing::Test {
protected:
    ComputationGraph graph;
};

using ComputationGraphDeathTest = ComputationGraphTest;

TEST_F(ComputationGraphTest, AddNodeReturnsSequentialIds) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    NodeId b = graph.add_node(OpType::Activation, Shape({4}), std::nullopt, {a});

    EXPECT_EQ(a, 0u);
    EXPECT_EQ(b, 1u);
}

TEST_F(ComputationGraphTest, NodeCountMatchesNumberAdded) {
    graph.add_node(OpType::Linear, Shape({4}));
    graph.add_node(OpType::Activation, Shape({4}));
    EXPECT_EQ(graph.node_count(), 2u);
}

TEST_F(ComputationGraphTest, AddNodeWithoutParentsHasEmptyParentList) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    EXPECT_TRUE(graph.node(a).parents().empty());
}

TEST_F(ComputationGraphTest, AddNodeWiresParentChildEdgesBothWays) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    NodeId b = graph.add_node(OpType::Activation, Shape({4}), std::nullopt, {a});

    ASSERT_EQ(graph.node(b).parents().size(), 1u);
    EXPECT_EQ(graph.node(b).parents()[0], &graph.node(a));

    ASSERT_EQ(graph.node(a).children().size(), 1u);
    EXPECT_EQ(graph.node(a).children()[0], &graph.node(b));
}

TEST_F(ComputationGraphTest, NodeWithMultipleParentsWiresAllEdges) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    NodeId b = graph.add_node(OpType::Linear, Shape({4}));
    NodeId c = graph.add_node(OpType::Elementwise, Shape({4}), std::nullopt, {a, b});

    ASSERT_EQ(graph.node(c).parents().size(), 2u);
    EXPECT_EQ(graph.node(c).parents()[0], &graph.node(a));
    EXPECT_EQ(graph.node(c).parents()[1], &graph.node(b));
}

TEST_F(ComputationGraphTest, NodeAccessorReturnsCorrectNode) {
    NodeId a = graph.add_node(OpType::Conv, Shape({3, 3}), "conv1");
    EXPECT_EQ(graph.node(a).op_type(), OpType::Conv);
    ASSERT_TRUE(graph.node(a).label().has_value());
    EXPECT_EQ(*graph.node(a).label(), "conv1");
}

TEST_F(ComputationGraphTest, NodesByOpTypeReturnsOnlyMatchingIds) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    graph.add_node(OpType::Activation, Shape({4}), std::nullopt, {a});
    NodeId c = graph.add_node(OpType::Linear, Shape({4}));

    std::vector<NodeId> linears = graph.nodes_by_op_type(OpType::Linear);
    ASSERT_EQ(linears.size(), 2u);
    EXPECT_NE(std::find(linears.begin(), linears.end(), a), linears.end());
    EXPECT_NE(std::find(linears.begin(), linears.end(), c), linears.end());
}

TEST_F(ComputationGraphTest, NodesByOpTypeReturnsEmptyWhenNoneMatch) {
    graph.add_node(OpType::Linear, Shape({4}));
    EXPECT_TRUE(graph.nodes_by_op_type(OpType::Conv).empty());
}

TEST_F(ComputationGraphTest, TopologicalOrderPlacesEveryNodeAfterItsParents) {
    NodeId a = graph.add_node(OpType::Linear, Shape({4}));
    NodeId b = graph.add_node(OpType::Activation, Shape({4}), std::nullopt, {a});
    NodeId c = graph.add_node(OpType::Elementwise, Shape({4}), std::nullopt, {b});

    std::vector<NodeId> order = graph.topological_order();
    ASSERT_EQ(order.size(), 3u);

    auto pos_a = std::find(order.begin(), order.end(), a);
    auto pos_b = std::find(order.begin(), order.end(), b);
    auto pos_c = std::find(order.begin(), order.end(), c);
    EXPECT_LT(pos_a, pos_b);
    EXPECT_LT(pos_b, pos_c);
}

TEST_F(ComputationGraphDeathTest, NodeAccessorAbortsOnOutOfRangeId) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    EXPECT_DEATH({ (void)graph.node(999); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
