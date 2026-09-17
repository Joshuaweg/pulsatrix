#include <gtest/gtest.h>

#include "exai/node.hpp"

namespace exai {
namespace {

TEST(NodeTest, ConstructionSetsOpTypeShapeAndId) {
    Node n(NodeId{0}, OpType::Linear, Shape({4, 8}));
    EXPECT_EQ(n.id(), NodeId{0});
    EXPECT_EQ(n.op_type(), OpType::Linear);
    EXPECT_EQ(n.shape(), Shape({4, 8}));
}

TEST(NodeTest, LabelDefaultsToNullopt) {
    Node n(NodeId{0}, OpType::Linear, Shape({4}));
    EXPECT_FALSE(n.label().has_value());
}

TEST(NodeTest, LabelIsSettableAtConstruction) {
    Node n(NodeId{0}, OpType::Conv, Shape({1}), "conv1");
    ASSERT_TRUE(n.label().has_value());
    EXPECT_EQ(*n.label(), "conv1");
}

TEST(NodeTest, NewNodeHasNoParentsOrChildren) {
    Node n(NodeId{0}, OpType::Linear, Shape({1}));
    EXPECT_TRUE(n.parents().empty());
    EXPECT_TRUE(n.children().empty());
}

TEST(NodeTest, AddParentAppendsToParentList) {
    Node parent(NodeId{0}, OpType::Linear, Shape({1}));
    Node child(NodeId{1}, OpType::Activation, Shape({1}));

    child.add_parent(&parent);

    ASSERT_EQ(child.parents().size(), 1u);
    EXPECT_EQ(child.parents()[0], &parent);
}

TEST(NodeTest, AddChildAppendsToChildList) {
    Node parent(NodeId{0}, OpType::Linear, Shape({1}));
    Node child(NodeId{1}, OpType::Activation, Shape({1}));

    parent.add_child(&child);

    ASSERT_EQ(parent.children().size(), 1u);
    EXPECT_EQ(parent.children()[0], &child);
}

}  // namespace
}  // namespace exai
