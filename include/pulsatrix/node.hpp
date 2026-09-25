/** @file node.hpp
 *  @brief Computation graph node -- op type, shape, optional label, parent/child edges.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/op_type.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {

/** @brief Stable identifier for a Node within its owning ComputationGraph. */
using NodeId = size_t;

/**
 * @brief A single computation graph node. Owned exclusively by its ComputationGraph
 *        (see computation_graph.hpp); parent/child edges here are non-owning raw
 *        pointers into nodes the same graph owns.
 * @note Nodes never expose a way to remove an edge or reset parents/children to empty --
 *       this is deliberate. ComputationGraph's persistence guarantee (graph survives past
 *       the backward pass, Mission 3) depends on nothing anywhere in this codebase being
 *       able to silently discard graph structure.
 */
class Node {
public:
    /**
     * @brief Constructs a node.
     * @param id This node's stable identifier, assigned by the owning ComputationGraph.
     * @param op_type The operation category this node represents.
     * @param shape This node's output shape.
     * @param label Optional human-readable label (e.g. a layer name), for debugging/display.
     */
    Node(NodeId id, OpType op_type, Shape shape, std::optional<std::string> label = std::nullopt)
        : id_(id), op_type_(op_type), shape_(std::move(shape)), label_(std::move(label)) {}

    [[nodiscard]] NodeId id() const { return id_; }
    [[nodiscard]] OpType op_type() const { return op_type_; }
    [[nodiscard]] const Shape& shape() const { return shape_; }
    [[nodiscard]] const std::optional<std::string>& label() const { return label_; }

    /** @brief This node's parent nodes (edges point from parent to this node). */
    [[nodiscard]] const std::vector<Node*>& parents() const { return parents_; }

    /** @brief This node's child nodes. */
    [[nodiscard]] const std::vector<Node*>& children() const { return children_; }

    /** @brief Registers parent as a parent of this node. Non-owning; parent must outlive this node. */
    void add_parent(Node* parent) { parents_.push_back(parent); }

    /** @brief Registers child as a child of this node. Non-owning; child must outlive this node. */
    void add_child(Node* child) { children_.push_back(child); }

private:
    NodeId id_;
    OpType op_type_;
    Shape shape_;
    std::optional<std::string> label_;
    std::vector<Node*> parents_;
    std::vector<Node*> children_;
};

}  // namespace pulsatrix
