/** @file computation_graph.hpp
 *  @brief Owns and exposes graph structure -- the interpretability substrate every
 *         explainer (Phase 2+) walks.
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "exai/node.hpp"
#include "exai/op_type.hpp"
#include "exai/shape.hpp"

namespace exai {

/**
 * @brief Owns every Node in a computation graph and exposes read access for graph-walking
 *        code (autograd's backward pass, Phase 2+ explainers).
 * @note This class exposes no method that removes a node or clears the graph. That is
 *       deliberate, not an oversight -- the charter's central Phase 0 requirement is that
 *       the graph survives past the backward pass (Mission 3 exercises this directly).
 *       Nothing in this codebase should be able to silently discard graph structure.
 */
class ComputationGraph {
public:
    /**
     * @brief Adds a node to the graph and wires it to its parents.
     * @param op_type The operation category this node represents.
     * @param shape This node's output shape.
     * @param label Optional human-readable label.
     * @param parent_ids Ids of this node's parents. Every id must already exist in the
     *        graph (i.e. refer to a node added by an earlier call) -- this constraint is
     *        what guarantees insertion order is already a valid topological order.
     * @return The new node's id.
     */
    NodeId add_node(OpType op_type, Shape shape, std::optional<std::string> label = std::nullopt,
                     std::vector<NodeId> parent_ids = {});

    /**
     * @brief Looks up a node by id.
     * @param id Node id, must be < node_count().
     * @return The node.
     */
    [[nodiscard]] const Node& node(NodeId id) const;

    /** @brief Number of nodes currently in the graph. */
    [[nodiscard]] size_t node_count() const { return nodes_.size(); }

    /**
     * @brief Finds every node with the given op type.
     * @param op_type Op type to match.
     * @return Ids of matching nodes, in insertion order. Empty if none match.
     * @note This is the op-type-tagging query mechanism from charter Part 2 SS3 -- e.g.
     *       Grad-CAM finding "the last conv layer" queries this, never by layer name.
     */
    [[nodiscard]] std::vector<NodeId> nodes_by_op_type(OpType op_type) const;

    /**
     * @brief Returns every node id in a valid topological order (every node after all its
     *        parents).
     * @return Node ids in topological order.
     * @note Simply returns insertion order -- add_node()'s constraint that parent_ids must
     *       already exist in the graph guarantees insertion order is already topological.
     *       If a future relaxation of that constraint is ever considered, this method's
     *       implementation would need to change to an explicit sort; it does not today.
     */
    [[nodiscard]] std::vector<NodeId> topological_order() const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace exai
