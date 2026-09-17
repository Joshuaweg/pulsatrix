#include "exai/computation_graph.hpp"

#include "exai/assert.hpp"

namespace exai {

NodeId ComputationGraph::add_node(OpType op_type, Shape shape, std::optional<std::string> label,
                                   std::vector<NodeId> parent_ids) {
    NodeId new_id = nodes_.size();
    auto new_node = std::make_unique<Node>(new_id, op_type, std::move(shape), std::move(label));
    Node* new_node_ptr = new_node.get();

    for (NodeId parent_id : parent_ids) {
        EXAI_ASSERT(parent_id < nodes_.size());
        Node* parent_ptr = nodes_[parent_id].get();
        new_node_ptr->add_parent(parent_ptr);
        parent_ptr->add_child(new_node_ptr);
    }

    nodes_.push_back(std::move(new_node));
    return new_id;
}

const Node& ComputationGraph::node(NodeId id) const {
    EXAI_ASSERT(id < nodes_.size());
    return *nodes_[id];
}

std::vector<NodeId> ComputationGraph::nodes_by_op_type(OpType op_type) const {
    std::vector<NodeId> result;
    for (const auto& n : nodes_) {
        if (n->op_type() == op_type) {
            result.push_back(n->id());
        }
    }
    return result;
}

std::vector<NodeId> ComputationGraph::topological_order() const {
    std::vector<NodeId> order;
    order.reserve(nodes_.size());
    for (const auto& n : nodes_) {
        order.push_back(n->id());
    }
    return order;
}

}  // namespace exai
