#include "exai/autograd.hpp"

#include "exai/assert.hpp"

namespace exai {

void Autograd::register_backward(NodeId id, BackwardFn fn) {
    backward_fns_[id] = std::move(fn);
}

void Autograd::backward(const ComputationGraph& graph, NodeId root, const Tensor& grad_output) {
    std::vector<std::pair<NodeId, Tensor>> seeds;
    seeds.emplace_back(root, Tensor(grad_output));
    backward(graph, std::move(seeds));
}

void Autograd::backward(const ComputationGraph& graph, std::vector<std::pair<NodeId, Tensor>> seeds) {
    gradients_.clear();  // each backward() call starts fresh -- see class-level @note

    for (auto& [id, seed] : seeds) {
        auto it = gradients_.find(id);
        if (it == gradients_.end()) {
            gradients_.emplace(id, std::move(seed));
        } else {
            it->second.accumulate(seed);
        }
    }

    std::vector<NodeId> order = graph.topological_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        NodeId id = *it;
        if (!has_gradient(id)) {
            continue;  // nothing flowed here -- disconnected from every seed
        }
        auto fn_it = backward_fns_.find(id);
        if (fn_it == backward_fns_.end()) {
            continue;  // leaf/input node, or no backward function registered for it
        }

        const Node& node = graph.node(id);
        EXAI_ASSERT(node.parents().size() <= 1);  // single-parent scope -- see class-level @note
        if (node.parents().empty()) {
            continue;
        }

        Tensor local_grad = fn_it->second(gradients_.at(id));
        NodeId parent_id = node.parents()[0]->id();

        auto parent_it = gradients_.find(parent_id);
        if (parent_it == gradients_.end()) {
            gradients_.emplace(parent_id, std::move(local_grad));
        } else {
            parent_it->second.accumulate(local_grad);
        }
    }
}

const Tensor& Autograd::gradient(NodeId id) const {
    EXAI_ASSERT(has_gradient(id));
    return gradients_.at(id);
}

}  // namespace exai
