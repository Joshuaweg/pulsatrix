/** @file explainer_context.hpp
 *  @brief Stable interface every explainer gets, regardless of type (charter Part 2 SS2).
 */
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exai/assert.hpp"
#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/module.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief Wraps an ordered chain of Modules, running them via Module::forward_traced to
 *        build a real ComputationGraph and Autograd backward wiring -- graph-native
 *        explainers (Missions 2-3) use graph()/backward_pass()/activation(); surrogate
 *        explainers (Phase 3) would use only forward_pass(), per the charter's stated
 *        interface segregation.
 * @note Owns a std::vector<Module*>, not a new Sequential container -- no generic
 *       module-sequence type exists yet in this codebase (XorNetwork hardcodes its own
 *       chain), and this mission doesn't need one beyond what ExplainerContext itself
 *       requires. Modules are not owned; every pointer must outlive this ExplainerContext.
 * @note forward_pass() replaces its ComputationGraph/Autograd members with fresh instances
 *       on every call, rather than mutating a shared graph -- ComputationGraph deliberately
 *       has no clear/reset method (Phase 0's persistence-past-backward guarantee applies
 *       within one forward_pass()'s lifetime), but Integrated Gradients (Mission 2) needs
 *       multiple independent forward passes at different interpolated inputs, each with
 *       its own graph. See mission_explainer_context.md's Recon.
 */
class ExplainerContext {
public:
    /**
     * @brief Constructs a context over an ordered module chain.
     * @param modules The network, in forward-pass order. Not owned; each must outlive
     *        this ExplainerContext.
     * @throws std::invalid_argument if modules is empty or contains a nullptr -- external
     *         boundary (campaign_exai_dl_library_adversarial_hardening.md, Mission 2,
     *         finding 5): without this check, a nullptr element was an unconditional
     *         null-pointer dereference on the next forward_pass() call.
     */
    explicit ExplainerContext(std::vector<Module*> modules) : modules_(std::move(modules)) {
        if (modules_.empty()) {
            throw std::invalid_argument("ExplainerContext: modules must not be empty");
        }
        for (Module* m : modules_) {
            if (m == nullptr) {
                throw std::invalid_argument("ExplainerContext: modules must not contain a null pointer");
            }
        }
    }

    /**
     * @brief Runs the full module chain forward, building a fresh graph and caching every
     *        node's activation value as it goes.
     * @param input Input to the first module in the chain.
     * @return The final module's output.
     */
    Tensor forward_pass(const Tensor& input) {
        graph_ = ComputationGraph{};
        autograd_ = Autograd{};
        activations_.clear();

        input_node_ = graph_.add_node(OpType::Elementwise, input.shape(), "input");
        activations_.emplace(input_node_, Tensor(input));

        Tensor current = input;
        NodeId current_node = input_node_;
        for (Module* module : modules_) {
            auto [output, node_id] = module->forward_traced(current, current_node, graph_, autograd_);
            activations_.emplace(node_id, Tensor(output));
            current = std::move(output);
            current_node = node_id;
        }
        output_node_ = current_node;
        return current;
    }

    /**
     * @brief Runs Autograd::backward from the most recent forward_pass()'s output node.
     * @param output_grad Gradient w.r.t. the chain's output.
     * @return Gradient w.r.t. the chain's input.
     * @note Must be called after forward_pass().
     */
    Tensor backward_pass(const Tensor& output_grad) {
        autograd_.backward(graph_, output_node_, output_grad);
        return Tensor(autograd_.gradient(input_node_));
    }

    /** @brief The current graph (from the most recent forward_pass() call). */
    [[nodiscard]] const ComputationGraph& graph() const { return graph_; }

    /**
     * @brief The cached activation value at a node, from the most recent forward_pass().
     * @param id Node id. Must have been produced by the most recent forward_pass() call.
     */
    [[nodiscard]] const Tensor& activation(NodeId id) const {
        auto it = activations_.find(id);
        EXAI_ASSERT(it != activations_.end());
        return it->second;
    }

    /**
     * @brief The cached gradient at a node, from the most recent backward_pass() call.
     * @param id Node id. Must have accumulated a gradient during the most recent
     *        backward_pass() call (i.e. lie on the path between the seeded output and
     *        the input).
     * @note Mirrors activation()'s shape -- Autograd::backward() already populates a
     *       gradient at every node it walks through, not just the input node
     *       backward_pass() itself returns; this exposes that directly, needed for
     *       Grad-CAM's target-conv-layer gradient (Phase 2 Mission 3).
     */
    [[nodiscard]] const Tensor& gradient(NodeId id) const { return autograd_.gradient(id); }

    /** @brief Passthrough to Node::label() -- e.g. a layer name, for debugging/display. */
    [[nodiscard]] std::optional<std::string> layer_label(NodeId id) const { return graph_.node(id).label(); }

private:
    std::vector<Module*> modules_;
    ComputationGraph graph_;
    Autograd autograd_;
    std::unordered_map<NodeId, Tensor> activations_;
    NodeId input_node_ = 0;
    NodeId output_node_ = 0;
};

}  // namespace exai
