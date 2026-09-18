/** @file autograd.hpp
 *  @brief Reverse-mode autodiff -- walks a ComputationGraph backward, accumulating
 *         gradients via per-node backward functions supplied by the caller.
 */
#pragma once

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exai/computation_graph.hpp"
#include "exai/node.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief Computes gradients by walking a ComputationGraph in reverse topological order.
 * @note Phase 0 has no Module/layer abstraction yet -- backward functions are supplied
 *       directly by the caller via register_backward() rather than being derived from an
 *       op-type dispatch table. Phase 1's Module::forward() is expected to call
 *       register_backward() when it builds graph nodes; this class doesn't need to know
 *       anything about Linear/Conv/etc. to do its job.
 * @note Scoped to single-parent nodes: a registered backward function takes the gradient
 *       w.r.t. this node's output and returns the gradient w.r.t. its one input. Nodes
 *       that combine multiple parents (branching/residual composition) are explicitly out
 *       of scope here -- charter Part 2 SS6 defers that composition to Module-level design
 *       (a TransformerBlock "knows its own composition"), which is Phase 1+/2 work.
 * @note backward() is not noexcept: Tensor/DeviceBackend allocation can throw (see
 *       cpp_style_guide's error-handling table), so this gives only the basic exception
 *       safety guarantee, not nothrow -- campaign Decision Point 3, resolved here rather
 *       than assumed.
 */
class Autograd {
public:
    /** @brief A function computing the gradient w.r.t. a node's single input, given the gradient w.r.t. its output. */
    using BackwardFn = std::function<Tensor(const Tensor& grad_output)>;

    /**
     * @brief Registers how to compute this node's input gradient from its output gradient.
     * @param id Node to register a backward function for. Must have at most one parent
     *        (see class-level @note on single-parent scope).
     * @param fn The backward function.
     */
    void register_backward(NodeId id, BackwardFn fn);

    /**
     * @brief Runs backward from a single root, seeding its gradient with grad_output.
     * @param graph The graph to walk. Must outlive this call.
     * @param root Node to seed.
     * @param grad_output Gradient to seed at root.
     */
    void backward(const ComputationGraph& graph, NodeId root, const Tensor& grad_output);

    /**
     * @brief Runs backward from multiple seeded roots in one pass, so gradients that
     *        converge on a shared ancestor accumulate correctly within a single call.
     * @param graph The graph to walk.
     * @param seeds (node id, seed gradient) pairs to start from.
     */
    void backward(const ComputationGraph& graph, std::vector<std::pair<NodeId, Tensor>> seeds);

    /**
     * @brief Retrieves the accumulated gradient for a node after backward() has run.
     * @param id Node id. Must satisfy has_gradient(id).
     * @return The accumulated gradient.
     */
    [[nodiscard]] const Tensor& gradient(NodeId id) const;

    /** @brief Whether a gradient was accumulated for this node during the last backward() call. */
    [[nodiscard]] bool has_gradient(NodeId id) const { return gradients_.find(id) != gradients_.end(); }

private:
    std::unordered_map<NodeId, BackwardFn> backward_fns_;
    std::unordered_map<NodeId, Tensor> gradients_;
};

}  // namespace exai
