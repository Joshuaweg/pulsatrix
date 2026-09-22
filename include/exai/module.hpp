/** @file module.hpp
 *  @brief Abstract base every layer subclasses -- NVI forward(), pure-virtual LRP contract.
 */
#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "exai/assert.hpp"
#include "exai/autograd.hpp"
#include "exai/computation_graph.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/node.hpp"
#include "exai/op_type.hpp"
#include "exai/tensor.hpp"

namespace exai {

/** @brief A trainable parameter and its accumulated gradient, as owned by some Module. */
struct ParamRef {
    Tensor* value;
    Tensor* grad;
};

/**
 * @brief Base class for every layer type (LinearModule, Conv2DModule, activations, ...).
 * @note `forward()`/`forward_impl()` is the NVI (non-virtual interface) idiom
 *       (`oop_design/context_oop_design_patterns.md`'s Template Method section): the
 *       public entry point enforces preconditions every subclass gets for free; subclasses
 *       only implement the part that actually varies.
 * @note `propagate_relevance` is pure-virtual -- charter non-negotiable #5. A module type
 *       without a defined LRP rule is a compile error, not a runtime "no default rule"
 *       exception (the Captum/Zennit failure mode this project exists to avoid).
 * @note A module does not build or own graph structure itself -- ComputationGraph node
 *       registration and Autograd backward-function wiring are done generically by
 *       whichever caller opts into the traced path (Phase 2 Mission 0's
 *       Module::forward_traced), using backward()/op_type() below polymorphically. A
 *       module exposes plain tensor-in/tensor-out operations plus these two facts about
 *       itself; graph bookkeeping stays a separate concern (single responsibility).
 */
class Module {
public:
    virtual ~Module() = default;

    /**
     * @brief Runs this module's forward computation.
     * @param input Input tensor. Must be non-empty.
     * @return The module's output.
     * @throws std::invalid_argument if input is empty -- external boundary
     *         (campaign_exai_dl_library_adversarial_hardening.md, Mission 2, finding 15
     *         systemic sweep): the single most external-facing check in the whole system,
     *         since every Module::forward() call -- including from Phase 5's Python
     *         bindings -- passes through this NVI wrapper first. Escalated from
     *         EXAI_ASSERT-only.
     */
    [[nodiscard]] Tensor forward(const Tensor& input) {
        if (input.numel() <= 0) {
            throw std::invalid_argument("Module::forward: input must not be empty");
        }
        return forward_impl(input);
    }

    /**
     * @brief Runs forward() while also registering a ComputationGraph node (tagged with
     *        this module's op_type(), parented to input_node) and wiring an Autograd
     *        backward function that reuses this module's own backward() -- the opt-in
     *        traced/explainable path, per Phase 2 Mission 0.
     * @param input Input tensor. Must be non-empty (same precondition as forward()).
     * @param input_node Id of the graph node producing input. Must already exist in graph.
     * @param graph Graph to add this module's output node to. Must outlive the returned
     *        node id's use (graph structure, per ComputationGraph's own design, persists
     *        past this call and past any subsequent backward pass).
     * @param autograd Autograd instance to register this node's backward function with.
     * @return The output tensor (identical to what forward(input) alone would return) and
     *         the new node's id, so a caller chaining multiple modules can thread node ids
     *         the same way it already threads tensors.
     * @note Strictly additive: forward()/backward() are completely unaffected by this
     *       method's existence or use. Every graph-free call site (XorNetwork, every
     *       Phase 0/1 test) needs no changes.
     */
    [[nodiscard]] std::pair<Tensor, NodeId> forward_traced(const Tensor& input, NodeId input_node,
                                                             ComputationGraph& graph, Autograd& autograd) {
        Tensor output = forward(input);
        NodeId node_id = graph.add_node(op_type(), output.shape(), std::nullopt, {input_node});
        autograd.register_backward(node_id, [this](const Tensor& grad_output) { return this->backward(grad_output); });
        return {std::move(output), node_id};
    }

    /**
     * @brief Computes this module's contribution to LRP relevance propagation.
     * @param relevance_out Relevance at this module's output.
     * @param config Selects the LRP rule variant.
     * @return Relevance at this module's input.
     */
    [[nodiscard]] virtual Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) = 0;

    /**
     * @brief Computes the gradient w.r.t. this module's input, given the gradient w.r.t.
     *        its output. Must be called after forward() -- uses state cached from that call.
     * @param grad_output Gradient w.r.t. this module's output.
     * @return Gradient w.r.t. this module's input.
     * @note Promoted to the base class in Phase 2 Mission 0 -- every existing subclass
     *       (LinearModule/ReluModule/Conv2DModule) already implemented this exact
     *       signature independently; making it virtual lets graph-wiring code
     *       (Module::forward_traced) call it polymorphically through a Module* without
     *       knowing the concrete subclass, the same way propagate_relevance already works.
     */
    [[nodiscard]] virtual Tensor backward(const Tensor& grad_output) = 0;

    /**
     * @brief This module's operation-category tag, for ComputationGraph node tagging.
     * @return This module's OpType (see op_type.hpp's closed set).
     * @note Added in Phase 2 Mission 0 alongside backward() -- lets graph-wiring code tag
     *       nodes generically through a Module* rather than switching on concrete subclass.
     */
    [[nodiscard]] virtual OpType op_type() const = 0;

    /**
     * @brief This module's trainable parameters and their gradients, for an optimizer to
     *        update uniformly across module types.
     * @return {value, grad} pairs pointing directly at this module's own members. Default:
     *         empty (a parameterless module like ReluModule needs no override).
     * @note Not pure-virtual -- unlike propagate_relevance, there is no charter
     *       non-negotiable requiring every module to define this; "no parameters" is a
     *       legitimate, common answer that shouldn't need restating per module type.
     */
    [[nodiscard]] virtual std::vector<ParamRef> parameters() { return {}; }

    /**
     * @brief Sets this module's training/eval mode. Defaults to training (matches every
     *        mainstream framework's Module default).
     * @note Virtual since Phase 6 Mission 6 (SequentialModule) -- Mission 5 originally
     *       shipped this as plain non-virtual state ("no composite container exists yet to
     *       cascade through"); SequentialModule overrides this to cascade to every
     *       contained layer, and needs virtual dispatch to do so correctly even when
     *       accessed through a Module* base pointer, not just its own concrete type.
     * @note Deliberately NOT extended to BatchNormModule's running-mean/variance question
     *       (flagged, still open) -- that is additive numerical-tracking state, a
     *       genuinely different scope than this boolean toggle, and touching a closed
     *       mission's module is its own decision, not bundled in here.
     */
    virtual void set_training(bool training) { training_ = training; }

    /** @brief Whether this module is currently in training mode. */
    [[nodiscard]] bool is_training() const { return training_; }

protected:
    /** @brief The actual forward computation. Called by forward() after precondition checks. */
    [[nodiscard]] virtual Tensor forward_impl(const Tensor& input) = 0;

private:
    bool training_ = true;
};

}  // namespace exai
