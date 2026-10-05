/** @file module.hpp
 *  @brief Abstract base every layer subclasses -- NVI forward(), pure-virtual LRP contract.
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/autograd.hpp"
#include "pulsatrix/computation_graph.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/node.hpp"
#include "pulsatrix/op_type.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief A trainable parameter and its accumulated gradient, as owned by some Module. */
struct ParamRef {
    Tensor* value;
    Tensor* grad;
};

/**
 * @brief A parameter together with its hierarchical, dot-separated name relative to the
 *        module that reported it (`weight`, `mha.q_proj.bias`, `0.weight`).
 * @note Leaf names follow PyTorch where a PyTorch analog exists (`weight`/`bias`, including
 *       Conv2D's kernel and the norms' gamma/beta); otherwise they are the C++ member name
 *       without its trailing underscore. A container prefixes each child's names with that
 *       child's accessor name (or its index, for SequentialModule). Roadmap FND-1.
 */
struct NamedParamRef {
    std::string name;
    ParamRef ref;
};

/**
 * @brief Module state that is saved with a model but never trained, such as BatchNorm's running
 *        statistics, with its hierarchical name (roadmap IO-2). Named like parameters.
 */
struct NamedBufferRef {
    std::string name;
    Tensor* value;
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
     *         PULSATRIX_ASSERT-only.
     */
    [[nodiscard]] Tensor forward(const Tensor& input) {
        if (input.numel() <= 0) {
            throw std::invalid_argument("Module::forward: input must not be empty");
        }
        if (const std::optional<DeviceType> device = compute_device()) {
            require_device(input, *device, "Module::forward");
        }
        return forward_impl(input);
    }

    /**
     * @brief The device this module computes on, so forward() can reject an input on another
     *        device before any kernel sees it (roadmap FND-8).
     * @return std::nullopt (the default) skips the check: a container whose layers check their
     *         own inputs, or a user module written before this existed. Every in-tree layer
     *         returns its device; EmbeddingModule doesn't, because it reads its indices through
     *         their own backend and so accepts them from any device.
     */
    [[nodiscard]] virtual std::optional<DeviceType> compute_device() const { return std::nullopt; }

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
     * @brief Whether propagate_relevance() implements `rule` (no silent fallback: callers such
     *        as ExplainerContext::relevance_pass() throw rather than run a module on a rule it
     *        does not implement).
     * @note Default: only LRPRule::Epsilon, the rule every module implements. Overridden by
     *       LinearModule / Conv2DModule (all rules), by parameter-free pass-through modules whose
     *       relevance rule does not read the config at all, and by SequentialModule (all layers).
     */
    [[nodiscard]] virtual bool supports_lrp_rule(LRPRule rule) const { return rule == LRPRule::Epsilon; }

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
     * @brief This module's trainable parameters, each with its hierarchical name -- the one
     *        place a module declares its parameters (roadmap FND-1).
     * @return {name, {value, grad}} entries pointing directly at this module's own members,
     *         in a fixed order. Names are unique within the module tree. Default: empty (a
     *         parameterless module like ReluModule needs no override).
     * @note Override this, not parameters(): saving, loading, freezing by name and optimizer
     *       parameter groups all key on these names.
     */
    [[nodiscard]] virtual std::vector<NamedParamRef> named_parameters() { return {}; }

    /**
     * @brief This module's buffers: state a checkpoint must save that no optimizer updates.
     * @return {name, tensor} entries, named like named_parameters(), in a fixed order. Default:
     *         empty. BatchNormModule reports its running statistics; containers prefix their
     *         layers' buffers the way they prefix parameters.
     */
    [[nodiscard]] virtual std::vector<NamedBufferRef> named_buffers() { return {}; }

    /**
     * @brief This module's trainable parameters and their gradients, for an optimizer to
     *        update uniformly across module types.
     * @return named_parameters() without the names -- same tensors, same order.
     * @note Not pure-virtual -- unlike propagate_relevance, there is no charter
     *       non-negotiable requiring every module to define this; "no parameters" is a
     *       legitimate, common answer that shouldn't need restating per module type.
     * @note Still virtual only so subclasses written before named_parameters() existed keep
     *       compiling and training. Such a subclass reports no names, so name-keyed features
     *       can't see its parameters; new code overrides named_parameters() instead.
     */
    [[nodiscard]] virtual std::vector<ParamRef> parameters() {
        std::vector<ParamRef> params;
        for (const NamedParamRef& p : named_parameters()) {
            params.push_back(p.ref);
        }
        return params;
    }

    /**
     * @brief Freezes (`false`) or unfreezes (`true`) parameters by name (roadmap FND-2).
     * @param requires_grad The flag to set on every selected parameter's value tensor.
     * @param prefix Empty selects every parameter. Otherwise selects the parameter named
     *        exactly `prefix`, and every parameter under it (`mha.q_proj` selects
     *        `mha.q_proj.weight` and `mha.q_proj.bias`, but `mha.q` selects nothing).
     * @throws std::invalid_argument if a non-empty prefix selects nothing -- a mistyped name
     *         would otherwise silently leave the model trainable. Nothing is changed then.
     * @note A frozen parameter's gradient is not accumulated by backward() and is not
     *       updated by an optimizer; the gradient w.r.t. the module's input is unchanged.
     */
    void set_requires_grad(bool requires_grad, const std::string& prefix = "") {
        if (prefix.empty()) {
            // parameters(), not named_parameters(): also reaches a legacy module that has no names.
            for (ParamRef p : parameters()) {
                p.value->set_requires_grad(requires_grad);
            }
            return;
        }
        std::vector<Tensor*> selected;
        for (const NamedParamRef& p : named_parameters()) {
            if (p.name == prefix || p.name.rfind(prefix + ".", 0) == 0) {
                selected.push_back(p.ref.value);
            }
        }
        if (selected.empty()) {
            throw std::invalid_argument("Module::set_requires_grad: no parameter named or under '" + prefix + "'");
        }
        for (Tensor* value : selected) {
            value->set_requires_grad(requires_grad);
        }
    }

    /**
     * @brief Sets this module's training/eval mode. Defaults to training (matches every
     *        mainstream framework's Module default).
     * @note Virtual since Phase 6 Mission 6 (SequentialModule) -- Mission 5 originally
     *       shipped this as plain non-virtual state ("no composite container exists yet to
     *       cascade through"); SequentialModule overrides this to cascade to every
     *       contained layer, and needs virtual dispatch to do so correctly even when
     *       accessed through a Module* base pointer, not just its own concrete type.
     * @note BatchNormModule keeps running statistics in training mode and normalizes with
     *       them in eval mode; DropoutModule is the identity in eval mode. Call
     *       set_training(false) before explaining a model that has either, so a sample's
     *       explanation doesn't depend on the rest of its batch.
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

/**
 * @brief Appends `child`'s named parameters to `out`, each renamed to `prefix.name` -- the
 *        one step every container's named_parameters() repeats per child.
 * @note A child that overrides only the legacy parameters() hook reports no names; its
 *       parameters are appended under positional names (`prefix.0`, `prefix.1`, ...) so a
 *       container never hides them from an optimizer that used to see them.
 */
inline void append_named_parameters(std::vector<NamedParamRef>& out, const std::string& prefix, Module& child) {
    std::vector<NamedParamRef> named = child.named_parameters();
    if (named.empty()) {
        std::vector<ParamRef> plain = child.parameters();
        for (size_t i = 0; i < plain.size(); ++i) {
            named.push_back({std::to_string(i), plain[i]});
        }
    }
    for (NamedParamRef& p : named) {
        out.push_back({prefix + "." + p.name, p.ref});
    }
}

/** @brief Appends `child`'s named buffers to `out`, each renamed to `prefix.name`. */
inline void append_named_buffers(std::vector<NamedBufferRef>& out, const std::string& prefix, Module& child) {
    for (NamedBufferRef& b : child.named_buffers()) {
        out.push_back({prefix + "." + b.name, b.value});
    }
}

}  // namespace pulsatrix
