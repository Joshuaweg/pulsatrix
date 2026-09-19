/** @file module.hpp
 *  @brief Abstract base every layer subclasses -- NVI forward(), pure-virtual LRP contract.
 */
#pragma once

#include <vector>

#include "exai/assert.hpp"
#include "exai/lrp_rule_config.hpp"
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
     */
    [[nodiscard]] Tensor forward(const Tensor& input) {
        EXAI_ASSERT(input.numel() > 0);
        return forward_impl(input);
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

protected:
    /** @brief The actual forward computation. Called by forward() after precondition checks. */
    [[nodiscard]] virtual Tensor forward_impl(const Tensor& input) = 0;
};

}  // namespace exai
