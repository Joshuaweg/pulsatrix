/** @file module.hpp
 *  @brief Abstract base every layer subclasses -- NVI forward(), pure-virtual LRP contract.
 */
#pragma once

#include "exai/assert.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief Base class for every layer type (LinearModule, Conv2DModule, activations, ...).
 * @note `forward()`/`forward_impl()` is the NVI (non-virtual interface) idiom
 *       (`oop_design/context_oop_design_patterns.md`'s Template Method section): the
 *       public entry point enforces preconditions every subclass gets for free; subclasses
 *       only implement the part that actually varies.
 * @note `propagate_relevance` is pure-virtual -- charter non-negotiable #5. A module type
 *       without a defined LRP rule is a compile error, not a runtime "no default rule"
 *       exception (the Captum/Zennit failure mode this project exists to avoid).
 * @note A module does not know about ComputationGraph/Autograd -- graph node registration
 *       and backward-function wiring are the caller's responsibility (the training loop,
 *       Mission 4). A module exposes plain tensor-in/tensor-out operations; graph
 *       bookkeeping is a separate concern (single responsibility).
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

protected:
    /** @brief The actual forward computation. Called by forward() after precondition checks. */
    [[nodiscard]] virtual Tensor forward_impl(const Tensor& input) = 0;
};

}  // namespace exai
