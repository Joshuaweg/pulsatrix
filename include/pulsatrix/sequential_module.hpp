/** @file sequential_module.hpp
 *  @brief Model container -- chains a sequence of existing Modules.
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Composes layers_[0..n-1] in forward() order; backward()/propagate_relevance()
 *        chain layers_[n-1..0] in reverse -- correct reverse-mode composition order.
 * @note layers_ is not owned -- "not owned; must outlive this object", the same
 *       convention as every DeviceBackend* member in this codebase. A caller constructing
 *       a SequentialModule keeps its constituent modules alive independently.
 * @note propagate_relevance needs no new LRP theory -- pure delegation to each contained
 *       layer's own already-cited rule. Conservation holds end-to-end because composing
 *       conserving functions conserves; this is verified numerically by an
 *       LRPConservationTest TEST_P case, not just asserted in prose.
 * @note set_training() overrides the (now virtual, since this mission) base method: sets
 *       its own flag and cascades to every contained layer -- correct even when accessed
 *       through a Module* base pointer, not just SequentialModule's own concrete type.
 */
class SequentialModule : public Module {
public:
    /**
     * @brief Constructs a container over an ordered sequence of layers.
     * @param layers Layers in forward-execution order. Not owned; must outlive this object.
     * @throws std::invalid_argument if layers is empty, or any entry is nullptr -- both
     *         external boundaries (a caller-constructed vector, e.g. from Phase 5's Python
     *         bindings, could be empty or contain a null entry with no upstream validation).
     */
    explicit SequentialModule(std::vector<Module*> layers);

    /**
     * @brief Chains backward() across layers_ in reverse order.
     * @param grad_output Gradient w.r.t. this container's output. Must match the shape of
     *        the most recent forward() call's output (validated by the last layer's own
     *        backward(), not by SequentialModule itself).
     * @return Gradient w.r.t. this container's input.
     * @throws std::logic_error if forward() has never been called.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Composite per charter's closed OpType set -- a container wrapping several
     *         ops is genuinely not any single existing category. */
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }

    /**
     * @brief Chains propagate_relevance() across layers_ in reverse order.
     * @param relevance_out Relevance at this container's output. Must match the shape of
     *        the most recent forward() call's output (validated by the last layer's own
     *        propagate_relevance(), not by SequentialModule itself).
     * @param config Forwarded unchanged to every contained layer's own rule.
     * @return Relevance at this container's input.
     * @throws std::logic_error if forward() has never been called.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief A rule is supported iff every contained layer supports it (the config is forwarded to all). */
    [[nodiscard]] bool supports_lrp_rule(LRPRule rule) const override {
        for (const Module* layer : layers_) {
            if (!layer->supports_lrp_rule(rule)) {
                return false;
            }
        }
        return true;
    }

    /** @brief Every contained layer's named_parameters(), prefixed with its index (`0.weight`). */
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;

    /**
     * @brief Sets this container's own training flag and cascades to every contained layer.
     * @param training New training/eval mode.
     */
    void set_training(bool training) override;

    /** @brief The contained layers, in forward-execution order. */
    [[nodiscard]] const std::vector<Module*>& layers() const { return layers_; }

protected:
    /**
     * @brief Chains forward() across layers_ in order.
     * @throws Whatever the first layer whose own forward() rejects input throws --
     *         SequentialModule adds no extra validation of its own beyond delegation.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    std::vector<Module*> layers_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
