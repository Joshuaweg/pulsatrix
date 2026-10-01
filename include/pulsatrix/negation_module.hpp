/** @file negation_module.hpp
 *  @brief Standard fuzzy negation `y = 1 - x` -- Phase 1 Mission 0 of
 *         campaign_exai_dl_library_neuro_symbolic (differentiable fuzzy-logic core).
 *  @ingroup neuro_symbolic
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `y = 1 - x`, elementwise. No parameters, no parameter gradients.
 * @note Unlike ConjunctionModule/DisjunctionModule (genuinely binary, two independent
 *       operand tensors), negation is a plain unary elementwise map -- it fits
 *       Module::forward()'s existing single-Tensor contract with no design question to
 *       resolve (see mission_0_tnorm_operators.md's Stage 3 section, which is scoped to
 *       the binary operators only).
 */
class NegationModule : public Module {
public:
    /**
     * @brief Constructs a negation module.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     */
    explicit NegationModule(DeviceBackend* backend);

    /**
     * @brief Gradient w.r.t. this module's input: `dy/dx = -1`, so `grad_x = -grad_output`.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached forward shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own single-input, weight-free operation. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation: pass-through, unchanged.
     * @note Not a placeholder -- this is the correct rule for negation given this
     *       codebase's own convention (ReluModule's identical justification): LRP rules are
     *       defined across combinations of multiple relevance-bearing inputs (the genuinely
     *       novel case ConjunctionModule/DisjunctionModule need); a single-input, monotonic,
     *       bijective elementwise reparametrization like `1 - x` passes relevance through
     *       unchanged, exactly like ReluModule's pointwise nonlinearity does. The `1` in
     *       `y = -x + 1` is a constant bias, absorbed rather than distributed -- same
     *       bias-exclusion convention LinearModule's own epsilon rule already uses (its
     *       denominator is the pre-bias `z_j`, not the post-bias output).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    Tensor last_input_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
