/** @file disjunction_module.hpp
 *  @brief Differentiable fuzzy disjunction (t-conorm) -- Phase 1 Mission 0 of
 *         campaign_exai_dl_library_neuro_symbolic (differentiable fuzzy-logic core,
 *         Logic Tensor Networks-shaped).
 *  @ingroup neuro_symbolic
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `y = a S b` for a selected t-conorm S, over two independent fuzzy-truth-valued
 *        operand tensors (values intended in [0,1]; out-of-range values are not rejected --
 *        see conjunction_module.hpp's identical note).
 *
 * @note Same Stack-based binary-input design as ConjunctionModule (mission_0_tnorm_operators.md
 *       Stage 3) -- see that header's design-decision note; not re-derived here.
 */
class DisjunctionModule : public Module {
public:
    /** @brief Which t-conorm this instance computes. Product is the campaign's primary case. */
    enum class TConorm {
        Product,      ///< a + b - a*b
        Lukasiewicz,  ///< min(1, a + b)
        Godel         ///< max(a, b)
    };

    /**
     * @brief Constructs a disjunction module.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param t_conorm Which t-conorm to compute. Defaults to Product.
     */
    explicit DisjunctionModule(DeviceBackend* backend, TConorm t_conorm = TConorm::Product);

    using Module::forward;

    /**
     * @brief Convenience two-operand entry point -- see ConjunctionModule::forward(a, b)'s
     *        identical convention.
     */
    [[nodiscard]] Tensor forward(const Tensor& a, const Tensor& b);

    /** @brief See ConjunctionModule::stack_operands()'s identical convention. */
    [[nodiscard]] static Tensor stack_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend);

    /**
     * @brief Gradient w.r.t. this module's (stacked) input.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached output shape.
     * @note Same Godel/Lukasiewicz tie-breaking convention as ConjunctionModule: ties go to
     *       the first operand (a).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own op_type() convention. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation for the selected t-conorm -- genuinely novel, no
     *        prior art (research_2026_neuro_symbolic_ai.md §3); hand-derived and
     *        conservation-tested before this implementation existed.
     *
     * - **Product** (`y = a + b - a*b`): unlike conjunction's pure bilinear product, this
     *   has a mixed additive+bilinear structure -- neither a plain weighted sum nor a plain
     *   bilinear form, so neither LinearModule's nor AttnLRP Eq. 15's rule applies directly.
     *   This is this operator's own genuinely new derivation: `y` decomposes exactly two
     *   ways as a two-term "weighted sum" (the shape every other rule in this codebase
     *   already knows how to split) by picking which operand plays the fixed-weight-1 role:
     *     `y = a*1 + b*(1-a)`   (decomposition 1: weight_a=1, weight_b=(1-a))
     *     `y = b*1 + a*(1-b)`   (decomposition 2: weight_b=1, weight_a=(1-b))
     *   Both are exact (`a*1+b*(1-a) = a+b-ab = y`, symmetric for decomposition 2) but each
     *   is individually *asymmetric* in a/b -- decomposition 1 privileges a, decomposition 2
     *   privileges b. Averaging the two term-by-term restores the symmetry the operator
     *   itself has (disjunction doesn't distinguish its operands):
     *     `z_a = (a + a*(1-b)) / 2 = a*(2-b)/2`,  `z_b = (b + b*(1-a)) / 2 = b*(2-a)/2`
     *     `z_a + z_b = (a*(2-b) + b*(2-a)) / 2 = (2a - ab + 2b - ab) / 2 = a + b - ab = y`
     *   exactly, for every a, b -- not just near-exact. The averaged terms are then run
     *   through this codebase's standard epsilon-rule split against y (LinearModule-shaped):
     *   `R_a = z_a/(y+eps*sign(y)) * R_out`, `R_b = z_b/(y+eps*sign(y)) * R_out`, giving
     *   near-exact conservation (`R_a+R_b = y/(y+eps) * R_out`), the same near-exactness
     *   class as every other epsilon-stabilized rule in this codebase.
     * - **Lukasiewicz** (`y = min(1, a+b)`): same active/inactive split as
     *   ConjunctionModule's Lukasiewicz rule, mirrored -- active (`a+b < 1`): bias-free
     *   epsilon split on `z = a+b` (`R_a = a/(z+eps)*R_out`, `R_b = b/(z+eps)*R_out`);
     *   saturated (`a+b >= 1`, `y = 1` locally constant): both operands' derivative is 0,
     *   both receive 0, consistent with backward().
     * - **Godel** (`y = max(a, b)`): winning (larger, tie -> a) operand receives all of
     *   R_out exactly; the other receives 0 -- exact conservation, mirroring
     *   ConjunctionModule's Godel rule.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief Splits input (leading dim 2) into the two operands and computes the selected
     *        t-conorm elementwise.
     * @throws std::invalid_argument if input's rank is 0 or its leading dimension isn't 2.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    TConorm t_conorm_;
    Tensor last_input_;
    Tensor last_output_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
