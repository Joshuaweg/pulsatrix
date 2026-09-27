/** @file conjunction_module.hpp
 *  @brief Differentiable fuzzy conjunction (t-norm) -- Phase 1 Mission 0 of
 *         campaign_exai_dl_library_neuro_symbolic (differentiable fuzzy-logic core,
 *         Logic Tensor Networks-shaped).
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `y = a T b` for a selected t-norm T, over two independent fuzzy-truth-valued
 *        operand tensors (values intended in [0,1]; out-of-range values are not rejected --
 *        see propagate_relevance()'s note and mission_0_tnorm_operators.md's adversarial
 *        section).
 *
 * @note **Design decision (mission_0_tnorm_operators.md Stage 3, resolved): Option 1 --
 *       Stack-based.** Conjunction is genuinely binary (two independent external operand
 *       tensors), but Module::forward() is single-Tensor in/out. The two operands are
 *       combined via `stack_operands()` below (a thin wrapper around the already-existing
 *       `Tensor::Stack`, tensor.hpp:84) into one leading-dim-2 tensor before entering the
 *       ordinary `Module::forward()`/`forward_impl()` NVI path; `forward_impl` splits it
 *       back into the two operands internally, and `backward()`/`propagate_relevance()`
 *       re-split the incoming gradient/relevance the same way. Chosen over the
 *       MSELoss-shaped free-function alternative (Option 2, mission file's own comparison)
 *       specifically because this keeps ConjunctionModule a genuine `Module` subclass --
 *       needed for `op_type()`/`forward_traced()` participation once Mission 1's
 *       `AggregatorModule` and the satisfaction loss compose these operators inside a
 *       traced graph (per campaign scope's explicit "native Modules" framing) -- at the
 *       cost of an internal split/re-split each call, which is O(n) elementwise work, not
 *       a new execution model.
 * @note `forward(const Tensor&, const Tensor&)` is the convenience two-operand entry point;
 *       `Module::forward(const Tensor&)` (the single-Tensor NVI base method) remains
 *       reachable via the `using` declaration below and expects an already-stacked tensor
 *       (leading dim exactly 2) -- exactly what `stack_operands()`/`forward(a, b)` build,
 *       and what `forward_traced()` threads through when this module participates in a
 *       ComputationGraph.
 */
class ConjunctionModule : public Module {
public:
    /** @brief Which t-norm this instance computes. Product is the campaign's primary case. */
    enum class TNorm {
        Product,      ///< a * b
        Lukasiewicz,  ///< max(0, a + b - 1)
        Godel         ///< min(a, b)
    };

    /**
     * @brief Constructs a conjunction module.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param t_norm Which t-norm to compute. Defaults to Product (campaign's primary case).
     */
    explicit ConjunctionModule(DeviceBackend* backend, TNorm t_norm = TNorm::Product);

    using Module::forward;

    /**
     * @brief Convenience two-operand entry point: builds the stacked input via
     *        stack_operands() and delegates to Module::forward().
     * @param a First operand tensor.
     * @param b Second operand tensor. Must match a's rank and every non-leading-adjusted
     *        dimension -- see stack_operands()'s note; a shape mismatch surfaces as
     *        Tensor::Stack's own std::invalid_argument (external boundary, already
     *        implemented there, not re-validated here).
     * @return The conjunction, same shape as a (and b).
     */
    [[nodiscard]] Tensor forward(const Tensor& a, const Tensor& b);

    /**
     * @brief Combines two independent operand tensors into the leading-dim-2 stacked
     *        tensor forward_impl()/backward()/propagate_relevance() expect.
     * @param a First operand.
     * @param b Second operand.
     * @param backend Backend to allocate the stacked tensor through.
     * @return A tensor of shape (2, a.shape()...).
     * @throws std::invalid_argument if a and b differ in rank or any dimension -- delegated
     *         to Tensor::Stack, which already implements exactly this external-boundary check.
     */
    [[nodiscard]] static Tensor stack_operands(const Tensor& a, const Tensor& b, DeviceBackend* backend);

    /**
     * @brief Gradient w.r.t. this module's (stacked) input, via the selected t-norm's
     *        closed-form partial derivatives.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output (the per-operand shape, not the stacked shape).
     * @return Gradient w.r.t. the stacked input, shape (2, output.shape()...).
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached output shape.
     * @note Tie-breaking convention (Godel, and Lukasiewicz's boundary): when the two
     *       operands are exactly equal, or exactly at Lukasiewicz's clip boundary, the whole
     *       gradient is attributed to the first operand (a) -- a deterministic, documented
     *       choice, the same style as ReluModule's own x == 0 "treated as blocked" convention
     *       rather than an unhandled tie.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own op_type() convention (ReluModule/ResidualModule). */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation for the selected t-norm -- genuinely novel, no prior
     *        art (research_2026_neuro_symbolic_ai.md §3's honest finding); hand-derived here,
     *        conservation-tested before this implementation existed.
     *
     * - **Product** (`y = a*b`): the bilinear/"uniform" split this codebase already uses for
     *   Q@K^T-shaped matmul products (AttnLRP Eq. 15, `bilinear_lrp_eq15` in
     *   multihead_attention_module.cpp) -- applied here to the elementwise (inner
     *   dimension 1) case: `R_a = R_b = (a*b) / (2*y + eps*sign(y)) * R_out`. Extending an
     *   already-vetted-in-this-codebase bilinear rule from a matmul's inner-product
     *   structure to a t-norm's elementwise product is the novel step (no literature
     *   applies AttnLRP-style bilinear splitting to fuzzy logic operators).
     * - **Lukasiewicz** (`y = max(0, a+b-1)`): active region (`a+b-1 > 0`) is exactly
     *   LinearModule's own bias-excluded epsilon rule with `z = a+b-1` (the `-1` bias
     *   absorbed, not distributed, same convention as LinearModule's pre-bias `z_j`):
     *   `R_a = a/(z+eps*sign(z)) * R_out`, `R_b = b/(z+eps*sign(z)) * R_out`. Inactive region
     *   (`a+b-1 <= 0`, `y = 0`): both operands' local derivative is 0 (matches backward()'s
     *   own gradient there), so both receive 0 -- ReluModule's "blocked" convention,
     *   deliberately kept consistent with backward() rather than force-conserving through a
     *   dead branch.
     * - **Godel** (`y = min(a, b)`): the winning (smaller, tie -> a) operand receives all of
     *   `R_out` exactly (`R_a = a/y * R_out = R_out` when a wins, since a == y there); the
     *   other receives 0. Exact (not merely near-exact) conservation, since min's active
     *   branch is a pure identity map on the winning operand.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief Splits input (leading dim 2) into the two operands and computes the selected
     *        t-norm elementwise.
     * @param input A stacked tensor, shape (2, ...), as built by stack_operands().
     * @return The t-norm result, shape equal to input's shape with the leading dim dropped.
     * @throws std::invalid_argument if input's rank is 0 or its leading dimension isn't 2 --
     *         external boundary: forward_impl is reachable directly through the inherited,
     *         non-virtual Module::forward(const Tensor&) by any caller bypassing
     *         forward(a, b)/stack_operands() (e.g. Phase 5's Python bindings, or
     *         forward_traced()'s own single-Tensor threading).
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    TNorm t_norm_;
    Tensor last_input_;   ///< Cached stacked input, shape (2, ...).
    Tensor last_output_;  ///< Cached t-norm result, shape (...).
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
