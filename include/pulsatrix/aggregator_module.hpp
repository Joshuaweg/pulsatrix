/** @file aggregator_module.hpp
 *  @brief Differentiable p-mean quantifier aggregator -- Phase 1 Mission 1 of
 *         campaign_exai_dl_library_neuro_symbolic (Logic Tensor Networks' Real Logic:
 *         `agg_p(x) = (mean(x^p))^(1/p)`, standing in for a fuzzy universal/existential
 *         quantifier over a batch of groundings).
 *  @ingroup neuro_symbolic
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `y = (mean(x^p))^(1/p)`, reduced over the *leading* (batch/grounding) axis --
 *        not the last axis. Rank-agnostic: input shape `(N, ...rest)` reduces to output
 *        shape `(...rest)` (rank 0 -- a scalar -- when input is rank 1, i.e. a single
 *        formula's groundings with no other axes).
 *
 * @note **Design decision (mission_1_aggregator_satisfaction_loss.md Stage 3, resolved):
 *       no Stack-based wrapper needed.** Unlike ConjunctionModule/DisjunctionModule
 *       (genuinely binary, two independent external operand tensors, needing
 *       stack_operands()), AggregatorModule is unary -- one batch of groundings in, one
 *       reduced tensor out -- and fits Module::forward()'s existing single-Tensor contract
 *       directly, with no design question to resolve (mirrors NegationModule's identical
 *       "no wrapper needed" note, mission_0's own Stage 3). Verified explicitly, not
 *       assumed silently, per this mission file's own carried-forward Stage 3 prompt.
 * @note **Reduction axis is the *leading* dimension (axis 0), not the last axis** --
 *       deliberately the opposite convention from SoftmaxModule (which reduces the *last*
 *       axis, per-row, because that axis is the class/feature axis in softmax's own
 *       problem). Here the axis being aggregated over is the *batch* axis (§2's "batch of
 *       groundings" framing) -- this codebase's own standing convention is that the batch
 *       dimension is the leading one (Tensor::Stack concatenates along the leading
 *       dimension; GroupNormModule computes per-batch-row statistics indexed by the
 *       leading `n`). Reducing axis 0 is therefore the semantically correct choice for
 *       "aggregate a formula's per-grounding truth degrees into one satisfaction degree",
 *       not an arbitrary pick between two equally-valid axes.
 * @note **`p == 0` is rejected at construction** (`std::invalid_argument`) -- the pure
 *       power-mean formula is undefined there (`mean(x^0) == 1` identically, then
 *       `1^(1/0)` is undefined; the true `p -> 0` limit is the *geometric* mean, a
 *       genuinely different formula this class does not implement). Checked once at
 *       construction, not per-call, since it is a fixed configuration value, not a
 *       per-element runtime input -- cheaper and earlier than a hot-path check.
 * @note **Negative `x` with non-integer `p`, or `x == 0` with negative `p`, are
 *       deliberately NOT validated or rejected** -- same policy as
 *       ConjunctionModule/DisjunctionModule's out-of-range-input note
 *       (mission_0_tnorm_operators.md Stage 3): `std::pow` already gives the honest
 *       floating-point answer for these domains (`NaN` for a negative base raised to a
 *       non-integer exponent, `+-inf` for `0` raised to a negative exponent), and adding a
 *       per-element domain check to every forward/backward/propagate_relevance call would
 *       cost hot-path time this codebase's convention avoids elsewhere. Verified by test
 *       (`*HandlesNegativeInputWithNonIntegerPWithoutThrowing`-style adversarial cases in
 *       aggregator_module_test.cpp), not enforced by code.
 */
class AggregatorModule : public Module {
public:
    /**
     * @brief Constructs a p-mean aggregator.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param p Aggregator exponent. `p == 1` is the arithmetic mean; `p == 2` is the RMS
     *        (root-mean-square, a common LTN "soft-forall" choice, since larger `p` weighs
     *        low-truth groundings more heavily, approaching a true min/forall as
     *        `p -> +inf`). Defaults to 2.0.
     * @throws std::invalid_argument if p == 0 -- see the class note above.
     */
    explicit AggregatorModule(DeviceBackend* backend, float p = 2.0f);

    /**
     * @brief Gradient w.r.t. this module's input, via `agg_p`'s closed-form partial
     *        derivative.
     *
     * Derivation: let `m = mean(x^p) = (1/n) * sum_i(x_i^p)`, `y = m^(1/p)`. Then
     * `dy/dx_i = (1/p) * m^(1/p - 1) * dm/dx_i = (1/p) * m^(1/p - 1) * (p/n) * x_i^(p-1)
     *          = (1/n) * m^(1/p - 1) * x_i^(p-1)`.
     * Implemented directly in this closed form (not rearranged into a `y`-based
     * division) so it stays well-defined at `y == 0` for every `p` where the direct
     * `m`-based powers themselves are well-defined (e.g. `p == 1`, where this reduces
     * to the constant `1/n` independent of `m`/`y` entirely -- the ordinary mean's own
     * gradient).
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output (input shape with the leading dim dropped).
     * @return Gradient w.r.t. this module's input, same shape as that forward() call's input.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached output shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 3).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Reduction, per charter's closed OpType set -- this is the campaign's first
     *         reduction-shaped (rather than elementwise) operator. */
    [[nodiscard]] OpType op_type() const override { return OpType::Reduction; }

    /**
     * @brief LRP relevance propagation -- genuinely novel, no prior art
     *        (research_2026_neuro_symbolic_ai.md §3's honest finding, same standing as
     *        every Mission 0 rule); hand-derived here, conservation-tested before this
     *        implementation existed.
     *
     * **Derivation, via Euler's homogeneous-function theorem (the same gradient-times-input
     * identity this codebase's other epsilon-rule derivations already rest on, e.g.
     * LinearModule's weighted sum):** for any fixed `t > 0`, `agg_p(t*x) = t * agg_p(x)`
     * (scaling every grounding by `t` scales the aggregated result by the same `t`) --
     * i.e. `agg_p` is positively homogeneous of **degree 1** in `x`. Euler's theorem for a
     * degree-1-homogeneous, differentiable function states `sum_i(x_i * dy/dx_i) == y`
     * *exactly*. Substituting this module's own `dy/dx_i` (backward()'s closed form) gives
     * the local (gradient x input) relevance term:
     * `z_i = x_i * dy/dx_i = x_i * (1/n) * m^(1/p-1) * x_i^(p-1) = (1/n) * x_i^p * m^(1/p-1)`.
     * Using `m^(1/p-1) = m^(1/p)/m = y/m` (valid for `m != 0`) collapses this to
     * `z_i = (1/n) * x_i^p * y/m = (x_i^p / (n*m)) * y`, and since `n*m == sum_j(x_j^p)`
     * exactly (`m`'s own definition), `sum_i(z_i) = (sum_i(x_i^p) / sum_j(x_j^p)) * y == y`
     * -- Euler's theorem confirmed directly from the closed form, not just cited abstractly.
     * The final rule redistributes `R_out` by each grounding's share of `sum_j(x_j^p)`
     * (an epsilon-stabilized denominator, this codebase's standard stabilization
     * convention):
     * `R_i = (x_i^p / (sum_j(x_j^p) + eps*sign(sum_j(x_j^p)))) * R_out`.
     * This is a genuinely new derivation (no literature applies an Euler-homogeneity
     * gradient-times-input decomposition to a p-mean aggregator; research doc §3's
     * literature search found no LRP-style rule for any quantifier aggregator at all) --
     * the campaign's second, after Mission 0's t-norm/t-conorm rules, novel contribution.
     * @param relevance_out Relevance at this module's output. Must match forward()'s shape.
     * @param config Selects the epsilon stabilizer (same role as every other
     *        epsilon-stabilized rule in this codebase).
     * @return Relevance at this module's input, same shape as the cached forward() input.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached
     *         output shape.
     * @note Conserves *near*-exactly (`sum_i(R_i) ~= R_out`), same class as every other
     *       epsilon-stabilized rule in this codebase (LinearModule, ResidualModule,
     *       DisjunctionModule's Product rule) -- not exact, since the epsilon stabilizer
     *       introduces a bounded, provably small gap. Exact when `config.epsilon == 0` and
     *       `sum_j(x_j^p) != 0` (verified by a dedicated test using `p == 1`, where the
     *       derivation collapses to the same exact-conservation shape as an ordinary mean).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief This aggregator's configured exponent. */
    [[nodiscard]] float p() const { return p_; }

protected:
    /**
     * @brief Computes `agg_p` over the leading (batch) axis, per remaining "column" (every
     *        fixed combination of the trailing dimensions).
     * @param input Input tensor. Must be rank >= 1 (the leading batch axis) -- external
     *        boundary: forward_impl is reachable directly through the inherited,
     *        non-virtual Module::forward(const Tensor&) by any caller, and a rank-0 tensor
     *        (a bare scalar, no batch axis at all) has no leading dimension to reduce.
     * @return `agg_p(input)`, shape equal to input's shape with the leading dim dropped.
     * @throws std::invalid_argument if input's rank is 0.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    float p_;
    Tensor last_input_;   ///< Cached forward input, shape (N, ...rest).
    Tensor last_mean_;    ///< Cached per-column m = mean(x^p), shape (...rest).
    Tensor last_output_;  ///< Cached per-column y = m^(1/p), shape (...rest).
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
