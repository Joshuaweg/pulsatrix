/** @file swiglu_module.hpp
 *  @brief SwiGLU gated feedforward block -- second module composed from real `LinearModule`
 *         sub-objects, plus a diagonal specialization of AttnLRP's Eq. 15 bilinear rule.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `down_proj(silu(gate_proj(x)) * up_proj(x))`, the gated feedforward block used in
 *        place of a plain two-linear-layer MLP in most modern transformers. Rank-agnostic
 *        over `(..., d_model) -> (..., d_model)`, matching `MultiHeadAttentionModule`'s I/O
 *        contract so Phase 3 Mission 5 (`TransformerBlock`) can chain them directly.
 *
 * **Composition over inheritance**, same principle `MultiHeadAttentionModule` established:
 * `gate_proj_`/`up_proj_`/`down_proj_` are real `LinearModule` sub-objects driven through
 * their own `forward()`/`backward()`/`propagate_relevance()`. The only hand-written
 * operation is the elementwise `silu(gate) * up` gate -- its LRP rule is a diagonal
 * specialization of AttnLRP's Eq. 15 (Achtibat et al. 2024): for `c[i] = a[i]*b[i]`, the
 * `sum_k` over a matmul's shared contraction index collapses to a single term since there is
 * no contraction, giving `R_a[i] += (a[i]*b[i]/(2*c[i]+eps*sign(c[i]))) * R_c[i]`,
 * symmetrically for `R_b[i]` -- the same rule `MultiHeadAttentionModule` uses for its two
 * batched matmuls, specialized to a diagonal case instead of a real contraction.
 *
 * @note **SiLU itself gets this codebase's existing pointwise-nonlinearity identity
 *       pass-through treatment** (Montavon et al. 2019 -- the same convention already
 *       applied to `ReluModule`/`DropoutModule`/RMSNorm's-and-LayerNorm's Eq. 19 identity
 *       rule), extended here to a smooth, non-piecewise-linear nonlinearity for the first
 *       time in this codebase. A known approximation (not an exact DTD derivation for SiLU
 *       specifically), consistent with this codebase's own established simplification for
 *       every other pointwise nonlinearity so far -- see mission_swiglu.md's Recon.
 * @note **Conserves near-exactly** (unlike `SoftmaxModule`'s/`MultiHeadAttentionModule`'s
 *       large by-design gap): the diagonal split above satisfies
 *       `R_a[i]+R_b[i] = 2*(c[i]/(2c[i]+eps))*R_c[i] ~= R_c[i]` (exact at `eps=0`), and
 *       SiLU's pass-through is exact by construction -- so the whole composed module should
 *       conserve up to an epsilon residual only, same category as `RoPEModule`. Measured and
 *       confirmed in tests/swiglu_module_test.cpp, not just asserted.
 * @note `op_type()` reuses `OpType::Elementwise` -- same resolution `RoPEModule` made
 *       (the hand-written novel part is an elementwise gate multiply, not a new operation
 *       category). Contrast `MultiHeadAttentionModule`, which earned a new category for
 *       genuinely novel cross-position mixing this module does not do -- every position
 *       here is processed independently.
 */
class SwiGLUModule : public Module {
public:
    /**
     * @brief Constructs a SwiGLU block with zero-initialized projections.
     * @param d_model Input/output width.
     * @param d_ff Hidden (gate/up) width.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if d_model <= 0 or d_ff <= 0 -- external boundary
     *         (construction arguments can originate from Phase 5's Python bindings with no
     *         upstream validation), same convention as every other constructor.
     */
    SwiGLUModule(int64_t d_model, int64_t d_ff, DeviceBackend* backend);

    /**
     * @brief Gradient w.r.t. this module's input; sub-module parameter gradients accumulate
     *        inside gate_proj_/up_proj_/down_proj_ (reachable through parameters()).
     * @param grad_output Gradient w.r.t. this module's output, matching the cached forward
     *        shape.
     * @return Gradient w.r.t. this module's input, same shape as grad_output.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached forward shape.
     * @note Raw host loops for the gate-derivative combine and the two elementwise
     *       multiplies/add; PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu) guarded --
     *       see mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per this module's own op_type() note above. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief LRP relevance propagation: down_proj_'s epsilon rule, then the diagonal Eq. 15
     *        split into gate/up shares, then SiLU's identity pass-through, then
     *        gate_proj_'s/up_proj_'s epsilon rules summed.
     * @param relevance_out Relevance at this module's output, matching the cached forward shape.
     * @param config Supplies the epsilon stabilizer for both the diagonal split and the
     *        three LinearModule epsilon rules.
     * @return Relevance at this module's input, same shape as relevance_out.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached forward shape.
     * @note Raw host loops; PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu) guarded.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief gate_proj_'s, up_proj_'s, and down_proj_'s parameters, flattened. */
    [[nodiscard]] std::vector<ParamRef> parameters() override;

    [[nodiscard]] int64_t d_model() const { return d_model_; }
    [[nodiscard]] int64_t d_ff() const { return d_ff_; }

    /** @name Sub-module access -- weight initialization from tests/loaders, and inspection. */
    ///@{
    [[nodiscard]] LinearModule& gate_proj() { return gate_proj_; }
    [[nodiscard]] LinearModule& up_proj() { return up_proj_; }
    [[nodiscard]] LinearModule& down_proj() { return down_proj_; }
    ///@}

protected:
    /**
     * @brief Runs: project (gate, up) -> silu(gate) -> gate*up -> project (down).
     * @param input `(..., d_model)`, rank >= 2, Cpu-resident.
     * @return `(..., d_model)`, same shape as input.
     * @throws std::invalid_argument if input's rank < 2 or final dimension != d_model.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    int64_t d_ff_;
    DeviceBackend* backend_;

    LinearModule gate_proj_;
    LinearModule up_proj_;
    LinearModule down_proj_;

    // Forward caches -- the operands the gate multiply's forward/backward/LRP rule needs.
    // gate_proj_/up_proj_/down_proj_ also cache their own inputs internally; this module
    // additionally caches these three for its own composed math, same choice
    // MultiHeadAttentionModule made for its Q/K/V/scores caches.
    Shape last_input_shape_ = Shape({0});
    int64_t last_n_flat_ = 0;
    Tensor last_gate_pre_;   ///< (N_flat, d_ff) -- gate_proj_'s raw output, pre-SiLU.
    Tensor last_gate_post_;  ///< (N_flat, d_ff) -- silu(last_gate_pre_).
    Tensor last_up_;         ///< (N_flat, d_ff) -- up_proj_'s output.
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
