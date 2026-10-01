/** @file rwkv_module.hpp
 *  @brief RWKV-4 time-mixing (WKV linear-attention) recurrence, with an original derived
 *         LRP rule adapting MambaLRP's detach-the-gate technique to the WKV quotient (see
 *         the class-level note).
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Core RWKV-4 time-mixing block (Peng et al. 2023, arXiv:2305.13048),
 *        input (N, L, d_model) -> output (N, L, d_model).
 *
 *        Per timestep t (1-based below), batch row b, channel d in [0, d_model), with
 *        x_0 = 0 (the zero-initialized "previous token" for the token-shift at t = 1) and
 *        a_0 = b_0 = 0 (the zero-initialized running WKV numerator/denominator state):
 *          xr_t[b,d]   = mu_r[d]*x_t[b,d] + (1-mu_r[d])*x_{t-1}[b,d]   (and xk_t/xv_t alike)
 *          r_t[b,d]    = sigmoid( sum_e xr_t[b,e]*W_r[e,d] )           (Linear, NO bias)
 *          k_t[b,d]    = sum_e xk_t[b,e]*W_k[e,d]                      (Linear, NO bias)
 *          v_t[b,d]    = sum_e xv_t[b,e]*W_v[e,d]                      (Linear, NO bias)
 *          e_t[b,d]    = exp(u[d] + k_t[b,d])                          (current-token bonus)
 *          num_t[b,d]  = a_{t-1}[b,d] + e_t[b,d]*v_t[b,d]
 *          den_t[b,d]  = b_{t-1}[b,d] + e_t[b,d]
 *          wkv_t[b,d]  = num_t[b,d] / den_t[b,d]
 *          decay[d]    = exp(-w[d]);   kk_t[b,d] = exp(k_t[b,d])
 *          a_t[b,d]    = decay[d]*a_{t-1}[b,d] + kk_t[b,d]*v_t[b,d]
 *          b_t[b,d]    = decay[d]*b_{t-1}[b,d] + kk_t[b,d]
 *          o_t[b,d]    = sum_e ( r_t[b,e]*wkv_t[b,e] ) * W_o[e,d]      (Linear, NO bias)
 *
 * @note Scope cut, mirroring every prior recurrent module's own: this is the *time-mixing
 *       (WKV) block only*. A full RWKV layer additionally has a separate channel-mixing
 *       feedforward block, not built here -- the same "core mechanism only, not the full
 *       published block" convention RNNModule/LSTMModule/GRUModule/MambaModule each follow.
 * @note Numerical scope cut: the recurrence above is RWKV's *direct, unstabilized* running-
 *       sum form. RWKV's reference implementation additionally carries a running maximum
 *       M_t purely to keep exp() from overflowing float on very long sequences -- a pure
 *       numerical-conditioning device that does not change the mathematical function being
 *       computed. It is deliberately omitted here (safe and exact at this project's test-
 *       scale sequence lengths); it would have to be added back before production-scale
 *       long-sequence use. Same class of documented simplification as RNNModule's zero h_0
 *       and MambaModule's Euler-approximated Bbar.
 * @note a_0 = b_0 = 0 and x_0 = 0, zero-initialized and not learnable -- the same scope cut
 *       every prior recurrent module makes for its own initial state.
 * @note exp and sigmoid are computed in raw host loops here, PULSATRIX_ASSERT(... .device() ==
 *       DeviceType::Cpu)-guarded on every entry point. DeviceBackend::elementwise has no
 *       Exp/Sigmoid op; this follows MambaModule's first-occurrence-per-need disposition
 *       (which did the same for softplus/exp) rather than speculatively adding backend
 *       primitives in a module mission.
 * @note **LRP rule -- original derivation (2026-09-27, operator-directed reassessment
 *       following campaign_exai_dl_library_phase6_modern_architectures's Decision Point 2
 *       and RetNet's own resolved derivation).** The mission's a-priori hypothesis was that
 *       the WKV num/den quotient would inherit SoftmaxModule's non-conserving DTD-
 *       approximation shape; fresh re-derivation (not a re-run of the same guess) found
 *       otherwise. Unrolling: `num_t = a_{t-1} + e_t*v_t` and `den_t = b_{t-1} + e_t`, with
 *       `a_t = decay*a_{t-1} + kk_t*v_t` (`kk_t = exp(k_t)`, no bonus) -- so
 *       `wkv_t = num_t/den_t` is, at every step, a **two-term weighted sum of `a_{t-1}` and
 *       `v_t`** with weights `1/den_t` and `e_t/den_t` (which sum to exactly 1 by
 *       construction), and `a_t` is itself a two-term weighted sum of `a_{t-1}` and `v_t`
 *       with weights `decay` and `kk_t`. This is structurally MambaModule's own
 *       `h_t = Abar_t*h_{t-1} + Bbar_t*x_t` shape, not softmax's cross-normalizing shape --
 *       so the *same* MambaLRP technique applies: **detach `e_t`, `kk_t`, `decay` (and the
 *       receptance gate `r_t`) as constants** (they are already data-dependent gates in
 *       Mamba's Abar/Bbar/C, detached there for the same reason), and apply the standard
 *       weighted-sum epsilon/z-rule to the two surviving weighted-sum nodes (`wkv_t`'s
 *       quotient and `a_t`'s carry), threading a state-relevance carry backward across `t`
 *       exactly the way MambaModule's own `r_h_carry` does. Consequently `w_r_`, `mu_r_`,
 *       `w_k_`, `mu_k_`, `u_` never appear in propagate_relevance() at all -- receptance
 *       and the key path are consumed only through their cached, *detached* forward values
 *       (`last_r_`, `last_e_`, `last_kk_`), mirroring MambaModule's own identical treatment
 *       of `w_delta_`/`bias_delta_`/`w_b_`/`w_c_`. `w_` (the decay rate) is the one
 *       exception: `decay = exp(-w_)` is recomputed from the live parameter rather than
 *       cached, so it is technically referenced -- but only to reconstruct a detached
 *       forward *value* used as a fixed weight, the same role every other detached gate
 *       plays, not to compute `w_`'s own relevance share (there is no such concept for any
 *       weight tensor in this codebase's LRP rules). None of these six parameters ever
 *       receives a bilinear-split or weighted-sum relevance share of its own. Verified on a
 *       hand-worked single-channel,
 *       L=3 example before implementation (see the mission's Completion Summary): the
 *       composed rule conserves exactly there (mod the usual epsilon stabilizers), so
 *       RWKVModule -- like RetNetModule, and unlike SoftmaxModule/MultiHeadAttentionModule
 *       -- belongs in lrp_conservation_test.cpp's AllModuleTypeCases(). See the campaign's
 *       Decision Point 2 addendum for the full outcome, distinguishing this resolved case
 *       from RetNet's independently-resolved (and approximation-free) one.
 */
class RWKVModule : public Module {
public:
    /**
     * @brief Constructs an RWKV time-mixing layer with zero-initialized parameters.
     * @param d_model Model/channel dimension (also the input and output feature dimension).
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if d_model <= 0 -- external boundary (construction
     *         arguments can originate from the Python bindings with no upstream validation).
     */
    RWKVModule(int64_t d_model, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time across the WKV recurrence: accumulates all
     *        nine parameter gradients across every timestep into the same buffers via
     *        Tensor::accumulate(). Differentiates through the sigmoid receptance gate,
     *        through both exp() nonlinearities (e_t and kk_t), through the num/den quotient
     *        and through the decayed a/b state carry, plus the three token-shift mixes.
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, d_model)
     *        matching the most recent forward() call's output shape.
     * @return Gradient w.r.t. this module's input, shape (N, L, d_model).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached forward
     *         output shape.
     * @note Not yet backend-generic -- raw host loops for the recurrence and the exp/sigmoid
     *       math. PULSATRIX_REQUIRE_HOST(grad_output) guards against silent
     *       UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Recurrent per charter's closed OpType set -- a compound accumulate-over-time
     *         operation, shared with RNNModule/LSTMModule/GRUModule/MambaModule (no enum
     *         change). */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the receptance projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_r(std::initializer_list<float> values);
    /** @brief Overwrites the key projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_k(std::initializer_list<float> values);
    /** @brief Overwrites the value projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_v(std::initializer_list<float> values);
    /** @brief Overwrites the output projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_o(std::initializer_list<float> values);
    /** @brief Overwrites the per-channel decay rate w (d_model,), decay = exp(-w) -- test/initialization use only. */
    void set_w(std::initializer_list<float> values);
    /** @brief Overwrites the per-channel current-token bonus u (d_model,) -- test/initialization use only. */
    void set_u(std::initializer_list<float> values);
    /** @brief Overwrites the receptance token-shift mix ratio (d_model,) -- test/initialization use only. */
    void set_mu_r(std::initializer_list<float> values);
    /** @brief Overwrites the key token-shift mix ratio (d_model,) -- test/initialization use only. */
    void set_mu_k(std::initializer_list<float> values);
    /** @brief Overwrites the value token-shift mix ratio (d_model,) -- test/initialization use only. */
    void set_mu_v(std::initializer_list<float> values);

    /** @brief std::vector overload of set_W_r() -- for callers building values programmatically. */
    void set_W_r(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_k(). */
    void set_W_k(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_v(). */
    void set_W_v(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_o(). */
    void set_W_o(const std::vector<float>& values);
    /** @brief std::vector overload of set_w(). */
    void set_w(const std::vector<float>& values);
    /** @brief std::vector overload of set_u(). */
    void set_u(const std::vector<float>& values);
    /** @brief std::vector overload of set_mu_r(). */
    void set_mu_r(const std::vector<float>& values);
    /** @brief std::vector overload of set_mu_k(). */
    void set_mu_k(const std::vector<float>& values);
    /** @brief std::vector overload of set_mu_v(). */
    void set_mu_v(const std::vector<float>& values);

    /** @brief The receptance projection weight (d_model, d_model). */
    [[nodiscard]] const Tensor& W_r() const { return w_r_; }
    /** @brief The key projection weight (d_model, d_model). */
    [[nodiscard]] const Tensor& W_k() const { return w_k_; }
    /** @brief The value projection weight (d_model, d_model). */
    [[nodiscard]] const Tensor& W_v() const { return w_v_; }
    /** @brief The output projection weight (d_model, d_model). */
    [[nodiscard]] const Tensor& W_o() const { return w_o_; }
    /** @brief The per-channel decay rate w (d_model,); the applied decay is exp(-w). */
    [[nodiscard]] const Tensor& w() const { return w_; }
    /** @brief The per-channel current-token bonus u (d_model,). */
    [[nodiscard]] const Tensor& u() const { return u_; }
    /** @brief The receptance token-shift mix ratio (d_model,). */
    [[nodiscard]] const Tensor& mu_r() const { return mu_r_; }
    /** @brief The key token-shift mix ratio (d_model,). */
    [[nodiscard]] const Tensor& mu_k() const { return mu_k_; }
    /** @brief The value token-shift mix ratio (d_model,). */
    [[nodiscard]] const Tensor& mu_v() const { return mu_v_; }

    /** @brief Accumulated gradient w.r.t. W_r. */
    [[nodiscard]] const Tensor& W_r_grad() const { return w_r_grad_; }
    /** @brief Accumulated gradient w.r.t. W_k. */
    [[nodiscard]] const Tensor& W_k_grad() const { return w_k_grad_; }
    /** @brief Accumulated gradient w.r.t. W_v. */
    [[nodiscard]] const Tensor& W_v_grad() const { return w_v_grad_; }
    /** @brief Accumulated gradient w.r.t. W_o. */
    [[nodiscard]] const Tensor& W_o_grad() const { return w_o_grad_; }
    /** @brief Accumulated gradient w.r.t. w. */
    [[nodiscard]] const Tensor& w_grad() const { return w_grad_; }
    /** @brief Accumulated gradient w.r.t. u. */
    [[nodiscard]] const Tensor& u_grad() const { return u_grad_; }
    /** @brief Accumulated gradient w.r.t. mu_r. */
    [[nodiscard]] const Tensor& mu_r_grad() const { return mu_r_grad_; }
    /** @brief Accumulated gradient w.r.t. mu_k. */
    [[nodiscard]] const Tensor& mu_k_grad() const { return mu_k_grad_; }
    /** @brief Accumulated gradient w.r.t. mu_v. */
    [[nodiscard]] const Tensor& mu_v_grad() const { return mu_v_grad_; }

    /**
     * @brief The original derived LRP rule (see the class-level note): MambaLRP's
     *        detach-the-gate technique adapted to the WKV quotient. Detaches the receptance
     *        gate `r_t` and the num/den weights (`e_t`, `kk_t`, `decay`) as constants, then
     *        applies the standard weighted-sum epsilon/z-rule to `wkv_t = num_t/den_t` and
     *        to the state carry `a_t = decay*a_{t-1} + kk_t*v_t`, threading a state-
     *        relevance carry backward across `t` (mirroring MambaModule's `r_h_carry`),
     *        then the output/value projections' own no-bias z-rule and the value
     *        token-shift's weighted-sum split.
     * @param relevance_out Relevance at this module's output. Must be (N, L, d_model)
     *        matching the most recent forward() call's output shape.
     * @param config Supplies the epsilon stabilizer used throughout.
     * @return Relevance at this module's input, shape (N, L, d_model).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     * @note `w_r_`, `mu_r_`, `w_k_`, `mu_k_`, `u_` never appear below at all; `w_` appears
     *       only to reconstruct the detached `decay` value, not to compute its own
     *       relevance share -- see the class-level note. Only `w_v_`, `w_o_`, `mu_v_`
     *       (and `w_` for `decay`) participate.
     * @note Not yet backend-generic -- raw host loops, mirroring forward_impl()/backward().
     *       PULSATRIX_REQUIRE_HOST(relevance_out) guards against silent
     *       UB on a CUDA-backed Tensor.
     * @note Conserves near-exactly (measured in rwkv_module_test.cpp), gated only by the
     *       usual epsilon stabilizers -- NOT a known non-conserving approximation like
     *       SoftmaxModule's Eq. 13.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&w_r_, &w_r_grad_},   {&w_k_, &w_k_grad_},   {&w_v_, &w_v_grad_},
                {&w_o_, &w_o_grad_},   {&w_, &w_grad_},       {&u_, &u_grad_},
                {&mu_r_, &mu_r_grad_}, {&mu_k_, &mu_k_grad_}, {&mu_v_, &mu_v_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation -- per-timestep tied-weight WKV recurrence.
     * @throws std::invalid_argument if input isn't rank-3 (N, L, d_model), or its last
     *         dimension doesn't match d_model.
     * @note Not yet backend-generic -- raw host loops (exp/sigmoid have no backend
     *       primitive). PULSATRIX_REQUIRE_HOST(input) guards against
     *       silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    DeviceBackend* backend_;
    Tensor w_r_;   // (d_model, d_model)
    Tensor w_k_;   // (d_model, d_model)
    Tensor w_v_;   // (d_model, d_model)
    Tensor w_o_;   // (d_model, d_model)
    Tensor w_;     // (d_model,); decay = exp(-w)
    Tensor u_;     // (d_model,)
    Tensor mu_r_;  // (d_model,)
    Tensor mu_k_;  // (d_model,)
    Tensor mu_v_;  // (d_model,)
    Tensor w_r_grad_;
    Tensor w_k_grad_;
    Tensor w_v_grad_;
    Tensor w_o_grad_;
    Tensor w_grad_;
    Tensor u_grad_;
    Tensor mu_r_grad_;
    Tensor mu_k_grad_;
    Tensor mu_v_grad_;
    Tensor last_input_;   // (N, L, d_model)
    Tensor last_xr_;      // (N, L, d_model); token-shifted receptance input
    Tensor last_xk_;      // (N, L, d_model)
    Tensor last_xv_;      // (N, L, d_model)
    Tensor last_r_;       // (N, L, d_model); sigmoid POST-activation (its own derivative source)
    Tensor last_k_;       // (N, L, d_model)
    Tensor last_v_;       // (N, L, d_model)
    Tensor last_e_;       // (N, L, d_model); exp(u + k_t)
    Tensor last_kk_;      // (N, L, d_model); exp(k_t)
    Tensor last_num_;     // (N, L, d_model)
    Tensor last_den_;     // (N, L, d_model)
    Tensor last_wkv_;     // (N, L, d_model)
    Tensor last_a_;       // (N, L+1, d_model); index 0 along dim 1 = a_0 = 0
    Tensor last_b_;       // (N, L+1, d_model); index 0 along dim 1 = b_0 = 0
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
