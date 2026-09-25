/** @file rwkv_module.hpp
 *  @brief RWKV-4 time-mixing (WKV linear-attention) recurrence. LRP rule deliberately
 *         deferred -- propagate_relevance throws, by design (see the class-level note).
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
 * @note **No LRP rule -- deliberate, logged charter deviation.** Per the Phase 5 amendment
 *       of campaign_exai_dl_library_phase6_modern_architectures (2026-09-23, operator-
 *       directed), this module ships its real forward()/backward() (full BPTT, finite-
 *       difference verified, same rigor as every prior module) but *no* relevance rule:
 *       propagate_relevance is a real override satisfying Module's pure-virtual contract
 *       whose body unconditionally throws std::logic_error. A loud, explicit throw is the
 *       honest signal; returning zeros, an approximation or a gradient-based substitute
 *       would be a silent wrong answer, which is exactly the Captum/Zennit failure mode
 *       this project exists to avoid. Consequently RWKVModule is deliberately absent from
 *       lrp_conservation_test.cpp's AllModuleTypeCases() -- there is no relevance to
 *       conserve. See the campaign's Decision Point 2.
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
     *       math. PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu) guards against silent
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
     * @brief **Not implemented, by design.** Unconditionally throws -- this module ships
     *        without an LRP rule under the campaign's logged Phase 5 charter deviation (see
     *        the class-level note). The override exists so Module's pure-virtual contract is
     *        satisfied honestly rather than weakened; its body redistributes nothing.
     * @param relevance_out Relevance at this module's output. Only its device is inspected
     *        (by the debug-only device guard); no relevance is read or redistributed.
     * @param config Selects the LRP rule variant. Unused -- there is no rule to configure.
     * @return Never returns.
     * @throws std::logic_error always, with the message "RWKVModule::propagate_relevance:
     *         LRP rule not yet implemented -- see campaign Decision Point 2".
     * @note The PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu) device guard is kept
     *       *ahead* of the throw so this entry point stays consistent with forward()/
     *       backward() (and so it keeps firing for the death-test convention every module
     *       here follows). It becomes load-bearing the moment a real rule lands.
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
     *       primitive). PULSATRIX_ASSERT(input.device() == DeviceType::Cpu) guards against
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
