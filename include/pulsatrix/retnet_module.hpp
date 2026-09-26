/** @file retnet_module.hpp
 *  @brief RetNet retention mechanism (recurrent mode). LRP rule deliberately deferred --
 *         propagate_relevance throws, by design (see the class-level note).
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Core RetNet retention block (Sun et al. 2023, arXiv:2307.08621), recurrent mode,
 *        input (N, L, d_model) -> output (N, L, d_model).
 *
 *        Per timestep t (1-based below), batch row b, key index i in [0, key_dim), value
 *        index j in [0, d_model), with S_0[b,i,j] = 0 (the zero-initialized retention
 *        state) and a fixed scalar decay gamma:
 *          Q_t[b,i]   = sum_e x_t[b,e]*W_Q[e,i]                 (Linear, NO bias)
 *          K_t[b,i]   = sum_e x_t[b,e]*W_K[e,i]                 (Linear, NO bias)
 *          V_t[b,j]   = sum_e x_t[b,e]*W_V[e,j]                 (Linear, NO bias)
 *          S_t[b,i,j] = gamma*S_{t-1}[b,i,j] + K_t[b,i]*V_t[b,j]
 *          o_t[b,j]   = sum_i( Q_t[b,i]*S_t[b,i,j] )
 *
 * @note Scope cut -- **no rotary position encoding.** The paper applies an xpos/RoPE-style
 *       rotation Theta_t to Q_t/K_t. RoPE already exists here as its own composable module
 *       (RoPEModule, Phase 3 Mission 2), exactly the way MultiHeadAttentionModule optionally
 *       composes a separate RoPEModule instance rather than baking rotation into the
 *       attention math. This module does not wire one in either -- that composition is a
 *       future mission's job, once this module's own core recurrence is proven (mirroring
 *       how MultiHeadAttentionModule itself came after RoPEModule existed standalone).
 *       Q_t/K_t here are plain linear projections, unrotated.
 * @note Scope cut -- **single head only.** The paper's own per-head formula is complete and
 *       citable standalone; multi-head is an orthogonal expressivity choice (like
 *       MultiHeadAttentionModule's head-splitting), not a structural requirement. Consistent
 *       with RNNModule/LSTMModule/GRUModule/MambaModule/RWKVModule all being single-layer,
 *       single-mechanism cores.
 * @note Scope cut -- **no separate output projection.** V's own dimension is d_model, so
 *       o_t is already this module's output shape; the same "no unnecessary extra
 *       projection" simplification MambaModule made by not needing a W_O. (RWKVModule does
 *       need one, because its r_t*wkv_t intermediate isn't already the output shape.)
 * @note gamma is a *fixed constructor hyperparameter*, a plain float, NOT a learned Tensor
 *       -- that is what the paper specifies (unlike Mamba's A, which is learned). No
 *       gradient is accumulated for it. 0 < gamma < 1 is expected but deliberately NOT
 *       enforced: the same "no stability constraint enforced" disposition MambaModule takes
 *       for A.
 * @note S_0 = 0, zero-initialized and not learnable -- the same scope cut every prior
 *       recurrent module makes for its own initial state.
 * @note There is no nonlinearity anywhere in this recurrence (it is purely linear in x
 *       through the projections and bilinear in Q/K/V through the state), so unlike
 *       MambaModule/RWKVModule no raw-loop-nonlinearity workaround is needed. The
 *       recurrence itself is still a raw host loop over Tensor::data(), so every entry point
 *       carries the PULSATRIX_ASSERT(... .device() == DeviceType::Cpu) guard the rest of the
 *       not-yet-backend-generic modules use.
 * @note **No LRP rule -- deliberate, logged charter deviation.** Per the Phase 5 amendment
 *       of campaign_exai_dl_library_phase6_modern_architectures (2026-09-23, operator-
 *       directed), this module ships its real forward()/backward() (full BPTT, finite-
 *       difference verified, same rigor as every prior module) but *no* relevance rule:
 *       propagate_relevance is a real override satisfying Module's pure-virtual contract
 *       whose body unconditionally throws std::logic_error. A loud, explicit throw is the
 *       honest signal; returning zeros, an approximation or a gradient-based substitute
 *       would be a silent wrong answer, which is exactly the Captum/Zennit failure mode this
 *       project exists to avoid. Consequently RetNetModule is deliberately absent from
 *       lrp_conservation_test.cpp's AllModuleTypeCases() -- there is no relevance to
 *       conserve. See the campaign's Decision Point 2.
 */
class RetNetModule : public Module {
public:
    /**
     * @brief Constructs a retention layer with zero-initialized parameters.
     * @param d_model Model dimension (also the value dimension and the output dimension).
     * @param key_dim Query/key projection dimension.
     * @param gamma Fixed retention decay applied to the carried state each timestep. Not
     *        learned, and not constrained -- see the class-level note.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if d_model <= 0 or key_dim <= 0 -- external boundary
     *         (construction arguments can originate from the Python bindings with no
     *         upstream validation).
     */
    RetNetModule(int64_t d_model, int64_t key_dim, float gamma, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time across the retention recurrence: accumulates
     *        all three parameter gradients across every timestep into the same buffers via
     *        Tensor::accumulate(). Threads a (key_dim, d_model) state-gradient accumulator
     *        backwards through the gamma-decayed carry, then backprops the three no-bias
     *        linear projections onto the shared grad_input slot.
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, d_model)
     *        matching the most recent forward() call's output shape.
     * @return Gradient w.r.t. this module's input, shape (N, L, d_model).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached forward
     *         output shape.
     * @note Not yet backend-generic -- raw host loops for the recurrence. The projections
     *       themselves do go through DeviceBackend::gemm. PULSATRIX_ASSERT(grad_output.device()
     *       == DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Recurrent per charter's closed OpType set -- a compound accumulate-over-time
     *         operation, shared with RNNModule/LSTMModule/GRUModule/MambaModule/RWKVModule
     *         (no enum change). */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the query projection weight (d_model, key_dim) -- test/initialization use only. */
    void set_W_Q(std::initializer_list<float> values);
    /** @brief Overwrites the key projection weight (d_model, key_dim) -- test/initialization use only. */
    void set_W_K(std::initializer_list<float> values);
    /** @brief Overwrites the value projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_V(std::initializer_list<float> values);

    /** @brief std::vector overload of set_W_Q() -- for callers building values programmatically. */
    void set_W_Q(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_K(). */
    void set_W_K(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_V(). */
    void set_W_V(const std::vector<float>& values);

    /** @brief The query projection weight (d_model, key_dim). */
    [[nodiscard]] const Tensor& W_Q() const { return w_q_; }
    /** @brief The key projection weight (d_model, key_dim). */
    [[nodiscard]] const Tensor& W_K() const { return w_k_; }
    /** @brief The value projection weight (d_model, d_model). */
    [[nodiscard]] const Tensor& W_V() const { return w_v_; }

    /** @brief Accumulated gradient w.r.t. W_Q. */
    [[nodiscard]] const Tensor& W_Q_grad() const { return w_q_grad_; }
    /** @brief Accumulated gradient w.r.t. W_K. */
    [[nodiscard]] const Tensor& W_K_grad() const { return w_k_grad_; }
    /** @brief Accumulated gradient w.r.t. W_V. */
    [[nodiscard]] const Tensor& W_V_grad() const { return w_v_grad_; }

    /** @brief The fixed retention decay. A hyperparameter, not a parameter -- it has no
     *         gradient and is absent from parameters(). */
    [[nodiscard]] float gamma() const { return gamma_; }

    /**
     * @brief **Not implemented, by design.** Unconditionally throws -- this module ships
     *        without an LRP rule under the campaign's logged Phase 5 charter deviation (see
     *        the class-level note). The override exists so Module's pure-virtual contract is
     *        satisfied honestly rather than weakened; its body redistributes nothing.
     * @param relevance_out Relevance at this module's output. Only its device is inspected
     *        (by the debug-only device guard); no relevance is read or redistributed.
     * @param config Selects the LRP rule variant. Unused -- there is no rule to configure.
     * @return Never returns.
     * @throws std::logic_error always, with the message "RetNetModule::propagate_relevance:
     *         LRP rule not yet implemented -- see campaign Decision Point 2".
     * @note The PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu) device guard is kept
     *       *ahead* of the throw so this entry point stays consistent with forward()/
     *       backward() (and so it keeps firing for the death-test convention every module
     *       here follows). It becomes load-bearing the moment a real rule lands.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&w_q_, &w_q_grad_}, {&w_k_, &w_k_grad_}, {&w_v_, &w_v_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation -- per-timestep tied-weight retention recurrence.
     * @throws std::invalid_argument if input isn't rank-3 (N, L, d_model), or its last
     *         dimension doesn't match d_model.
     * @note Not yet backend-generic -- raw host loops for the state recurrence.
     *       PULSATRIX_ASSERT(input.device() == DeviceType::Cpu) guards against silent UB on a
     *       CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    int64_t key_dim_;
    float gamma_;
    DeviceBackend* backend_;
    Tensor w_q_;  // (d_model, key_dim)
    Tensor w_k_;  // (d_model, key_dim)
    Tensor w_v_;  // (d_model, d_model)
    Tensor w_q_grad_;
    Tensor w_k_grad_;
    Tensor w_v_grad_;
    Tensor last_input_;   // (N, L, d_model)
    Tensor last_q_;       // (N, L, key_dim)
    Tensor last_k_;       // (N, L, key_dim)
    Tensor last_v_;       // (N, L, d_model)
    Tensor last_states_;  // (N, L+1, key_dim, d_model); index 0 along dim 1 = S_0 = 0
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
