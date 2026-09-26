/** @file mamba_module.hpp
 *  @brief Mamba/S6 selective-state-space recurrence with the MambaLRP relevance rule.
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Core Mamba/S6 selective-scan recurrence (Gu & Dao 2023, arXiv:2312.00752),
 *        input (N, L, d_model) -> output (N, L, d_model).
 *
 *        Per timestep t, with batch row b, channel d in [0, d_model) and state index n in
 *        [0, state_size):
 *          z_delta_t[b,d] = sum_e x_t[b,e]*W_delta[e,d] + bias_delta[d]   (Linear, WITH bias)
 *          Delta_t[b,d]   = softplus(z_delta_t[b,d])                      (> 0, stable step)
 *          B_t[b,n]       = sum_e x_t[b,e]*W_B[e,n]                       (Linear, NO bias)
 *          C_t[b,n]       = sum_e x_t[b,e]*W_C[e,n]                       (Linear, NO bias)
 *          Abar_t[b,d,n]  = exp(Delta_t[b,d] * A[d,n])
 *          Bbar_t[b,d,n]  = Delta_t[b,d] * B_t[b,n]
 *          h_t[b,d,n]     = Abar_t[b,d,n]*h_{t-1}[b,d,n] + Bbar_t[b,d,n]*x_t[b,d]
 *          y_t[b,d]       = sum_n( C_t[b,n]*h_t[b,d,n] ) + D[d]*x_t[b,d]
 *
 * @note Scope cut, mirroring every prior recurrent module's own: this is the *core
 *       selective-scan recurrence only*, not the full Mamba block. The full block
 *       additionally wraps this in an input Conv1D, an outer SiLU-gated multiplicative
 *       branch and a final output projection -- none of which are built here, for the same
 *       reason RNNModule/LSTMModule/GRUModule each implement only their own core
 *       recurrence. MultiHeadAttentionModule/SwiGLUModule/TransformerBlock already
 *       established that small, independently-correct pieces get composed later.
 * @note Discretization scope cut: `Bbar` uses the first-order/Euler approximation Mamba's
 *       own reference implementation uses, not the full ZOH integral
 *       (A^{-1}(exp(Delta*A)-I)*Delta*B). `Abar` is the exact ZOH form exp(Delta*A).
 * @note h_0 = 0, zero-initialized and not learnable -- same scope cut as every prior
 *       recurrent module, and the thing that makes this module's conservation exact rather
 *       than merely approximate (see the LRP note below).
 * @note softplus (and its derivative sigmoid) and exp are computed in raw host loops here,
 *       PULSATRIX_ASSERT(... .device() == DeviceType::Cpu)-guarded on every entry point.
 *       DeviceBackend::elementwise has no Softplus/Exp op; this is the *first* consumer
 *       needing softplus, so it gets the first-occurrence workaround RNNModule's Mission 0
 *       used for tanh, not a speculative new backend primitive. Adding one is the
 *       repeat-trigger's job, not this mission's.
 * @note LRP rule (MambaLRP -- Rezaei Jafari, Montavon, Müller, Eberle,
 *       "MambaLRP: Explaining Selective State Space Sequence Models", arXiv:2406.07592,
 *       NeurIPS 2024). MambaLRP's central finding: naive LRP breaks conservation on the
 *       selective scan because the discrete recurrence parameters (Abar_t, Bbar_t, C_t) are
 *       themselves functions of the input via their own learned projections, and
 *       redistributing "through" that input-dependence corrupts the redistribution. The fix
 *       is to treat Abar_t/Bbar_t/C_t (and A, D) as *detached* fixed multiplicative
 *       constants during relevance propagation -- exactly the way every existing module here
 *       already treats its weight matrices. That collapses the recurrence into the familiar
 *       weighted-sum shape, so no new rule *shape* is needed; the novelty is specifically
 *       *what gets detached*. Per timestep, in reverse time order:
 *       1. y_t = sum_n(C_t*h_t) + D*x_t is an (state_size + 1)-way weighted sum sharing one
 *          output; the epsilon/z-rule (denominator y_t itself, epsilon-stabilized) splits
 *          R(y_t) across the state_size state terms and the D skip term in proportion to
 *          their values. The skip share lands directly on R(x_t).
 *       2. h_t = Abar_t*h_{t-1} + Bbar_t*x_t is the two-weighted-source epsilon/z-rule,
 *          identical in shape to RNNModule's own, with denominator h_t itself; the
 *          h_{t-1} share accumulates into the carried t-1 relevance accumulator and the
 *          x_t share adds onto R(x_t).
 *       3. W_delta/bias_delta/W_B/W_C -- the selective projections' own upstream
 *          computation -- are **never** touched by propagate_relevance. Delta_t/B_t/C_t are
 *          pure conductors, the same convention LSTMModule's/GRUModule's gates use.
 *       4. Because h_0 = 0 (this module's own scope cut), the relevance that would
 *          otherwise leak into the nonexistent state before t=0 is provably exactly zero
 *          (the t=0 numerator is Abar_0*h_{-1} with h_{-1} == 0). End-to-end conservation
 *          therefore holds up to the epsilon stabilizers only -- measured at 2.5e-5 against
 *          a sum(R_out) of 5.3 (4.7e-6 relative) in mamba_module_test.cpp's
 *          PropagateRelevanceConservationGapIsMeasuredNotAssumed, i.e. the near-exact
 *          category (like RoPEModule/SwiGLUModule), NOT SoftmaxModule's large by-design DTD
 *          gap. This is the expected outcome: restoring conservation that naive LRP breaks
 *          is MambaLRP's whole point, so a large gap here would be a bug, not a property.
 */
class MambaModule : public Module {
public:
    /**
     * @brief Constructs a selective-scan layer with zero-initialized parameters.
     * @param d_model Model/channel dimension (also the input and output feature dimension).
     * @param state_size Latent SSM state dimension N.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if d_model <= 0 or state_size <= 0 -- external boundary
     *         (construction arguments can originate from the Python bindings with no
     *         upstream validation).
     */
    MambaModule(int64_t d_model, int64_t state_size, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time across the selective scan: accumulates
     *        W_delta/bias_delta/W_B/W_C/A/D gradients across every timestep into the same
     *        buffers via Tensor::accumulate(). Unlike propagate_relevance, this is the
     *        genuine *undetached* gradient -- it differentiates through Delta_t's softplus,
     *        through Abar_t = exp(Delta_t*A) and through all three selective projections.
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, d_model)
     *        matching the most recent forward() call's output shape.
     * @return Gradient w.r.t. this module's input, shape (N, L, d_model).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached forward
     *         output shape.
     * @note Not yet backend-generic -- raw host loops for the recurrence and the exp/sigmoid
     *       math. PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu) guards against
     *       silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Recurrent per charter's closed OpType set -- a compound accumulate-over-time
     *         operation, shared with RNNModule/LSTMModule/GRUModule (no enum change). */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the input-to-Delta projection weight (d_model, d_model) -- test/initialization use only. */
    void set_W_delta(std::initializer_list<float> values);
    /** @brief Overwrites the Delta projection bias (d_model,) -- test/initialization use only. */
    void set_bias_delta(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-B projection weight (d_model, state_size) -- test/initialization use only. */
    void set_W_B(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-C projection weight (d_model, state_size) -- test/initialization use only. */
    void set_W_C(std::initializer_list<float> values);
    /** @brief Overwrites the continuous state matrix A (d_model, state_size) -- test/initialization use only. */
    void set_A(std::initializer_list<float> values);
    /** @brief Overwrites the skip/feedthrough vector D (d_model,) -- test/initialization use only. */
    void set_D(std::initializer_list<float> values);

    /** @brief std::vector overload of set_W_delta() -- for callers building values programmatically. */
    void set_W_delta(const std::vector<float>& values);
    /** @brief std::vector overload of set_bias_delta(). */
    void set_bias_delta(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_B(). */
    void set_W_B(const std::vector<float>& values);
    /** @brief std::vector overload of set_W_C(). */
    void set_W_C(const std::vector<float>& values);
    /** @brief std::vector overload of set_A(). */
    void set_A(const std::vector<float>& values);
    /** @brief std::vector overload of set_D(). */
    void set_D(const std::vector<float>& values);

    [[nodiscard]] const Tensor& W_delta() const { return w_delta_; }
    [[nodiscard]] const Tensor& bias_delta() const { return bias_delta_; }
    [[nodiscard]] const Tensor& W_B() const { return w_b_; }
    [[nodiscard]] const Tensor& W_C() const { return w_c_; }
    [[nodiscard]] const Tensor& A() const { return a_; }
    [[nodiscard]] const Tensor& D() const { return d_; }

    [[nodiscard]] const Tensor& W_delta_grad() const { return w_delta_grad_; }
    [[nodiscard]] const Tensor& bias_delta_grad() const { return bias_delta_grad_; }
    [[nodiscard]] const Tensor& W_B_grad() const { return w_b_grad_; }
    [[nodiscard]] const Tensor& W_C_grad() const { return w_c_grad_; }
    [[nodiscard]] const Tensor& A_grad() const { return a_grad_; }
    [[nodiscard]] const Tensor& D_grad() const { return d_grad_; }

    /**
     * @brief MambaLRP relevance propagation -- Abar_t/Bbar_t/C_t/A/D detached and treated as
     *        fixed multiplicative constants, epsilon/z-rule over the resulting weighted
     *        sums, processed in reverse time order through a carried state-relevance
     *        accumulator. See the class-level note for the full per-timestep redistribution.
     * @param relevance_out Relevance at this module's output. Must be (N, L, d_model)
     *        matching the most recent forward() call's output shape.
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (N, L, d_model). Conserves up to the
     *         epsilon stabilizers -- see the class-level note for the measured gap.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     * @note Not yet backend-generic -- raw host loop. PULSATRIX_ASSERT(relevance_out.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&w_delta_, &w_delta_grad_}, {&bias_delta_, &bias_delta_grad_}, {&w_b_, &w_b_grad_},
                {&w_c_, &w_c_grad_},         {&a_, &a_grad_},                   {&d_, &d_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation -- per-timestep tied-weight selective scan.
     * @throws std::invalid_argument if input isn't rank-3 (N, L, d_model), or its last
     *         dimension doesn't match d_model.
     * @note Not yet backend-generic -- raw host loops (softplus/exp have no backend
     *       primitive). PULSATRIX_ASSERT(input.device() == DeviceType::Cpu) guards against
     *       silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t d_model_;
    int64_t state_size_;
    DeviceBackend* backend_;
    Tensor w_delta_;      // (d_model, d_model)
    Tensor bias_delta_;   // (d_model,)
    Tensor w_b_;          // (d_model, state_size)
    Tensor w_c_;          // (d_model, state_size)
    Tensor a_;            // (d_model, state_size)
    Tensor d_;            // (d_model,)
    Tensor w_delta_grad_;
    Tensor bias_delta_grad_;
    Tensor w_b_grad_;
    Tensor w_c_grad_;
    Tensor a_grad_;
    Tensor d_grad_;
    Tensor last_input_;    // (N, L, d_model)
    Tensor last_states_;   // (N, L+1, d_model, state_size); index 1 along dim 1 = h_0 = 0
    Tensor last_abar_;     // (N, L, d_model, state_size)
    Tensor last_bbar_;     // (N, L, d_model, state_size)
    Tensor last_z_delta_;  // (N, L, d_model); softplus PRE-activation, for its derivative
    Tensor last_delta_;    // (N, L, d_model)
    Tensor last_b_;        // (N, L, state_size)
    Tensor last_c_;        // (N, L, state_size)
    Tensor last_output_;   // (N, L, d_model); y_t, the LRP epsilon-rule denominator at step 1
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
