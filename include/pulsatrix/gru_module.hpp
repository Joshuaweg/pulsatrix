/** @file gru_module.hpp
 *  @brief Single-layer GRU -- gated recurrence with a reset-gated candidate.
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Standard GRU recurrence (Cho et al. 2014), h_0 = 0 (zero-initialized, not
 *        learnable -- the same deliberate scope cut RNNModule/LSTMModule made, and the same
 *        thing that makes this module's conservation exact; see the LRP note below):
 *        z_t = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)   (update gate),
 *        r_t = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)   (reset gate),
 *        hn_prev_t = h_{t-1} @ W_hn                          (internal projection, no bias),
 *        n_t = tanh(x_t @ W_xn + r_t * hn_prev_t + b_n)      (candidate),
 *        h_t = (1 - z_t) * h_{t-1} + z_t * n_t.
 *        Note the reset gate multiplies the *projected* previous hidden state
 *        `hn_prev_t`, not `h_{t-1}` itself -- that projection is a distinct cached
 *        intermediate, and it is what gives GRU's LRP rule a different shape from LSTM's.
 *        Input (N, L, input_size) -> output (N, L, hidden_size), the full hidden-state
 *        sequence (matches RNNModule's/LSTMModule's convention). Single layer, no
 *        bidirectional/multi-layer/variable-length support.
 * @note Device-generic (GPU-native-kernels Mission 5): forward(), backward() and
 *       propagate_relevance() run entirely through DeviceBackend primitives (gemm/gemm_ex,
 *       copy_2d timestep slicing, elementwise Sigmoid/Tanh plus add/mul/axpby for the gate
 *       blend, recurrent_cell(GruBackward / GruLrp), accumulate_rows, lrp_linear,
 *       gru_lrp_hprev), reproducing the former host loops' evaluation order so CPU results
 *       are bit-identical.
 * @note LRP rule (gate-signal principle of Arras et al. 2019, "Explaining Recurrent Neural
 *       Network Predictions in Sentiment Analysis", extended here to GRU's
 *       reset-gate-inside-the-preactivation structure -- Arras et al. cover LSTM explicitly,
 *       so the same principle is re-derived below against GRU's equation shape). The gates
 *       z_t and r_t are pure *conductors*, never relevance *recipients*; every
 *       multiplicative interaction is an exact bilinear identity summing to its own result,
 *       so relevance splits across the *signal* operands only, with gate values acting as
 *       fixed multiplicative weights. Per timestep, in reverse time order:
 *       1. h_t = (1-z_t)*h_{t-1} + z_t*n_t is a two-term weighted sum whose denominator is
 *          exactly h_t; R(h_t) splits between R(h_{t-1})_direct and R(n_t) in proportion to
 *          (1-z_t)*h_{t-1} and z_t*n_t, via the same epsilon/z-rule shape RNNModule uses.
 *       2. n_t = tanh(n_pre) is identity pass-through (same pointwise-nonlinearity
 *          precedent as ReluModule/RNNModule, Montavon et al. 2019).
 *       3/4. n_pre (bias excluded, same "bias has no input feature to redistribute to"
 *          convention as every prior module) = x_t@W_xn + r_t*hn_prev_t. R(n_pre) splits
 *          between the two terms in proportion to their values; the x_t@W_xn share is then
 *          redistributed across x_t's features weighted by W_xn. Those two steps compose
 *          into one epsilon-stabilized redistribution over the shared n_pre denominator.
 *       5. r_t is a pure gate, so ALL of the r_t*hn_prev_t term's relevance passes to
 *          R(hn_prev_t) and none to R(r_t).
 *       6. hn_prev_t = h_{t-1}@W_hn is a standard single-source linear combination;
 *          R(hn_prev_t) is redistributed across h_{t-1}'s features weighted by W_hn
 *          (epsilon rule). Call this R(h_{t-1})_via_candidate.
 *       7. **The structural feature that distinguishes GRU from LSTM here**: the carried
 *          relevance accumulator for h_{t-1} is fed by TWO separate paths and must SUM
 *          them -- R(h_{t-1}) = R(h_{t-1})_direct (step 1) + R(h_{t-1})_via_candidate
 *          (step 6). LSTMModule's cell carry is single-path per accumulator; GRU's is not.
 *          Letting the second contribution overwrite rather than accumulate onto the first
 *          silently breaks conservation.
 *       8. z_pre/r_pre (the gates' own linear combinations) never receive or emit relevance
 *          at all -- consistent with LSTMModule's i_t/f_t/o_t. The gate values are consumed
 *          purely as precomputed multiplicative coefficients during the backward pass.
 *       Because h_0 is zero (this module's own scope cut), both contributions to the
 *       relevance that would "leak" into the non-existent state before t=0 are provably
 *       exactly zero (both numerators are proportional to h_prev, which is 0 at t=0) --
 *       end-to-end conservation holds up to the epsilon stabilizers, verified numerically
 *       by dedicated conservation tests.
 */
class GRUModule : public Module {
public:
    /**
     * @brief Constructs a GRU layer with zero-initialized weights/biases.
     * @param input_size Input feature dimension.
     * @param hidden_size Hidden state dimension.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if input_size <= 0 or hidden_size <= 0 -- external
     *         boundary (construction arguments can originate from Phase 5's Python
     *         bindings with no upstream validation).
     */
    GRUModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time (BPTT) across both gates, the reset-gated
     *        candidate and the convex (1-z_t)/z_t hidden carry: accumulates every weight and
     *        bias gradient across every timestep into the same buffers via
     *        Tensor::accumulate(). The gradient w.r.t. h_{t-1} sums three contributions
     *        (direct (1-z_t) carry, both gates' recurrent branches, and the candidate's
     *        W_hn projection branch), mirroring the two-path relevance structure documented
     *        on the class.
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, hidden_size)
     *        matching the most recent forward() call's output shape.
     * @return Gradient w.r.t. this module's input, shape (N, L, input_size).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached
     *         forward output shape.
     * @note Device-generic (GPU-native-kernels Mission 5): every step runs through
     *       DeviceBackend primitives, bit-identical to the former host loops on CPU.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Recurrent per charter's closed OpType set -- a compound accumulate-over-time
     *         operation, shared with RNNModule/LSTMModule (no enum change for this mission). */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the input-to-update-gate weight buffer -- test/initialization use only. */
    void set_weight_xz(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-update-gate weight buffer -- test/initialization use only. */
    void set_weight_hz(std::initializer_list<float> values);
    /** @brief Overwrites the update-gate bias buffer -- test/initialization use only. */
    void set_bias_z(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-reset-gate weight buffer -- test/initialization use only. */
    void set_weight_xr(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-reset-gate weight buffer -- test/initialization use only. */
    void set_weight_hr(std::initializer_list<float> values);
    /** @brief Overwrites the reset-gate bias buffer -- test/initialization use only. */
    void set_bias_r(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-candidate weight buffer -- test/initialization use only. */
    void set_weight_xn(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-candidate projection weight buffer (the W_hn whose
     *         output the reset gate multiplies) -- test/initialization use only. */
    void set_weight_hn(std::initializer_list<float> values);
    /** @brief Overwrites the candidate bias buffer -- test/initialization use only. */
    void set_bias_n(std::initializer_list<float> values);

    [[nodiscard]] const Tensor& weight_xz() const { return weight_xz_; }
    [[nodiscard]] const Tensor& weight_hz() const { return weight_hz_; }
    [[nodiscard]] const Tensor& bias_z() const { return bias_z_; }
    [[nodiscard]] const Tensor& weight_xr() const { return weight_xr_; }
    [[nodiscard]] const Tensor& weight_hr() const { return weight_hr_; }
    [[nodiscard]] const Tensor& bias_r() const { return bias_r_; }
    [[nodiscard]] const Tensor& weight_xn() const { return weight_xn_; }
    [[nodiscard]] const Tensor& weight_hn() const { return weight_hn_; }
    [[nodiscard]] const Tensor& bias_n() const { return bias_n_; }

    [[nodiscard]] const Tensor& weight_xz_grad() const { return weight_xz_grad_; }
    [[nodiscard]] const Tensor& weight_hz_grad() const { return weight_hz_grad_; }
    [[nodiscard]] const Tensor& bias_z_grad() const { return bias_z_grad_; }
    [[nodiscard]] const Tensor& weight_xr_grad() const { return weight_xr_grad_; }
    [[nodiscard]] const Tensor& weight_hr_grad() const { return weight_hr_grad_; }
    [[nodiscard]] const Tensor& bias_r_grad() const { return bias_r_grad_; }
    [[nodiscard]] const Tensor& weight_xn_grad() const { return weight_xn_grad_; }
    [[nodiscard]] const Tensor& weight_hn_grad() const { return weight_hn_grad_; }
    [[nodiscard]] const Tensor& bias_n_grad() const { return bias_n_grad_; }

    /**
     * @brief Arras et al. 2019 gate-signal LRP, extended to GRU: gates conduct, signals
     *        receive, and the carried h_{t-1} relevance sums a direct and a candidate-path
     *        contribution. See the class-level note for the full per-timestep redistribution.
     * @param relevance_out Relevance at this module's output. Must be (N, L, hidden_size)
     *        matching the most recent forward() call's output shape.
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (N, L, input_size). Conserves up to
     *         the epsilon stabilizers -- see the class-level note.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override {
        return {
            {"weight_xz", {&weight_xz_, &weight_xz_grad_}},
            {"weight_hz", {&weight_hz_, &weight_hz_grad_}},
            {"bias_z", {&bias_z_, &bias_z_grad_}},
            {"weight_xr", {&weight_xr_, &weight_xr_grad_}},
            {"weight_hr", {&weight_hr_, &weight_hr_grad_}},
            {"bias_r", {&bias_r_, &bias_r_grad_}},
            {"weight_xn", {&weight_xn_, &weight_xn_grad_}},
            {"weight_hn", {&weight_hn_, &weight_hn_grad_}},
            {"bias_n", {&bias_n_, &bias_n_grad_}},
        };
    }


    /** @brief Where this layer computes, so forward() rejects an input on another device (FND-8). */
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

protected:
    /**
     * @brief The actual forward computation -- per-timestep tied-weight gated recurrence.
     * @throws std::invalid_argument if input isn't rank-3 (N, L, input_size), or its last
     *         dimension doesn't match input_size.
     * @note Device-generic (GPU-native-kernels Mission 5): every step runs through
     *       DeviceBackend primitives, bit-identical to the former host loops on CPU.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t input_size_;
    int64_t hidden_size_;
    DeviceBackend* backend_;
    Tensor weight_xz_;  // (input_size, hidden_size)
    Tensor weight_hz_;  // (hidden_size, hidden_size)
    Tensor bias_z_;     // (hidden_size,)
    Tensor weight_xr_;
    Tensor weight_hr_;
    Tensor bias_r_;
    Tensor weight_xn_;
    Tensor weight_hn_;
    Tensor bias_n_;
    Tensor weight_xz_grad_;
    Tensor weight_hz_grad_;
    Tensor bias_z_grad_;
    Tensor weight_xr_grad_;
    Tensor weight_hr_grad_;
    Tensor bias_r_grad_;
    Tensor weight_xn_grad_;
    Tensor weight_hn_grad_;
    Tensor bias_n_grad_;
    Tensor last_input_;           // (N, L, input_size)
    Tensor last_hidden_states_;   // (N, L+1, hidden_size); index 0 = h_0 = 0
    Tensor last_gate_z_;          // (N, L, hidden_size); update gate
    Tensor last_gate_r_;          // (N, L, hidden_size); reset gate
    Tensor last_candidate_n_;     // (N, L, hidden_size); n_t, the tanh candidate signal
    // hn_prev_t = h_{t-1} @ W_hn is a DISTINCT intermediate from h_{t-1} itself (the reset
    // gate multiplies the projection, not the raw state), and it is the node the candidate
    // path's relevance lands on before being redistributed back onto h_{t-1} -- so it is
    // cached in its own right, for both backward() and propagate_relevance()'s step 6.
    Tensor last_hn_prev_;         // (N, L, hidden_size)
    Tensor last_pre_activation_n_;  // (N, L, hidden_size); n_t's pre-activation EXCLUDING bias
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
