/** @file lstm_module.hpp
 *  @brief Single-layer LSTM -- this codebase's first *gated* recurrent module.
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Standard 4-gate LSTM recurrence, h_0 = c_0 = 0 (zero-initialized, not learnable --
 *        the same deliberate scope cut RNNModule made, and the same thing that makes this
 *        module's conservation exact; see the LRP note below):
 *        i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i),
 *        f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f),
 *        g_t = tanh(x_t @ W_xg + h_{t-1} @ W_hg + b_g),
 *        o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o),
 *        c_t = f_t * c_{t-1} + i_t * g_t, h_t = o_t * tanh(c_t).
 *        Input (N, L, input_size) -> output (N, L, hidden_size), the full hidden-state
 *        sequence (matches RNNModule's convention). Single layer, no bidirectional/
 *        multi-layer/variable-length/peephole support.
 * @note Device-generic (GPU-native-kernels Mission 5): forward(), backward() and
 *       propagate_relevance() run entirely through DeviceBackend primitives (gemm/gemm_ex,
 *       copy_2d timestep slicing, elementwise Sigmoid/Tanh, recurrent_cell(LstmForward /
 *       LstmBackward / LstmLrp), accumulate_rows, lrp_linear), reproducing the former host
 *       loops' evaluation order so CPU results are bit-identical.
 * @note LRP rule (Arras et al. 2019, "Explaining Recurrent Neural Network Predictions in
 *       Sentiment Analysis"): the gates i_t, f_t, o_t are pure *conductors*, never relevance
 *       *recipients*. Every multiplicative interaction in the recurrence is an exact
 *       bilinear identity summing to its own result, so relevance splits across the
 *       *signal* operands only, with the gate values acting as fixed multiplicative
 *       weights:
 *       - c_t = f_t*c_{t-1} + i_t*g_t is a two-term weighted sum (weights f_t/i_t, signals
 *         c_{t-1}/g_t); R(c_t) splits between R(c_{t-1}) and R(g_t) in proportion to
 *         f_t*c_{t-1} and i_t*g_t, via the same epsilon/z-rule redistribution RNNModule
 *         applies to its two-source pre-activation.
 *       - h_t = o_t * tanh(c_t) is a single-signal product, so ALL of R(h_t) passes to
 *         R(tanh(c_t)) and none to R(o_t).
 *       - tanh(c_t) -> c_t is identity pass-through (same pointwise-nonlinearity precedent
 *         as ReluModule/RNNModule, Montavon et al. 2019).
 *       - R(g_t) is then redistributed across x_t and h_{t-1} by one epsilon/z-rule step
 *         over g_t's two weighted sources, exactly like RNNModule's.
 *       Net flow per timestep, processed in reverse time order threading both a hidden and
 *       a cell relevance accumulator (mirroring backward()'s BPTT accumulators):
 *       h_t -> c_t -> { c_{t-1} (carried to t-1), g_t -> { x_t, h_{t-1} (carried to t-1) } }.
 *       Because h_0 and c_0 are zero (this module's own scope cut), the relevance that
 *       would otherwise "leak" into the non-existent state before t=0 is provably exactly
 *       zero (both epsilon-rule numerators are proportional to c_prev/h_prev, both == 0 at
 *       t=0) -- end-to-end conservation is exact up to the epsilon stabilizer, verified
 *       numerically by dedicated conservation tests.
 */
class LSTMModule : public Module {
public:
    /**
     * @brief Constructs an LSTM layer with zero-initialized weights/biases.
     * @param input_size Input feature dimension.
     * @param hidden_size Hidden (and cell) state dimension.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if input_size <= 0 or hidden_size <= 0 -- external
     *         boundary (construction arguments can originate from Phase 5's Python
     *         bindings with no upstream validation).
     */
    LSTMModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time (BPTT) across all four gates and the cell
     *        carry: accumulates every per-gate input weight, recurrent weight and bias
     *        gradient across every timestep into the same buffers via Tensor::accumulate().
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
     *         operation, shared with RNNModule (no enum change needed for this mission). */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the input-to-input-gate weight buffer -- test/initialization use only. */
    void set_weight_xi(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-input-gate weight buffer -- test/initialization use only. */
    void set_weight_hi(std::initializer_list<float> values);
    /** @brief Overwrites the input-gate bias buffer -- test/initialization use only. */
    void set_bias_i(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-forget-gate weight buffer -- test/initialization use only. */
    void set_weight_xf(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-forget-gate weight buffer -- test/initialization use only. */
    void set_weight_hf(std::initializer_list<float> values);
    /** @brief Overwrites the forget-gate bias buffer -- test/initialization use only. */
    void set_bias_f(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-cell-candidate weight buffer -- test/initialization use only. */
    void set_weight_xg(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-cell-candidate weight buffer -- test/initialization use only. */
    void set_weight_hg(std::initializer_list<float> values);
    /** @brief Overwrites the cell-candidate bias buffer -- test/initialization use only. */
    void set_bias_g(std::initializer_list<float> values);
    /** @brief Overwrites the input-to-output-gate weight buffer -- test/initialization use only. */
    void set_weight_xo(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-output-gate weight buffer -- test/initialization use only. */
    void set_weight_ho(std::initializer_list<float> values);
    /** @brief Overwrites the output-gate bias buffer -- test/initialization use only. */
    void set_bias_o(std::initializer_list<float> values);

    [[nodiscard]] const Tensor& weight_xi() const { return weight_xi_; }
    [[nodiscard]] const Tensor& weight_hi() const { return weight_hi_; }
    [[nodiscard]] const Tensor& bias_i() const { return bias_i_; }
    [[nodiscard]] const Tensor& weight_xf() const { return weight_xf_; }
    [[nodiscard]] const Tensor& weight_hf() const { return weight_hf_; }
    [[nodiscard]] const Tensor& bias_f() const { return bias_f_; }
    [[nodiscard]] const Tensor& weight_xg() const { return weight_xg_; }
    [[nodiscard]] const Tensor& weight_hg() const { return weight_hg_; }
    [[nodiscard]] const Tensor& bias_g() const { return bias_g_; }
    [[nodiscard]] const Tensor& weight_xo() const { return weight_xo_; }
    [[nodiscard]] const Tensor& weight_ho() const { return weight_ho_; }
    [[nodiscard]] const Tensor& bias_o() const { return bias_o_; }

    [[nodiscard]] const Tensor& weight_xi_grad() const { return weight_xi_grad_; }
    [[nodiscard]] const Tensor& weight_hi_grad() const { return weight_hi_grad_; }
    [[nodiscard]] const Tensor& bias_i_grad() const { return bias_i_grad_; }
    [[nodiscard]] const Tensor& weight_xf_grad() const { return weight_xf_grad_; }
    [[nodiscard]] const Tensor& weight_hf_grad() const { return weight_hf_grad_; }
    [[nodiscard]] const Tensor& bias_f_grad() const { return bias_f_grad_; }
    [[nodiscard]] const Tensor& weight_xg_grad() const { return weight_xg_grad_; }
    [[nodiscard]] const Tensor& weight_hg_grad() const { return weight_hg_grad_; }
    [[nodiscard]] const Tensor& bias_g_grad() const { return bias_g_grad_; }
    [[nodiscard]] const Tensor& weight_xo_grad() const { return weight_xo_grad_; }
    [[nodiscard]] const Tensor& weight_ho_grad() const { return weight_ho_grad_; }
    [[nodiscard]] const Tensor& bias_o_grad() const { return bias_o_grad_; }

    /**
     * @brief Arras et al. 2019 gate-signal LRP: gates conduct, signals receive. See the
     *        class-level note for the full per-timestep redistribution.
     * @param relevance_out Relevance at this module's output. Must be (N, L, hidden_size)
     *        matching the most recent forward() call's output shape.
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (N, L, input_size). Conserves up to
     *         the epsilon stabilizer -- see the class-level note.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override {
        return {
            {"weight_xi", {&weight_xi_, &weight_xi_grad_}},
            {"weight_hi", {&weight_hi_, &weight_hi_grad_}},
            {"bias_i", {&bias_i_, &bias_i_grad_}},
            {"weight_xf", {&weight_xf_, &weight_xf_grad_}},
            {"weight_hf", {&weight_hf_, &weight_hf_grad_}},
            {"bias_f", {&bias_f_, &bias_f_grad_}},
            {"weight_xg", {&weight_xg_, &weight_xg_grad_}},
            {"weight_hg", {&weight_hg_, &weight_hg_grad_}},
            {"bias_g", {&bias_g_, &bias_g_grad_}},
            {"weight_xo", {&weight_xo_, &weight_xo_grad_}},
            {"weight_ho", {&weight_ho_, &weight_ho_grad_}},
            {"bias_o", {&bias_o_, &bias_o_grad_}},
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
    Tensor weight_xi_;  // (input_size, hidden_size)
    Tensor weight_hi_;  // (hidden_size, hidden_size)
    Tensor bias_i_;     // (hidden_size,)
    Tensor weight_xf_;
    Tensor weight_hf_;
    Tensor bias_f_;
    Tensor weight_xg_;
    Tensor weight_hg_;
    Tensor bias_g_;
    Tensor weight_xo_;
    Tensor weight_ho_;
    Tensor bias_o_;
    Tensor weight_xi_grad_;
    Tensor weight_hi_grad_;
    Tensor bias_i_grad_;
    Tensor weight_xf_grad_;
    Tensor weight_hf_grad_;
    Tensor bias_f_grad_;
    Tensor weight_xg_grad_;
    Tensor weight_hg_grad_;
    Tensor bias_g_grad_;
    Tensor weight_xo_grad_;
    Tensor weight_ho_grad_;
    Tensor bias_o_grad_;
    Tensor last_input_;           // (N, L, input_size)
    Tensor last_hidden_states_;   // (N, L+1, hidden_size); index 0 = h_0 = 0
    Tensor last_cell_states_;     // (N, L+1, hidden_size); index 0 = c_0 = 0
    Tensor last_gate_i_;          // (N, L, hidden_size)
    Tensor last_gate_f_;          // (N, L, hidden_size)
    Tensor last_gate_g_;          // (N, L, hidden_size); the cell candidate (tanh) signal
    Tensor last_gate_o_;          // (N, L, hidden_size)
    Tensor last_cell_tanh_;       // (N, L, hidden_size); tanh(c_t), cached for backward
    Tensor last_pre_activation_g_;  // (N, L, hidden_size); g_t's z EXCLUDING bias, for LRP
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
