/** @file rnn_module.hpp
 *  @brief Vanilla (Elman) recurrent layer -- this codebase's first recurrent module.
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "exai/module.hpp"

namespace exai {

/**
 * @brief h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h), h_0 = 0 (zero-initialized, not
 *        learnable -- a deliberate scope cut, see the class-level conservation note
 *        below). Input (N, L, input_size) -> output (N, L, hidden_size), the full
 *        hidden-state sequence. Single layer, tanh only, no bidirectional/multi-layer/
 *        variable-length support.
 * @note No DeviceBackend::Tanh primitive -- ElementwiseOp has only Relu/Neg. tanh and its
 *       derivative are computed via a raw host loop, EXAI_ASSERT-guarded like every other
 *       not-yet-backend-generic method in this codebase; this is the first module needing
 *       a nonlinearity beyond ReLU, not yet the repeated-pattern threshold that would
 *       justify a new backend primitive.
 * @note LRP rule (Arras et al. 2017, already cited in this charter for RNN/LSTM):
 *       epsilon/z-rule generalized to two weighted sources sharing one pre-activation
 *       (x_t's branch and h_{t-1}'s branch), tanh treated as identity pass-through (same
 *       precedent as ReluModule/DropoutModule, Montavon et al. 2019). Processed in
 *       reverse time order, threading a carried relevance accumulator exactly like
 *       backward()'s BPTT threads a gradient accumulator. Because h_0 is zero (this
 *       module's own scope cut), the relevance that would otherwise "leak" into the
 *       non-existent input before h_0 is provably exactly zero (the epsilon-rule
 *       numerator is h_prev*weight, and h_prev == 0 at t=0) -- end-to-end conservation is
 *       exact, not approximate, verified numerically by a dedicated conservation test.
 */
class RNNModule : public Module {
public:
    /**
     * @brief Constructs an RNN layer with zero-initialized weights/bias.
     * @param input_size Input feature dimension.
     * @param hidden_size Hidden state dimension.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if input_size <= 0 or hidden_size <= 0 -- external
     *         boundary (construction arguments can originate from Phase 5's Python
     *         bindings with no upstream validation).
     */
    RNNModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend);

    /**
     * @brief Real backpropagation-through-time (BPTT): accumulates W_xh/W_hh/b_h
     *        gradients across every timestep into the same buffers via Tensor::accumulate().
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, hidden_size)
     *        matching the most recent forward() call's output shape.
     * @return Gradient w.r.t. this module's input, shape (N, L, input_size).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached
     *         forward output shape.
     * @note Not yet backend-generic -- raw host loop. EXAI_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Recurrent per charter's closed OpType set -- a compound accumulate-over-time
     *         operation, not any existing category. */
    [[nodiscard]] OpType op_type() const override { return OpType::Recurrent; }

    /** @brief Overwrites the input-to-hidden weight buffer -- test/initialization use only. */
    void set_weight_xh(std::initializer_list<float> values);
    /** @brief Overwrites the hidden-to-hidden weight buffer -- test/initialization use only. */
    void set_weight_hh(std::initializer_list<float> values);
    /** @brief Overwrites the hidden bias buffer -- test/initialization use only. */
    void set_bias(std::initializer_list<float> values);

    [[nodiscard]] const Tensor& weight_xh() const { return weight_xh_; }
    [[nodiscard]] const Tensor& weight_hh() const { return weight_hh_; }
    [[nodiscard]] const Tensor& bias() const { return bias_; }
    [[nodiscard]] const Tensor& weight_xh_grad() const { return weight_xh_grad_; }
    [[nodiscard]] const Tensor& weight_hh_grad() const { return weight_hh_grad_; }
    [[nodiscard]] const Tensor& bias_grad() const { return bias_grad_; }

    /**
     * @brief Epsilon/z-rule LRP relevance propagation, generalized to two weighted
     *        sources, tanh treated as identity (Arras et al. 2017).
     * @param relevance_out Relevance at this module's output. Must be (N, L, hidden_size)
     *        matching the most recent forward() call's output shape.
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (N, L, input_size). Conserves
     *         exactly -- see the class-level note.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&weight_xh_, &weight_xh_grad_}, {&weight_hh_, &weight_hh_grad_}, {&bias_, &bias_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation -- per-timestep tied-weight recurrence.
     * @throws std::invalid_argument if input isn't rank-3 (N, L, input_size), or its last
     *         dimension doesn't match input_size.
     * @note Not yet backend-generic -- raw host loop. EXAI_ASSERT(input.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t input_size_;
    int64_t hidden_size_;
    DeviceBackend* backend_;
    Tensor weight_xh_;       // (input_size, hidden_size)
    Tensor weight_hh_;       // (hidden_size, hidden_size)
    Tensor bias_;            // (hidden_size,)
    Tensor weight_xh_grad_;
    Tensor weight_hh_grad_;
    Tensor bias_grad_;
    Tensor last_input_;             // (N, L, input_size)
    Tensor last_hidden_states_;     // (N, L+1, hidden_size); index 0 = h_0 = 0
    Tensor last_pre_activation_;    // (N, L, hidden_size); z_t EXCLUDING bias, cached for LRP
    int64_t last_L_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace exai
