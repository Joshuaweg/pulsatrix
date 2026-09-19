/** @file linear_module.hpp
 *  @brief Dense/fully-connected layer -- the reference Module implementation.
 */
#pragma once

#include <initializer_list>

#include "exai/module.hpp"

namespace exai {

/**
 * @brief y = x @ W + b, unbatched (x, y are rank-1 tensors; batching is Mission 4's
 *        concern, not built here -- see campaign Decision 1 in this mission's Notes).
 * @note Weight layout is (in_features, out_features), not the more common
 *       (out_features, in_features) PyTorch convention -- chosen specifically so forward
 *       (x @ W) and the weight-gradient step (outer(x, grad_y)) both use
 *       DeviceBackend::gemm directly with no transpose. Only the input-gradient step
 *       (grad_y @ W^T) needs an actual transpose, handled internally.
 * @note Weights/biases are owned here as member Tensors, not ComputationGraph nodes.
 *       Parameter gradients accumulate via Tensor::accumulate() across backward() calls
 *       until something (Mission 3's optimizer) resets them -- there is no zero_grad()
 *       yet because nothing needs one until the optimizer exists.
 */
class LinearModule : public Module {
public:
    /**
     * @brief Constructs a linear layer with zero-initialized weight/bias.
     * @param in_features Input dimension.
     * @param out_features Output dimension.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     */
    LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates the
     *        weight/bias gradients internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of
     *        the most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @note Must be called after forward() -- uses the input cached from that call.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output);

    /** @brief Overwrites the weight buffer -- test/initialization use only. */
    void set_weight(std::initializer_list<float> values);

    /** @brief Overwrites the bias buffer -- test/initialization use only. */
    void set_bias(std::initializer_list<float> values);

    [[nodiscard]] const Tensor& weight() const { return weight_; }
    [[nodiscard]] const Tensor& bias() const { return bias_; }
    [[nodiscard]] const Tensor& weight_grad() const { return weight_grad_; }
    [[nodiscard]] const Tensor& bias_grad() const { return bias_grad_; }

    /**
     * @brief Epsilon-rule LRP relevance propagation (Bach et al. 2015).
     * @param relevance_out Relevance at this module's output. Must match out_features.
     * @param config Selects epsilon. Larger epsilon trades a small amount of conservation
     *        for numerical stability when a pre-bias output is near zero.
     * @return Relevance at this module's input.
     * @note Uses the pre-bias linear output (x @ W, not x @ W + b) as z_j -- bias has no
     *       associated input feature to redistribute relevance to, so it is excluded from
     *       the rule entirely rather than approximated. This is what makes relevance
     *       conservation exact (up to the epsilon stabilizer) rather than merely
     *       approximate. Must be called after forward() -- uses the cached pre-bias output.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&weight_, &weight_grad_}, {&bias_, &bias_grad_}};
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t in_features_;
    int64_t out_features_;
    DeviceBackend* backend_;
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    Tensor last_input_;
    Tensor last_pre_bias_output_;
};

}  // namespace exai
