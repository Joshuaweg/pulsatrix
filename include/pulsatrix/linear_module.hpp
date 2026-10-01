/** @file linear_module.hpp
 *  @brief Dense/fully-connected layer -- the reference Module implementation.
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y = x @ W + b, batched (x is (N, in_features), y is (N, out_features)) --
 *        migrated from the original unbatched (rank-1) scope by
 *        campaign_exai_dl_library_batch_dimension_support (breaking migration to
 *        always-batched; a single example is N=1, not a structurally different case).
 * @note Weight layout is (in_features, out_features), not the more common
 *       (out_features, in_features) PyTorch convention -- chosen specifically so forward
 *       (x @ W) and the weight-gradient step (X^T @ grad_Y) both use
 *       DeviceBackend::gemm directly with no transpose of the *weight* operand. The
 *       input-gradient step (grad_Y @ W^T) and the weight-gradient step (X^T @ grad_Y, the
 *       batched sum of outer products) read their transposed operand in place through
 *       DeviceBackend::gemm_ex -- no transposed copy is built.
 * @note Weights/biases are owned here as member Tensors, not ComputationGraph nodes.
 *       Parameter gradients accumulate via Tensor::accumulate() across backward() calls
 *       until something (the optimizer) resets them. Batch-dimension gradient reduction
 *       (summing weight/bias gradient contributions across the N examples in a batch)
 *       needs no new Tensor primitive -- see
 *       campaign_exai_dl_library_batch_dimension_support's mission_tensor_shape_foundation.md:
 *       weight_grad's reduction falls out of gemm's own k-dimension summation (accumulated
 *       in place, beta = 1); bias_grad's is DeviceBackend::column_sums, likewise
 *       accumulated in place. Both are device-resident (GPU-native-kernels Mission 1).
 */
class LinearModule : public Module {
public:
    /**
     * @brief Constructs a linear layer with zero-initialized weight/bias.
     * @param in_features Input dimension.
     * @param out_features Output dimension.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param device Which device every internal Tensor member (weight, bias, gradients,
     *        forward-pass caches) is tagged as. Must match whatever device backend actually
     *        allocates on, or Tensor's own device-based dispatch (e.g. CopyDirection
     *        selection) will be wrong. See campaign_exai_dl_library_phase1_5_cuda_backend.md's
     *        Mission 3.
     */
    LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend, DeviceType device);

    /**
     * @brief As above, on backend's own device (backend->device()).
     * @note Previously the device defaulted to Cpu, so a LinearModule built on a GPU backend
     *       without an explicit tag -- e.g. SwiGLUModule's three projections -- held
     *       Cpu-tagged weights in device memory (GPU-native-kernels Mission 0).
     */
    LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates the
     *        weight/bias gradients internally (summed across the batch).
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, out_features),
     *        with N matching the most recent forward() call's batch size.
     * @return Gradient w.r.t. this module's input, shape (N, in_features).
     * @note Must be called after forward() -- uses the input cached from that call.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1).
     * @throws std::logic_error if forward() has never been called -- see
     *         campaign_exai_dl_library_adversarial_hardening.md, finding 12.
     * @throws std::invalid_argument if grad_output's rank/shape don't match (N, out_features)
     *         for the cached N -- external boundary, batch-size-mismatch is a new adversarial
     *         case this migration introduces (campaign_exai_dl_library_batch_dimension_support).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Linear per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Linear; }

    /** @brief Overwrites the weight buffer -- test/initialization use only. */
    void set_weight(std::initializer_list<float> values);

    /** @brief Overwrites the bias buffer -- test/initialization use only. */
    void set_bias(std::initializer_list<float> values);

    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_weight(const std::vector<float>& values);

    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_bias(const std::vector<float>& values);

    [[nodiscard]] const Tensor& weight() const { return weight_; }
    [[nodiscard]] const Tensor& bias() const { return bias_; }
    [[nodiscard]] const Tensor& weight_grad() const { return weight_grad_; }
    [[nodiscard]] const Tensor& bias_grad() const { return bias_grad_; }

    /**
     * @brief Epsilon-rule LRP relevance propagation (Bach et al. 2015), applied
     *        independently per example in the batch.
     * @param relevance_out Relevance at this module's output. Must be (N, out_features),
     *        matching the cached forward() batch size.
     * @param config Selects epsilon. Larger epsilon trades a small amount of conservation
     *        for numerical stability when a pre-bias output is near zero.
     * @return Relevance at this module's input, shape (N, in_features).
     * @note Uses the pre-bias linear output (x @ W, not x @ W + b) as z_j -- bias has no
     *       associated input feature to redistribute relevance to, so it is excluded from
     *       the rule entirely rather than approximated. This is what makes relevance
     *       conservation exact (up to the epsilon stabilizer) rather than merely
     *       approximate. Must be called after forward() -- uses the cached pre-bias output.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 3).
     * @throws std::logic_error if forward() has never been called -- see
     *         campaign_exai_dl_library_adversarial_hardening.md, finding 12.
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
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
