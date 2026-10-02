/** @file conv2d_module.hpp
 *  @brief 2D convolution -- implemented via im2col + DeviceBackend::gemm (no new backend primitive).
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief 2D convolution, batched (input/output are rank-4: N x channels x H x W) --
 *        migrated from the original unbatched (rank-3) scope by
 *        campaign_exai_dl_library_batch_dimension_support. Stride 1, no padding, no
 *        dilation -- deferred until a real use case needs them, same pattern as
 *        LinearModule's original unbatched scope cut.
 * @note Implemented via im2col + gemm per campaign Decision 3 (see
 *       campaign_exai_dl_library_phase1_core_layers_training.md) rather than a dedicated
 *       DeviceBackend::conv2d primitive -- Phase 1.5's cuDNN integration will need its own
 *       conv path anyway, so a CPU-side primitive now would likely be thrown away, not reused.
 * @note Batching implemented as a per-example loop over the existing unbatched im2col/gemm/
 *       col2im pipeline (each output column of the im2col matrix is independent of every
 *       other, so this is exactly equivalent to a single larger gemm with the Q axis
 *       extended to N*Q -- the per-example loop is the simpler, correctness-first choice;
 *       see campaign recon's noted alternative), not a new backend primitive.
 */
class Conv2DModule : public Module {
public:
    /**
     * @brief Constructs a conv layer with zero-initialized kernel/bias.
     * @param in_channels Input channel count.
     * @param out_channels Output channel count.
     * @param kernel_h Kernel height.
     * @param kernel_w Kernel width.
     * @param backend Backend to compute through. Not owned; must outlive this module.
     */
    Conv2DModule(int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
                 DeviceBackend* backend);

    /**
     * @brief Computes gradients w.r.t. input, kernel, and bias.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of
     *        the most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @note Must be called after forward() -- uses the input/im2col cached from that call.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     * @throws std::logic_error if forward() has never been called -- see
     *         campaign_exai_dl_library_adversarial_hardening.md, finding 12.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Conv per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Conv; }

    /** @brief Overwrites the kernel buffer -- test/initialization use only. */
    void set_kernel(std::initializer_list<float> values);

    /** @brief Overwrites the bias buffer -- test/initialization use only. */
    void set_bias(std::initializer_list<float> values);

    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_kernel(const std::vector<float>& values);

    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_bias(const std::vector<float>& values);

    [[nodiscard]] const Tensor& kernel() const { return kernel_; }
    [[nodiscard]] const Tensor& bias() const { return bias_; }
    [[nodiscard]] const Tensor& kernel_grad() const { return kernel_grad_; }
    [[nodiscard]] const Tensor& bias_grad() const { return bias_grad_; }

    /**
     * @brief Epsilon-rule LRP relevance propagation.
     * @param relevance_out Relevance at this module's output. Shape (N, out_channels, out_h, out_w).
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (N, in_channels, H, W).
     * @note Structurally the same rule as LinearModule's, applied independently per output
     *       position via the im2col representation (each output position/channel pair is
     *       a "virtual Linear neuron" over its own receptive-field patch), then col2im'd
     *       back to input space -- which sums relevance from every output position that
     *       touched a given input pixel, the same overlap-handling backward() already
     *       needed for gradients. Bias is excluded from z, same rationale as LinearModule.
     *       Must be called after forward().
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     * @note config.rule selects Epsilon (default, as above), Gamma, AlphaBeta or ZBox: the
     *       LinearModule formulas (Zennit 1.0.0) applied in patch space, where each output
     *       position is K @ patch + b, then folded back with col2im. Epsilon with
     *       config.epsilon_bias_in_denominator uses z = K @ patch + b (Zennit's Epsilon).
     * @throws std::logic_error if forward() has never been called -- see
     *         campaign_exai_dl_library_adversarial_hardening.md, finding 12.
     * @throws std::invalid_argument if config's rule parameters are invalid (AlphaBeta needs
     *         alpha, beta >= 0 and alpha - beta == 1).
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief Implements every LRPRule. */
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&kernel_, &kernel_grad_}, {&bias_, &bias_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation (im2col + gemm + per-channel bias add).
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     * @throws std::invalid_argument if input isn't rank-4 (N, in_channels, H, W), its
     *         channel count doesn't match in_channels_, or the kernel is larger than the
     *         input (kernel_h_ &gt; H or kernel_w_ &gt; W) -- see
     *         campaign_exai_dl_library_adversarial_hardening.md, findings 1 and 6.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_h_;
    int64_t kernel_w_;
    DeviceBackend* backend_;
    Tensor kernel_;       // shape (out_channels, in_channels, kernel_h, kernel_w)
    Tensor bias_;         // shape (out_channels,)
    Tensor kernel_grad_;
    Tensor bias_grad_;
    Tensor last_input_;       // (N, in_channels, H, W)
    Tensor last_im2col_;      // (N, patch_size, out_h*out_w), cached for backward
    Tensor last_pre_bias_output_;  // (N, out_channels, out_h, out_w), cached for LRP
    int64_t last_out_h_ = 0;
    int64_t last_out_w_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
