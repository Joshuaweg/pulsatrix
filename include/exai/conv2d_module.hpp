/** @file conv2d_module.hpp
 *  @brief 2D convolution -- implemented via im2col + DeviceBackend::gemm (no new backend primitive).
 */
#pragma once

#include <initializer_list>

#include "exai/module.hpp"

namespace exai {

/**
 * @brief 2D convolution, unbatched (input/output are rank-3: channels x H x W). Stride 1,
 *        no padding, no dilation -- deferred until a real use case needs them, same
 *        pattern as LinearModule's unbatched scope cut.
 * @note Implemented via im2col + gemm per campaign Decision 3 (see
 *       campaign_exai_dl_library_phase1_core_layers_training.md) rather than a dedicated
 *       DeviceBackend::conv2d primitive -- Phase 1.5's cuDNN integration will need its own
 *       conv path anyway, so a CPU-side primitive now would likely be thrown away, not reused.
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
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in raw host
     *       loops (via transpose2d()/col2im() helpers). EXAI_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output);

    /** @brief Overwrites the kernel buffer -- test/initialization use only. */
    void set_kernel(std::initializer_list<float> values);

    /** @brief Overwrites the bias buffer -- test/initialization use only. */
    void set_bias(std::initializer_list<float> values);

    [[nodiscard]] const Tensor& kernel() const { return kernel_; }
    [[nodiscard]] const Tensor& bias() const { return bias_; }
    [[nodiscard]] const Tensor& kernel_grad() const { return kernel_grad_; }
    [[nodiscard]] const Tensor& bias_grad() const { return bias_grad_; }

    /**
     * @brief Epsilon-rule LRP relevance propagation.
     * @param relevance_out Relevance at this module's output. Shape (out_channels, out_h, out_w).
     * @param config Selects epsilon.
     * @return Relevance at this module's input, shape (in_channels, H, W).
     * @note Structurally the same rule as LinearModule's, applied independently per output
     *       position via the im2col representation (each output position/channel pair is
     *       a "virtual Linear neuron" over its own receptive-field patch), then col2im'd
     *       back to input space -- which sums relevance from every output position that
     *       touched a given input pixel, the same overlap-handling backward() already
     *       needed for gradients. Bias is excluded from z, same rationale as LinearModule.
     *       Must be called after forward().
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in raw host
     *       loops (via col2im()). EXAI_ASSERT(relevance_out.device() == DeviceType::Cpu)
     *       guards against silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&kernel_, &kernel_grad_}, {&bias_, &bias_grad_}};
    }

protected:
    /**
     * @brief The actual forward computation (im2col + gemm + per-channel bias add).
     * @note Not yet backend-generic -- unlike LinearModule/ReluModule's forward_impl, this
     *       one dereferences Tensor::data() directly (its bias-add loop, plus the im2col()
     *       helper). EXAI_ASSERT(input.device() == DeviceType::Cpu) guards against silent
     *       UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
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
    Tensor last_input_;       // (in_channels, H, W)
    Tensor last_im2col_;      // (patch_size, out_h*out_w), cached for backward
    Tensor last_pre_bias_output_;  // (out_channels, out_h, out_w), cached for LRP
    int64_t last_out_h_ = 0;
    int64_t last_out_w_ = 0;
};

}  // namespace exai
