/** @file avg_pool2d_module.hpp
 *  @brief 2D average pooling, non-overlapping windows (stride == kernel), no padding.
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Average pooling, rank-4 (N, channels, H, W), matching Conv2DModule's convention.
 *        Stride fixed equal to kernel size (non-overlapping windows), no padding, no
 *        dilation -- same minimal-cut discipline as MaxPool2DModule/Conv2DModule.
 * @note propagate_relevance is the epsilon/z-rule (Bach et al. 2015's generalized-linear-
 *       layer treatment of pooling, the same family LinearModule/Conv2DModule already use):
 *       average pooling is y = (1/K) * sum(x_i), i.e. a Linear layer with K uniform 1/K
 *       weights and no bias. z = sum(x_i/K) + eps*sign(z), R_i = (x_i * (1/K) / z) * R_out
 *       -- proportional to each input's own activation, not a uniform 1/K split (that would
 *       be the *gradient* distribution, a weaker rule deliberately not used here).
 *       backward() is the real, undetached training gradient (uniform 1/K per input,
 *       independent of activation value -- the correct derivative of a mean).
 */
class AvgPool2DModule : public Module {
public:
    /**
     * @brief Constructs an average-pool layer.
     * @param kernel_h Window height. Also the vertical stride (non-overlapping).
     * @param kernel_w Window width. Also the horizontal stride (non-overlapping).
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param eps Stabilizer added inside the LRP z-rule's denominator. Defaults to 1e-6,
     *        matching every other epsilon-rule module in this codebase.
     * @throws std::invalid_argument if kernel_h <= 0 or kernel_w <= 0 -- external boundary,
     *         per cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    AvgPool2DModule(int64_t kernel_h, int64_t kernel_w, DeviceBackend* backend, float eps = 1e-6f);

    /**
     * @brief Computes the gradient w.r.t. this module's input -- uniform 1/K per input
     *        position in each window (the true derivative of a mean, independent of
     *        activation value).
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached forward
     *         output shape.
     * @note Not yet backend-generic -- raw host loop. PULSATRIX_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Pooling per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Pooling; }

    /**
     * @brief Epsilon/z-rule LRP relevance propagation, weight = 1/K (Bach et al. 2015).
     * @param relevance_out Relevance at this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @param config Selects epsilon.
     * @return Relevance at this module's input, proportional to each input's own cached
     *         activation within its window.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief The actual forward computation -- per-window mean.
     * @throws std::invalid_argument if input isn't rank-4 (N, C, H, W), or the kernel is
     *         larger than the input (kernel_h &gt; H or kernel_w &gt; W).
     * @note Not yet backend-generic -- raw host loop. PULSATRIX_ASSERT(input.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t kernel_h_;
    int64_t kernel_w_;
    DeviceBackend* backend_;
    float eps_;
    Tensor last_input_;  // (N, C, H, W)
    int64_t last_out_h_ = 0;
    int64_t last_out_w_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
