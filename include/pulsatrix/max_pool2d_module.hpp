/** @file max_pool2d_module.hpp
 *  @brief 2D max pooling with a stride and padding (non-overlapping windows by default).
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Max pooling, rank-4 (N, channels, H, W), matching Conv2DModule's convention, as
 *        PyTorch's MaxPool2d without dilation: out = (n + 2 padding - kernel) / stride + 1.
 *        Windows may overlap (stride < kernel), as in ResNet's stem (3x3, stride 2, padding 1);
 *        padded positions never win. No dilation or ceil mode.
 * @note propagate_relevance is winner-take-all (Bach et al. 2015's supplementary
 *       treatment of max-pooling, the same rule iNNvestigate/zennit ship): all relevance
 *       at an output position flows to the single input position that was the argmax in
 *       forward(); every other position in that window gets zero. An input that wins several
 *       overlapping windows receives all of their relevance. Conserves exactly by
 *       construction. backward() routes gradient the same way (only the argmax position
 *       receives grad_output; standard max-pool gradient semantics).
 */
class MaxPool2DModule : public Module {
public:
    /**
     * @brief Constructs a max-pool layer.
     * @param kernel_h Window height. Also the vertical stride (non-overlapping).
     * @param kernel_w Window width. Also the horizontal stride (non-overlapping).
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if kernel_h <= 0 or kernel_w <= 0 -- external boundary
     *         (construction arguments can originate from Phase 5's Python bindings with no
     *         upstream validation), per cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    MaxPool2DModule(int64_t kernel_h, int64_t kernel_w, DeviceBackend* backend);

    /**
     * @brief A max-pool layer with its own stride and padding (KS-9), as PyTorch's
     *        `MaxPool2d(kernel, stride, padding)`.
     * @throws std::invalid_argument for a non-positive kernel or stride, a negative padding, or
     *         padding above half the kernel (PyTorch's rule: every window must touch the input).
     */
    MaxPool2DModule(int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
                    DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input -- only the cached argmax
     *        position within each window receives grad_output; every other position is 0.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached forward
     *         output shape.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Pooling per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Pooling; }

    /**
     * @brief Winner-take-all LRP relevance propagation (Bach et al. 2015).
     * @param relevance_out Relevance at this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @param config Unused -- winner-take-all has no tunable parameter.
     * @return Relevance at this module's input: relevance_out's value at each window's
     *         cached argmax position, zero everywhere else. Conserves trivially by
     *         construction (no epsilon stabilizer needed).
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    /** @brief Winner-take-all ignores the config: the same under every rule, so supports all of them. */
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }


    /** @brief Where this layer computes, so forward() rejects an input on another device (FND-8). */
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

protected:
    /**
     * @brief The actual forward computation -- per-window max, argmax cached per output
     *        element for backward()/propagate_relevance() to reuse.
     * @throws std::invalid_argument if input isn't rank-4 (N, C, H, W), or the padded input is
     *         smaller than the kernel.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t kernel_h_;
    int64_t kernel_w_;
    int64_t stride_h_;
    int64_t stride_w_;
    int64_t pad_h_ = 0;
    int64_t pad_w_ = 0;
    DeviceBackend* backend_;
    Shape last_input_shape_ = Shape({0});
    int64_t last_out_h_ = 0;
    int64_t last_out_w_ = 0;
    // Flat (within-window) offset of the argmax for each output element, indexed the same
    // way as the output buffer (n, c, oh, ow) -- row-major flat index into (H, W) input
    // plane, i.e. ih * W + iw. Sized N*C*out_h*out_w after a real forward() call.
    // Stored as whole-number floats on the input's device (exact below 2^24 -- far above any
    // plane this module pools), consumed by DeviceBackend::max_unpool (GPU-native-kernels
    // Mission 4).
    Tensor argmax_flat_index_ = Tensor(Shape({0}), backend_);
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
