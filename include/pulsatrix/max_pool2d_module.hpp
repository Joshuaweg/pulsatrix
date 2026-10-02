/** @file max_pool2d_module.hpp
 *  @brief 2D max pooling, non-overlapping windows (stride == kernel), no padding.
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Max pooling, rank-4 (N, channels, H, W), matching Conv2DModule's convention.
 *        Stride fixed equal to kernel size (non-overlapping windows), no padding, no
 *        dilation -- deferred until a real use case needs them, same minimal-cut
 *        discipline as Conv2DModule's original stride-1/no-padding scope cut.
 * @note propagate_relevance is winner-take-all (Bach et al. 2015's supplementary
 *       treatment of max-pooling, the same rule iNNvestigate/zennit ship): all relevance
 *       at an output position flows to the single input position that was the argmax in
 *       forward(); every other position in that window gets zero. Conserves exactly by
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

protected:
    /**
     * @brief The actual forward computation -- per-window max, argmax cached per output
     *        element for backward()/propagate_relevance() to reuse.
     * @throws std::invalid_argument if input isn't rank-4 (N, C, H, W), or the kernel is
     *         larger than the input (kernel_h &gt; H or kernel_w &gt; W).
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t kernel_h_;
    int64_t kernel_w_;
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
