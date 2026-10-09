/** @file adaptive_avg_pool2d_module.hpp
 *  @brief Average pooling to a fixed output size, as PyTorch's AdaptiveAvgPool2d (KS-9).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Pools a rank-4 (N, C, H, W) input to (N, C, out_h, out_w), as `AdaptiveAvgPool2d((out_h,
 *        out_w))`: ResNet's global pooling is (1, 1), VGG's is (7, 7).
 * @note Only sizes that divide evenly are supported (H divisible by out_h, W by out_w): the
 *       windows are then the non-overlapping `(H / out_h, W / out_w)` of AvgPool2DModule, with its
 *       backward and its LRP z-rule. PyTorch's uneven windows (overlapping, of different sizes)
 *       are refused rather than approximated.
 */
class AdaptiveAvgPool2DModule : public Module {
public:
    /** @throws std::invalid_argument if out_h or out_w < 1. */
    AdaptiveAvgPool2DModule(int64_t out_h, int64_t out_w, DeviceBackend* backend, float eps = 1e-6f);

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief AvgPool2DModule's z-rule over each window. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Pooling; }
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    [[nodiscard]] int64_t out_h() const { return out_h_; }
    [[nodiscard]] int64_t out_w() const { return out_w_; }

protected:
    /** @throws std::invalid_argument unless the input is rank 4 with H and W divisible by the
     *          output size. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t out_h_, out_w_;
    DeviceBackend* backend_;
    float eps_;
    std::unique_ptr<AvgPool2DModule> pool_;  ///< for the last input size
    int64_t kernel_h_ = 0, kernel_w_ = 0;
};

}  // namespace pulsatrix
