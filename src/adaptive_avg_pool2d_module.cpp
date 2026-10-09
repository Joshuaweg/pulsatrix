#include "pulsatrix/adaptive_avg_pool2d_module.hpp"

#include <stdexcept>

namespace pulsatrix {

AdaptiveAvgPool2DModule::AdaptiveAvgPool2DModule(int64_t out_h, int64_t out_w, DeviceBackend* backend, float eps)
    : out_h_(out_h), out_w_(out_w), backend_(backend), eps_(eps) {
    if (out_h < 1 || out_w < 1) throw std::invalid_argument("AdaptiveAvgPool2DModule: the output size must be positive");
}

Tensor AdaptiveAvgPool2DModule::forward_impl(const Tensor& input) {
    if (input.rank() != 4) throw std::invalid_argument("AdaptiveAvgPool2DModule::forward: input must be rank-4 (N, C, H, W)");
    const int64_t H = input.shape().dim(2), W = input.shape().dim(3);
    if (H % out_h_ != 0 || W % out_w_ != 0) {
        throw std::invalid_argument("AdaptiveAvgPool2DModule::forward: the input's height and width must be multiples of the output's");
    }
    if (!pool_ || kernel_h_ != H / out_h_ || kernel_w_ != W / out_w_) {
        kernel_h_ = H / out_h_;
        kernel_w_ = W / out_w_;
        pool_ = std::make_unique<AvgPool2DModule>(kernel_h_, kernel_w_, backend_, eps_);
    }
    return pool_->forward(input);
}

Tensor AdaptiveAvgPool2DModule::backward(const Tensor& grad_output) {
    if (!pool_) throw std::logic_error("AdaptiveAvgPool2DModule::backward: called before any forward()");
    return pool_->backward(grad_output);
}

Tensor AdaptiveAvgPool2DModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!pool_) throw std::logic_error("AdaptiveAvgPool2DModule::propagate_relevance: called before any forward()");
    return pool_->propagate_relevance(relevance_out, config);
}

}  // namespace pulsatrix
