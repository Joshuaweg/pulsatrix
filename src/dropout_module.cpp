#include "pulsatrix/dropout_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

DropoutModule::DropoutModule(float p, DeviceBackend* backend, uint64_t seed)
    : p_(p),
      scale_(p < 1.0f ? 1.0f / (1.0f - p) : 1.0f),
      backend_(backend),
      seed_(seed),
      last_mask_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation). p == 1 would make scale diverge.
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("DropoutModule: p must be in [0, 1)");
    }
}

Tensor DropoutModule::forward_impl(const Tensor& input) {
    // Device-generic (GPU-native-kernels Mission 1b): mask drawn and applied on the input's
    // device by one fused kernel.
    last_shape_ = input.shape();
    has_forwarded_ = true;

    if (!is_training() || p_ == 0.0f) {
        last_was_identity_ = true;
        return Tensor(input);
    }

    const auto n = static_cast<size_t>(input.numel());
    Tensor output(input.shape(), backend_, input.device());
    last_mask_ = Tensor(input.shape(), backend_, input.device());
    backend_->dropout_forward(input.data(), output.data(), last_mask_.data(), n, p_, scale_, seed_, draws_);
    draws_ += n;
    last_was_identity_ = false;
    return output;
}

Tensor DropoutModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("DropoutModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_shape_) {
        throw std::invalid_argument("DropoutModule::backward: grad_output must match the cached forward shape");
    }
    if (last_was_identity_) {
        return Tensor(grad_output);
    }
    // grad * mask * scale, in that order (as the original host loop multiplied).
    const auto n = static_cast<size_t>(grad_output.numel());
    Tensor grad_input(last_shape_, backend_, grad_output.device());
    backend_->mul(grad_output.data(), last_mask_.data(), grad_input.data(), n);
    backend_->axpby(scale_, grad_input.data(), 0.0f, grad_input.data(), grad_input.data(), n);
    return grad_input;
}

Tensor DropoutModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("DropoutModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_shape_) {
        throw std::invalid_argument(
            "DropoutModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
