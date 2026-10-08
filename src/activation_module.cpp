#include "pulsatrix/activation_module.hpp"

#include <stdexcept>

namespace pulsatrix {

ActivationModule::ActivationModule(ElementwiseOp op, DeviceBackend* backend)
    : op_(op), backend_(backend), last_input_(Shape({0}), backend) {
    if (op == ElementwiseOp::Neg || op == ElementwiseOp::Exp) {
        throw std::invalid_argument("ActivationModule: Neg and Exp aren't activations");
    }
}

Tensor ActivationModule::forward_impl(const Tensor& input) {
    last_input_ = input;
    has_forwarded_ = true;
    Tensor output(input.shape(), backend_, input.device());
    backend_->elementwise(op_, input.data(), output.data(), static_cast<size_t>(input.numel()));
    return output;
}

Tensor ActivationModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "ActivationModule::backward");
    if (!has_forwarded_) throw std::logic_error("ActivationModule::backward: called before any forward()");
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("ActivationModule::backward: grad_output must match the cached forward shape");
    }
    Tensor grad_input(grad_output.shape(), backend_, grad_output.device());
    backend_->elementwise_backward(op_, last_input_.data(), grad_output.data(), grad_input.data(),
                                   static_cast<size_t>(grad_output.numel()));
    return grad_input;
}

Tensor ActivationModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    require_device(relevance_out, *compute_device(), "ActivationModule::propagate_relevance");
    return Tensor(relevance_out);
}

void ActivationModule::release_activations() {
    release_tensor(last_input_);
    has_forwarded_ = false;
}

}  // namespace pulsatrix
