#include "pulsatrix/negation_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

NegationModule::NegationModule(DeviceBackend* backend) : backend_(backend), last_input_(Shape({0}), backend) {}

Tensor NegationModule::forward_impl(const Tensor& input) {
    // Device-generic (GPU-native-kernels Mission 1b): 1 - x as (-1)*x + 1*ones, which rounds
    // identically to the original host loop's 1.0f - x.
    last_input_ = input;
    has_forwarded_ = true;

    const auto n = static_cast<size_t>(input.numel());
    Tensor output(input.shape(), backend_, input.device());
    output.fill(1.0f);
    backend_->axpby(-1.0f, input.data(), 1.0f, output.data(), output.data(), n);
    return output;
}

Tensor NegationModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "NegationModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("NegationModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("NegationModule::backward: grad_output must match the cached forward shape");
    }
    Tensor grad_input(grad_output.shape(), backend_, grad_output.device());
    backend_->elementwise(ElementwiseOp::Neg, grad_output.data(), grad_input.data(),
                          static_cast<size_t>(grad_output.numel()));
    return grad_input;
}

Tensor NegationModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    require_device(relevance_out, *compute_device(), "NegationModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("NegationModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.shape() != last_input_.shape()) {
        throw std::invalid_argument(
            "NegationModule::propagate_relevance: relevance_out must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
