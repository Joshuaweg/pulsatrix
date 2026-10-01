#include "pulsatrix/relu_module.hpp"

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

ReluModule::ReluModule(DeviceBackend* backend) : ReluModule(backend, backend->device()) {}

ReluModule::ReluModule(DeviceBackend* backend, DeviceType device)
    : backend_(backend), last_input_(Shape({0}), backend, device) {}
// last_input_ starts as a zero-element placeholder -- ReLU has no fixed shape (unlike
// LinearModule's in_features/out_features), so the real shape is only known once
// forward_impl() is first called and reassigns it wholesale.

Tensor ReluModule::forward_impl(const Tensor& input) {
    last_input_ = input;

    // Tagged with input's own device, not a stored module-level device -- ReLU has no
    // parameter Tensor to anchor one, and input.device() is always the correct, current
    // source of truth. See mission_forward_pass_equivalence.md, Objective 3.
    Tensor output(input.shape(), backend_, input.device());
    backend_->elementwise(ElementwiseOp::Relu, input.data(), output.data(), static_cast<size_t>(input.numel()));
    return output;
}

Tensor ReluModule::backward(const Tensor& grad_output) {
    // Device-generic (GPU-native-kernels Mission 1): grad masked where the cached forward
    // input was <= 0.
    Tensor grad_input(grad_output.shape(), backend_, grad_output.device());
    backend_->elementwise_backward(ElementwiseOp::Relu, last_input_.data(), grad_output.data(), grad_input.data(),
                                   static_cast<size_t>(grad_output.numel()));
    return grad_input;
}

Tensor ReluModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
