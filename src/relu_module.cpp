#include "exai/relu_module.hpp"

#include "exai/assert.hpp"

namespace exai {

ReluModule::ReluModule(DeviceBackend* backend)
    : backend_(backend), last_input_(Shape({0}), backend) {}
// last_input_ starts as a zero-element placeholder -- ReLU has no fixed shape (unlike
// LinearModule's in_features/out_features), so the real shape is only known once
// forward_impl() is first called and reassigns it wholesale.

Tensor ReluModule::forward_impl(const Tensor& input) {
    last_input_ = input;

    Tensor output(input.shape(), backend_);
    backend_->elementwise(ElementwiseOp::Relu, input.data(), output.data(), static_cast<size_t>(input.numel()));
    return output;
}

Tensor ReluModule::backward(const Tensor& grad_output) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    Tensor grad_input(grad_output.shape(), backend_);
    for (int64_t i = 0; i < grad_output.numel(); ++i) {
        grad_input.data()[i] = (last_input_.data()[i] > 0.0f) ? grad_output.data()[i] : 0.0f;
    }
    return grad_input;
}

Tensor ReluModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    return Tensor(relevance_out);
}

}  // namespace exai
