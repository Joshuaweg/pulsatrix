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
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);

    Tensor grad_input(grad_output.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    for (int64_t i = 0; i < grad_output.numel(); ++i) {
        grad_input.data()[i] = (last_input_.data()[i] > 0.0f) ? grad_output.data()[i] : 0.0f;
    }
    return grad_input;
}

Tensor ReluModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
