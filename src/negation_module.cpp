#include "pulsatrix/negation_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

NegationModule::NegationModule(DeviceBackend* backend) : backend_(backend), last_input_(Shape({0}), backend) {}

Tensor NegationModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(input);
    last_input_ = input;
    has_forwarded_ = true;

    Tensor output(input.shape(), backend_, input.device());
    PULSATRIX_REQUIRE_HOST(output);
    for (int64_t i = 0; i < input.numel(); ++i) {
        output.data()[i] = 1.0f - input.data()[i];
    }
    return output;
}

Tensor NegationModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("NegationModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("NegationModule::backward: grad_output must match the cached forward shape");
    }
    // Raw host loop -- not yet backend-generic. See mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(grad_output);

    Tensor grad_input(grad_output.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    for (int64_t i = 0; i < grad_output.numel(); ++i) {
        grad_input.data()[i] = -grad_output.data()[i];
    }
    return grad_input;
}

Tensor NegationModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
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
