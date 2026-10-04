#include "pulsatrix/rms_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

RMSNormModule::RMSNormModule(int64_t num_features, DeviceBackend* backend)
    : RMSNormModule(num_features, backend, backend->device()) {}

RMSNormModule::RMSNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device, float eps)
    : num_features_(num_features),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      gamma_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_input_(Shape({1, num_features > 0 ? num_features : 1}), backend, device),
      last_rms_(Shape({1}), backend, device) {
    // External boundary (constructor arguments can originate from Phase 5's Python
    // bindings with no upstream validation) -- a non-positive num_features would make
    // every subsequent Shape construction either empty or nonsensical.
    if (num_features <= 0) {
        throw std::invalid_argument("RMSNormModule: num_features must be positive");
    }
}

void RMSNormModule::set_gamma(std::initializer_list<float> values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void RMSNormModule::set_gamma(const std::vector<float>& values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

Tensor RMSNormModule::forward_impl(const Tensor& input) {
    if (input.rank() != 2 || input.shape().dim(1) != num_features_) {
        throw std::invalid_argument("RMSNormModule::forward: input must be rank-2 (N, num_features)");
    }
    const int64_t N = input.shape().dim(0);

    // Device-generic (GPU-native-kernels Mission 2): one fused row kernel.
    last_input_ = input;
    const DeviceType device = gamma_.device();
    last_rms_ = Tensor(Shape({N}), backend_, device);
    Tensor output(Shape({N, num_features_}), backend_, device);
    backend_->rms_norm_forward(input.data(), gamma_.data(), output.data(), last_rms_.data(), static_cast<size_t>(N),
                               static_cast<size_t>(num_features_), eps_);

    has_forwarded_ = true;
    return output;
}

Tensor RMSNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RMSNormModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    if (grad_output.rank() != 2 || grad_output.shape().dim(1) != num_features_ || grad_output.shape().dim(0) != N) {
        throw std::invalid_argument(
            "RMSNormModule::backward: grad_output must be rank-2 (N, num_features) matching the cached batch size");
    }

    // Device-generic (GPU-native-kernels Mission 2): the row kernel also emits each row's
    // gamma-gradient terms; column_sums reduces them in row order, as the original loop did.
    const auto rows = static_cast<size_t>(N);
    const auto cols = static_cast<size_t>(num_features_);
    const DeviceType device = gamma_.device();
    Tensor grad_input(Shape({N, num_features_}), backend_, device);
    Tensor gamma_terms(Shape({N, num_features_}), backend_, device);
    backend_->rms_norm_backward(grad_output.data(), gamma_.data(), last_input_.data(), last_rms_.data(),
                                grad_input.data(), gamma_terms.data(), rows, cols);
    Tensor local_gamma_grad(Shape({num_features_}), backend_, device);
    backend_->column_sums(gamma_terms.data(), local_gamma_grad.data(), rows, cols, 0.0f);
    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (gamma_.requires_grad()) {
        gamma_grad_.accumulate(local_gamma_grad);
    }
    return grad_input;
}

Tensor RMSNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("RMSNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != last_input_.numel()) {
        throw std::invalid_argument(
            "RMSNormModule::propagate_relevance: relevance_out size must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
