#include "pulsatrix/layer_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

LayerNormModule::LayerNormModule(int64_t num_features, DeviceBackend* backend)
    : LayerNormModule(num_features, backend, backend->device()) {}

LayerNormModule::LayerNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device, float eps)
    : num_features_(num_features),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      beta_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      gamma_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      beta_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_input_(Shape({1, num_features > 0 ? num_features : 1}), backend, device),
      last_xhat_(Shape({1, num_features > 0 ? num_features : 1}), backend, device),
      last_std_(Shape({1}), backend, device) {
    if (num_features <= 0) {
        throw std::invalid_argument("LayerNormModule: num_features must be positive");
    }
}

void LayerNormModule::set_gamma(std::initializer_list<float> values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void LayerNormModule::set_beta(std::initializer_list<float> values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

void LayerNormModule::set_gamma(const std::vector<float>& values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void LayerNormModule::set_beta(const std::vector<float>& values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

Tensor LayerNormModule::forward_impl(const Tensor& input) {
    if (input.rank() != 2 || input.shape().dim(1) != num_features_) {
        throw std::invalid_argument("LayerNormModule::forward: input must be rank-2 (N, num_features)");
    }
    const int64_t N = input.shape().dim(0);

    // Device-generic (GPU-native-kernels Mission 2): one fused row kernel computes mean, std,
    // xhat and the affine output per row, in the original host loop's order.
    last_input_ = input;
    const DeviceType device = gamma_.device();
    last_std_ = Tensor(Shape({N}), backend_, device);
    Tensor xhat(Shape({N, num_features_}), backend_, device);
    Tensor output(Shape({N, num_features_}), backend_, device);
    backend_->layer_norm_forward(input.data(), gamma_.data(), beta_.data(), xhat.data(), output.data(),
                                 last_std_.data(), static_cast<size_t>(N), static_cast<size_t>(num_features_), eps_);
    last_xhat_ = xhat;

    has_forwarded_ = true;
    return output;
}

Tensor LayerNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("LayerNormModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    if (grad_output.rank() != 2 || grad_output.shape().dim(1) != num_features_ || grad_output.shape().dim(0) != N) {
        throw std::invalid_argument(
            "LayerNormModule::backward: grad_output must be rank-2 (N, num_features) matching the cached batch "
            "size");
    }

    // Device-generic (GPU-native-kernels Mission 2). Parameter gradients: per-element terms,
    // then column_sums over rows in row order into a zeroed local, then accumulate -- the
    // original loop's exact association.
    const auto rows = static_cast<size_t>(N);
    const auto cols = static_cast<size_t>(num_features_);
    const DeviceType device = gamma_.device();

    Tensor gamma_terms(Shape({N, num_features_}), backend_, device);
    backend_->mul(grad_output.data(), last_xhat_.data(), gamma_terms.data(), rows * cols);
    Tensor local_gamma_grad(Shape({num_features_}), backend_, device);
    backend_->column_sums(gamma_terms.data(), local_gamma_grad.data(), rows, cols, 0.0f);
    Tensor local_beta_grad(Shape({num_features_}), backend_, device);
    backend_->column_sums(grad_output.data(), local_beta_grad.data(), rows, cols, 0.0f);

    Tensor grad_input(Shape({N, num_features_}), backend_, device);
    backend_->layer_norm_backward(grad_output.data(), gamma_.data(), last_xhat_.data(), last_std_.data(),
                                  grad_input.data(), rows, cols);

    gamma_grad_.accumulate(local_gamma_grad);
    beta_grad_.accumulate(local_beta_grad);
    return grad_input;
}

Tensor LayerNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("LayerNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != last_input_.numel()) {
        throw std::invalid_argument(
            "LayerNormModule::propagate_relevance: relevance_out size must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
