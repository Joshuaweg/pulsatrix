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
      last_xhat_(Shape({1, num_features > 0 ? num_features : 1}), backend, device) {
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
    const float D = static_cast<float>(num_features_);

    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(input);
    PULSATRIX_REQUIRE_HOST(gamma_);
    PULSATRIX_REQUIRE_HOST(beta_);

    last_input_ = input;
    last_std_.assign(static_cast<size_t>(N), 0.0f);

    Tensor xhat(Shape({N, num_features_}), backend_, gamma_.device());
    Tensor output(Shape({N, num_features_}), backend_, gamma_.device());
    PULSATRIX_REQUIRE_HOST(xhat);
    PULSATRIX_REQUIRE_HOST(output);

    for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t i = 0; i < num_features_; ++i) {
            sum += input.data()[n * num_features_ + i];
        }
        float mu = sum / D;

        float sum_sq_diff = 0.0f;
        for (int64_t i = 0; i < num_features_; ++i) {
            float d = input.data()[n * num_features_ + i] - mu;
            sum_sq_diff += d * d;
        }
        float var = sum_sq_diff / D;
        float std_dev = std::sqrt(var + eps_);
        last_std_[static_cast<size_t>(n)] = std_dev;

        for (int64_t i = 0; i < num_features_; ++i) {
            int64_t idx = n * num_features_ + i;
            float xh = (input.data()[idx] - mu) / std_dev;
            xhat.data()[idx] = xh;
            output.data()[idx] = gamma_.data()[i] * xh + beta_.data()[i];
        }
    }
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
    PULSATRIX_REQUIRE_HOST(grad_output);

    const float D = static_cast<float>(num_features_);

    Tensor local_gamma_grad(Shape({num_features_}), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(local_gamma_grad);
    Tensor local_beta_grad(Shape({num_features_}), backend_);
    local_gamma_grad.fill(0.0f);
    local_beta_grad.fill(0.0f);

    Tensor grad_input(Shape({N, num_features_}), backend_);

    for (int64_t n = 0; n < N; ++n) {
        const float std_dev = last_std_[static_cast<size_t>(n)];

        // dL/dxhat_{n,i} = dL/dy_{n,i} * gamma_i
        std::vector<float> grad_xhat(static_cast<size_t>(num_features_));
        float sum_grad_xhat = 0.0f;
        float sum_grad_xhat_xhat = 0.0f;
        for (int64_t i = 0; i < num_features_; ++i) {
            int64_t idx = n * num_features_ + i;
            float g = grad_output.data()[idx] * gamma_.data()[i];
            grad_xhat[static_cast<size_t>(i)] = g;
            sum_grad_xhat += g;
            sum_grad_xhat_xhat += g * last_xhat_.data()[idx];

            // dL/dgamma_i = sum over batch of dL/dy_{n,i} * xhat_{n,i}
            local_gamma_grad.data()[i] += grad_output.data()[idx] * last_xhat_.data()[idx];
            // dL/dbeta_i = sum over batch of dL/dy_{n,i}
            local_beta_grad.data()[i] += grad_output.data()[idx];
        }

        for (int64_t i = 0; i < num_features_; ++i) {
            int64_t idx = n * num_features_ + i;
            // dL/dx_{n,i} = (1/(D*std_n)) * [D*grad_xhat_i - sum(grad_xhat) - xhat_i*sum(grad_xhat*xhat)]
            grad_input.data()[idx] =
                (D * grad_xhat[static_cast<size_t>(i)] - sum_grad_xhat - last_xhat_.data()[idx] * sum_grad_xhat_xhat) /
                (D * std_dev);
        }
    }
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
