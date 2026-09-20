#include "exai/layer_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

LayerNormModule::LayerNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device, float eps)
    : num_features_(num_features),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      beta_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      gamma_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      beta_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_input_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_xhat_(Shape({num_features > 0 ? num_features : 1}), backend, device) {
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
    if (input.rank() != 1 || input.numel() != num_features_) {
        throw std::invalid_argument("LayerNormModule::forward: input must be rank-1 with num_features elements");
    }

    last_input_ = input;

    float sum = 0.0f;
    for (int64_t i = 0; i < num_features_; ++i) {
        sum += input.data()[i];
    }
    const float D = static_cast<float>(num_features_);
    float mu = sum / D;

    float sum_sq_diff = 0.0f;
    for (int64_t i = 0; i < num_features_; ++i) {
        float d = input.data()[i] - mu;
        sum_sq_diff += d * d;
    }
    float var = sum_sq_diff / D;
    float std_dev = std::sqrt(var + eps_);

    last_mu_ = mu;
    last_std_ = std_dev;

    Tensor xhat(Shape({num_features_}), backend_, gamma_.device());
    Tensor output(Shape({num_features_}), backend_, gamma_.device());
    for (int64_t i = 0; i < num_features_; ++i) {
        float xh = (input.data()[i] - mu) / std_dev;
        xhat.data()[i] = xh;
        output.data()[i] = gamma_.data()[i] * xh + beta_.data()[i];
    }
    last_xhat_ = xhat;

    has_forwarded_ = true;
    return output;
}

Tensor LayerNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("LayerNormModule::backward: called before any forward()");
    }
    if (grad_output.rank() != 1 || grad_output.numel() != num_features_) {
        throw std::invalid_argument("LayerNormModule::backward: grad_output must be rank-1 with num_features elements");
    }
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    const float D = static_cast<float>(num_features_);
    const float std_dev = last_std_;

    // dL/dxhat_i = dL/dy_i * gamma_i
    Tensor grad_xhat(Shape({num_features_}), backend_);
    float sum_grad_xhat = 0.0f;
    float sum_grad_xhat_xhat = 0.0f;
    for (int64_t i = 0; i < num_features_; ++i) {
        float g = grad_output.data()[i] * gamma_.data()[i];
        grad_xhat.data()[i] = g;
        sum_grad_xhat += g;
        sum_grad_xhat_xhat += g * last_xhat_.data()[i];
    }

    Tensor local_gamma_grad(Shape({num_features_}), backend_);
    Tensor grad_input(Shape({num_features_}), backend_);
    for (int64_t i = 0; i < num_features_; ++i) {
        // dL/dgamma_i = dL/dy_i * xhat_i ; dL/dbeta_i = dL/dy_i (accumulated below)
        local_gamma_grad.data()[i] = grad_output.data()[i] * last_xhat_.data()[i];

        // dL/dx_i = (1/(D*std)) * [D*grad_xhat_i - sum(grad_xhat) - xhat_i*sum(grad_xhat*xhat)]
        grad_input.data()[i] = (D * grad_xhat.data()[i] - sum_grad_xhat - last_xhat_.data()[i] * sum_grad_xhat_xhat) /
                                (D * std_dev);
    }
    gamma_grad_.accumulate(local_gamma_grad);
    beta_grad_.accumulate(grad_output);

    return grad_input;
}

Tensor LayerNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("LayerNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != num_features_) {
        throw std::invalid_argument(
            "LayerNormModule::propagate_relevance: relevance_out size must match num_features");
    }
    return Tensor(relevance_out);
}

}  // namespace exai
