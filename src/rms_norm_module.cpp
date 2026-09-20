#include "exai/rms_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

RMSNormModule::RMSNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device, float eps)
    : num_features_(num_features),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      gamma_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_input_(Shape({num_features > 0 ? num_features : 1}), backend, device) {
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
    // External boundary: input can originate from Phase 5's Python bindings with no
    // upstream validation -- matches LinearModule::forward_impl's identical check.
    if (input.rank() != 1 || input.numel() != num_features_) {
        throw std::invalid_argument("RMSNormModule::forward: input must be rank-1 with num_features elements");
    }

    last_input_ = input;

    float sum_sq = 0.0f;
    for (int64_t i = 0; i < num_features_; ++i) {
        float xi = input.data()[i];
        sum_sq += xi * xi;
    }
    float ms = sum_sq / static_cast<float>(num_features_);
    last_rms_ = std::sqrt(ms + eps_);

    Tensor output(Shape({num_features_}), backend_, gamma_.device());
    for (int64_t i = 0; i < num_features_; ++i) {
        output.data()[i] = gamma_.data()[i] * input.data()[i] / last_rms_;
    }

    has_forwarded_ = true;
    return output;
}

Tensor RMSNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RMSNormModule::backward: called before any forward()");
    }
    if (grad_output.rank() != 1 || grad_output.numel() != num_features_) {
        throw std::invalid_argument("RMSNormModule::backward: grad_output must be rank-1 with num_features elements");
    }
    // Not yet backend-generic -- raw host loop below. See every existing Module
    // subclass's identical Phase 1.5 scope decision.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    const float rms = last_rms_;
    const float D = static_cast<float>(num_features_);

    // dot = sum_i(dL/dy_i * gamma_i * x_i)
    float dot = 0.0f;
    for (int64_t i = 0; i < num_features_; ++i) {
        dot += grad_output.data()[i] * gamma_.data()[i] * last_input_.data()[i];
    }

    Tensor local_gamma_grad(Shape({num_features_}), backend_);
    Tensor grad_input(Shape({num_features_}), backend_);
    for (int64_t i = 0; i < num_features_; ++i) {
        // dL/dgamma_i = dL/dy_i * x_i / rms
        local_gamma_grad.data()[i] = grad_output.data()[i] * last_input_.data()[i] / rms;

        // dL/dx_i = dL/dy_i * gamma_i / rms  -  (x_i / (D * rms^3)) * dot
        grad_input.data()[i] =
            grad_output.data()[i] * gamma_.data()[i] / rms - (last_input_.data()[i] / (D * rms * rms * rms)) * dot;
    }
    gamma_grad_.accumulate(local_gamma_grad);

    return grad_input;
}

Tensor RMSNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("RMSNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != num_features_) {
        throw std::invalid_argument("RMSNormModule::propagate_relevance: relevance_out size must match num_features");
    }
    return Tensor(relevance_out);
}

}  // namespace exai
