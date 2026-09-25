#include "pulsatrix/rms_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

RMSNormModule::RMSNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device, float eps)
    : num_features_(num_features),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      gamma_grad_(Shape({num_features > 0 ? num_features : 1}), backend, device),
      last_input_(Shape({1, num_features > 0 ? num_features : 1}), backend, device) {
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
    // upstream validation -- shape generalized to (N, num_features) by
    // campaign_exai_dl_library_batch_dimension_support (matches LinearModule's identical
    // migration).
    if (input.rank() != 2 || input.shape().dim(1) != num_features_) {
        throw std::invalid_argument("RMSNormModule::forward: input must be rank-2 (N, num_features)");
    }
    const int64_t N = input.shape().dim(0);

    last_input_ = input;
    last_rms_.assign(static_cast<size_t>(N), 0.0f);

    Tensor output(Shape({N, num_features_}), backend_, gamma_.device());
    for (int64_t n = 0; n < N; ++n) {
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < num_features_; ++i) {
            float xi = input.data()[n * num_features_ + i];
            sum_sq += xi * xi;
        }
        float ms = sum_sq / static_cast<float>(num_features_);
        float rms = std::sqrt(ms + eps_);
        last_rms_[static_cast<size_t>(n)] = rms;

        for (int64_t i = 0; i < num_features_; ++i) {
            output.data()[n * num_features_ + i] = gamma_.data()[i] * input.data()[n * num_features_ + i] / rms;
        }
    }

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
    // Not yet backend-generic -- raw host loop below. See every existing Module
    // subclass's identical Phase 1.5 scope decision.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    const float D = static_cast<float>(num_features_);

    Tensor local_gamma_grad(Shape({num_features_}), backend_);
    local_gamma_grad.fill(0.0f);
    Tensor grad_input(Shape({N, num_features_}), backend_);

    for (int64_t n = 0; n < N; ++n) {
        const float rms = last_rms_[static_cast<size_t>(n)];

        // dot = sum_i(dL/dy_i * gamma_i * x_i), per row
        float dot = 0.0f;
        for (int64_t i = 0; i < num_features_; ++i) {
            dot += grad_output.data()[n * num_features_ + i] * gamma_.data()[i] * last_input_.data()[n * num_features_ + i];
        }

        for (int64_t i = 0; i < num_features_; ++i) {
            int64_t idx = n * num_features_ + i;
            // dL/dgamma_i = sum over batch of dL/dy_{n,i} * x_{n,i} / rms_n
            local_gamma_grad.data()[i] += grad_output.data()[idx] * last_input_.data()[idx] / rms;

            // dL/dx_{n,i} = dL/dy_{n,i} * gamma_i / rms_n  -  (x_{n,i} / (D * rms_n^3)) * dot_n
            grad_input.data()[idx] =
                grad_output.data()[idx] * gamma_.data()[i] / rms - (last_input_.data()[idx] / (D * rms * rms * rms)) * dot;
        }
    }
    gamma_grad_.accumulate(local_gamma_grad);

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
