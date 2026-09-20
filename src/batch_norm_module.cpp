#include "exai/batch_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

namespace {
int64_t safe_channels(int64_t num_channels) { return num_channels > 0 ? num_channels : 1; }
}  // namespace

BatchNormModule::BatchNormModule(int64_t num_channels, DeviceBackend* backend, DeviceType device, float eps)
    : num_channels_(num_channels),
      eps_(eps),
      backend_(backend),
      gamma_(Shape({safe_channels(num_channels)}), backend, device),
      beta_(Shape({safe_channels(num_channels)}), backend, device),
      gamma_grad_(Shape({safe_channels(num_channels)}), backend, device),
      beta_grad_(Shape({safe_channels(num_channels)}), backend, device),
      last_input_(Shape({1, safe_channels(num_channels), 1, 1}), backend, device),
      last_xhat_(Shape({1, safe_channels(num_channels), 1, 1}), backend, device) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (num_channels <= 0) {
        throw std::invalid_argument("BatchNormModule: num_channels must be positive");
    }
}

void BatchNormModule::set_gamma(std::initializer_list<float> values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void BatchNormModule::set_beta(std::initializer_list<float> values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

void BatchNormModule::set_gamma(const std::vector<float>& values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void BatchNormModule::set_beta(const std::vector<float>& values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

Tensor BatchNormModule::forward_impl(const Tensor& input) {
    // External boundary: input can originate from Phase 5's Python bindings with no
    // upstream validation -- matches Conv2DModule/GroupNormModule's identical check,
    // shape generalized to (N, num_channels, H, W).
    if (input.rank() != 4 || input.shape().dim(1) != num_channels_) {
        throw std::invalid_argument("BatchNormModule::forward: input must be rank-4 (N, num_channels, H, W)");
    }

    const int64_t N = input.shape().dim(0);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    const int64_t spatial = H * W;
    const int64_t per_example = num_channels_ * spatial;
    const int64_t M = N * spatial;  // elements per channel, across the whole batch

    last_input_ = input;
    last_std_.assign(static_cast<size_t>(num_channels_), 0.0f);

    Tensor xhat(input.shape(), backend_, gamma_.device());
    Tensor output(input.shape(), backend_, gamma_.device());

    for (int64_t c = 0; c < num_channels_; ++c) {
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                sum += input.data()[n * per_example + c * spatial + s];
            }
        }
        float mu = sum / static_cast<float>(M);

        float sum_sq_diff = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                float d = input.data()[n * per_example + c * spatial + s] - mu;
                sum_sq_diff += d * d;
            }
        }
        float var = sum_sq_diff / static_cast<float>(M);
        float std_dev = std::sqrt(var + eps_);
        last_std_[static_cast<size_t>(c)] = std_dev;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = n * per_example + c * spatial + s;
                float xh = (input.data()[idx] - mu) / std_dev;
                xhat.data()[idx] = xh;
                output.data()[idx] = gamma_.data()[c] * xh + beta_.data()[c];
            }
        }
    }
    last_xhat_ = xhat;

    has_forwarded_ = true;
    return output;
}

Tensor BatchNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("BatchNormModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("BatchNormModule::backward: grad_output must match the cached forward shape");
    }
    // Not yet backend-generic -- raw host loop below. See every existing Module
    // subclass's identical Phase 1.5 scope decision.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    const int64_t N = last_input_.shape().dim(0);
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t spatial = H * W;
    const int64_t per_example = num_channels_ * spatial;
    const int64_t M = N * spatial;

    Tensor local_gamma_grad(gamma_.shape(), backend_);
    Tensor local_beta_grad(beta_.shape(), backend_);
    Tensor grad_input(last_input_.shape(), backend_);

    for (int64_t c = 0; c < num_channels_; ++c) {
        const float std_c = last_std_[static_cast<size_t>(c)];

        // dL/dgamma_c = sum over (n,h,w) of dL/dy * xhat ; dL/dbeta_c likewise of dL/dy
        float gsum = 0.0f;
        float bsum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = n * per_example + c * spatial + s;
                gsum += grad_output.data()[idx] * last_xhat_.data()[idx];
                bsum += grad_output.data()[idx];
            }
        }
        local_gamma_grad.data()[c] = gsum;
        local_beta_grad.data()[c] = bsum;

        // dL/dxhat_{n,h,w} = dL/dy_{n,h,w} * gamma_c
        float sum_grad_xhat = 0.0f;
        float sum_grad_xhat_xhat = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = n * per_example + c * spatial + s;
                float gxh = grad_output.data()[idx] * gamma_.data()[c];
                sum_grad_xhat += gxh;
                sum_grad_xhat_xhat += gxh * last_xhat_.data()[idx];
            }
        }

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = n * per_example + c * spatial + s;
                float gxh = grad_output.data()[idx] * gamma_.data()[c];
                grad_input.data()[idx] = (static_cast<float>(M) * gxh - sum_grad_xhat -
                                           last_xhat_.data()[idx] * sum_grad_xhat_xhat) /
                                          (static_cast<float>(M) * std_c);
            }
        }
    }
    gamma_grad_.accumulate(local_gamma_grad);
    beta_grad_.accumulate(local_beta_grad);

    return grad_input;
}

Tensor BatchNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("BatchNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != last_input_.numel()) {
        throw std::invalid_argument(
            "BatchNormModule::propagate_relevance: relevance_out size must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace exai
