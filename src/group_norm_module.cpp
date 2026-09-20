#include "exai/group_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

namespace {
int64_t safe_channels(int64_t num_channels) { return num_channels > 0 ? num_channels : 1; }
}  // namespace

GroupNormModule::GroupNormModule(int64_t num_groups, int64_t num_channels, DeviceBackend* backend,
                                  DeviceType device, float eps)
    : num_groups_(num_groups),
      num_channels_(num_channels),
      group_size_(num_groups > 0 ? safe_channels(num_channels) / num_groups : 1),
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
    if (num_groups <= 0) {
        throw std::invalid_argument("GroupNormModule: num_groups must be positive");
    }
    if (num_channels <= 0) {
        throw std::invalid_argument("GroupNormModule: num_channels must be positive");
    }
    if (num_channels % num_groups != 0) {
        throw std::invalid_argument("GroupNormModule: num_channels must be evenly divisible by num_groups");
    }
}

void GroupNormModule::set_gamma(std::initializer_list<float> values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void GroupNormModule::set_beta(std::initializer_list<float> values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

void GroupNormModule::set_gamma(const std::vector<float>& values) {
    gamma_ = Tensor(gamma_.shape(), backend_, values, gamma_.device());
}

void GroupNormModule::set_beta(const std::vector<float>& values) {
    beta_ = Tensor(beta_.shape(), backend_, values, beta_.device());
}

Tensor GroupNormModule::forward_impl(const Tensor& input) {
    // External boundary: input can originate from Phase 5's Python bindings with no
    // upstream validation -- shape generalized to (N, num_channels, H, W) by
    // campaign_exai_dl_library_batch_dimension_support.
    if (input.rank() != 4 || input.shape().dim(1) != num_channels_) {
        throw std::invalid_argument(
            "GroupNormModule::forward: input must be rank-4 (N, num_channels, H, W)");
    }

    const int64_t N = input.shape().dim(0);
    const int64_t H = input.shape().dim(2);
    const int64_t W = input.shape().dim(3);
    const int64_t spatial = H * W;
    const int64_t N_g = group_size_ * spatial;
    const int64_t per_example = num_channels_ * spatial;

    last_input_ = input;
    last_h_ = H;
    last_w_ = W;
    last_group_std_.assign(static_cast<size_t>(N * num_groups_), 0.0f);

    Tensor xhat(input.shape(), backend_, gamma_.device());
    Tensor output(input.shape(), backend_, gamma_.device());

    for (int64_t n = 0; n < N; ++n) {
        const int64_t base = n * per_example;

        std::vector<float> group_mean(static_cast<size_t>(num_groups_), 0.0f);
        std::vector<float> group_std(static_cast<size_t>(num_groups_), 0.0f);

        for (int64_t g = 0; g < num_groups_; ++g) {
            float sum = 0.0f;
            int64_t c_start = g * group_size_;
            int64_t c_end = c_start + group_size_;
            for (int64_t c = c_start; c < c_end; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    sum += input.data()[base + c * spatial + s];
                }
            }
            float mu = sum / static_cast<float>(N_g);

            float sum_sq_diff = 0.0f;
            for (int64_t c = c_start; c < c_end; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    float d = input.data()[base + c * spatial + s] - mu;
                    sum_sq_diff += d * d;
                }
            }
            float var = sum_sq_diff / static_cast<float>(N_g);
            group_mean[static_cast<size_t>(g)] = mu;
            group_std[static_cast<size_t>(g)] = std::sqrt(var + eps_);
            last_group_std_[static_cast<size_t>(n * num_groups_ + g)] = group_std[static_cast<size_t>(g)];
        }

        for (int64_t c = 0; c < num_channels_; ++c) {
            int64_t g = c / group_size_;
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = base + c * spatial + s;
                float xh = (input.data()[idx] - group_mean[static_cast<size_t>(g)]) /
                           group_std[static_cast<size_t>(g)];
                xhat.data()[idx] = xh;
                output.data()[idx] = gamma_.data()[c] * xh + beta_.data()[c];
            }
        }
    }
    last_xhat_ = xhat;

    has_forwarded_ = true;
    return output;
}

Tensor GroupNormModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("GroupNormModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("GroupNormModule::backward: grad_output must match the cached forward shape");
    }
    // Not yet backend-generic -- raw host loop below. See every existing Module
    // subclass's identical Phase 1.5 scope decision.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    const int64_t N = last_input_.shape().dim(0);
    const int64_t H = last_h_;
    const int64_t W = last_w_;
    const int64_t spatial = H * W;
    const int64_t N_g = group_size_ * spatial;
    const int64_t per_example = num_channels_ * spatial;

    Tensor local_gamma_grad(gamma_.shape(), backend_);
    Tensor local_beta_grad(beta_.shape(), backend_);
    local_gamma_grad.fill(0.0f);
    local_beta_grad.fill(0.0f);

    Tensor grad_input(last_input_.shape(), backend_);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t base = n * per_example;

        // dL/dgamma_c = sum over batch,spatial of dL/dy * xhat ; dL/dbeta_c likewise of dL/dy
        for (int64_t c = 0; c < num_channels_; ++c) {
            float gsum = 0.0f;
            float bsum = 0.0f;
            for (int64_t s = 0; s < spatial; ++s) {
                int64_t idx = base + c * spatial + s;
                gsum += grad_output.data()[idx] * last_xhat_.data()[idx];
                bsum += grad_output.data()[idx];
            }
            local_gamma_grad.data()[c] += gsum;
            local_beta_grad.data()[c] += bsum;
        }

        for (int64_t g = 0; g < num_groups_; ++g) {
            int64_t c_start = g * group_size_;
            int64_t c_end = c_start + group_size_;
            const float std_g = last_group_std_[static_cast<size_t>(n * num_groups_ + g)];

            // dL/dxhat_i = dL/dy_i * gamma_{c(i)}
            float sum_grad_xhat = 0.0f;
            float sum_grad_xhat_xhat = 0.0f;
            for (int64_t c = c_start; c < c_end; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = base + c * spatial + s;
                    float gxh = grad_output.data()[idx] * gamma_.data()[c];
                    sum_grad_xhat += gxh;
                    sum_grad_xhat_xhat += gxh * last_xhat_.data()[idx];
                }
            }

            for (int64_t c = c_start; c < c_end; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    int64_t idx = base + c * spatial + s;
                    float gxh = grad_output.data()[idx] * gamma_.data()[c];
                    grad_input.data()[idx] = (static_cast<float>(N_g) * gxh - sum_grad_xhat -
                                               last_xhat_.data()[idx] * sum_grad_xhat_xhat) /
                                              (static_cast<float>(N_g) * std_g);
                }
            }
        }
    }

    gamma_grad_.accumulate(local_gamma_grad);
    beta_grad_.accumulate(local_beta_grad);

    return grad_input;
}

Tensor GroupNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("GroupNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != last_input_.numel()) {
        throw std::invalid_argument(
            "GroupNormModule::propagate_relevance: relevance_out size must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace exai
