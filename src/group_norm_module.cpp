#include "pulsatrix/group_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
int64_t safe_channels(int64_t num_channels) { return num_channels > 0 ? num_channels : 1; }
}  // namespace

GroupNormModule::GroupNormModule(int64_t num_groups, int64_t num_channels, DeviceBackend* backend)
    : GroupNormModule(num_groups, num_channels, backend, backend->device()) {}

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
      last_xhat_(Shape({1, safe_channels(num_channels), 1, 1}), backend, device),
      last_group_std_(Shape({1}), backend, device) {
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
    // Device-generic (GPU-native-kernels Mission 4): one thread per (example, group).
    last_group_std_ = Tensor(Shape({N, num_groups_}), backend_, gamma_.device());
    Tensor xhat(input.shape(), backend_, gamma_.device());
    Tensor output(input.shape(), backend_, gamma_.device());
    backend_->group_norm_forward(input.data(), gamma_.data(), beta_.data(), xhat.data(), output.data(),
                                 last_group_std_.data(), static_cast<size_t>(N), static_cast<size_t>(num_channels_),
                                 static_cast<size_t>(spatial), static_cast<size_t>(num_groups_), eps_);
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

    const int64_t N = last_input_.shape().dim(0);
    const int64_t H = last_h_;
    const int64_t W = last_w_;
    const int64_t spatial = H * W;
    const int64_t N_g = group_size_ * spatial;
    const int64_t per_example = num_channels_ * spatial;

    // Device-generic: parameter gradients one thread per channel (examples summed in order),
    // input gradient one thread per (example, group).
    Tensor local_gamma_grad(gamma_.shape(), backend_, gamma_.device());
    Tensor local_beta_grad(beta_.shape(), backend_, gamma_.device());
    Tensor grad_input(last_input_.shape(), backend_, gamma_.device());
    backend_->group_norm_backward(grad_output.data(), gamma_.data(), last_xhat_.data(), last_group_std_.data(),
                                  grad_input.data(), local_gamma_grad.data(), local_beta_grad.data(),
                                  static_cast<size_t>(N), static_cast<size_t>(num_channels_),
                                  static_cast<size_t>(spatial), static_cast<size_t>(num_groups_));
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

}  // namespace pulsatrix
