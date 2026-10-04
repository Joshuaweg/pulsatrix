#include "pulsatrix/batch_norm_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
int64_t safe_channels(int64_t num_channels) { return num_channels > 0 ? num_channels : 1; }
}  // namespace

BatchNormModule::BatchNormModule(int64_t num_channels, DeviceBackend* backend)
    : BatchNormModule(num_channels, backend, backend->device()) {}

BatchNormModule::BatchNormModule(int64_t num_channels, DeviceBackend* backend, DeviceType device, float eps,
                                 float momentum)
    : num_channels_(num_channels),
      eps_(eps),
      momentum_(momentum),
      backend_(backend),
      gamma_(Shape({safe_channels(num_channels)}), backend, device),
      beta_(Shape({safe_channels(num_channels)}), backend, device),
      gamma_grad_(Shape({safe_channels(num_channels)}), backend, device),
      beta_grad_(Shape({safe_channels(num_channels)}), backend, device),
      last_input_(Shape({1, safe_channels(num_channels), 1, 1}), backend, device),
      last_xhat_(Shape({1, safe_channels(num_channels), 1, 1}), backend, device),
      last_std_(Shape({safe_channels(num_channels)}), backend, device),
      running_mean_(Shape({safe_channels(num_channels)}), backend, device),
      running_var_(Shape({safe_channels(num_channels)}), backend, device) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (num_channels <= 0) {
        throw std::invalid_argument("BatchNormModule: num_channels must be positive");
    }
    if (!(momentum > 0.0f && momentum <= 1.0f)) {
        throw std::invalid_argument("BatchNormModule: momentum must be in (0, 1]");
    }
    running_mean_.fill(0.0f);
    running_var_.fill(1.0f);
}

void BatchNormModule::set_running_mean(const std::vector<float>& values) {
    if (static_cast<int64_t>(values.size()) != num_channels_) {
        throw std::invalid_argument("BatchNormModule::set_running_mean: need one value per channel");
    }
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("BatchNormModule::set_running_mean: values must be finite");
        }
    }
    running_mean_ = Tensor(running_mean_.shape(), backend_, values, running_mean_.device());
}

void BatchNormModule::set_running_var(const std::vector<float>& values) {
    if (static_cast<int64_t>(values.size()) != num_channels_) {
        throw std::invalid_argument("BatchNormModule::set_running_var: need one value per channel");
    }
    for (float v : values) {
        if (!std::isfinite(v) || v < 0.0f) {
            throw std::invalid_argument("BatchNormModule::set_running_var: values must be finite and non-negative");
        }
    }
    running_var_ = Tensor(running_var_.shape(), backend_, values, running_var_.device());
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
    has_forwarded_ = true;
    if (folded_) {
        // BatchNormFold moved this layer's affine map into the preceding Conv2D.
        last_mode_ = Mode::Folded;
        return Tensor(input);
    }

    // Device-generic (GPU-native-kernels Mission 4): one thread per channel, same order.
    Tensor xhat(input.shape(), backend_, gamma_.device());
    Tensor output(input.shape(), backend_, gamma_.device());
    const auto n = static_cast<size_t>(N);
    const auto c = static_cast<size_t>(num_channels_);
    const auto s = static_cast<size_t>(spatial);
    if (is_training()) {
        last_mode_ = Mode::Batch;
        backend_->batch_norm_forward(input.data(), gamma_.data(), beta_.data(), xhat.data(), output.data(),
                                     last_std_.data(), n, c, s, eps_);
        backend_->batch_norm_update_running(input.data(), running_mean_.data(), running_var_.data(), n, c, s,
                                            momentum_);
    } else {
        last_mode_ = Mode::Running;
        backend_->batch_norm_eval_forward(input.data(), gamma_.data(), beta_.data(), running_mean_.data(),
                                          running_var_.data(), xhat.data(), output.data(), last_std_.data(), n, c, s,
                                          eps_);
    }
    last_xhat_ = xhat;
    return output;
}

Tensor BatchNormModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "BatchNormModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("BatchNormModule::backward: called before any forward()");
    }
    if (grad_output.shape() != last_input_.shape()) {
        throw std::invalid_argument("BatchNormModule::backward: grad_output must match the cached forward shape");
    }

    const int64_t N = last_input_.shape().dim(0);
    const int64_t H = last_input_.shape().dim(2);
    const int64_t W = last_input_.shape().dim(3);
    const int64_t spatial = H * W;
    const int64_t per_example = num_channels_ * spatial;
    const int64_t M = N * spatial;

    if (last_mode_ == Mode::Folded) {
        return Tensor(grad_output);
    }

    Tensor local_gamma_grad(gamma_.shape(), backend_, gamma_.device());
    Tensor local_beta_grad(beta_.shape(), backend_, gamma_.device());
    Tensor grad_input(last_input_.shape(), backend_, gamma_.device());
    if (last_mode_ == Mode::Batch) {
        backend_->batch_norm_backward(grad_output.data(), gamma_.data(), last_xhat_.data(), last_std_.data(),
                                      grad_input.data(), local_gamma_grad.data(), local_beta_grad.data(),
                                      static_cast<size_t>(N), static_cast<size_t>(num_channels_),
                                      static_cast<size_t>(spatial));
    } else {
        backend_->batch_norm_eval_backward(grad_output.data(), gamma_.data(), last_xhat_.data(), last_std_.data(),
                                           grad_input.data(), local_gamma_grad.data(), local_beta_grad.data(),
                                           static_cast<size_t>(N), static_cast<size_t>(num_channels_),
                                           static_cast<size_t>(spatial));
    }
    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (gamma_.requires_grad()) {
        gamma_grad_.accumulate(local_gamma_grad);
    }
    if (beta_.requires_grad()) {
        beta_grad_.accumulate(local_beta_grad);
    }

    return grad_input;
}

Tensor BatchNormModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    require_device(relevance_out, *compute_device(), "BatchNormModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("BatchNormModule::propagate_relevance: called before any forward()");
    }
    if (relevance_out.numel() != last_input_.numel()) {
        throw std::invalid_argument(
            "BatchNormModule::propagate_relevance: relevance_out size must match the cached forward shape");
    }
    return Tensor(relevance_out);
}

}  // namespace pulsatrix
