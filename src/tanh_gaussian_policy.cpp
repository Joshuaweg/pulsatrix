#include "pulsatrix/tanh_gaussian_policy.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// 0.5 * log(2*pi), the Gaussian log-density's normalizing constant. Spelled as a literal
// (rather than 0.5*std::log(2*M_PI)) because M_PI is not standard C++ and this value is
// otherwise recomputed identically on every element.
constexpr double kHalfLogTwoPi = 0.91893853320467274178;

}  // namespace

TanhGaussianPolicy::TanhGaussianPolicy(DeviceBackend* backend)
    : backend_(backend),
      last_action_(Shape({0}), backend),
      last_std_(Shape({0}), backend),
      last_epsilon_(Shape({0}), backend) {}

TanhGaussianSample TanhGaussianPolicy::forward(const Tensor& mean, const Tensor& log_std, const Tensor& epsilon) {
    if (mean.rank() != 2 || mean.shape().dim(0) < 1 || mean.shape().dim(1) < 1) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::forward: mean must have shape (N, action_dim) with N >= 1 and action_dim >= 1");
    }
    if (!(mean.shape() == log_std.shape()) || !(mean.shape() == epsilon.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::forward: mean, log_std and epsilon must all have the same shape");
    }
    if (mean.device() != log_std.device() || mean.device() != epsilon.device()) {
        throw std::invalid_argument("TanhGaussianPolicy::forward: mean, log_std and epsilon must be on the same device");
    }

    // Device-generic (GPU-native-kernels Mission 2): one fused row kernel; the row
    // log-probability is still accumulated in double, on the device too.
    const int64_t batch_size = mean.shape().dim(0);
    const int64_t action_dim = mean.shape().dim(1);
    const DeviceType device = mean.device();
    Tensor action(mean.shape(), backend_, device);
    Tensor std_cache(mean.shape(), backend_, device);
    Tensor log_prob(Shape({batch_size, 1}), backend_, device);
    backend_->tanh_gaussian_forward(mean.data(), log_std.data(), epsilon.data(), action.data(), std_cache.data(),
                                    log_prob.data(), static_cast<size_t>(batch_size), static_cast<size_t>(action_dim),
                                    kLogProbStabilizer, kHalfLogTwoPi);

    last_action_ = action;
    last_std_ = std::move(std_cache);
    last_epsilon_ = epsilon;
    has_forwarded_ = true;

    return TanhGaussianSample{std::move(action), std::move(log_prob)};
}

TanhGaussianGrad TanhGaussianPolicy::backward(const Tensor& grad_action, const Tensor& grad_log_prob) const {
    if (!has_forwarded_) {
        throw std::logic_error("TanhGaussianPolicy::backward called before forward");
    }
    if (!(grad_action.shape() == last_action_.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::backward: grad_action shape must match the cached forward shape");
    }
    if (!(grad_log_prob.shape() == last_action_.shape())) {
        throw std::invalid_argument(
            "TanhGaussianPolicy::backward: grad_log_prob shape must match the cached forward shape "
            "(N, action_dim) -- log_prob's own (N, 1) gradient broadcast across the row");
    }
    if (grad_action.device() != last_action_.device() || grad_log_prob.device() != last_action_.device()) {
        throw std::invalid_argument("TanhGaussianPolicy::backward: gradients must be on the forward pass's device");
    }

    Tensor grad_mean(last_action_.shape(), backend_, last_action_.device());
    Tensor grad_log_std(last_action_.shape(), backend_, last_action_.device());
    backend_->tanh_gaussian_backward(last_action_.data(), last_std_.data(), last_epsilon_.data(), grad_action.data(),
                                     grad_log_prob.data(), grad_mean.data(), grad_log_std.data(),
                                     static_cast<size_t>(last_action_.numel()), kLogProbStabilizer);
    return TanhGaussianGrad{std::move(grad_mean), std::move(grad_log_std)};
}

}  // namespace pulsatrix
