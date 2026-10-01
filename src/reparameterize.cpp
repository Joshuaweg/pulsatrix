#include "pulsatrix/reparameterize.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

Reparameterize::Reparameterize(DeviceBackend* backend)
    : backend_(backend), last_log_sigma_(Shape({0}), backend), last_epsilon_(Shape({0}), backend) {}

Tensor Reparameterize::forward(const Tensor& mu, const Tensor& log_sigma, const Tensor& epsilon) {
    if (!(mu.shape() == log_sigma.shape()) || !(mu.shape() == epsilon.shape())) {
        throw std::invalid_argument("Reparameterize::forward: mu, log_sigma and epsilon must all have the same shape");
    }
    if (mu.device() != log_sigma.device() || mu.device() != epsilon.device()) {
        throw std::invalid_argument("Reparameterize::forward: mu, log_sigma and epsilon must be on the same device");
    }
    last_log_sigma_ = log_sigma;
    last_epsilon_ = epsilon;
    has_forwarded_ = true;

    // z = mu + exp(ls) * eps (GPU-native-kernels Mission 1b), same rounding order as before.
    const auto n = static_cast<size_t>(mu.numel());
    Tensor z(mu.shape(), backend_, mu.device());
    backend_->elementwise(ElementwiseOp::Exp, log_sigma.data(), z.data(), n);
    backend_->mul(z.data(), epsilon.data(), z.data(), n);
    backend_->add(mu.data(), z.data(), z.data(), n);
    return z;
}

ReparamGrad Reparameterize::backward(const Tensor& grad_z) const {
    if (!has_forwarded_) {
        throw std::logic_error("Reparameterize::backward called before forward");
    }
    if (!(grad_z.shape() == last_log_sigma_.shape())) {
        throw std::invalid_argument("Reparameterize::backward: grad_z shape must match the cached forward shape");
    }
    if (grad_z.device() != last_log_sigma_.device()) {
        throw std::invalid_argument("Reparameterize::backward: grad_z must be on the forward pass's device");
    }
    // grad_mu = grad_z; grad_log_sigma = (grad_z * exp(ls)) * eps.
    const auto n = static_cast<size_t>(grad_z.numel());
    Tensor grad_mu(grad_z);
    Tensor grad_log_sigma(grad_z.shape(), backend_, grad_z.device());
    backend_->elementwise(ElementwiseOp::Exp, last_log_sigma_.data(), grad_log_sigma.data(), n);
    backend_->mul(grad_z.data(), grad_log_sigma.data(), grad_log_sigma.data(), n);
    backend_->mul(grad_log_sigma.data(), last_epsilon_.data(), grad_log_sigma.data(), n);
    return ReparamGrad{std::move(grad_mu), std::move(grad_log_sigma)};
}

}  // namespace pulsatrix
