#include "pulsatrix/kl_divergence_loss.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

KLDivergenceLoss::KLDivergenceLoss(DeviceBackend* backend)
    : backend_(backend), last_mu_(Shape({0}), backend), last_log_sigma_(Shape({0}), backend) {}

float KLDivergenceLoss::forward(const Tensor& mu, const Tensor& log_sigma) {
    require_device(mu, backend_->device(), "KLDivergenceLoss::forward");
    require_device(log_sigma, backend_->device(), "KLDivergenceLoss::forward");
    if (mu.rank() != 2) {
        throw std::invalid_argument("KLDivergenceLoss::forward: mu must be rank-2 (N, latent_dim)");
    }
    if (!(mu.shape() == log_sigma.shape())) {
        throw std::invalid_argument("KLDivergenceLoss::forward: mu and log_sigma must have the same shape");
    }
    if (mu.device() != log_sigma.device()) {
        throw std::invalid_argument("KLDivergenceLoss::forward: mu and log_sigma must be on the same device");
    }
    last_mu_ = mu;
    last_log_sigma_ = log_sigma;
    has_forwarded_ = true;

    // Device-generic (GPU-native-kernels Mission 1b). Each step is one rounding of the
    // original per-element expression ((mu^2 + exp(2 ls)) - 2 ls) - 1, in the same order, so
    // CPU results are unchanged.
    const auto n = static_cast<size_t>(mu.numel());
    const DeviceType device = mu.device();
    Tensor term(mu.shape(), backend_, device);
    Tensor scratch(mu.shape(), backend_, device);
    backend_->mul(mu.data(), mu.data(), term.data(), n);                                    // mu^2
    backend_->axpby(2.0f, log_sigma.data(), 0.0f, nullptr, scratch.data(), n);             // 2 ls
    backend_->elementwise(ElementwiseOp::Exp, scratch.data(), scratch.data(), n);           // exp(2 ls)
    backend_->add(term.data(), scratch.data(), term.data(), n);                             // + exp(2 ls)
    backend_->axpby(1.0f, term.data(), -2.0f, log_sigma.data(), term.data(), n);           // - 2 ls
    scratch.fill(1.0f);
    backend_->axpby(1.0f, term.data(), -1.0f, scratch.data(), term.data(), n);             // - 1

    const int64_t batch = mu.shape().dim(0);
    return 0.5f * backend_->sum(term.data(), n) / static_cast<float>(batch);
}

ReparamGrad KLDivergenceLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("KLDivergenceLoss::backward called before forward");
    }
    const float inv_batch = 1.0f / static_cast<float>(last_mu_.shape().dim(0));
    const auto n = static_cast<size_t>(last_mu_.numel());
    const DeviceType device = last_mu_.device();

    // grad_mu = mu / N; grad_log_sigma = (exp(2 ls) - 1) / N.
    Tensor grad_mu(last_mu_.shape(), backend_, device);
    backend_->axpby(inv_batch, last_mu_.data(), 0.0f, nullptr, grad_mu.data(), n);

    Tensor grad_log_sigma(last_mu_.shape(), backend_, device);
    Tensor ones(last_mu_.shape(), backend_, device);
    ones.fill(1.0f);
    backend_->axpby(2.0f, last_log_sigma_.data(), 0.0f, nullptr, grad_log_sigma.data(), n);
    backend_->elementwise(ElementwiseOp::Exp, grad_log_sigma.data(), grad_log_sigma.data(), n);
    backend_->axpby(1.0f, grad_log_sigma.data(), -1.0f, ones.data(), grad_log_sigma.data(), n);
    backend_->axpby(inv_batch, grad_log_sigma.data(), 0.0f, nullptr, grad_log_sigma.data(), n);
    return ReparamGrad{std::move(grad_mu), std::move(grad_log_sigma)};
}

}  // namespace pulsatrix
