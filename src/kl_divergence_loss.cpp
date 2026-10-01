#include "pulsatrix/kl_divergence_loss.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

KLDivergenceLoss::KLDivergenceLoss(DeviceBackend* backend)
    : backend_(backend), last_mu_(Shape({0}), backend), last_log_sigma_(Shape({0}), backend) {}

float KLDivergenceLoss::forward(const Tensor& mu, const Tensor& log_sigma) {
    // Dereferences Tensor::data() directly in a raw host loop (exp() has no backend
    // primitive) -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(mu);
    PULSATRIX_REQUIRE_HOST(log_sigma);

    if (mu.rank() != 2) {
        throw std::invalid_argument("KLDivergenceLoss::forward: mu must be rank-2 (N, latent_dim)");
    }
    if (!(mu.shape() == log_sigma.shape())) {
        throw std::invalid_argument("KLDivergenceLoss::forward: mu and log_sigma must have the same shape");
    }

    last_mu_ = mu;
    last_log_sigma_ = log_sigma;
    has_forwarded_ = true;

    // Summed over every element, then divided by N: the per-example sum over latent_dim
    // and the batch mean collapse into one pass because the shape is exactly (N, latent_dim).
    const int64_t batch = mu.shape().dim(0);
    float total = 0.0f;
    for (int64_t i = 0; i < mu.numel(); ++i) {
        const float m = mu.data()[i];
        const float ls = log_sigma.data()[i];
        total += m * m + std::exp(2.0f * ls) - 2.0f * ls - 1.0f;
    }
    return 0.5f * total / static_cast<float>(batch);
}

ReparamGrad KLDivergenceLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("KLDivergenceLoss::backward called before forward");
    }

    const float inv_batch = 1.0f / static_cast<float>(last_mu_.shape().dim(0));
    Tensor grad_mu(last_mu_.shape(), backend_);
    Tensor grad_log_sigma(last_mu_.shape(), backend_);
    for (int64_t i = 0; i < last_mu_.numel(); ++i) {
        // d/dmu of 0.5*mu^2 is mu; d/dlog_sigma of 0.5*(exp(2*ls) - 2*ls) is exp(2*ls) - 1.
        grad_mu.data()[i] = last_mu_.data()[i] * inv_batch;
        grad_log_sigma.data()[i] = (std::exp(2.0f * last_log_sigma_.data()[i]) - 1.0f) * inv_batch;
    }
    return ReparamGrad{std::move(grad_mu), std::move(grad_log_sigma)};
}

}  // namespace pulsatrix
