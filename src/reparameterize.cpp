#include "pulsatrix/reparameterize.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

Reparameterize::Reparameterize(DeviceBackend* backend)
    : backend_(backend), last_log_sigma_(Shape({0}), backend), last_epsilon_(Shape({0}), backend) {}

Tensor Reparameterize::forward(const Tensor& mu, const Tensor& log_sigma, const Tensor& epsilon) {
    // Dereferences Tensor::data() directly in a raw host loop (exp() has no backend
    // primitive) -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_REQUIRE_HOST(mu);
    PULSATRIX_REQUIRE_HOST(log_sigma);
    PULSATRIX_REQUIRE_HOST(epsilon);

    if (!(mu.shape() == log_sigma.shape()) || !(mu.shape() == epsilon.shape())) {
        throw std::invalid_argument("Reparameterize::forward: mu, log_sigma and epsilon must all have the same shape");
    }

    last_log_sigma_ = log_sigma;
    last_epsilon_ = epsilon;
    has_forwarded_ = true;

    Tensor z(mu.shape(), backend_);
    for (int64_t i = 0; i < mu.numel(); ++i) {
        z.data()[i] = mu.data()[i] + std::exp(log_sigma.data()[i]) * epsilon.data()[i];
    }
    return z;
}

ReparamGrad Reparameterize::backward(const Tensor& grad_z) const {
    if (!has_forwarded_) {
        throw std::logic_error("Reparameterize::backward called before forward");
    }
    if (!(grad_z.shape() == last_log_sigma_.shape())) {
        throw std::invalid_argument("Reparameterize::backward: grad_z shape must match the cached forward shape");
    }

    Tensor grad_mu(grad_z.shape(), backend_);
    Tensor grad_log_sigma(grad_z.shape(), backend_);
    for (int64_t i = 0; i < grad_z.numel(); ++i) {
        // dz/dmu is exactly 1, so grad_mu is a straight pass-through of grad_z.
        grad_mu.data()[i] = grad_z.data()[i];
        // dz/dlog_sigma = exp(log_sigma) * epsilon (chain rule through exp), reusing the
        // cached log_sigma/epsilon rather than recomputing them from a re-run forward.
        grad_log_sigma.data()[i] =
            grad_z.data()[i] * std::exp(last_log_sigma_.data()[i]) * last_epsilon_.data()[i];
    }
    return ReparamGrad{std::move(grad_mu), std::move(grad_log_sigma)};
}

}  // namespace pulsatrix
