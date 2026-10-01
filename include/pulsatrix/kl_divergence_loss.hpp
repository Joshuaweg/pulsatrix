/** @file kl_divergence_loss.hpp
 *  @brief VAE KL-divergence-to-standard-normal loss term.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/reparameterize.hpp"  // ReparamGrad -- the shared (grad_mu, grad_log_sigma) pair
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Closed-form KL(N(mu, sigma^2) || N(0, I)) for a diagonal Gaussian posterior
 *        (Kingma & Welling 2013, arXiv:1312.6114, Appendix B):
 *          loss = mean_b( 0.5 * sum_d( mu[b,d]^2 + exp(2*log_sigma[b,d]) - 2*log_sigma[b,d] - 1 ) )
 *        i.e. summed over the latent dimension per example, averaged over the batch -- the
 *        standard VAE ELBO normalization, matching how the reconstruction term is typically
 *        summed-per-example/averaged-over-batch too.
 *
 * @note Not a Module subclass, for exactly MSELoss's reason: losses are the seed point
 *       relevance/gradient propagation starts *from*, not something a `propagate_relevance`
 *       rule is defined for. Same shape as MSELoss/CrossEntropyLoss -- forward() computes
 *       and caches, backward() consumes the cache.
 * @note backward() takes no incoming gradient: like MSELoss, this loss is a graph root.
 * @note Pairs with MSELoss (Gaussian decoder reconstruction term) to form the full VAE
 *       objective; no separate reconstruction loss is built for VAE, MSELoss is reused
 *       directly.
 */
class KLDivergenceLoss {
public:
    /**
     * @brief Constructs a KL divergence loss.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     */
    explicit KLDivergenceLoss(DeviceBackend* backend);

    /**
     * @brief Computes the KL term and caches mu/log_sigma for backward().
     * @param mu Posterior mean, shape (N, latent_dim).
     * @param log_sigma Posterior log-standard-deviation. Must match mu's shape.
     * @return The scalar KL value.
     * @throws std::invalid_argument if mu/log_sigma shapes don't match, or either isn't
     *         rank-2 (N, latent_dim) -- external boundary, same classification as
     *         MSELoss::forward's shape check.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
     */
    [[nodiscard]] float forward(const Tensor& mu, const Tensor& log_sigma);

    /**
     * @brief Gradients of the KL term w.r.t. its two inputs:
     *          grad_mu[b,d]        = mu[b,d] / N
     *          grad_log_sigma[b,d] = (exp(2*log_sigma[b,d]) - 1) / N
     *        (N is the batch size; the 0.5 factor cancels against the derivative of the
     *        squared/exponential terms).
     * @return Both gradients, each the shape of the mu passed to forward().
     * @throws std::logic_error if forward() has never been called -- uses the cached
     *         mu/log_sigma.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
     */
    [[nodiscard]] ReparamGrad backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_mu_;
    Tensor last_log_sigma_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
