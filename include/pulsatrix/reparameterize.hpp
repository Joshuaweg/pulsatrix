/** @file reparameterize.hpp
 *  @brief VAE reparameterization trick: z = mu + exp(log_sigma) * epsilon.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The (grad_mu, grad_log_sigma) pair both VAE building blocks produce.
 * @note Shared by Reparameterize::backward() and KLDivergenceLoss::backward() -- both
 *       differentiate w.r.t. exactly the same two tensors, so duplicating the struct would
 *       mean a caller combining reconstruction and KL gradients (the whole point of the VAE
 *       ELBO) juggling two structurally identical but unrelated types. Declared here rather
 *       than in a new "VAE types" header: one pair struct does not justify a header of its
 *       own, and mission_vae_module.md explicitly rules out a speculative larger shared
 *       header beyond this type.
 */
struct ReparamGrad {
    Tensor grad_mu;
    Tensor grad_log_sigma;
};

/**
 * @brief VAE reparameterization trick (Kingma & Welling 2013, arXiv:1312.6114),
 *        z[b,d] = mu[b,d] + exp(log_sigma[b,d]) * epsilon[b,d].
 *
 * @note Not a Module subclass. Module::forward is a single-tensor-in/single-tensor-out
 *       contract; this operation takes *three* tensors (two learned, one sampled) and would
 *       have to be deformed to fit. It therefore mirrors MSELoss's shape instead --
 *       forward(...) computes and caches, backward() consumes the cache -- which is this
 *       codebase's established pattern for a computation that is genuinely not a layer.
 *       Deliberate design decision (mission_vae_module.md's "Design decision" section), not
 *       an oversight, and independent of the LRP note below.
 * @note No LRP rule. VAE's own research spike found that no credible native rule exists in
 *       the literature: stochastic reparameterization plus the dual (reconstruction + KL)
 *       loss structure breaks the single-deterministic-seed-point assumption every LRP rule
 *       in this codebase relies on. There is nothing to stub either, since this is not a
 *       Module. See the campaign's Phase 5 Amendment (2026-09-23).
 * @note epsilon is caller-supplied, never sampled internally. That keeps this class
 *       deterministic and therefore finite-difference testable; wiring real Gaussian
 *       sampling (e.g. std::normal_distribution) into a demo is separate, out-of-scope
 *       future work -- the
 *       same disposition as RNNModule's h_0 = 0 being a scope cut, not a missing feature.
 */
class Reparameterize {
public:
    /**
     * @brief Constructs a reparameterization step.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this object.
     */
    explicit Reparameterize(DeviceBackend* backend);

    /**
     * @brief Computes z = mu + exp(log_sigma) * epsilon, caching log_sigma/epsilon for backward().
     * @param mu Mean tensor, shape (N, latent_dim).
     * @param log_sigma Log-standard-deviation tensor. Must match mu's shape.
     * @param epsilon Sampled noise, supplied by the caller. Must match mu's shape.
     * @return The sampled latent z, same shape as mu.
     * @throws std::invalid_argument if the three shapes don't all match -- external boundary,
     *         same classification as MSELoss::forward's prediction/target shape check.
     * @note Not yet backend-generic -- exp() has no DeviceBackend::elementwise op, so this
     *       is a raw host loop dereferencing Tensor::data() directly.
     *       PULSATRIX_REQUIRE_HOST on all three inputs guards against
     *       silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] Tensor forward(const Tensor& mu, const Tensor& log_sigma, const Tensor& epsilon);

    /**
     * @brief Gradients w.r.t. mu and log_sigma, given the gradient w.r.t. z.
     *        grad_mu = grad_z; grad_log_sigma = grad_z * exp(log_sigma) * epsilon.
     * @param grad_z Gradient w.r.t. this step's output. Must match the cached shape.
     * @return Both gradients, each the shape of the mu passed to forward().
     * @throws std::logic_error if forward() has never been called -- uses the cached
     *         log_sigma/epsilon.
     * @throws std::invalid_argument if grad_z's shape doesn't match the cached forward shape.
     * @note Dereferences Tensor::data() directly, so it carries its own PULSATRIX_REQUIRE_HOST
     *       guards on the incoming gradient(s), the cached state and the freshly allocated
     *       gradients. forward()'s guard covers only forward()'s own arguments; the gradients
     *       are allocated through backend_, which a GPU backend tags Cuda/Hip
     *       (GPU-native-kernels campaign, Mission 0 O4).
     */
    [[nodiscard]] ReparamGrad backward(const Tensor& grad_z) const;

private:
    DeviceBackend* backend_;
    Tensor last_log_sigma_;
    Tensor last_epsilon_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
