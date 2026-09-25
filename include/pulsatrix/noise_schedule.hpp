/** @file noise_schedule.hpp
 *  @brief DDPM linear noise schedule: precomputed beta/alpha/alpha_bar, forward noising and
 *         reverse sampling steps.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The DDPM (Ho et al. 2020, arXiv:2006.11239) linear variance schedule plus the two
 *        tensor operations defined directly on top of it -- closed-form forward noising and
 *        the reverse (sampling) step.
 *
 * `beta_t` is linearly spaced over `t = 1 .. T` from `beta_start` to `beta_end` (the paper's
 * own choice of schedule), with `alpha_t = 1 - beta_t` and
 * `alpha_bar_t = prod_{s=1..t} alpha_s` precomputed once at construction.
 *
 * @note **1-indexed timesteps.** `t` runs over `[1, T]`, matching the published math
 *       verbatim, not the 0-based indexing of the underlying storage. Every accessor takes
 *       the published `t` and does the offset internally; nothing outside this class ever
 *       sees the 0-based index.
 * @note **Not a Module, and has no parameters.** Nothing here is learned and nothing here
 *       has a backward pass of its own -- the schedule is a fixed, deterministic function of
 *       its three constructor arguments, precomputed at construction in the same spirit as
 *       RoPEModule's fixed non-learned rotation angles. The training objective a diffusion
 *       model is actually optimized against is a plain MSELoss between the true and
 *       predicted noise (`L_simple`), which needs no new loss class; the gradient path runs
 *       entirely through the caller's `epsilon_theta` network, never through this object.
 * @note **No LRP rule, nothing to stub.** Diffusion's research spike found no credible
 *       native LRP rule (iterative stochastic denoising has no single conserved relevance
 *       seed per timestep) -- and since this is not a Module, there is no
 *       `propagate_relevance` to stub either. Same disposition as Reparameterize /
 *       KLDivergenceLoss; see the campaign's Phase 5 Amendment (2026-09-23).
 * @note **`epsilon` and `z` are caller-supplied, never sampled internally.** Identical
 *       convention to Reparameterize's `epsilon`: it keeps both operations deterministic and
 *       therefore directly testable against hand-computed values. Wiring real Gaussian
 *       sampling in is a demo's job, not this class's.
 */
class NoiseSchedule {
public:
    /**
     * @brief Precomputes the linear beta schedule and its alpha / alpha_bar derivatives.
     * @param num_timesteps Number of diffusion steps T. Must be positive.
     * @param beta_start Variance at t = 1.
     * @param beta_end Variance at t = T. Must be strictly greater than beta_start.
     * @throws std::invalid_argument if num_timesteps <= 0 or beta_start >= beta_end --
     *         external boundary (constructor arguments can originate from the Python
     *         bindings with no upstream validation), same convention as every module
     *         constructor's argument checks.
     * @note T == 1 is a degenerate but legal schedule: there is no interval to interpolate
     *       across, so `beta_1 = beta_start` (using the linear formula's own t = 1 endpoint
     *       rather than dividing by T - 1 == 0).
     */
    explicit NoiseSchedule(int64_t num_timesteps, float beta_start = 1e-4f, float beta_end = 0.02f);

    /** @brief Number of diffusion steps T this schedule was built for. */
    [[nodiscard]] int64_t num_timesteps() const { return num_timesteps_; }

    /**
     * @brief Variance beta_t at timestep t.
     * @param t Timestep in [1, T].
     * @throws std::out_of_range if t is outside [1, T].
     */
    [[nodiscard]] float beta(int64_t t) const;

    /**
     * @brief alpha_t = 1 - beta_t.
     * @param t Timestep in [1, T].
     * @throws std::out_of_range if t is outside [1, T].
     */
    [[nodiscard]] float alpha(int64_t t) const;

    /**
     * @brief alpha_bar_t = prod_{s=1..t} alpha_s (the cumulative product).
     * @param t Timestep in [1, T].
     * @throws std::out_of_range if t is outside [1, T].
     */
    [[nodiscard]] float alpha_bar(int64_t t) const;

    /**
     * @brief Closed-form forward (noising) sample:
     *        `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon`.
     *
     * The closed form is why a diffusion model can be trained on a randomly chosen `t`
     * without ever simulating the intermediate steps 1..t-1.
     *
     * @param x0 Clean data tensor.
     * @param epsilon Caller-supplied noise, nominally ~ N(0, I). Must match x0's shape.
     * @param t Timestep in [1, T].
     * @return The noised tensor x_t, same shape as x0.
     * @throws std::invalid_argument if x0 and epsilon have different shapes -- external
     *         boundary, same classification as MSELoss::forward's shape check.
     * @throws std::out_of_range if t is outside [1, T].
     * @note Not backend-generic -- a raw host loop dereferencing Tensor::data() directly
     *       (a scaled two-tensor combination has no DeviceBackend elementwise primitive).
     *       PULSATRIX_ASSERT(device() == DeviceType::Cpu) on both inputs guards against silent UB
     *       on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
     *       mission_host_loop_guards.md. Do not remove this guard without actually
     *       retrofitting the method to route through DeviceBackend.
     */
    [[nodiscard]] Tensor add_noise(const Tensor& x0, const Tensor& epsilon, int64_t t) const;

    /**
     * @brief One reverse (sampling) step, `x_{t-1} = mean_term + sqrt(beta_t) * z` where the
     *        mean term is `(1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta)`.
     *
     * @param x_t Current noisy sample.
     * @param predicted_epsilon The network's noise prediction at this timestep. Must match
     *        x_t's shape.
     * @param z Caller-supplied noise, nominally ~ N(0, I) for t > 1 and exactly zero at
     *        t == 1 (the final step is deterministic in the published algorithm). Must match
     *        x_t's shape. **Zeroing z at t == 1 is the caller's responsibility and is
     *        deliberately not enforced here** -- consistent with every other sampled input
     *        in this mission being a caller-supplied deterministic value rather than
     *        something this class decides.
     * @param t Timestep in [1, T].
     * @return x_{t-1}, same shape as x_t.
     * @throws std::invalid_argument if the three shapes don't all match.
     * @throws std::out_of_range if t is outside [1, T].
     * @note Same raw-host-loop device guard rationale as add_noise().
     */
    [[nodiscard]] Tensor denoise_step(const Tensor& x_t, const Tensor& predicted_epsilon, const Tensor& z,
                                      int64_t t) const;

private:
    /**
     * @brief Converts a published 1-based timestep to a storage index, validating the range.
     * @throws std::out_of_range if t is outside [1, T].
     */
    [[nodiscard]] size_t index_of(int64_t t, const char* context) const;

    int64_t num_timesteps_;
    std::vector<float> beta_;       ///< beta_[i] is beta_{i+1}.
    std::vector<float> alpha_;      ///< alpha_[i] is alpha_{i+1}.
    std::vector<float> alpha_bar_;  ///< alpha_bar_[i] is alpha_bar_{i+1}.
};

}  // namespace pulsatrix
