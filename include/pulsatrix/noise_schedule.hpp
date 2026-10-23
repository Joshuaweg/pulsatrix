/** @file noise_schedule.hpp
 *  @brief DDPM noise schedules (linear and cosine): precomputed beta/alpha/alpha_bar, forward
 *         noising, and DDPM and DDIM reverse steps.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief One reverse step as `x_prev = skip * x_t - network * eps_theta + noise * z` (DIFF-5).
 *        DDPM and DDIM steps both have this form; only the coefficients differ.
 */
struct ReverseStepCoefficients {
    float skip = 0.0f;
    float network = 0.0f;
    float noise = 0.0f;
};

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
 * @note **No LRP rule of its own.** A reverse step is `skip * x_t - network * eps + noise * z`
 *       (ReverseStepCoefficients), and trajectory LRP (trajectory_lrp.hpp, DIFF-5) splits
 *       relevance across those three terms. The schedule only supplies the coefficients.
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

    /**
     * @brief The cosine schedule of Nichol & Dhariwal 2021 (arXiv:2102.09672):
     *        `alpha_bar_t = f(t) / f(0)` with `f(t) = cos^2(((t / T) + offset) / (1 + offset) * pi / 2)`,
     *        and `beta_t = min(1 - alpha_bar_t / alpha_bar_{t-1}, max_beta)`. Computed in double.
     * @throws std::invalid_argument if num_timesteps <= 0, offset < 0, or max_beta is outside (0, 1].
     * @note alpha_bar() then follows the clipped betas, as in the paper's code: only the last steps
     *       are clipped, where f falls to zero.
     */
    [[nodiscard]] static NoiseSchedule Cosine(int64_t num_timesteps, double offset = 0.008, double max_beta = 0.999);

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
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
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

    /**
     * @brief denoise_step()'s coefficients: skip `1/sqrt(alpha_t)`, network
     *        `beta_t / (sqrt(alpha_t) sqrt(1 - alpha_bar_t))`, noise `sqrt(beta_t)`.
     * @throws std::out_of_range if t is outside [1, T].
     */
    [[nodiscard]] ReverseStepCoefficients ddpm_coefficients(int64_t t) const;

    /**
     * @brief A DDIM step (Song et al. 2021, arXiv:2010.02502, eq. 12) from t to an earlier s:
     *        `x_s = sqrt(alpha_bar_s) x0_hat + sqrt(1 - alpha_bar_s - sigma^2) eps + sigma z` with
     *        `x0_hat = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t)` and
     *        `sigma = eta sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t)) sqrt(1 - alpha_bar_t / alpha_bar_s)`.
     *        `s = 0` is the clean sample, `alpha_bar_0 = 1`, where sigma is zero.
     * @param eta 0 is deterministic DDIM; 1 with s = t - 1 has the mean of a DDPM step and the
     *        posterior variance beta_tilde_t as its noise.
     * @throws std::out_of_range if t is outside [1, T] or s outside [0, t).
     * @throws std::invalid_argument if eta < 0.
     */
    [[nodiscard]] ReverseStepCoefficients ddim_coefficients(int64_t t, int64_t s, float eta = 0.0f) const;

    /**
     * @brief Applies ddim_coefficients(t, s, eta): `skip * x_t - network * eps + noise * z`.
     * @throws as ddim_coefficients(), and std::invalid_argument if the shapes or devices differ.
     */
    [[nodiscard]] Tensor ddim_step(const Tensor& x_t, const Tensor& predicted_epsilon, const Tensor& z, int64_t t,
                                   int64_t s, float eta = 0.0f) const;

private:
    struct FromBetas {};
    /** @brief Builds the schedule from its betas, computing alpha and alpha_bar in double. */
    NoiseSchedule(FromBetas, const std::vector<double>& betas);

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
