/** @file tanh_gaussian_policy.hpp
 *  @brief SAC's reparameterized, tanh-squashed Gaussian policy sampling (action + log-prob).
 *  @ingroup rl
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The (action, log_prob) pair TanhGaussianPolicy::forward() produces.
 * @note A return-pair struct rather than two out-parameters, mirroring ReparamGrad's
 *       precedent: the two outputs are produced by one indivisible computation (log_prob's
 *       squash-correction term is a function of the very action being returned), so splitting
 *       them across two calls would either duplicate the forward or need a cache-then-fetch
 *       protocol no other class in this codebase uses.
 * @note `action` is (N, action_dim) -- the bounded action handed to the environment.
 *       `log_prob` is (N, 1): the per-sample log-density of the *joint* action, i.e. already
 *       summed over action_dim (a diagonal Gaussian's dimensions are independent, so the
 *       joint log-density is the sum of the per-dimension ones). For action_dim == 1 the sum
 *       is a one-term sum, not a special case.
 */
struct TanhGaussianSample {
    Tensor action;
    Tensor log_prob;
};

/**
 * @brief The (grad_mean, grad_log_std) pair TanhGaussianPolicy::backward() produces -- both
 *        (N, action_dim), the shape of the forward's own `mean`/`log_std`.
 */
struct TanhGaussianGrad {
    Tensor grad_mean;
    Tensor grad_log_std;
};

/**
 * @brief SAC's reparameterized, tanh-squashed Gaussian policy sample (Haarnoja et al. 2018,
 *        arXiv:1801.01290, Appendix C "Enforcing Action Bounds"):
 *        `u = mean + exp(log_std) * epsilon`, `action = tanh(u)`, with the change-of-variables
 *        corrected log-density
 *        `log_prob = sum_d [ -0.5*epsilon^2 - log_std - 0.5*log(2*pi) - log(1 - action^2 + 1e-6) ]`.
 *
 * Every discrete-action policy in this codebase (DQNAgent's greedy pick,
 * CategoricalPolicyAgent's categorical sample) gets away with never differentiating *through*
 * the sampling step -- DQN doesn't backprop through action selection at all, and
 * REINFORCE/A2C/PPO only need the log-density of the action that was taken, never the sample
 * itself. SAC's actor loss does need it: the gradient of a sampled, *continuous* action
 * w.r.t. the policy's own parameters. That is what the reparameterization trick buys, and the
 * tanh squash (plus its log-density correction) is what keeps the action inside (-1, 1).
 *
 * @note Not a Module subclass -- the same reasoning Reparameterize documents. Module::forward
 *       is a single-tensor-in/single-tensor-out contract; this takes three tensors (two
 *       learned, one sampled) and returns two, and would have to be deformed to fit. It
 *       therefore mirrors Reparameterize's/MSELoss's shape instead: forward(...) computes and
 *       caches, backward() consumes the cache.
 * @note Structurally novel in one respect Reparameterize is not: forward() has *two* outputs
 *       and backward() takes *two* incoming gradients (one arriving via the action from the
 *       critic, one arriving via the log-probability from the entropy term), which it combines
 *       into one gradient per learned tensor. No other class here has that shape.
 * @note `epsilon` is caller-supplied, never sampled internally -- this codebase's established
 *       "randomness is a caller-supplied, deterministic-for-testing input" convention
 *       (Reparameterize, NoiseSchedule). That is what makes the backward finite-difference
 *       testable.
 * @note No LRP rule and nothing to stub: this is not a Module. Same disposition as
 *       Reparameterize (see the campaign's Phase 5 Amendment, 2026-09-23) -- stochastic
 *       reparameterization has no credible native LRP rule in the literature.
 */
class TanhGaussianPolicy {
public:
    /**
     * @brief The fixed stabilizer added inside `log(1 - action^2 + eps_stab)`.
     * @note 1e-6, exactly. `tanh` saturates hard: an |u| of ~9 already gives `1 - action^2`
     *       below float epsilon, and `log(0)` is -inf, which would poison both the log-prob
     *       and (via the same denominator in backward) the gradient. The stabilizer bounds the
     *       term at `-log(1e-6) ~= 13.8` instead. It is a documented constant, not a tunable:
     *       changing it changes the log-density this class reports, so a caller comparing
     *       log-probs across runs needs it pinned.
     */
    static constexpr float kLogProbStabilizer = 1e-6f;

    /**
     * @brief Constructs a tanh-Gaussian policy sampling step.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this object.
     */
    explicit TanhGaussianPolicy(DeviceBackend* backend);

    /**
     * @brief Samples `action = tanh(mean + exp(log_std)*epsilon)` and its corrected
     *        log-density, caching what backward() needs.
     * @param mean Gaussian mean, shape (N, action_dim) with N >= 1 and action_dim >= 1.
     * @param log_std Log-standard-deviation. Must match mean's shape.
     * @param epsilon Standard-normal noise, supplied by the caller. Must match mean's shape.
     * @return The squashed action (N, action_dim) and its per-sample log-probability (N, 1).
     * @throws std::invalid_argument if mean is not rank-2 with N >= 1 and action_dim >= 1, or
     *         if the three shapes don't all match -- external boundary, same classification as
     *         Reparameterize::forward's own shape check. The rank-2 requirement is not
     *         incidental: log_prob is summed over the *last* axis and emitted per row, which
     *         is only defined for a (N, action_dim) block.
     * @note The Gaussian log-density's quadratic term is computed as `-0.5*epsilon^2`, not as
     *       `-0.5*((u - mean)/exp(log_std))^2`. Those are identical by construction (that
     *       ratio *is* epsilon -- the reparameterization identity), but the epsilon form is
     *       numerically cleaner and, more importantly, makes visible that this term carries
     *       **zero** direct gradient w.r.t. mean/log_std: it depends on them only through an
     *       identity that holds by construction, not through a live dependency. backward()'s
     *       derivation relies on exactly that.
     * @note Not backend-generic -- exp/tanh/log have no DeviceBackend::elementwise op, so this
     *       is a raw host loop dereferencing Tensor::data() directly.
     *       PULSATRIX_REQUIRE_HOST on all three inputs guards against silent
     *       UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not remove
     *       this guard without actually retrofitting the method to route through DeviceBackend.
     */
    [[nodiscard]] TanhGaussianSample forward(const Tensor& mean, const Tensor& log_std, const Tensor& epsilon);

    /**
     * @brief Gradients w.r.t. mean and log_std, combining the two gradients that arrive at
     *        this step's two outputs.
     *
     * With `a` the cached action, `std = exp(log_std)`, `c = kLogProbStabilizer`:
     * - `d(a)/d(u) = 1 - a^2` (tanh's derivative).
     * - `d(log_prob)/d(u) = 2*a*(1 - a^2) / (1 - a^2 + c)`, from the `-log(1 - a^2 + c)` term
     *   alone -- the `-0.5*epsilon^2` term has no live dependency on u (see forward()'s note),
     *   and the `-log_std` term's dependency is on log_std directly, handled below.
     * - `grad_u = grad_action*(1 - a^2) + grad_log_prob * 2*a*(1 - a^2)/(1 - a^2 + c)`.
     * - `grad_mean = grad_u` (since `d(u)/d(mean) = 1`).
     * - `grad_log_std = grad_u*std*epsilon - grad_log_prob` (since `d(u)/d(log_std) =
     *   std*epsilon`, plus the `-log_std` term's own direct derivative of -1).
     *
     * @param grad_action Gradient w.r.t. `action` -- in SAC, what the critic's `Q(s, a)`
     *        backward pass delivers. Must match the cached (N, action_dim) forward shape.
     * @param grad_log_prob Gradient w.r.t. `log_prob`, **broadcast to (N, action_dim)** rather
     *        than log_prob's own (N, 1). log_prob is a *sum* over action_dim, so every element
     *        of row n receives the identical incoming scalar; asking the caller for the
     *        already-broadcast block keeps this method a single flat loop and keeps both of
     *        backward()'s arguments one shape instead of two. In SAC this is the per-sample
     *        constant `alpha/N` repeated across the row.
     * @return Both gradients, each the shape of the mean passed to forward().
     * @throws std::logic_error if forward() has never been called -- it consumes that call's
     *         cache.
     * @throws std::invalid_argument if either gradient's shape doesn't match the cached
     *         forward shape. Both are genuine external inputs to backward(), the same
     *         classification Reparameterize::backward()'s grad_z argument got.
     * @note Also dereferences Tensor::data() directly, but deliberately not independently
     *       device-guarded: it only reads state forward() already validated before caching,
     *       and forward()'s guard is the only way a non-Cpu tensor could reach that cache -- a
     *       second guard here would be untestable dead code, not a real safety net. Identical
     *       reasoning to Reparameterize::backward()/MSELoss::backward(); see
     *       mission_host_loop_guards.md.
     */
    [[nodiscard]] TanhGaussianGrad backward(const Tensor& grad_action, const Tensor& grad_log_prob) const;

private:
    DeviceBackend* backend_;
    // The cached action (not u) -- every backward term is expressible in a, std and epsilon,
    // and caching a avoids recomputing tanh in backward().
    Tensor last_action_;
    Tensor last_std_;
    Tensor last_epsilon_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
