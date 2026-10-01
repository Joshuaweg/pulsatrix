/** @file ppo_clipped_loss.hpp
 *  @brief PPO's clipped surrogate objective (Schulman et al. 2017) over a batched rollout.
 *  @ingroup rl
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief loss = mean_b( -min( r_b * A_b, clamp(r_b, 1-eps, 1+eps) * A_b ) ), where
 *        `r_b = pi_new(a_b|s_b) / pi_old(a_b|s_b)` -- the clipped surrogate objective of
 *        Schulman et al. 2017 (arXiv:1707.06347), negated so that minimizing it maximizes the
 *        objective the paper states.
 *
 * The ratio compares the *current* policy's probability for the action against the probability
 * assigned by the policy that actually *collected* the data. That comparison is what lets PPO
 * take several gradient epochs over one rollout: the ratio measures how far the policy has
 * drifted from the data-collecting one, and the clip refuses to pay for drift past
 * `1 +/- clip_epsilon`. At the first minibatch of the first epoch the policy has not moved yet,
 * every ratio is exactly 1, and the loss degenerates to an advantage-weighted negative
 * log-likelihood -- PolicyGradientLoss's own surrogate with `advantages` in place of `returns`.
 * That degeneracy is directly tested.
 *
 * @note This is a *surrogate*: its value has no interpretation as an error, and a negative
 *       advantage legitimately makes a step's contribution negative. Same disposition as
 *       PolicyGradientLoss.
 * @note The `min` is pessimistic, not symmetric: the clip only bites in the direction that
 *       would *improve* the surrogate. A row whose ratio has run past the trust region in the
 *       direction that already over-rewards it contributes a flat term, so its gradient is
 *       **exactly zero** -- see backward(). A row that has run past the trust region in the
 *       direction that *hurts* the objective keeps its full, unclipped gradient, because PPO
 *       does want to pull such a step back.
 * @note Like PolicyGradientLoss, this consumes raw logits and computes the numerically stable
 *       row-wise softmax itself (subtract the row max before exponentiating), for the reason
 *       CrossEntropyLoss fuses softmax into NLL rather than composing two modules.
 * @note Advantages are consumed exactly as given. PPO implementations commonly standardize
 *       advantages per minibatch, but that is a training-loop decision; baking it in here would
 *       silently change the estimator every caller gets -- the same scope cut RolloutBuffer and
 *       PolicyGradientLoss already made.
 * @note Not a Module subclass, for MSELoss/DQNLoss/PolicyGradientLoss's reason: losses are the
 *       seed point relevance/gradient propagation starts *from*, not something a
 *       propagate_relevance rule is defined for.
 * @note backward() takes no incoming gradient: like every loss here, this is a graph root. It
 *       returns the gradient w.r.t. `new_logits` only -- `advantages` and `old_log_probs` are
 *       treated as constants, which they are: the advantage comes from a separately-trained
 *       critic (and is detached in every reference implementation), and the old log-probability
 *       belongs to a policy snapshot that is by definition not being differentiated.
 */
class PPOClippedLoss {
public:
    /**
     * @brief Constructs a PPO clipped-surrogate loss.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     */
    explicit PPOClippedLoss(DeviceBackend* backend);

    /**
     * @brief Computes the clipped surrogate loss and caches what backward() needs.
     * @param new_logits The *current* policy network's raw logits over the rollout, shape
     *        (N, action_dim), N >= 1, action_dim >= 1. Raw logits, not probabilities.
     * @param actions Float-encoded action indices, shape (N, 1) -- the same discrete-action
     *        encoding Environment::step() accepts and CategoricalPolicyAgent::act() produces,
     *        identical to PolicyGradientLoss's.
     * @param old_log_probs Log-probability the *data-collecting* policy assigned to each
     *        action, shape (N, 1) -- exactly what RolloutBatch::log_probs holds.
     * @param advantages Per-step advantage estimate, shape (N, 1) -- typically
     *        ComputeGAE()'s `advantages`.
     * @param clip_epsilon Trust-region half-width. Must be > 0 and < 1. The paper's value is
     *        0.2. Validated as `(0, 1)`: a non-positive epsilon would clamp the ratio to the
     *        single point 1 (killing every gradient), and an epsilon >= 1 would admit a lower
     *        clip bound of <= 0, which no probability ratio can reach -- neither is a
     *        trust region.
     * @return The scalar mean clipped surrogate loss.
     * @throws std::invalid_argument if any tensor has the wrong rank, if the four batch
     *         dimensions disagree, if an encoded action is not within 1e-4 of a whole number,
     *         if a decoded index falls outside [0, action_dim), or if clip_epsilon is outside
     *         (0, 1) -- all external boundaries. The integer tolerance is byte-for-byte
     *         PolicyGradientLoss::forward()'s (itself DQNLoss's, itself CartPoleEnv::step()'s).
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in raw host loops
     *       (a row-wise stabilized softmax and a per-row gather at a data-dependent column have
     *       no DeviceBackend primitive). PULSATRIX_REQUIRE_HOST on all four
     *       inputs guards against silent UB on a CUDA-backed Tensor; see
     *       mission_host_loop_guards.md. Do not remove these guards without actually
     *       retrofitting the method to route through DeviceBackend.
     */
    [[nodiscard]] float forward(const Tensor& new_logits, const Tensor& actions, const Tensor& old_log_probs,
                                const Tensor& advantages, float clip_epsilon);

    /**
     * @brief Gradient w.r.t. `new_logits`:
     *        `-mask[b] * advantages[b,0] * ratio[b] * (1{k == a_b} - p[b,k]) / N`.
     *
     * `mask[b] == 0` exactly when (`advantages[b,0] >= 0` **and** `ratio[b] > 1+eps`) or
     * (`advantages[b,0] < 0` **and** `ratio[b] < 1-eps`); otherwise `mask[b] == 1`. Those two
     * cases are the ones where the row has already exceeded the trust region *in the direction
     * that increases the objective further*, so the `min` selects the clipped branch, which is
     * constant in the ratio and therefore has zero derivative. The zero is structural, not a
     * small residual, and is asserted as exact equality in the tests.
     *
     * @return Gradient tensor, shape (N, action_dim) -- the shape of the logits passed to
     *         forward(), so it feeds straight into the policy network's own backward().
     * @throws std::logic_error if forward() has never been called -- uses the cached state.
     * @note **Every column is written**, including the non-taken actions, for
     *       PolicyGradientLoss::backward()'s reason: softmax is a normalized distribution, so
     *       raising the taken action's probability necessarily takes mass from every other
     *       action, and each of those columns gets a real `+mask * A * ratio * p[b,k] / N`.
     *       Contrast DQNLoss::backward(), whose non-selected columns are exactly 0.0f.
     * @note The sign convention: `(1{k == a_b} - p[b,k])` is `d log p_a / d logit_k`, and the
     *       leading minus is the loss's own negation of the maximized objective. This is the
     *       *negative* of PolicyGradientLoss's `(p - 1{...})` bracket ordering combined with
     *       its own leading sign; the two agree at `ratio == 1` with `advantages` playing
     *       `returns`' role, which is exactly what the ratio-equals-one test cross-checks.
     * @note The `1/N` factor is the batch-mean scaling; N is the number of rollout steps, not
     *       the element count, because the mean is taken over steps -- identical reasoning to
     *       PolicyGradientLoss's and DQNLoss's.
     * @note Dereferences Tensor::data() directly, so it carries its own PULSATRIX_REQUIRE_HOST
     *       guards on the cached state and on the freshly allocated gradient(s). forward()'s
     *       guard covers only the caller's tensors; the gradient is allocated through backend_,
     *       which a GPU backend tags Cuda/Hip (GPU-native-kernels campaign, Mission 0 O4).
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    /** @brief Row-wise softmax probabilities of the last forward()'s logits, (N, action_dim). */
    Tensor last_probs_;
    Tensor last_advantages_;
    /** @brief Per-row probability ratio from the last forward() -- cached rather than
     *         recomputed, since backward() needs the exact value forward() masked against. */
    std::vector<float> last_ratios_;
    /** @brief Per-row clip mask, 0.0f or 1.0f, decided in forward() where `clip_epsilon` is in
     *         scope. Caching the decision rather than the epsilon keeps backward() free of the
     *         region logic and makes the mask directly assertable in tests via the gradient. */
    std::vector<float> last_masks_;
    /** @brief Decoded, already-validated action index per rollout step. */
    std::vector<int64_t> last_action_indices_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
