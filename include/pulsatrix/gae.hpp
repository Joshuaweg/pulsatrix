/** @file gae.hpp
 *  @brief Generalized Advantage Estimation (Schulman et al. 2016) over a stored rollout.
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief ComputeGAE()'s two outputs: the advantage estimate per step and the critic's
 *        regression target per step.
 * @note Plain data, no behavior -- the same small named return-struct precedent as ReparamGrad
 *       (reparameterize.hpp), StepResult (environment.hpp) and RolloutBatch
 *       (rollout_buffer.hpp). A std::pair<Tensor, Tensor> would make every call site read
 *       `.first`/`.second` for two tensors that are emphatically not interchangeable.
 * @note Both are (N, 1), row t of each belonging to stored step t -- the same leading
 *       dimension as the rewards/dones/values passed in.
 */
struct GAEResult {
    /** @brief `A_t` -- the GAE(gamma, lambda) advantage estimate, the actor's per-step weight. */
    Tensor advantages;
    /** @brief `A_t + V(s_t)` -- the critic's regression target. Deliberately *not* the raw
     *         Monte-Carlo return RolloutBuffer::compute_returns() produces; see ComputeGAE(). */
    Tensor returns;
};

/**
 * @brief Generalized Advantage Estimation (Schulman et al. 2016, arXiv:1506.02438):
 *        the exponentially-weighted average of k-step advantage estimators, computed in one
 *        reverse pass.
 *
 * Per step:
 * - `delta[t] = rewards[t] + gamma * (1 - dones[t]) * V_next[t] - values[t]`, where
 *   `V_next[t] = values[t+1]` for `t < N-1` and `V_next[N-1] = bootstrap_value`.
 * - `advantages[t] = delta[t] + gamma * lambda * (1 - dones[t]) * advantages[t+1]`, with
 *   `advantages[N] := 0` (nothing exists past the end of the rollout).
 * - `returns[t] = advantages[t] + values[t]`.
 *
 * `lambda` interpolates between the two ends of the bias/variance trade-off: `lambda == 0`
 * collapses the recursion to `advantages[t] == delta[t]`, the single-step TD residual (low
 * variance, biased by whatever the critic gets wrong); `lambda == 1` accumulates every
 * discounted residual to the end of the episode, which telescopes to the Monte-Carlo
 * advantage `sum gamma^k r - V(s_t)` (unbiased, high variance). Both ends are directly tested.
 *
 * @param rewards Immediate rewards, shape (N, 1).
 * @param dones Episode-termination flags as 0.0f/1.0f floats, shape (N, 1) -- the same
 *        encoding ReplayBatch and ComputeDQNTarget use.
 * @param values The critic's value estimate `V(s_t)` for each *stored* step, shape (N, 1).
 * @param bootstrap_value `V` of the state immediately following the last stored step. Supplied
 *        by the caller as a plain scalar rather than read from the buffer: RolloutBuffer stores
 *        no `next_observation` (unlike ReplayBuffer), and the training loop already holds that
 *        observation from its own most recent env.step(). Ignored in effect when the last step
 *        is terminal, since `(1 - dones[N-1])` zeroes it exactly. Conventionally 0.0f when the
 *        rollout ends on a terminal step.
 * @param gamma Discount factor, in [0, 1].
 * @param lambda GAE trace-decay parameter, in [0, 1].
 * @param backend Backend to allocate the results through. Not owned; must outlive them.
 *        Passed explicitly for ComputeDQNTarget()'s reason: Tensor exposes no accessor for the
 *        backend it was built against, and this codebase injects a non-owned DeviceBackend*
 *        rather than consulting a global default.
 * @return The advantages and the critic's targets, both (N, 1).
 * @throws std::invalid_argument if `rewards` is not rank-2 (N, 1) with N >= 1, if `dones` or
 *         `values` is not (N, 1) for that same N, or if `gamma` or `lambda` is outside [0, 1]
 *         -- all external boundaries.
 * @note `returns` is `advantages + values`, **not** the raw discounted return-to-go
 *       RolloutBuffer::compute_returns() computes, and this function never calls that method:
 *       it consumes raw `rewards`/`dones` directly. The GAE identity is the whole point --
 *       regressing the critic onto `A_t + V(s_t)` keeps the critic's target consistent with the
 *       very advantage the actor is being updated with, at the same lambda. Feeding
 *       compute_returns()'s Monte-Carlo return instead would silently train the critic against
 *       a lambda == 1 target while the actor used a different one.
 * @note A `dones[t] == 1` row cuts *both* the bootstrap and the trace recursion exactly -- an
 *       advantage must not propagate backwards across a terminal state, the same hard cut (not
 *       heavy discount) RolloutBuffer::compute_returns()'s reset-at-done performs. The
 *       recurrence shape is that reverse pass's; the recurrence itself is different.
 * @note `dones` is used as a plain multiplier, not thresholded -- 0.0f/1.0f is the documented
 *       encoding, and silently reinterpreting anything else would hide a caller's bug. Same
 *       convention as ComputeDQNTarget().
 * @note A free function rather than a class: a pure reduction with no state to carry, matching
 *       ComputeDQNTarget()/ComputeDoubleDQNTarget()'s own free-function precedent.
 * @note Raw host loop over Tensor::data() (a reverse-order recursion has no DeviceBackend
 *       primitive) -- PULSATRIX_ASSERT(device() == DeviceType::Cpu) guards against silent UB on a
 *       CUDA-backed Tensor; see mission_host_loop_guards.md.
 */
[[nodiscard]] GAEResult ComputeGAE(const Tensor& rewards, const Tensor& dones, const Tensor& values,
                                   float bootstrap_value, float gamma, float lambda, DeviceBackend* backend);

}  // namespace pulsatrix
