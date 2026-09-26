/** @file gflownet_trajectory.hpp
 *  @brief Rolls out one full HyperGrid episode under a GFlowNetForwardPolicy.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief One full sampled trajectory: every state the forward policy acted from, the action
 *        taken at each, and the trajectory-level quantities Trajectory Balance (and Detailed
 *        Balance/SubTB, in later missions) need.
 * @note `states`/`actions` deliberately are NOT enough on their own to recompute
 *       `sum_log_pf`/`sum_log_pb` -- they exist so a training loop's second pass can
 *       re-`forward()` the policy network at each state (refreshing its single-most-recent-call
 *       cache) and apply the now-known per-step gradient. See
 *       mission_trajectory_balance_loss.md's Design section ("Two-pass rollout").
 */
struct GFlowNetTrajectory {
    /** @brief The state the policy acted from at each decision point, in order. */
    std::vector<Tensor> states;
    /** @brief The action index sampled at each corresponding entry of states. */
    std::vector<int64_t> actions;
    /** @brief `Σ_t log P_F(s_{t+1}|s_t)` over the whole trajectory, including the final
     *         `stop` decision. */
    float sum_log_pf = 0.0f;
    /** @brief `Σ_t log P_B(s_t|s_{t+1})` over every real move (the `stop` action does not
     *         move the state, so it contributes no P_B term). */
    float sum_log_pb = 0.0f;
    /** @brief `R(x)` at the trajectory's terminal state. */
    float terminal_reward = 0.0f;
};

/**
 * @brief Samples one full trajectory: resets env, then repeatedly samples a masked action from
 *        forward_policy and steps env until termination (an explicit `stop` or env's own
 *        max_steps cap).
 * @param env The environment to roll out on. Reset internally; any existing episode in
 *        progress is discarded.
 * @param forward_policy The forward policy sampling each step's action.
 * @return The full trajectory, per GFlowNetTrajectory's fields above.
 */
[[nodiscard]] GFlowNetTrajectory sample_gflownet_trajectory(HyperGridEnv& env, GFlowNetForwardPolicy& forward_policy);

}  // namespace pulsatrix
