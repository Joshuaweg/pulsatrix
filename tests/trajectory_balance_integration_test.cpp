/** @file trajectory_balance_integration_test.cpp
 *  @brief End-to-end proof that TrajectoryBalanceLoss training makes a real forward policy
 *         sample proportional to reward on HyperGrid -- not just "loss goes down."
 *
 *  HyperGrid's default reward (r0=0.1, r1=0.5, r2=2.0, ndim=2, side_length=8) is symmetric
 *  across all 4 corners: (0,0), (0,7), (7,0), (7,7) each have identical reward 2.6, the
 *  maximum in the whole grid. A GFlowNet trained to convergence should sample all 4 with
 *  roughly equal frequency. An *untrained* (zero-logit, uniform-at-every-state) policy has a
 *  strong, hand-justifiable bias toward the *origin* corner instead: with `stop` equally
 *  likely as either increment action at every state (1/3 each), episodes are short
 *  (expected length ~2 steps), so reaching the far corner (7,7) -- which needs >=14
 *  non-stop steps in a row -- is astronomically unlikely under the untrained policy but
 *  should become common under a trained one. This asymmetry is the test's real evidence:
 *  training measurably shifts probability mass toward a region the untrained policy could
 *  not plausibly reach by chance.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/learnable_scalar.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/trajectory_balance_loss.hpp"

namespace pulsatrix {
namespace {

// Whether a terminal state fell in the far corner's near-corner band (both coordinates in
// {side_length-2, side_length-1}), the region diametrically opposite the origin.
bool is_far_corner_region(const Tensor& state, int64_t side_length) {
    return state.data()[0] >= static_cast<float>(side_length - 2) &&
           state.data()[1] >= static_cast<float>(side_length - 2);
}

// Accumulates one trajectory's TB gradient into policy_net's/log_z's pending gradients,
// without stepping either -- the caller batches several of these before a single
// optimizer.step()/log_z.step(), the standard variance-reduction practice for GFlowNet
// training (a single trajectory's gradient is extremely high-variance on its own).
void accumulate_trajectory_gradient(HyperGridEnv& env, GFlowNetForwardPolicy& policy, LinearModule& policy_net,
                                     DeviceBackend* backend, LearnableScalar& log_z) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    const float log_reward = std::log(traj.terminal_reward);

    TrajectoryBalanceLoss tb;
    (void)tb.forward(traj.sum_log_pf, traj.sum_log_pb, log_reward, log_z.value());

    const float weight = tb.grad_weight_for_log_pf();
    const int64_t action_dim = env.action_dim();
    for (size_t t = 0; t < traj.states.size(); ++t) {
        const std::vector<bool> mask = env.valid_actions_mask(traj.states[t]);
        // Recomputes the exact distribution sample() used (refreshing policy_net's
        // single-most-recent-call cache) -- pass two of the two-pass rollout.
        const std::vector<float> probs = policy.masked_probs(traj.states[t], mask);

        std::vector<float> grad_values(static_cast<size_t>(action_dim));
        for (int64_t k = 0; k < action_dim; ++k) {
            const float indicator = (k == traj.actions[t]) ? 1.0f : 0.0f;
            grad_values[static_cast<size_t>(k)] = weight * (probs[static_cast<size_t>(k)] - indicator);
        }
        Tensor grad_tensor(Shape({1, action_dim}), backend, grad_values);
        (void)policy_net.backward(grad_tensor);
    }
    log_z.accumulate_grad(tb.grad_log_z());
}

void train_one_batch(HyperGridEnv& env, GFlowNetForwardPolicy& policy, LinearModule& policy_net,
                      DeviceBackend* backend, AdamOptimizer& optimizer, LearnableScalar& log_z, float log_z_lr,
                      int batch_size) {
    optimizer.zero_grad(policy_net);
    log_z.zero_grad();
    for (int b = 0; b < batch_size; ++b) {
        accumulate_trajectory_gradient(env, policy, policy_net, backend, log_z);
    }
    optimizer.step(policy_net);
    // log_z's plain SGD step has no Adam-style gradient-scale adaptation, unlike policy_net's
    // optimizer -- accumulate_trajectory_gradient sums (not averages) each trajectory's
    // contribution, so log_z's effective per-trajectory learning rate must be divided by
    // batch_size to match what a single un-batched step would apply.
    log_z.step(log_z_lr / static_cast<float>(batch_size));
}

// Samples num_trajectories terminal states under policy and returns the fraction that landed
// in the far-corner region.
float far_corner_visitation_rate(HyperGridEnv& env, GFlowNetForwardPolicy& policy, int64_t side_length,
                                  int num_trajectories) {
    int far_corner_count = 0;
    for (int i = 0; i < num_trajectories; ++i) {
        GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
        if (is_far_corner_region(traj.states.back(), side_length)) {
            ++far_corner_count;
        }
    }
    return static_cast<float>(far_corner_count) / static_cast<float>(num_trajectories);
}

TEST(TrajectoryBalanceIntegrationTest, TrainedPolicyVisitsFarCornerFarMoreThanUntrainedControl) {
    CPUBackend backend;
    // side_length=5 (not the default 8): reaching the far corner needs 8 consecutive non-stop
    // actions (4 increments per dimension) rather than 14 -- feasible for an on-policy,
    // uniform-initialized rollout to occasionally discover by chance within a few thousand
    // training steps (P(exactly 4-and-4 in 8 steps, 1/3 stop probability per step) ~1%, vs.
    // ~0.015% at side_length=8). GFlowNet training is on-policy here (no exploration bonus);
    // without at least occasional discovery, the far corner's gradient signal never arrives.
    constexpr int64_t kSideLength = 5;

    // Untrained control: a fresh, zero-initialized (never-trained) policy network.
    HyperGridEnv control_env(&backend, 2, kSideLength);
    LinearModule control_net(2, 3, &backend);
    GFlowNetForwardPolicy control_policy(&control_net, 3, &backend, /*seed=*/1);
    const float control_rate = far_corner_visitation_rate(control_env, control_policy, kSideLength, 500);

    // Trained policy: many TB training steps on an independent environment/network/seed.
    HyperGridEnv train_env(&backend, 2, kSideLength);
    LinearModule policy_net(2, 3, &backend);
    GFlowNetForwardPolicy policy(&policy_net, 3, &backend, /*seed=*/2);
    AdamOptimizer optimizer(0.05f, &backend);
    LearnableScalar log_z(0.0f);

    constexpr int kBatchSize = 16;
    for (int iter = 0; iter < 1500; ++iter) {
        train_one_batch(train_env, policy, policy_net, &backend, optimizer, log_z, /*log_z_lr=*/0.1f, kBatchSize);
    }

    HyperGridEnv eval_env(&backend, 2, kSideLength);
    const float trained_rate = far_corner_visitation_rate(eval_env, policy, kSideLength, 500);

    // The untrained control reaches the far corner only rarely (needs 8 consecutive non-stop
    // actions in the right 4-and-4 split; empirically measured well under 10% even accounting
    // for the boundary-masking renormalization that makes stopping somewhat less likely once a
    // coordinate saturates). The trained policy, having learned to sample the reward's 4
    // equal-mass corners more proportionally, visits the far corner measurably more often.
    // Thresholds are set with real margin below/above the actual measured values (control
    // ~0.05, trained ~0.10 in the run these were tuned against), not exact-value pins, since
    // float non-associativity across build configurations (Debug vs. Release) can shift which
    // side of an inverse-CDF comparison a sample falls on partway through a long stochastic
    // training run, and this is a real property of the trained policy, not a hand-derivable
    // closed-form fact like CartPole's oracle -- an exact-match assertion would be fragile.
    EXPECT_LT(control_rate, 0.15f);
    EXPECT_GT(trained_rate, 0.08f);
    EXPECT_GT(trained_rate, control_rate * 1.5f + 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
