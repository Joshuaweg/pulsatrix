/** @file gflownet_hypergrid.cpp
 *  @brief Recipe: trains a GFlowNet forward policy via Trajectory Balance on HyperGrid, and
 *         shows training shifts sampling toward the reward's far corner. Paired with
 *         docs/recipes/mechanistic-interpretability/gflownet_hypergrid.md.
 *  @note Reuses the exact training loop and hyperparameters from
 *        tests/trajectory_balance_integration_test.cpp -- see that file for the full
 *        statistical justification of the thresholds.
 */
#include <cmath>
#include <cstdio>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/learnable_scalar.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/trajectory_balance_loss.hpp"

namespace {
using namespace pulsatrix;

bool is_far_corner_region(const Tensor& state, int64_t side_length) {
    return state.data()[0] >= static_cast<float>(side_length - 2) &&
           state.data()[1] >= static_cast<float>(side_length - 2);
}

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
    log_z.step(log_z_lr / static_cast<float>(batch_size));
}

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

}  // namespace

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    constexpr int64_t kSideLength = 5;  // 4 equal-reward corners: (0,0),(0,4),(4,0),(4,4)

    std::printf("GFlowNet HyperGrid recipe -- Trajectory Balance, %lldx%lld grid\n\n",
                static_cast<long long>(kSideLength), static_cast<long long>(kSideLength));

    HyperGridEnv control_env(&backend, 2, kSideLength);
    LinearModule control_net(2, 3, &backend);
    GFlowNetForwardPolicy control_policy(&control_net, 3, &backend, /*seed=*/1);
    const float control_rate = far_corner_visitation_rate(control_env, control_policy, kSideLength, 500);
    std::printf("untrained policy:  far-corner visitation rate over 500 trajectories: %.1f%%\n",
                100.0f * control_rate);

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
    std::printf("trained policy:    far-corner visitation rate over 500 trajectories: %.1f%%\n",
                100.0f * trained_rate);

    std::printf(
        "\nAll 4 corners of this grid share the maximum reward, so a GFlowNet trained to\n"
        "convergence should sample all 4 with roughly equal frequency. The untrained (uniform)\n"
        "policy is heavily biased toward stopping early near the origin; Trajectory Balance\n"
        "training measurably shifts probability mass toward the far corner instead, without\n"
        "ever being told 'maximize reward' -- it learns to sample proportional to R(x).\n");

    return 0;
}
