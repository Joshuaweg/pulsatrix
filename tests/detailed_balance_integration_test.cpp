/** @file detailed_balance_integration_test.cpp
 *  @brief End-to-end proof that DetailedBalanceLoss training makes a real forward policy
 *         sample proportional to reward, and that the trained state-flow F_theta(s_0)
 *         converges toward the true log-partition-function log Z.
 *
 *  Reuses HyperGrid at side_length=5 and sample_gflownet_trajectory() unmodified from Mission
 *  2 -- see that mission's own AAR for why side_length=5 (not the default 8) is needed for a
 *  far reward mode to be discoverable by on-policy exploration in a tractable number of steps.
 *
 *  True log Z for ndim=2, side_length=5, r0=0.1/r1=0.5/r2=2.0 (hand-computed, independent of
 *  this implementation): 4 exact-corner cells at 2.6, 12 near-corner-band cells at 0.6, 9
 *  interior cells at 0.1 -> Z = 4*2.6 + 12*0.6 + 9*0.1 = 10.4 + 7.2 + 0.9 = 18.5,
 *  log(18.5) ~= 2.918.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/detailed_balance_loss.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

bool is_far_corner_region(const Tensor& state, int64_t side_length) {
    return state.data()[0] >= static_cast<float>(side_length - 2) &&
           state.data()[1] >= static_cast<float>(side_length - 2);
}

float log_flow_at(LinearModule& flow_net, const Tensor& state) { return flow_net.forward(state).data()[0]; }

// Accumulates one trajectory's DB gradient into flow_net's/policy_net's pending gradients,
// without stepping either. See mission_detailed_balance_loss.md's Design section for why
// intermediate states need their two adjacent Delta terms' coefficients summed into one
// combined gradient before flow_net.backward() is called (once per state).
void accumulate_trajectory_gradient(HyperGridEnv& env, GFlowNetForwardPolicy& policy, LinearModule& policy_net,
                                     LinearModule& flow_net, DeviceBackend* backend) {
    GFlowNetTrajectory traj = sample_gflownet_trajectory(env, policy);
    const size_t n = traj.actions.size() - 1;  // number of real moves (last action is stop)
    const float log_reward = std::log(traj.terminal_reward);

    // Pass 1: read every state's current log-flow (no backward yet).
    std::vector<float> log_flow(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        log_flow[i] = log_flow_at(flow_net, traj.states[i]);
    }

    // Pass 1 (continued): one DetailedBalanceLoss::forward() per transition (n real moves +
    // the exit transition), accumulating each state's combined flow-gradient coefficient and
    // recording each transition's own policy-gradient weight for pass 2.
    std::vector<float> flow_grad_coeff(n + 1, 0.0f);
    std::vector<float> pf_weight(n + 1);  // pf_weight[i] is the weight for the decision at states[i]

    for (size_t i = 0; i < n; ++i) {
        const std::vector<bool> mask = env.valid_actions_mask(traj.states[i]);
        const std::vector<float> probs = policy.masked_probs(traj.states[i], mask);
        const float log_pf = std::log(probs[static_cast<size_t>(traj.actions[i])]);
        const float log_pb = env.backward_log_prob(traj.states[i + 1], traj.actions[i]);

        DetailedBalanceLoss db;
        (void)db.forward(log_flow[i], log_pf, log_flow[i + 1], log_pb);
        flow_grad_coeff[i] += db.grad_log_flow_s();
        flow_grad_coeff[i + 1] += db.grad_log_flow_s_next();
        pf_weight[i] = db.grad_weight_for_log_pf();
    }
    {
        // Exit transition at states[n]: log_flow_s_next is the fixed log-reward, not a
        // learned flow -- no coefficient is added to flow_grad_coeff for it.
        const std::vector<bool> mask = env.valid_actions_mask(traj.states[n]);
        const std::vector<float> probs = policy.masked_probs(traj.states[n], mask);
        const float log_pf_stop = std::log(probs[static_cast<size_t>(traj.actions[n])]);

        DetailedBalanceLoss db;
        (void)db.forward(log_flow[n], log_pf_stop, log_reward, /*log_pb=*/0.0f);
        flow_grad_coeff[n] += db.grad_log_flow_s();
        pf_weight[n] = db.grad_weight_for_log_pf();
    }

    // Pass 2 (flow network): one combined backward() call per state.
    for (size_t i = 0; i <= n; ++i) {
        (void)flow_net.forward(traj.states[i]);
        Tensor grad(Shape({1, 1}), backend, {flow_grad_coeff[i]});
        (void)flow_net.backward(grad);
    }

    // Pass 2 (policy network): one backward() call per decision point (n real moves + stop).
    const int64_t action_dim = env.action_dim();
    for (size_t i = 0; i <= n; ++i) {
        const std::vector<bool> mask = env.valid_actions_mask(traj.states[i]);
        const std::vector<float> probs = policy.masked_probs(traj.states[i], mask);
        std::vector<float> grad_values(static_cast<size_t>(action_dim));
        for (int64_t k = 0; k < action_dim; ++k) {
            const float indicator = (k == traj.actions[i]) ? 1.0f : 0.0f;
            grad_values[static_cast<size_t>(k)] = pf_weight[i] * (probs[static_cast<size_t>(k)] - indicator);
        }
        Tensor grad_tensor(Shape({1, action_dim}), backend, grad_values);
        (void)policy_net.backward(grad_tensor);
    }
}

void train_one_batch(HyperGridEnv& env, GFlowNetForwardPolicy& policy, LinearModule& policy_net,
                      LinearModule& flow_net, DeviceBackend* backend, AdamOptimizer& policy_optimizer,
                      AdamOptimizer& flow_optimizer, int batch_size) {
    policy_optimizer.zero_grad(policy_net);
    flow_optimizer.zero_grad(flow_net);
    for (int b = 0; b < batch_size; ++b) {
        accumulate_trajectory_gradient(env, policy, policy_net, flow_net, backend);
    }
    policy_optimizer.step(policy_net);
    flow_optimizer.step(flow_net);
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

TEST(DetailedBalanceIntegrationTest, TrainedFlowAtOriginConvergesTowardTrueLogZ) {
    CPUBackend backend;
    constexpr int64_t kSideLength = 5;
    constexpr float kTrueLogZ = 2.918f;

    HyperGridEnv env(&backend, 2, kSideLength);
    LinearModule policy_net(2, 3, &backend);
    LinearModule flow_net(2, 1, &backend);
    GFlowNetForwardPolicy policy(&policy_net, 3, &backend, /*seed=*/2);
    AdamOptimizer policy_optimizer(0.05f, &backend);
    AdamOptimizer flow_optimizer(0.05f, &backend);

    constexpr int kBatchSize = 16;
    for (int iter = 0; iter < 1500; ++iter) {
        train_one_batch(env, policy, policy_net, flow_net, &backend, policy_optimizer, flow_optimizer, kBatchSize);
    }

    const Tensor origin(Shape({1, 2}), &backend, {0.0f, 0.0f});
    const float trained_log_z_at_origin = log_flow_at(flow_net, origin);
    EXPECT_NEAR(trained_log_z_at_origin, kTrueLogZ, 1.0f);
}

TEST(DetailedBalanceIntegrationTest, TrainedPolicyVisitsFarCornerMoreThanUntrainedControl) {
    CPUBackend backend;
    constexpr int64_t kSideLength = 5;

    HyperGridEnv control_env(&backend, 2, kSideLength);
    LinearModule control_net(2, 3, &backend);
    GFlowNetForwardPolicy control_policy(&control_net, 3, &backend, /*seed=*/1);
    const float control_rate = far_corner_visitation_rate(control_env, control_policy, kSideLength, 500);

    HyperGridEnv train_env(&backend, 2, kSideLength);
    LinearModule policy_net(2, 3, &backend);
    LinearModule flow_net(2, 1, &backend);
    GFlowNetForwardPolicy policy(&policy_net, 3, &backend, /*seed=*/2);
    AdamOptimizer policy_optimizer(0.05f, &backend);
    AdamOptimizer flow_optimizer(0.05f, &backend);

    constexpr int kBatchSize = 16;
    for (int iter = 0; iter < 1500; ++iter) {
        train_one_batch(train_env, policy, policy_net, flow_net, &backend, policy_optimizer, flow_optimizer,
                         kBatchSize);
    }

    HyperGridEnv eval_env(&backend, 2, kSideLength);
    const float trained_rate = far_corner_visitation_rate(eval_env, policy, kSideLength, 500);

    // Same reasoning and threshold margins as Mission 2's own TB correctness proof -- see
    // trajectory_balance_integration_test.cpp's comment for why exact-value pins would be
    // fragile across build configurations.
    EXPECT_LT(control_rate, 0.15f);
    EXPECT_GT(trained_rate, 0.08f);
    EXPECT_GT(trained_rate, control_rate * 1.5f + 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
