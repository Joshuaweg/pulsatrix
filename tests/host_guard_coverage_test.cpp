// Death tests for the PULSATRIX_REQUIRE_HOST guards added by GPU-native-kernels Mission 0 O4
// (host-dereferencing paths that had no guard, or guarded their inputs but not the scratch /
// output tensors they allocate through their backend).
//
// No GPU hardware is needed. Two techniques:
//   1. A mislabelled tensor: host memory from a CPUBackend, tagged DeviceType::Hip.
//   2. HipReportingBackend: a CPUBackend whose device() reports Hip. Every Tensor allocated
//      through it without an explicit device tag is Hip-tagged (Tensor(shape, backend) takes
//      backend->device()) while its memory stays host-resident, so the guard fires where a real
//      GPU backend would otherwise have handed a device pointer to a host loop.
// The guard is always on (not compiled out under NDEBUG), so none of these skip in Release.
// Each test matches the specific guarded expression, proving the new guard -- not some
// earlier one -- is what fires.

#include <gtest/gtest.h>

#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/categorical_policy_agent.hpp"
#include "pulsatrix/conjunction_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/dqn_agent.hpp"
#include "pulsatrix/dqn_loss.hpp"
#include "pulsatrix/dqn_target.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/explainer_stability.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"
#include "pulsatrix/polyak_update.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/reparameterize.hpp"
#include "pulsatrix/replay_buffer.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rollout_buffer.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/sinusoidal_timestep_embedding.hpp"
#include "pulsatrix/softmax_module.hpp"
#include "pulsatrix/tanh_gaussian_policy.hpp"

// Matches the guard's own message for one specific guarded expression.
#define EXPECT_HOST_GUARD_DEATH(statement, guarded_expr) \
    EXPECT_DEATH(statement, "PULSATRIX_REQUIRE_HOST\\(" guarded_expr "\\)")

namespace pulsatrix {
namespace {

class HipReportingBackend : public CPUBackend {
public:
    [[nodiscard]] DeviceType device() const noexcept override { return DeviceType::Hip; }
};

class HostGuardCoverageDeathTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    HipReportingBackend hip;

    // Host memory, Hip tag.
    Tensor mislabelled(const Shape& shape, const std::vector<float>& values) {
        return Tensor(shape, &cpu, values, DeviceType::Hip);
    }
    Tensor host(const Shape& shape, const std::vector<float>& values) { return Tensor(shape, &cpu, values); }
};

// ---- Normalization modules: forward_impl (mislabelled input) ----

// ---- Normalization / elementwise modules: backward scratch allocated through a GPU backend ----

// ---- Losses ----

TEST_F(HostGuardCoverageDeathTest, DQNLossBackwardAbortsOnGpuBackendGradient) {
    DQNLoss loss(&hip);
    (void)loss.forward(host(Shape({1, 2}), {0.3f, 0.7f}), host(Shape({1, 1}), {1}), host(Shape({1, 1}), {1}));
    EXPECT_HOST_GUARD_DEATH({ (void)loss.backward(); }, "grad");
}

// As with CrossEntropyLoss, forward()'s backend-allocated probs is the reachable guard.
TEST_F(HostGuardCoverageDeathTest, PolicyGradientLossForwardAbortsOnGpuBackendProbs) {
    PolicyGradientLoss loss(&hip);
    Tensor logits = host(Shape({1, 2}), {0.3f, 0.7f});
    Tensor actions = host(Shape({1, 1}), {1});
    Tensor returns = host(Shape({1, 1}), {1});
    EXPECT_HOST_GUARD_DEATH({ (void)loss.forward(logits, actions, returns); }, "probs");
}

TEST_F(HostGuardCoverageDeathTest, PPOClippedLossForwardAbortsOnGpuBackendProbs) {
    PPOClippedLoss loss(&hip);
    Tensor logits = host(Shape({1, 2}), {0.3f, 0.7f});
    Tensor actions = host(Shape({1, 1}), {1});
    Tensor old_log_probs = host(Shape({1, 1}), {-0.5f});
    Tensor advantages = host(Shape({1, 1}), {1});
    EXPECT_HOST_GUARD_DEATH({ (void)loss.forward(logits, actions, old_log_probs, advantages, 0.2f); }, "probs");
}

// ---- Reparameterization / squashed-Gaussian policy ----

// ---- Fuzzy-logic operators (split_operands / combine_operands paths) ----

// ---- Target-network updates and optimizer state ----

TEST_F(HostGuardCoverageDeathTest, SyncTargetNetworkAbortsOnNonCpuParameters) {
    LinearModule source(1, 1, &cpu, DeviceType::Hip);
    LinearModule destination(1, 1, &cpu, DeviceType::Hip);
    EXPECT_HOST_GUARD_DEATH({ SyncTargetNetwork(source, destination); }, "from");
}

TEST_F(HostGuardCoverageDeathTest, PolyakUpdateAbortsOnNonCpuParameters) {
    LinearModule source(1, 1, &cpu, DeviceType::Hip);
    LinearModule destination(1, 1, &cpu, DeviceType::Hip);
    EXPECT_HOST_GUARD_DEATH({ PolyakUpdate(source, destination, 0.5f); }, "from");
}

// ---- Explainability metrics ----

// ---- RL agents, GFlowNet sampling ----

TEST_F(HostGuardCoverageDeathTest, DQNAgentGreedyActionAbortsOnNonCpuQValues) {
    LinearModule q_network(2, 2, &cpu, DeviceType::Hip);  // q-values come out Hip-tagged
    DQNAgent agent(&q_network, 2, 0.0f, &cpu);
    Tensor obs = host(Shape({1, 2}), {0.5f, -0.5f});
    EXPECT_HOST_GUARD_DEATH({ (void)agent.act_greedy(obs); }, "q_values");
}

TEST_F(HostGuardCoverageDeathTest, CategoricalPolicyAgentAbortsOnNonCpuLogits) {
    LinearModule policy_network(2, 2, &cpu, DeviceType::Hip);
    CategoricalPolicyAgent agent(&policy_network, 2, &cpu);
    Tensor obs = host(Shape({1, 2}), {0.5f, -0.5f});
    EXPECT_HOST_GUARD_DEATH({ (void)agent.act_greedy(obs); }, "logits");
}

TEST_F(HostGuardCoverageDeathTest, GFlowNetForwardPolicySampleAbortsOnNonCpuLogits) {
    LinearModule policy_network(2, 3, &cpu, DeviceType::Hip);
    GFlowNetForwardPolicy policy(&policy_network, 3, &cpu);
    Tensor obs = host(Shape({1, 2}), {0, 0});
    const std::vector<bool> mask{true, true, true};
    EXPECT_HOST_GUARD_DEATH({ (void)policy.sample(obs, mask); }, "logits");
}

TEST_F(HostGuardCoverageDeathTest, SampleGFlowNetTrajectoryAbortsOnGpuBackendAction) {
    HyperGridEnv env(&cpu, 2, 8);
    LinearModule policy_network(2, 3, &cpu, DeviceType::Cpu);
    GFlowNetForwardPolicy policy(&policy_network, 3, &hip);  // sampled action is Hip-tagged
    EXPECT_HOST_GUARD_DEATH({ (void)sample_gflownet_trajectory(env, policy); }, "sampled.action");
}

// ---- Replay / rollout storage allocated through a GPU backend ----

TEST_F(HostGuardCoverageDeathTest, ReplayBufferAddAbortsOnGpuBackendStorage) {
    ReplayBuffer buffer(4, 2, 1, &hip);
    Tensor obs = host(Shape({1, 2}), {0, 1});
    Tensor action = host(Shape({1, 1}), {0});
    EXPECT_HOST_GUARD_DEATH({ buffer.add(obs, action, 1.0f, obs, false); }, "observations_");
}

TEST_F(HostGuardCoverageDeathTest, RolloutBufferAddAbortsOnGpuBackendStorage) {
    RolloutBuffer buffer(4, 2, 1, &hip);
    Tensor obs = host(Shape({1, 2}), {0, 1});
    Tensor action = host(Shape({1, 1}), {0});
    EXPECT_HOST_GUARD_DEATH({ buffer.add(obs, action, 1.0f, -0.5f, false); }, "observations_");
}

TEST_F(HostGuardCoverageDeathTest, RolloutBufferComputeReturnsAbortsOnGpuBackendStorage) {
    RolloutBuffer buffer(4, 2, 1, &hip);
    EXPECT_HOST_GUARD_DEATH({ (void)buffer.compute_returns(0.9f); }, "observations_");
}

TEST_F(HostGuardCoverageDeathTest, RolloutBufferRewardsAbortsOnGpuBackendStorage) {
    RolloutBuffer buffer(4, 2, 1, &hip);
    EXPECT_HOST_GUARD_DEATH({ (void)buffer.rewards(); }, "rewards_");
}

TEST_F(HostGuardCoverageDeathTest, RolloutBufferDonesAbortsOnGpuBackendStorage) {
    RolloutBuffer buffer(4, 2, 1, &hip);
    EXPECT_HOST_GUARD_DEATH({ (void)buffer.dones(); }, "dones_");
}

// ---- Diffusion timestep embedding ----

TEST_F(HostGuardCoverageDeathTest, SinusoidalTimestepEmbeddingAbortsOnGpuBackend) {
    EXPECT_HOST_GUARD_DEATH({ (void)SinusoidalTimestepEmbedding(3, 4, &hip); }, "embedding");
}

}  // namespace
}  // namespace pulsatrix
