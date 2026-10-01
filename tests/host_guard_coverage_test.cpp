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
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mamba_module.hpp"
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
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rollout_buffer.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
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

TEST_F(HostGuardCoverageDeathTest, BatchNormForwardAbortsOnNonCpuInput) {
    BatchNormModule bn(2, &cpu);
    Tensor x = mislabelled(Shape({1, 2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_HOST_GUARD_DEATH({ (void)bn.forward(x); }, "input");
}

TEST_F(HostGuardCoverageDeathTest, GroupNormForwardAbortsOnNonCpuInput) {
    GroupNormModule gn(1, 2, &cpu);
    Tensor x = mislabelled(Shape({1, 2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_HOST_GUARD_DEATH({ (void)gn.forward(x); }, "input");
}

TEST_F(HostGuardCoverageDeathTest, LayerNormForwardAbortsOnNonCpuInput) {
    LayerNormModule ln(3, &cpu);
    Tensor x = mislabelled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    EXPECT_HOST_GUARD_DEATH({ (void)ln.forward(x); }, "input");
}

TEST_F(HostGuardCoverageDeathTest, RMSNormForwardAbortsOnNonCpuInput) {
    RMSNormModule rms(3, &cpu);
    Tensor x = mislabelled(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    EXPECT_HOST_GUARD_DEATH({ (void)rms.forward(x); }, "input");
}

// ---- Normalization / elementwise modules: backward scratch allocated through a GPU backend ----

TEST_F(HostGuardCoverageDeathTest, BatchNormBackwardAbortsOnGpuBackendScratch) {
    BatchNormModule bn(2, &hip, DeviceType::Cpu);  // Cpu params, Hip-tagged scratch
    (void)bn.forward(host(Shape({1, 2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8}));
    Tensor grad = host(Shape({1, 2, 2, 2}), {1, 1, 1, 1, 1, 1, 1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)bn.backward(grad); }, "local_gamma_grad");
}

TEST_F(HostGuardCoverageDeathTest, GroupNormBackwardAbortsOnGpuBackendScratch) {
    GroupNormModule gn(1, 2, &hip, DeviceType::Cpu);  // Cpu params, Hip-tagged scratch
    (void)gn.forward(host(Shape({1, 2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8}));
    Tensor grad = host(Shape({1, 2, 2, 2}), {1, 1, 1, 1, 1, 1, 1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)gn.backward(grad); }, "local_gamma_grad");
}

TEST_F(HostGuardCoverageDeathTest, LayerNormBackwardAbortsOnGpuBackendScratch) {
    LayerNormModule ln(3, &hip, DeviceType::Cpu);  // Cpu params, Hip-tagged scratch
    (void)ln.forward(host(Shape({2, 3}), {1, 2, 3, 4, 5, 6}));
    Tensor grad = host(Shape({2, 3}), {1, 1, 1, 1, 1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)ln.backward(grad); }, "local_gamma_grad");
}

TEST_F(HostGuardCoverageDeathTest, RMSNormBackwardAbortsOnGpuBackendScratch) {
    RMSNormModule rms(3, &hip, DeviceType::Cpu);  // Cpu params, Hip-tagged scratch
    (void)rms.forward(host(Shape({2, 3}), {1, 2, 3, 4, 5, 6}));
    Tensor grad = host(Shape({2, 3}), {1, 1, 1, 1, 1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)rms.backward(grad); }, "local_gamma_grad");
}

TEST_F(HostGuardCoverageDeathTest, AggregatorBackwardAbortsOnGpuBackendScratch) {
    AggregatorModule agg(&hip);
    (void)agg.forward(host(Shape({3}), {0.2f, 0.5f, 0.8f}));
    Tensor grad = host(Shape({}), {1.0f});
    EXPECT_HOST_GUARD_DEATH({ (void)agg.backward(grad); }, "grad_input");
}

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

TEST_F(HostGuardCoverageDeathTest, TanhGaussianPolicyForwardAbortsOnGpuBackendOutput) {
    TanhGaussianPolicy policy(&hip);
    Tensor mean = host(Shape({1, 2}), {0, 0});
    Tensor log_std = host(Shape({1, 2}), {0, 0});
    Tensor eps = host(Shape({1, 2}), {0.5f, -0.5f});
    EXPECT_HOST_GUARD_DEATH({ (void)policy.forward(mean, log_std, eps); }, "action");
}

TEST_F(HostGuardCoverageDeathTest, TanhGaussianPolicyBackwardAbortsOnNonCpuGradAction) {
    TanhGaussianPolicy policy(&cpu);
    (void)policy.forward(host(Shape({1, 2}), {0, 0}), host(Shape({1, 2}), {0, 0}), host(Shape({1, 2}), {0.5f, -0.5f}));
    Tensor grad_action = mislabelled(Shape({1, 2}), {1, 1});
    Tensor grad_log_prob = host(Shape({1, 2}), {1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)policy.backward(grad_action, grad_log_prob); }, "grad_action");
}

// ---- Fuzzy-logic operators (split_operands / combine_operands paths) ----

TEST_F(HostGuardCoverageDeathTest, ConjunctionBackwardAbortsOnGpuBackendGradients) {
    ConjunctionModule conj(&hip);
    (void)conj.forward(host(Shape({2}), {0.2f, 0.8f}), host(Shape({2}), {0.6f, 0.4f}));
    Tensor grad = host(Shape({2}), {1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)conj.backward(grad); }, "grad_a");
}

TEST_F(HostGuardCoverageDeathTest, DisjunctionBackwardAbortsOnGpuBackendGradients) {
    DisjunctionModule disj(&hip);
    (void)disj.forward(host(Shape({2}), {0.2f, 0.8f}), host(Shape({2}), {0.6f, 0.4f}));
    Tensor grad = host(Shape({2}), {1, 1});
    EXPECT_HOST_GUARD_DEATH({ (void)disj.backward(grad); }, "grad_a");
}

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

TEST_F(HostGuardCoverageDeathTest, ComputeAttributionStabilityAbortsOnNonCpuValues) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"saliency", mislabelled(Shape({2}), {0.1f, 0.2f}), {}});
    runs.push_back(Attribution{"saliency", mislabelled(Shape({2}), {0.1f, 0.3f}), {}});
    EXPECT_HOST_GUARD_DEATH({ (void)ComputeAttributionStability(runs); }, "run.values");
}

TEST_F(HostGuardCoverageDeathTest, ComputeConservationAbortsOnNonCpuRelevance) {
    Tensor r_in = mislabelled(Shape({2}), {0.4f, 0.6f});
    Tensor r_out = host(Shape({1}), {1.0f});
    EXPECT_HOST_GUARD_DEATH({ (void)ComputeConservation(r_in, r_out); }, "relevance_in");
}

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

// ---- Layers whose output / scratch is allocated through a GPU backend ----

TEST_F(HostGuardCoverageDeathTest, Conv2DForwardAbortsOnGpuBackendOutput) {
    Conv2DModule conv(1, 1, 2, 2, &hip);
    Tensor x = host(Shape({1, 1, 3, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_HOST_GUARD_DEATH({ (void)conv.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, AvgPool2DForwardAbortsOnGpuBackendOutput) {
    AvgPool2DModule pool(2, 2, &hip);
    Tensor x = host(Shape({1, 1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)pool.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, MaxPool2DForwardAbortsOnGpuBackendOutput) {
    MaxPool2DModule pool(2, 2, &hip);
    Tensor x = host(Shape({1, 1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)pool.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, EmbeddingForwardAbortsOnGpuBackendOutput) {
    EmbeddingModule emb(4, 2, &hip);
    Tensor indices = host(Shape({1, 2}), {0, 3});
    EXPECT_HOST_GUARD_DEATH({ (void)emb.forward(indices); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, RoPEForwardAbortsOnGpuBackendOutput) {
    RoPEModule rope(2, &hip);
    Tensor x = host(Shape({2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)rope.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, RNNForwardAbortsOnGpuBackendOutput) {
    RNNModule rnn(2, 2, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)rnn.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, GRUForwardAbortsOnGpuBackendOutput) {
    GRUModule gru(2, 2, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)gru.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, LSTMForwardAbortsOnGpuBackendOutput) {
    LSTMModule lstm(2, 2, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)lstm.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, MambaForwardAbortsOnGpuBackendOutput) {
    MambaModule mamba(2, 2, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)mamba.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, RetNetForwardAbortsOnGpuBackendOutput) {
    RetNetModule retnet(2, 2, 0.7f, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)retnet.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, RWKVForwardAbortsOnGpuBackendOutput) {
    RWKVModule rwkv(2, &hip);
    Tensor x = host(Shape({1, 2, 2}), {1, 2, 3, 4});
    EXPECT_HOST_GUARD_DEATH({ (void)rwkv.forward(x); }, "output");
}

TEST_F(HostGuardCoverageDeathTest, MultiHeadAttentionForwardAbortsOnGpuBackendScratch) {
    MultiHeadAttentionModule mha(4, 2, &hip, /*use_rope=*/false, /*use_qk_norm=*/false);
    Tensor x = host(Shape({1, 2, 4}), {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_HOST_GUARD_DEATH({ (void)mha.forward(x); }, "q");
}

}  // namespace
}  // namespace pulsatrix
