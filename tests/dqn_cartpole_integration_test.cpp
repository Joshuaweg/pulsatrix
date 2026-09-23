/** @file dqn_cartpole_integration_test.cpp
 *  @brief Phase 2's exit gate: end-to-end proof that Mission 0's DQN building blocks
 *         (DQNAgent, DQNLoss, ComputeDoubleDQNTarget, SyncTargetNetwork) plus Phase 1's
 *         ReplayBuffer/CartPoleEnv compose into a training loop that actually learns to
 *         balance a pole -- to a threshold fixed in the mission *before* the run.
 *
 *  The per-class unit tests (dqn_agent_test, dqn_loss_test, dqn_target_test,
 *  replay_buffer_test, cartpole_env_test) each prove one piece in isolation. What none of
 *  them can prove is that the *procedure* composes: that the Bellman target really does
 *  bootstrap off a frozen snapshot, that DQNLoss's action-masked gradient routes back through
 *  the online network's parameters and nothing else, that the periodic hard sync actually
 *  moves the target network, and that the whole thing converges rather than diverging into
 *  NaN. That is this file's job, and it is the same role gan_integration_test.cpp plays for
 *  the GAN pieces.
 *
 *  @note The loop, hyperparameters and seeds all come from examples/dqn_cartpole_training.hpp,
 *        which examples/dqn_cartpole_demo.cpp includes too -- the mission requires the demo
 *        and the test to be the same run, not two configurations that can drift apart.
 *  @note Deterministic end to end (this codebase's LCG convention for weight init,
 *        exploration, replay sampling and env reset -- no `<random>` anywhere), so the
 *        threshold assertions below are reproducible, not "usually passing".
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "../examples/dqn_cartpole_training.hpp"

namespace exai {
namespace dqn_cartpole {
namespace {

/** @brief The relative-improvement bar, fixed by the mission before any run happened. */
constexpr double kMinImprovementRatio = 3.0;
/** @brief The absolute-competence bar, as a fraction of max_steps. Also fixed by the mission. */
constexpr double kMinFinalFractionOfMaxSteps = 0.5;

/** @brief The tuned default configuration -- the one the demo prints and this file asserts on. */
const TrainingConfig& shared_config() {
    static const TrainingConfig config;
    return config;
}

/**
 * @brief One shared training run for every assertion in this fixture.
 * @note Trained once, lazily, and reused: the run takes several seconds, and each test below
 *       inspects a different property of the *same* run. Re-training per test would multiply
 *       the suite's cost and prove nothing extra, precisely because the run is deterministic.
 */
const TrainingResult& shared_result() {
    static const TrainingResult result = RunTraining(shared_config());
    return result;
}

class DQNCartPoleIntegrationTest : public ::testing::Test {
protected:
    const TrainingConfig& config = shared_config();
    const TrainingResult& result = shared_result();
};

// ---------------------------------------------------------------------------------------
// THE EXIT GATE: the two performance bars, stated in the mission before this ran.
// ---------------------------------------------------------------------------------------
TEST_F(DQNCartPoleIntegrationTest, TrainedPolicyClearsBothPerformanceBars) {
    ASSERT_EQ(static_cast<int64_t>(result.episode_lengths.size()), config.num_episodes);

    const double ratio = result.last_window_average / result.first_window_average;
    std::cout << "[DQN] first " << kWindow << " episodes mean length " << result.first_window_average << " -> last "
              << kWindow << " episodes mean length " << result.last_window_average << " (" << ratio << "x; max_steps "
              << config.max_steps << ")" << std::endl;

    // Non-vacuity: a first window that was already at the ceiling would make the ratio bar
    // unreachable and the absolute bar free. It must genuinely start out bad.
    ASSERT_GT(result.first_window_average, 0.0);
    EXPECT_LT(result.first_window_average, 0.5 * static_cast<double>(config.max_steps))
        << "an untrained policy that already clears the absolute bar would make this test vacuous";

    // (a) Relative improvement.
    EXPECT_GE(ratio, kMinImprovementRatio)
        << "final-" << kWindow << " average episode length must be at least " << kMinImprovementRatio
        << "x the first-" << kWindow << " average";

    // (b) Absolute competence -- so "improved from awful to slightly-less-awful" cannot pass.
    EXPECT_GE(result.last_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "final-" << kWindow << " average episode length must be at least "
        << (100.0 * kMinFinalFractionOfMaxSteps) << "% of max_steps (" << config.max_steps << ")";
}

// The loss is checked inside the loop on every single update, not once at the end -- a run
// that went NaN mid-training and was rescued by Adam's epsilon would still be broken.
TEST_F(DQNCartPoleIntegrationTest, EveryUpdateLossIsFiniteAndNonNegative) {
    EXPECT_GT(result.update_count, 0) << "no gradient updates ran at all";
    EXPECT_TRUE(result.all_losses_finite)
        << "loss first became non-finite at update " << result.first_non_finite_update << " of "
        << result.update_count;
    EXPECT_EQ(result.first_non_finite_update, -1);

    // A mean of squares cannot be negative; finiteness alone would not catch a sign error.
    EXPECT_GE(result.first_loss, 0.0f);
    EXPECT_GE(result.last_loss, 0.0f);
    EXPECT_TRUE(std::isfinite(result.first_loss));
    EXPECT_TRUE(std::isfinite(result.last_loss));
    std::cout << "[DQN] " << result.update_count << " updates, all finite; loss " << result.first_loss << " -> "
              << result.last_loss << std::endl;
}

// SyncTargetNetwork is the only thing that can ever write to the target network -- it is never
// handed to the optimizer and never backpropagated through. So a target network whose weights
// differ from their initial values at the end is direct evidence a sync actually happened.
TEST_F(DQNCartPoleIntegrationTest, TargetNetworkIsActuallySyncedDuringTraining) {
    EXPECT_GT(result.sync_count, 0);
    ASSERT_EQ(result.initial_target_weight.size(), result.final_target_weight.size());
    ASSERT_FALSE(result.initial_target_weight.empty());

    std::cout << "[DQN] target-network syncs: " << result.sync_count
              << " | max |final - initial| first-layer weight change: " << result.max_target_weight_delta
              << std::endl;
    EXPECT_GT(result.max_target_weight_delta, 1e-3)
        << "the target network's weights never moved -- SyncTargetNetwork() cannot have run";
}

// The pre-training sync is load-bearing too: the target network must start as an exact copy of
// the online network, not an independent random draw. Pin that directly, since the training
// run above would still (more slowly) converge if it were violated.
TEST(DQNCartPoleInitializationTest, TargetNetworkStartsAsAnExactCopyOfTheOnlineNetwork) {
    CPUBackend backend;
    const TrainingConfig config;

    LinearModule online_in(4, config.hidden_size, &backend);
    ReluModule online_relu(&backend);
    LinearModule online_out(config.hidden_size, 2, &backend);
    SequentialModule online({&online_in, &online_relu, &online_out});

    LinearModule target_in(4, config.hidden_size, &backend);
    ReluModule target_relu(&backend);
    LinearModule target_out(config.hidden_size, 2, &backend);
    SequentialModule target({&target_in, &target_relu, &target_out});

    online_in.set_weight(lcg_weights(static_cast<size_t>(4 * config.hidden_size), config.weight_seed));
    online_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size * 2), config.weight_seed + 1u));

    // Non-vacuity: they genuinely differ before the sync (the online network is non-zero, the
    // freshly constructed target network is LinearModule's zero init).
    float max_before = 0.0f;
    for (int64_t i = 0; i < online_in.weight().numel(); ++i) {
        max_before = std::max(max_before, std::fabs(online_in.weight().data()[i] - target_in.weight().data()[i]));
    }
    ASSERT_GT(max_before, 1e-3f) << "weight init produced an all-zero online network -- no gradient signal exists";

    SyncTargetNetwork(online, target);
    for (int64_t i = 0; i < online_in.weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(target_in.weight().data()[i], online_in.weight().data()[i]) << "first-layer element " << i;
    }
    for (int64_t i = 0; i < online_out.weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(target_out.weight().data()[i], online_out.weight().data()[i]) << "output-layer element " << i;
    }
}

// The threshold assertions above are only meaningful if the run is reproducible. Two short
// runs of the same configuration must agree on every episode length, bit for bit -- if they
// ever don't, something has reached for non-deterministic randomness and the exit-gate test
// has quietly become a coin flip. Deliberately short (a fraction of the full run's episodes):
// a divergence in seeding shows up within the first few episodes, not only at the end.
TEST(DQNCartPoleDeterminismTest, TwoRunsOfTheSameConfigurationAreIdentical) {
    TrainingConfig config;
    config.num_episodes = 30;

    const TrainingResult a = RunTraining(config);
    const TrainingResult b = RunTraining(config);

    ASSERT_EQ(a.episode_lengths.size(), b.episode_lengths.size());
    for (size_t i = 0; i < a.episode_lengths.size(); ++i) {
        EXPECT_EQ(a.episode_lengths[i], b.episode_lengths[i]) << "episode " << i;
    }
    EXPECT_EQ(a.update_count, b.update_count);
    EXPECT_EQ(a.sync_count, b.sync_count);
    EXPECT_FLOAT_EQ(a.first_loss, b.first_loss);
    EXPECT_FLOAT_EQ(a.last_loss, b.last_loss);

    ASSERT_EQ(a.final_target_weight.size(), b.final_target_weight.size());
    for (size_t i = 0; i < a.final_target_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_target_weight[i], b.final_target_weight[i]) << "target weight element " << i;
    }
}

}  // namespace
}  // namespace dqn_cartpole
}  // namespace exai
