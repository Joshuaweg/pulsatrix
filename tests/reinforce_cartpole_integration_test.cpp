/** @file reinforce_cartpole_integration_test.cpp
 *  @brief Phase 3 Mission 1's acceptance criterion: end-to-end proof that Mission 0's
 *         policy-gradient building blocks (CategoricalPolicyAgent, PolicyGradientLoss) plus
 *         Phase 1's RolloutBuffer/CartPoleEnv compose into a REINFORCE training loop that
 *         actually learns to balance a pole -- to thresholds fixed in the mission *before* the
 *         run.
 *
 *  The per-class unit tests (categorical_policy_agent_test, policy_gradient_loss_test,
 *  rollout_buffer_test, cartpole_env_test) each prove one piece in isolation. What none of them
 *  can prove is that the *procedure* composes: that the log-probability the agent caches is the
 *  one the buffer stores and the one the loss's gradient is consistent with, that
 *  compute_returns()'s done-segmented returns really do carry a usable learning signal, that the
 *  dense (every-column) policy gradient routes back through the policy network's parameters, and
 *  that the whole thing climbs rather than collapsing into a degenerate deterministic policy or
 *  NaN. That is this file's job, and it is the same role dqn_cartpole_integration_test.cpp plays
 *  for the value-based pieces.
 *
 *  @note The loop, hyperparameters and seeds all come from
 *        examples/reinforce_cartpole_training.hpp, which examples/reinforce_cartpole_demo.cpp
 *        includes too -- the mission requires the demo and the test to be the same run, not two
 *        configurations that can drift apart.
 *  @note Deterministic end to end (this codebase's LCG convention for weight init, categorical
 *        action sampling and env reset -- no `<random>` anywhere), so the threshold assertions
 *        below are reproducible, not "usually passing".
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>

#include "../examples/reinforce_cartpole_training.hpp"

namespace pulsatrix {
namespace reinforce_cartpole {
namespace {

/** @brief The relative-improvement bar, fixed by the mission before any run happened. */
constexpr double kMinImprovementRatio = 3.0;
/**
 * @brief The absolute-competence bar, as a fraction of max_steps. Also fixed by the mission --
 *        30%, deliberately below DQN's 50%, because a vanilla policy gradient with no learned
 *        baseline is a higher-variance estimator than a replay-trained Q-network.
 */
constexpr double kMinFinalFractionOfMaxSteps = 0.3;

/** @brief The tuned default configuration -- the one the demo prints and this file asserts on. */
const TrainingConfig& shared_config() {
    static const TrainingConfig config;
    return config;
}

/**
 * @brief One shared training run for every assertion in this fixture.
 * @note Trained once, lazily, and reused: each test below inspects a different property of the
 *       *same* run. Re-training per test would multiply the suite's cost and prove nothing
 *       extra, precisely because the run is deterministic.
 */
const TrainingResult& shared_result() {
    static const TrainingResult result = RunTraining(shared_config());
    return result;
}

class ReinforceCartPoleIntegrationTest : public ::testing::Test {
protected:
    const TrainingConfig& config = shared_config();
    const TrainingResult& result = shared_result();
};

// ---------------------------------------------------------------------------------------
// THE EXIT GATE: the two performance bars, stated in the mission before this ran.
// ---------------------------------------------------------------------------------------
TEST_F(ReinforceCartPoleIntegrationTest, TrainedPolicyClearsBothPerformanceBars) {
    ASSERT_EQ(static_cast<int64_t>(result.episode_lengths.size()), config.num_episodes);
    // The window is 10% of the episode count, not DQN's fixed 20 -- the mission's own choice.
    ASSERT_EQ(result.window, window_size(config.num_episodes));
    ASSERT_GT(result.window, 1) << "a window of one episode would be pure noise, not an average";

    const double ratio = result.last_window_average / result.first_window_average;
    std::cout << "[REINFORCE] first " << result.window << " episodes mean length " << result.first_window_average
              << " -> last " << result.window << " episodes mean length " << result.last_window_average << " ("
              << ratio << "x; max_steps " << config.max_steps << ")" << std::endl;

    // Non-vacuity: a first window that was already competent would make the ratio bar
    // unreachable and the absolute bar free. It must genuinely start out bad.
    ASSERT_GT(result.first_window_average, 0.0);
    EXPECT_LT(result.first_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "an untrained policy that already clears the absolute bar would make this test vacuous";

    // (a) Relative improvement.
    EXPECT_GE(ratio, kMinImprovementRatio)
        << "final-10% average episode length must be at least " << kMinImprovementRatio
        << "x the first-10% average";

    // (b) Absolute competence -- so "improved from awful to slightly-less-awful" cannot pass.
    EXPECT_GE(result.last_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "final-10% average episode length must be at least " << (100.0 * kMinFinalFractionOfMaxSteps)
        << "% of max_steps (" << config.max_steps << ")";
}

// The loss is checked inside the loop on every single update, not once at the end -- a run that
// went NaN at update 12 and was rescued by Adam's epsilon would still be broken.
//
// Note what is deliberately *not* asserted: a sign. Unlike DQNLoss's mean-of-squares, the
// policy-gradient surrogate is return-weighted, and with standardized (zero-mean) returns
// roughly half of every batch's rows carry a negative weight. A negative loss value here is
// correct and expected; its magnitude and sign carry no information about policy quality, which
// is exactly why the performance bars above are measured in episode lengths instead.
TEST_F(ReinforceCartPoleIntegrationTest, EveryUpdateLossIsFinite) {
    EXPECT_GT(result.update_count, 0) << "no gradient updates ran at all";
    EXPECT_TRUE(result.all_losses_finite)
        << "loss first became non-finite at update " << result.first_non_finite_update << " of "
        << result.update_count;
    EXPECT_EQ(result.first_non_finite_update, -1);
    EXPECT_TRUE(std::isfinite(result.first_loss));
    EXPECT_TRUE(std::isfinite(result.last_loss));
    std::cout << "[REINFORCE] " << result.update_count << " updates, all finite; loss " << result.first_loss << " -> "
              << result.last_loss << std::endl;
}

// RolloutBuffer's whole usage protocol is the cycle fill -> compute_returns -> update -> clear.
// A run that filled the buffer once and stopped would exercise none of the cycling, and in
// particular would never prove that clear() leaves the buffer genuinely reusable (a clear() that
// forgot to reset the write index would throw std::logic_error on the very next add()).
// "Meaningfully > 1" is the bar the mission set; the tuned configuration runs far past it.
TEST_F(ReinforceCartPoleIntegrationTest, RolloutBufferIsCollectedAndClearedManyTimes) {
    EXPECT_GT(result.update_count, 10)
        << "only " << result.update_count << " collect-then-update cycles ran -- RolloutBuffer::clear() is barely "
                                             "exercised and the buffer's reuse path is effectively untested";

    // Every update consumes exactly rollout_length steps, and the run ends mid-rollout, so the
    // update count is the floor of total_steps / rollout_length. Pinning that identity is what
    // makes update_count trustworthy as a clear() counter rather than an incidental statistic.
    EXPECT_EQ(result.update_count, result.total_steps / config.rollout_length);
    std::cout << "[REINFORCE] " << result.total_steps << " env steps / " << config.rollout_length
              << " per rollout = " << result.update_count << " collect-update-clear cycles" << std::endl;
}

// The optimizer is the only thing that writes to the policy network's parameters, so first-layer
// weights that differ from their initial values are direct evidence the gradient actually
// reached them -- and not merely the output layer, which a broken SequentialModule::backward()
// could still have updated on its own.
TEST_F(ReinforceCartPoleIntegrationTest, PolicyNetworkParametersActuallyMove) {
    ASSERT_EQ(result.initial_policy_weight.size(), result.final_policy_weight.size());
    ASSERT_FALSE(result.initial_policy_weight.empty());
    std::cout << "[REINFORCE] policy first-layer max |final - initial| weight change: "
              << result.max_policy_weight_delta << std::endl;
    EXPECT_GT(result.max_policy_weight_delta, 1e-3)
        << "the policy network's first layer never moved -- the gradient did not reach it";
}

// Return standardization is a training-loop-level transform, so pin its contract directly rather
// than only through the end-to-end run: zero mean, unit variance, order preserved.
TEST(ReinforceReturnStandardizationTest, ProducesZeroMeanUnitVarianceAndPreservesOrder) {
    CPUBackend backend;
    const Tensor returns(Shape({5, 1}), &backend, {10.0f, 2.0f, 7.0f, 2.0f, 4.0f});
    const Tensor standardized = standardize_returns(returns, &backend);

    ASSERT_EQ(standardized.numel(), returns.numel());
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < standardized.numel(); ++i) {
        sum += static_cast<double>(standardized.data()[i]);
        sum_sq += static_cast<double>(standardized.data()[i]) * static_cast<double>(standardized.data()[i]);
    }
    EXPECT_NEAR(sum / 5.0, 0.0, 1e-5);
    EXPECT_NEAR(std::sqrt(sum_sq / 5.0), 1.0, 1e-5);

    // Strictly monotone, so the *relative* credit assignment REINFORCE depends on is untouched;
    // only the offset and the scale change. Equal inputs must stay equal.
    EXPECT_GT(standardized.data()[0], standardized.data()[2]);
    EXPECT_GT(standardized.data()[2], standardized.data()[4]);
    EXPECT_GT(standardized.data()[4], standardized.data()[1]);
    EXPECT_FLOAT_EQ(standardized.data()[1], standardized.data()[3]);
    // The largest return was above the batch mean and the smallest below it -- the sign flip is
    // the constant-baseline effect that makes this more than a rescaling.
    EXPECT_GT(standardized.data()[0], 0.0f);
    EXPECT_LT(standardized.data()[1], 0.0f);
}

// A degenerate rollout in which every return is identical has zero variance; standardizing it
// must produce all-zero weights (hence a zero gradient, i.e. no update), not a division by zero.
TEST(ReinforceReturnStandardizationTest, ConstantReturnsBecomeAllZeroRatherThanNaN) {
    CPUBackend backend;
    const Tensor returns(Shape({3, 1}), &backend, {5.0f, 5.0f, 5.0f});
    const Tensor standardized = standardize_returns(returns, &backend);
    for (int64_t i = 0; i < standardized.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(standardized.data()[i])) << "element " << i;
        EXPECT_FLOAT_EQ(standardized.data()[i], 0.0f) << "element " << i;
    }
}

// The threshold assertions above are only meaningful if the run is reproducible. Two short runs
// of the same configuration must agree on every episode length, bit for bit -- if they ever
// don't, something has reached for non-deterministic randomness and the exit-gate test has
// quietly become a coin flip. Deliberately short (a fraction of the full run's episodes): a
// divergence in seeding shows up within the first few episodes, not only at the end.
TEST(ReinforceCartPoleDeterminismTest, TwoRunsOfTheSameConfigurationAreIdentical) {
    TrainingConfig config;
    config.num_episodes = 60;

    const TrainingResult a = RunTraining(config);
    const TrainingResult b = RunTraining(config);

    ASSERT_EQ(a.episode_lengths.size(), b.episode_lengths.size());
    for (size_t i = 0; i < a.episode_lengths.size(); ++i) {
        EXPECT_EQ(a.episode_lengths[i], b.episode_lengths[i]) << "episode " << i;
    }
    EXPECT_EQ(a.total_steps, b.total_steps);
    EXPECT_EQ(a.update_count, b.update_count);
    ASSERT_GT(a.update_count, 1) << "too short to have exercised more than one update";
    EXPECT_FLOAT_EQ(a.first_loss, b.first_loss);
    EXPECT_FLOAT_EQ(a.last_loss, b.last_loss);

    ASSERT_EQ(a.final_policy_weight.size(), b.final_policy_weight.size());
    for (size_t i = 0; i < a.final_policy_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_policy_weight[i], b.final_policy_weight[i]) << "policy weight element " << i;
    }
}

}  // namespace
}  // namespace reinforce_cartpole
}  // namespace pulsatrix
