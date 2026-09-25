/** @file ppo_cartpole_integration_test.cpp
 *  @brief Phase 3 Mission 4's acceptance criterion, and Phase 3's exit gate: end-to-end proof
 *         that a multi-epoch PPO agent assembled out of CartPoleEnv, RolloutBuffer,
 *         CategoricalPolicyAgent, ComputeGAE, PPOClippedLoss, MSELoss and AdamOptimizer learns to
 *         balance a pole to thresholds fixed in the mission before the run -- and, separately,
 *         that its *multi-epoch* update is genuinely multi-epoch.
 *
 *  The per-class unit tests each prove one piece in isolation (gae_test.cpp pins the GAE
 *  recursion against hand-derived values; ppo_clipped_loss_test.cpp pins the clipped objective's
 *  six gradient regions), and a2c_cartpole_integration_test.cpp already proves the two-network
 *  actor-critic structure composes. What is new here, and what only an integration test can
 *  reach, is the thing that makes PPO PPO: that the *same* rollout can be consumed by several
 *  gradient steps because the probability ratio measures the policy's drift from the
 *  data-collecting one, and the clip bounds it.
 *
 *  That claim has a specific, checkable shape, and this file checks it directly rather than
 *  inferring it from the episode-length bars:
 *    - the ratio starts at 1 at epoch 0 of every rollout (nothing has been written yet),
 *    - it moves measurably away from 1 by the last epoch of *every* rollout,
 *    - and it stays bounded while it does, with the clip zeroing a real (small) fraction of rows.
 *  The `num_epochs = 1` control at the bottom is what turns that from three numbers into an
 *  argument: with one epoch the ratio never leaves 1 anywhere in the run and the clip never
 *  engages at all.
 *
 *  @note The one production change this mission made is the two additive RolloutBuffer accessors,
 *        rewards() and dones(); their own direct tests live in tests/rollout_buffer_test.cpp.
 *        Everything else under test here was committed before the mission started, and the only
 *        other new code is the training loop in examples/ppo_cartpole_training.hpp.
 *  @note The loop, hyperparameters and seeds all come from examples/ppo_cartpole_training.hpp,
 *        which examples/ppo_cartpole_demo.cpp includes too -- the mission requires the demo and
 *        the test to be the same run, not two configurations that can drift apart.
 *  @note Deterministic end to end (this codebase's LCG convention for both networks' weight init,
 *        categorical action sampling and env reset -- no `<random>` anywhere), so the threshold
 *        assertions below are reproducible, not "usually passing".
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>

#include "../examples/ppo_cartpole_training.hpp"

namespace pulsatrix {
namespace ppo_cartpole {
namespace {

/** @brief The relative-improvement bar, fixed by the mission before any run happened. */
constexpr double kMinImprovementRatio = 3.0;
/**
 * @brief The absolute-competence bar, as a fraction of max_steps. Also fixed by the mission --
 *        the same 30% REINFORCE and A2C were held to. The mission states explicitly that this is
 *        a floor, not a ceiling; it is deliberately not ratcheted up to match what the tuned run
 *        actually achieves, because the bar this mission is held to was set before the run.
 */
constexpr double kMinFinalFractionOfMaxSteps = 0.3;

/**
 * @brief Epoch 0's mean |ratio - 1| must be *this* close to zero. Not exact equality: the old
 *        log-probabilities came from CategoricalPolicyAgent's single-row forward passes during
 *        collection and epoch 0 recomputes them from one batched forward over the whole rollout,
 *        so float reassociation leaves a residual. Measured: 2.8e-8, against a 1e-4 bar.
 */
constexpr double kEpochZeroRatioTolerance = 1e-4;
/**
 * @brief By its last epoch, each rollout's mean |ratio - 1| must exceed its first epoch's by at
 *        least this much -- the "the policy genuinely moved across epochs" bar. Measured over 306
 *        rollouts: worst 0.0036, typical 0.044.
 */
constexpr double kMinPerRolloutRatioDrift = 1e-3;
/**
 * @brief ...and must not exceed this, at any update. The clip's entire job is to bound the
 *        per-rollout policy movement, so an unbounded drift is a *failure* of the mechanism even
 *        if the episode-length bars still pass. Measured maximum over all 1224 updates: 0.135.
 */
constexpr double kMaxRatioDrift = 0.5;
/**
 * @brief No single update's surrogate may exceed this in magnitude. With standardized advantages
 *        the surrogate sits at ~1e-2 throughout; measured maximum 0.015.
 */
constexpr double kMaxAbsActorLoss = 1.0;

/**
 * @brief The critic's explained variance against GAE's targets must end up non-negative, having
 *        started clearly worse than a constant predictor. Deliberately a loose bar and a
 *        secondary signal, not this mission's exit criterion: unlike A2C's Monte-Carlo target,
 *        GAE's return `A_t + V(s_t)` moves with the critic that produced it, so this number is
 *        not a clean "did the critic learn" statistic. See the header's TUNING NOTE (7).
 *        Measured: -1.38 -> +0.11.
 */
constexpr double kMinFinalExplainedVariance = -0.2;

/** @brief The tuned default configuration -- the one the demo prints and this file asserts on. */
const TrainingConfig& shared_config() {
    static const TrainingConfig config;
    return config;
}

/**
 * @brief One shared training run for every assertion in this fixture.
 * @note Trained once, lazily, and reused: each test below inspects a different property of the
 *       *same* run. Re-training per test would multiply the suite's cost and prove nothing extra,
 *       precisely because the run is deterministic.
 */
const TrainingResult& shared_result() {
    static const TrainingResult result = RunTraining(shared_config());
    return result;
}

class PPOCartPoleIntegrationTest : public ::testing::Test {
protected:
    const TrainingConfig& config = shared_config();
    const TrainingResult& result = shared_result();
};

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (1/2): the two performance bars, stated in the mission before this ran.
// ---------------------------------------------------------------------------------------
TEST_F(PPOCartPoleIntegrationTest, TrainedActorClearsBothPerformanceBars) {
    ASSERT_EQ(static_cast<int64_t>(result.episode_lengths.size()), config.num_episodes);
    ASSERT_EQ(result.window, window_size(config.num_episodes));
    ASSERT_GT(result.window, 1) << "a window of one episode would be pure noise, not an average";

    const double ratio = result.last_window_average / result.first_window_average;
    std::cout << "[PPO] first " << result.window << " episodes mean length " << result.first_window_average
              << " -> last " << result.window << " episodes mean length " << result.last_window_average << " ("
              << ratio << "x; max_steps " << config.max_steps << ")" << std::endl;

    // Non-vacuity: a first window that was already competent would make the ratio bar unreachable
    // and the absolute bar free. It must genuinely start out bad.
    ASSERT_GT(result.first_window_average, 0.0);
    EXPECT_LT(result.first_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "an untrained actor that already clears the absolute bar would make this test vacuous";

    // (a) Relative improvement.
    EXPECT_GE(ratio, kMinImprovementRatio)
        << "final-10% average episode length must be at least " << kMinImprovementRatio << "x the first-10% average";

    // (b) Absolute competence -- so "improved from awful to slightly-less-awful" cannot pass.
    EXPECT_GE(result.last_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "final-10% average episode length must be at least " << (100.0 * kMinFinalFractionOfMaxSteps)
        << "% of max_steps (" << config.max_steps << ")";
}

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (2/2): the PPO-specific regression check the mission names.
//
// The bars above are necessary but not sufficient, and the mission says so: A2C already cleared
// them with a single gradient step per rollout, so passing them proves nothing about whether the
// *multi-epoch* machinery is doing anything at all. A `num_epochs = 4` run in which the ratio
// never left 1.0 would be four unclipped policy-gradient steps wearing PPO's name -- the loss
// would still be finite, the episodes would still lengthen, and no performance bar would notice.
//
// So this asserts the mechanism directly, in three parts:
//   (a) the ratio starts at 1 (nothing has been written since the data was collected),
//   (b) it moves measurably away from 1 by the last epoch -- on every single rollout, not merely
//       on average, which is the difference between "the policy drifts" and "some rollout once
//       drifted",
//   (c) it stays bounded while doing so, and the clip zeroes a real fraction of rows. An
//       unbounded drift would mean the clip is not bounding the movement, which is a failure of
//       the mechanism even when training still succeeds.
// ---------------------------------------------------------------------------------------
TEST_F(PPOCartPoleIntegrationTest, ProbabilityRatioMovesAwayFromOneAcrossEpochsWithoutDiverging) {
    ASSERT_GT(config.num_epochs, 1) << "this check is vacuous at num_epochs == 1 -- see the control test below";
    ASSERT_EQ(static_cast<int64_t>(result.ratio_stats.size()), result.update_count);
    ASSERT_EQ(result.update_count, result.rollout_count * config.num_epochs);
    ASSERT_EQ(static_cast<int64_t>(result.mean_abs_deviation_by_epoch.size()), config.num_epochs);

    for (int64_t e = 0; e < config.num_epochs; ++e) {
        std::cout << "[PPO] epoch " << e << ": mean |r - 1| = " << result.mean_abs_deviation_by_epoch[static_cast<size_t>(e)]
                  << ", clipped rows " << (100.0 * result.clipped_fraction_by_epoch[static_cast<size_t>(e)]) << "%"
                  << std::endl;
    }

    // (a) Epoch 0 has not moved. Near-zero rather than exactly zero, deliberately -- see
    // kEpochZeroRatioTolerance.
    EXPECT_LT(result.mean_abs_deviation_by_epoch[0], kEpochZeroRatioTolerance)
        << "epoch 0's ratio must be 1: no parameter has been written since the data was collected";
    EXPECT_DOUBLE_EQ(result.clipped_fraction_by_epoch[0], 0.0)
        << "a ratio of 1 is inside every trust region, so nothing can be clipped at epoch 0";

    // (b) Strictly increasing drift with epoch index, in the rollout-averaged summary...
    for (int64_t e = 1; e < config.num_epochs; ++e) {
        EXPECT_GT(result.mean_abs_deviation_by_epoch[static_cast<size_t>(e)],
                  result.mean_abs_deviation_by_epoch[static_cast<size_t>(e - 1)])
            << "the policy stopped moving between epoch " << (e - 1) << " and epoch " << e;
    }
    EXPECT_GE(result.mean_abs_deviation_by_epoch[static_cast<size_t>(config.num_epochs - 1)],
              10.0 * kEpochZeroRatioTolerance)
        << "the final epoch's drift is indistinguishable from epoch 0's numerical residual -- the "
           "extra epochs are not actually changing the policy";

    // ...and on *every individual rollout*, which is the statement that rules out "one rollout
    // drifted and the average carried the rest".
    double worst_drift = 1e30;
    int64_t worst_rollout = -1;
    for (int64_t r = 0; r < result.rollout_count; ++r) {
        const size_t first = static_cast<size_t>(r * config.num_epochs);
        const size_t last = static_cast<size_t>(r * config.num_epochs + config.num_epochs - 1);
        const double drift = result.ratio_stats[last].mean_abs_deviation - result.ratio_stats[first].mean_abs_deviation;
        if (drift < worst_drift) {
            worst_drift = drift;
            worst_rollout = r;
        }
    }
    std::cout << "[PPO] worst per-rollout drift (last epoch minus first): " << worst_drift << " at rollout "
              << worst_rollout << " of " << result.rollout_count << std::endl;
    EXPECT_GE(worst_drift, kMinPerRolloutRatioDrift)
        << "rollout " << worst_rollout << " ran " << config.num_epochs
        << " epochs without the policy moving -- that rollout's updates were num_epochs = 1 in disguise";

    // (c) Bounded movement, bounded loss, and a clip that genuinely engages.
    double max_mean_drift = 0.0;
    double max_row_drift = 0.0;
    double max_abs_actor_loss = 0.0;
    for (int64_t i = 0; i < result.update_count; ++i) {
        const RatioStats& stats = result.ratio_stats[static_cast<size_t>(i)];
        ASSERT_TRUE(std::isfinite(stats.mean_abs_deviation)) << "ratio went non-finite at update " << i;
        max_mean_drift = std::fmax(max_mean_drift, stats.mean_abs_deviation);
        max_row_drift = std::fmax(max_row_drift, stats.max_abs_deviation);
        max_abs_actor_loss =
            std::fmax(max_abs_actor_loss, std::fabs(static_cast<double>(result.actor_losses[static_cast<size_t>(i)])));
    }
    std::cout << "[PPO] max over all " << result.update_count << " updates: mean |r - 1| = " << max_mean_drift
              << ", single-row |r - 1| = " << max_row_drift << ", |actor loss| = " << max_abs_actor_loss << std::endl;
    EXPECT_LE(max_mean_drift, kMaxRatioDrift)
        << "the policy's per-rollout movement is not being bounded -- the clip is not doing its job";
    EXPECT_LE(max_abs_actor_loss, kMaxAbsActorLoss) << "the clipped surrogate diverged";

    // The clip must bite somewhere, or `clip_epsilon` is a parameter with no effect on this run
    // and the 'bounded' result above would be a property of the step size rather than of PPO.
    EXPECT_GT(result.clipped_fraction_by_epoch[static_cast<size_t>(config.num_epochs - 1)], 0.0)
        << "no row was ever clipped -- the trust region is never reached, so this run does not "
           "actually exercise the clipped objective";
    // ...but only on a minority of rows. If most rows were clipped, most of the batch would carry
    // exactly zero gradient and the later epochs would be doing almost nothing.
    EXPECT_LT(result.clipped_fraction_by_epoch[static_cast<size_t>(config.num_epochs - 1)], 0.5)
        << "most of the batch's gradient is being zeroed by the clip";
}

// Both losses are checked inside the loop on every single update -- and for PPO "every update"
// means every epoch of every rollout, not once per rollout: a run that went NaN at epoch 3 of
// rollout 12 and was rescued by Adam's epsilon would still be broken, and a per-rollout check
// would sample only a quarter of the updates.
//
// Note what is deliberately *not* asserted about the actor's loss: a sign. The clipped surrogate
// is advantage-weighted, advantages are standardized to zero mean here, and the loss is the
// negation of a maximized objective -- so a negative actor loss is correct and expected and its
// magnitude carries no information about policy quality. The critic's MSE, by contrast, *is* a
// genuine error and must be non-negative, so that much is asserted.
TEST_F(PPOCartPoleIntegrationTest, EveryEpochOfEveryRolloutHasFiniteActorAndCriticLosses) {
    EXPECT_GT(result.update_count, 0) << "no gradient updates ran at all";
    EXPECT_EQ(result.update_count, result.rollout_count * config.num_epochs)
        << "the epoch loop did not run num_epochs times per rollout";
    ASSERT_EQ(static_cast<int64_t>(result.actor_losses.size()), result.update_count);
    ASSERT_EQ(static_cast<int64_t>(result.critic_losses.size()), result.update_count);

    EXPECT_TRUE(result.all_losses_finite) << "a loss first became non-finite at update "
                                          << result.first_non_finite_update << " of " << result.update_count;
    EXPECT_EQ(result.first_non_finite_update, -1);

    for (int64_t i = 0; i < result.update_count; ++i) {
        const int64_t rollout = i / config.num_epochs;
        const int64_t epoch = i % config.num_epochs;
        ASSERT_TRUE(std::isfinite(result.actor_losses[static_cast<size_t>(i)]))
            << "actor loss at rollout " << rollout << " epoch " << epoch;
        ASSERT_TRUE(std::isfinite(result.critic_losses[static_cast<size_t>(i)]))
            << "critic loss at rollout " << rollout << " epoch " << epoch;
        ASSERT_GE(result.critic_losses[static_cast<size_t>(i)], 0.0f)
            << "a mean of squares cannot be negative -- rollout " << rollout << " epoch " << epoch;
    }

    std::cout << "[PPO] " << result.update_count << " updates (" << result.rollout_count << " rollouts x "
              << config.num_epochs << " epochs), all actor+critic losses finite; actor " << result.first_actor_loss
              << " -> " << result.last_actor_loss << ", critic " << result.first_critic_loss << " -> "
              << result.last_critic_loss << std::endl;
}

// RolloutBuffer's whole usage protocol is the cycle fill -> read -> update -> clear. A run that
// filled the buffer once and stopped would exercise none of the cycling, and in particular would
// never prove that clear() leaves the buffer genuinely reusable (a clear() that forgot to reset
// the write index would throw std::logic_error on the very next add()).
//
// PPO adds a second thing worth pinning here: the two accessors this mission introduced,
// rewards() and dones(), are read *after* compute_returns() and *before* clear(), once per
// rollout, and must therefore survive several hundred fill/clear cycles reporting only the
// current rollout's steps. Their direct contract tests are in rollout_buffer_test.cpp; this is
// the end-to-end evidence that the cycle around them holds.
TEST_F(PPOCartPoleIntegrationTest, RolloutBufferIsCollectedAndClearedManyTimes) {
    EXPECT_GT(result.rollout_count, 10)
        << "only " << result.rollout_count
        << " collect-then-update cycles ran -- RolloutBuffer::clear() is barely exercised and the buffer's reuse "
           "path is effectively untested";

    // Every cycle consumes exactly rollout_length steps, and the run ends mid-rollout, so the
    // cycle count is the floor of total_steps / rollout_length. Pinning that identity is what
    // makes rollout_count trustworthy as a clear() counter rather than an incidental statistic.
    EXPECT_EQ(result.rollout_count, result.total_steps / config.rollout_length);
    std::cout << "[PPO] " << result.total_steps << " env steps / " << config.rollout_length << " per rollout = "
              << result.rollout_count << " collect-update-clear cycles, " << config.num_epochs
              << " gradient steps per network each" << std::endl;
}

// Both networks are written by separate optimizers, so one moving proves nothing about the other.
// The optimizer is the only thing that writes parameters, so a moved *first-layer* weight is
// direct evidence each loss's gradient reached all the way back through its network -- and not
// merely its output layer, which a broken SequentialModule::backward() could still have updated.
TEST_F(PPOCartPoleIntegrationTest, BothNetworksParametersActuallyMove) {
    ASSERT_EQ(result.initial_actor_weight.size(), result.final_actor_weight.size());
    ASSERT_FALSE(result.initial_actor_weight.empty());
    ASSERT_EQ(result.initial_critic_weight.size(), result.final_critic_weight.size());
    ASSERT_FALSE(result.initial_critic_weight.empty());

    std::cout << "[PPO] first-layer max |final - initial| weight change: actor " << result.max_actor_weight_delta
              << ", critic " << result.max_critic_weight_delta << std::endl;
    EXPECT_GT(result.max_actor_weight_delta, 1e-3)
        << "the actor network's first layer never moved -- the clipped surrogate's gradient did not reach it";
    EXPECT_GT(result.max_critic_weight_delta, 1e-3)
        << "the critic's first layer never moved -- MSELoss's gradient did not reach it";
}

// A secondary sanity signal, not the exit gate -- see kMinFinalExplainedVariance and the header's
// TUNING NOTE (7) for why GAE's self-referential target makes this weaker evidence than A2C's
// equivalent check. What it does rule out is a critic that is pure dead weight: the advantages
// the actor consumed must be real, non-degenerate numbers (an all-zero advantage vector would
// zero the policy gradient entirely and make every epoch a no-op), and the critic's fit must move
// from "much worse than predicting the batch mean" to "not worse than it".
TEST_F(PPOCartPoleIntegrationTest, CriticFitImprovesAndAdvantagesAreNonDegenerate) {
    ASSERT_EQ(static_cast<int64_t>(result.critic_explained_variances.size()), result.rollout_count);
    ASSERT_GT(result.critic_window, 1) << "too few rollouts to average a first/last window over";

    std::cout << "[PPO] critic explained variance over first/last " << result.critic_window << " of "
              << result.rollout_count << " rollouts: " << result.first_critic_explained_variance_average << " -> "
              << result.last_critic_explained_variance_average << std::endl;

    EXPECT_LT(result.first_critic_explained_variance_average, 0.0)
        << "the critic already explained variance before training -- this check would be vacuous";
    EXPECT_GE(result.last_critic_explained_variance_average, kMinFinalExplainedVariance);
    EXPECT_GE(result.last_critic_explained_variance_average - result.first_critic_explained_variance_average, 1.0)
        << "explained variance barely moved over training";

    for (int64_t i = 0; i < result.rollout_count; ++i) {
        ASSERT_TRUE(std::isfinite(result.critic_explained_variances[static_cast<size_t>(i)]))
            << "explained variance at rollout " << i;
    }

    std::cout << "[PPO] mean |advantage| (pre-standardization): " << result.first_mean_abs_advantage << " -> "
              << result.last_mean_abs_advantage << std::endl;
    EXPECT_GT(result.first_mean_abs_advantage, 0.0);
    EXPECT_GT(result.last_mean_abs_advantage, 0.0);
}

// ---------------------------------------------------------------------------------------
// The control that turns the ratio-movement numbers into an argument.
//
// With num_epochs = 1 the policy cannot possibly have drifted when the loss is evaluated: the
// single update happens *after* the only forward pass. So the ratio must be 1 everywhere in the
// entire run and the clip must never engage -- PPOClippedLoss degenerating, exactly as its own
// header documents, to an advantage-weighted policy gradient. If this test's numbers looked like
// the multi-epoch run's, the diagnostic above would be measuring something other than what it
// claims (a bug in probability_ratio_stats, say), and its passing would mean nothing.
//
// Deliberately short: a nonzero ratio would show up in the first handful of rollouts, not only
// at the end.
// ---------------------------------------------------------------------------------------
TEST(PPOSingleEpochControlTest, WithOneEpochPerRolloutTheRatioNeverLeavesOne) {
    TrainingConfig config;
    config.num_epochs = 1;
    config.num_episodes = 200;

    const TrainingResult result = RunTraining(config);

    ASSERT_GT(result.rollout_count, 1) << "too short to have run more than one rollout";
    ASSERT_EQ(result.update_count, result.rollout_count) << "one epoch per rollout means one update per rollout";

    double max_row_drift = 0.0;
    double max_clipped = 0.0;
    for (const RatioStats& stats : result.ratio_stats) {
        max_row_drift = std::fmax(max_row_drift, stats.max_abs_deviation);
        max_clipped = std::fmax(max_clipped, stats.clipped_fraction);
    }
    std::cout << "[PPO control] num_epochs=1 over " << result.rollout_count
              << " rollouts: max single-row |r - 1| = " << max_row_drift << ", max clipped fraction = " << max_clipped
              << std::endl;

    // Not even the single most extreme row in the whole run moves -- and this is the *max*, not a
    // mean, so it cannot be an averaging artifact.
    EXPECT_LT(max_row_drift, kEpochZeroRatioTolerance);
    EXPECT_DOUBLE_EQ(max_clipped, 0.0) << "nothing can be clipped when every ratio is exactly inside the trust region";

    // And the run still trains -- the point of the control is that the ratio diagnostic
    // distinguishes the two cases, not that one epoch fails to learn.
    EXPECT_GT(result.last_window_average, result.first_window_average);
}

// ---------------------------------------------------------------------------------------
// The two training-loop-level transforms, pinned directly rather than only through the
// end-to-end run.
// ---------------------------------------------------------------------------------------

// probability_ratio_stats() is load-bearing for the exit-gate check above, and it is an
// independent reimplementation of the ratio PPOClippedLoss computes internally, so its contract
// is worth pinning against hand-computable logits rather than trusted.
//
// Two actions, logits (0, 0) -> p = (0.5, 0.5) -> log p = -log 2. A row whose old log-prob is
// also -log 2 has ratio exactly 1; a row whose old log-prob was -log 4 (the policy used to assign
// the action a quarter, now assigns it a half) has ratio exactly 2.
TEST(PPOProbabilityRatioTest, MatchesHandComputedRatiosAndFlagsTheClippedRows) {
    CPUBackend backend;
    const float log_half = -std::log(2.0f);
    const float log_quarter = -std::log(4.0f);

    const Tensor logits(Shape({3, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    const Tensor actions(Shape({3, 1}), &backend, {0.0f, 1.0f, 0.0f});
    const Tensor old_log_probs(Shape({3, 1}), &backend, {log_half, log_quarter, log_quarter});
    // Row 1 has a positive advantage and ratio 2 > 1.2 -- clipped (its gradient is structurally
    // zeroed). Row 2 has ratio 2 as well, but a *negative* advantage, so the clip does not bite:
    // PPO does want to pull an over-rewarded bad step back. That asymmetry is the `min`'s whole
    // point, and it is exactly PPOClippedLoss::backward()'s documented mask condition.
    const Tensor advantages(Shape({3, 1}), &backend, {1.0f, 1.0f, -1.0f});

    const RatioStats stats = probability_ratio_stats(logits, actions, old_log_probs, advantages, 0.2f);

    EXPECT_NEAR(stats.mean_ratio, (1.0 + 2.0 + 2.0) / 3.0, 1e-6);
    EXPECT_NEAR(stats.mean_abs_deviation, (0.0 + 1.0 + 1.0) / 3.0, 1e-6);
    EXPECT_NEAR(stats.max_abs_deviation, 1.0, 1e-6);
    EXPECT_NEAR(stats.clipped_fraction, 1.0 / 3.0, 1e-9);
}

// The degenerate case the exit-gate check reads as its baseline: when the current policy is the
// data-collecting policy, every ratio is 1, nothing is clipped, and the drift is exactly zero.
// Asserted with unequal logits so this cannot pass by the uniform-distribution coincidence above.
TEST(PPOProbabilityRatioTest, IsIdenticallyOneWhenThePolicyHasNotMoved) {
    CPUBackend backend;
    const Tensor logits(Shape({2, 3}), &backend, {2.0f, -1.0f, 0.5f, -3.0f, 4.0f, 1.0f});
    const Tensor actions(Shape({2, 1}), &backend, {2.0f, 1.0f});

    // The stable log-softmax of each row at its own chosen action -- what CategoricalPolicyAgent
    // would have cached at collection time.
    std::vector<float> old_values(2);
    for (int64_t b = 0; b < 2; ++b) {
        const float* row = logits.data() + b * 3;
        float row_max = row[0];
        for (int64_t k = 1; k < 3; ++k) {
            row_max = row[k] > row_max ? row[k] : row_max;
        }
        double exp_sum = 0.0;
        for (int64_t k = 0; k < 3; ++k) {
            exp_sum += std::exp(static_cast<double>(row[k] - row_max));
        }
        const int64_t action = static_cast<int64_t>(actions.data()[b] + 0.5f);
        old_values[static_cast<size_t>(b)] =
            static_cast<float>(static_cast<double>(row[action] - row_max) - std::log(exp_sum));
    }
    const Tensor old_log_probs(Shape({2, 1}), &backend, old_values);
    const Tensor advantages(Shape({2, 1}), &backend, {3.0f, -3.0f});

    const RatioStats stats = probability_ratio_stats(logits, actions, old_log_probs, advantages, 0.2f);

    EXPECT_NEAR(stats.mean_ratio, 1.0, 1e-6);
    EXPECT_NEAR(stats.mean_abs_deviation, 0.0, 1e-6);
    EXPECT_DOUBLE_EQ(stats.clipped_fraction, 0.0);
}

TEST(PPOAdvantageStandardizationTest, ProducesZeroMeanUnitVarianceAndPreservesOrder) {
    CPUBackend backend;
    const Tensor advantages(Shape({5, 1}), &backend, {10.0f, 2.0f, 7.0f, 2.0f, 4.0f});
    const Tensor standardized = standardize_advantages(advantages, &backend);

    ASSERT_EQ(standardized.numel(), advantages.numel());
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < standardized.numel(); ++i) {
        sum += static_cast<double>(standardized.data()[i]);
        sum_sq += static_cast<double>(standardized.data()[i]) * static_cast<double>(standardized.data()[i]);
    }
    EXPECT_NEAR(sum / 5.0, 0.0, 1e-5);
    EXPECT_NEAR(std::sqrt(sum_sq / 5.0), 1.0, 1e-5);

    // Strictly monotone, so the relative credit assignment is untouched; only offset and scale
    // change. Equal inputs must stay equal.
    EXPECT_GT(standardized.data()[0], standardized.data()[2]);
    EXPECT_GT(standardized.data()[2], standardized.data()[4]);
    EXPECT_GT(standardized.data()[4], standardized.data()[1]);
    EXPECT_FLOAT_EQ(standardized.data()[1], standardized.data()[3]);
}

// A degenerate rollout in which every advantage is identical has zero variance; standardizing it
// must produce all-zero weights (hence a zero gradient, i.e. no update), not a division by zero.
TEST(PPOAdvantageStandardizationTest, ConstantAdvantagesBecomeAllZeroRatherThanNaN) {
    CPUBackend backend;
    const Tensor advantages(Shape({3, 1}), &backend, {5.0f, 5.0f, 5.0f});
    const Tensor standardized = standardize_advantages(advantages, &backend);
    for (int64_t i = 0; i < standardized.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(standardized.data()[i])) << "element " << i;
        EXPECT_FLOAT_EQ(standardized.data()[i], 0.0f) << "element " << i;
    }
}

TEST(PPOExplainedVarianceTest, ScoresPerfectFitAtOneAndMeanPredictorAtZero) {
    CPUBackend backend;
    // Var({1,2,3,4}) = 1.25 (population).
    const Tensor targets(Shape({4, 1}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_NEAR(explained_variance(0.0f, targets), 1.0, 1e-9);
    EXPECT_NEAR(explained_variance(1.25f, targets), 0.0, 1e-6);
    EXPECT_NEAR(explained_variance(2.5f, targets), -1.0, 1e-6);

    // A zero-variance batch has nothing to explain; report 0 rather than dividing by zero.
    const Tensor constant_targets(Shape({3, 1}), &backend, {7.0f, 7.0f, 7.0f});
    EXPECT_DOUBLE_EQ(explained_variance(0.0f, constant_targets), 0.0);
}

// The threshold assertions above are only meaningful if the run is reproducible. Two short runs
// of the same configuration must agree on every episode length, on both networks' final weights
// and on the ratio diagnostics, bit for bit -- if they ever don't, something has reached for
// non-deterministic randomness and the exit-gate tests have quietly become a coin flip.
// Deliberately short (a fraction of the full run's episodes): a divergence in seeding shows up
// within the first few episodes, not only at the end.
TEST(PPOCartPoleDeterminismTest, TwoRunsOfTheSameConfigurationAreIdentical) {
    TrainingConfig config;
    config.num_episodes = 60;

    const TrainingResult a = RunTraining(config);
    const TrainingResult b = RunTraining(config);

    ASSERT_EQ(a.episode_lengths.size(), b.episode_lengths.size());
    for (size_t i = 0; i < a.episode_lengths.size(); ++i) {
        EXPECT_EQ(a.episode_lengths[i], b.episode_lengths[i]) << "episode " << i;
    }
    EXPECT_EQ(a.total_steps, b.total_steps);
    EXPECT_EQ(a.rollout_count, b.rollout_count);
    EXPECT_EQ(a.update_count, b.update_count);
    ASSERT_GT(a.rollout_count, 1) << "too short to have exercised more than one rollout";

    ASSERT_EQ(a.actor_losses.size(), b.actor_losses.size());
    for (size_t i = 0; i < a.actor_losses.size(); ++i) {
        EXPECT_FLOAT_EQ(a.actor_losses[i], b.actor_losses[i]) << "actor loss at update " << i;
        EXPECT_FLOAT_EQ(a.critic_losses[i], b.critic_losses[i]) << "critic loss at update " << i;
        EXPECT_DOUBLE_EQ(a.ratio_stats[i].mean_abs_deviation, b.ratio_stats[i].mean_abs_deviation)
            << "ratio drift at update " << i;
    }

    ASSERT_EQ(a.final_actor_weight.size(), b.final_actor_weight.size());
    for (size_t i = 0; i < a.final_actor_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_actor_weight[i], b.final_actor_weight[i]) << "actor weight element " << i;
    }
    ASSERT_EQ(a.final_critic_weight.size(), b.final_critic_weight.size());
    for (size_t i = 0; i < a.final_critic_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_critic_weight[i], b.final_critic_weight[i]) << "critic weight element " << i;
    }
}

}  // namespace
}  // namespace ppo_cartpole
}  // namespace pulsatrix
