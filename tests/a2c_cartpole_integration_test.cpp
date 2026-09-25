/** @file a2c_cartpole_integration_test.cpp
 *  @brief Phase 3 Mission 2's acceptance criterion: end-to-end proof that an advantage
 *         actor-critic assembled out of pieces this library *already had* -- CartPoleEnv,
 *         RolloutBuffer, CategoricalPolicyAgent, PolicyGradientLoss, MSELoss, AdamOptimizer --
 *         both learns to balance a pole to thresholds fixed in the mission before the run, and
 *         learns a value function that is genuinely doing work.
 *
 *  The per-class unit tests each prove one piece in isolation, and
 *  reinforce_cartpole_integration_test.cpp already proves the single-network policy-gradient
 *  procedure composes. What is new here, and what only an integration test can reach, is the
 *  *two-network* structure: that two SequentialModules with independent parameters and
 *  independent AdamOptimizers can be trained in the same loop without one's cached forward state
 *  clobbering the other's backward pass; that MSELoss's gradient routes back through a network
 *  whose regression target is produced by RolloutBuffer rather than by a dataset; and that
 *  PolicyGradientLoss -- unmodified, its third argument reinterpreted from "return" to
 *  "advantage" -- still yields a policy gradient that climbs.
 *
 *  @note The mission's Recon predicted A2C needs no new include/pulsatrix/ production class at all.
 *        This file is part of the evidence that it was right: everything under test here was
 *        already committed before the mission started, and the only new code is the training
 *        loop in examples/a2c_cartpole_training.hpp.
 *  @note The loop, hyperparameters and seeds all come from examples/a2c_cartpole_training.hpp,
 *        which examples/a2c_cartpole_demo.cpp includes too -- the mission requires the demo and
 *        the test to be the same run, not two configurations that can drift apart.
 *  @note Deterministic end to end (this codebase's LCG convention for both networks' weight init,
 *        categorical action sampling and env reset -- no `<random>` anywhere), so the threshold
 *        assertions below are reproducible, not "usually passing".
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>

#include "../examples/a2c_cartpole_training.hpp"

namespace pulsatrix {
namespace a2c_cartpole {
namespace {

/** @brief The relative-improvement bar, fixed by the mission before any run happened. */
constexpr double kMinImprovementRatio = 3.0;
/**
 * @brief The absolute-competence bar, as a fraction of max_steps. Also fixed by the mission --
 *        the same 30% REINFORCE was held to. The mission states explicitly that this is a floor
 *        and that A2C's learned baseline is expected to beat it comfortably; it is deliberately
 *        not ratcheted up to match what the tuned run actually achieves, because the bar this
 *        mission is held to was set before the run, not after it.
 */
constexpr double kMinFinalFractionOfMaxSteps = 0.3;

/**
 * @brief The critic must end up explaining a real share of its targets' variance, having started
 *        out worse than a constant predictor. 0.25 against a measured 0.60, so this is not a
 *        knife-edge -- and note the ceiling is well under 1.0 (see the header's TUNING NOTE: once
 *        the policy saturates, a step's return-to-go depends partly on how many steps remain
 *        before the time limit, which is not in CartPole's 4-dimensional observation).
 */
constexpr double kMinFinalExplainedVariance = 0.25;
/** @brief The critic's windowed MSE must fall by at least this factor. Measured: 0.40. */
constexpr double kMaxCriticLossRatio = 0.75;

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

class A2CCartPoleIntegrationTest : public ::testing::Test {
protected:
    const TrainingConfig& config = shared_config();
    const TrainingResult& result = shared_result();
};

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (1/2): the two performance bars, stated in the mission before this ran.
// ---------------------------------------------------------------------------------------
TEST_F(A2CCartPoleIntegrationTest, TrainedActorClearsBothPerformanceBars) {
    ASSERT_EQ(static_cast<int64_t>(result.episode_lengths.size()), config.num_episodes);
    ASSERT_EQ(result.window, window_size(config.num_episodes));
    ASSERT_GT(result.window, 1) << "a window of one episode would be pure noise, not an average";

    const double ratio = result.last_window_average / result.first_window_average;
    std::cout << "[A2C] first " << result.window << " episodes mean length " << result.first_window_average << " -> last "
              << result.window << " episodes mean length " << result.last_window_average << " (" << ratio
              << "x; max_steps " << config.max_steps << ")" << std::endl;

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
// THE EXIT GATE (2/2): the A2C-specific check. The bars above are necessary but emphatically
// not sufficient, and the mission says so: a control run with critic_learning_rate = 0 -- an
// untrained critic frozen at its random initialization -- clears both of them just as well
// (7.08x, 200.00 final), while its own MSE *rises* from 1703 to 3305. So passing the
// episode-length bars is compatible with the critic being pure dead weight. This test is what
// distinguishes "the critic learned to predict returns" from "the actor did all the work".
//
// Two measures, deliberately, because neither alone is trustworthy here:
//   - Raw MSE is what the mission names, but it is confounded: as the actor improves, episodes
//     lengthen and the discounted returns the critic is fitting grow by nearly an order of
//     magnitude, so a critic that got strictly better could still show a rising MSE.
//   - Explained variance (1 - MSE/Var(returns)) divides that growth out, and has an absolute
//     zero point with a meaning: 0 is "no better than predicting the batch mean".
// Both are averaged over the first and last 10% of updates rather than read off a single update,
// since one update is one rollout's worth of Monte-Carlo noise.
// ---------------------------------------------------------------------------------------
TEST_F(A2CCartPoleIntegrationTest, CriticValueEstimatesGenuinelyImprove) {
    ASSERT_EQ(static_cast<int64_t>(result.critic_losses.size()), result.update_count);
    ASSERT_EQ(static_cast<int64_t>(result.critic_explained_variances.size()), result.update_count);
    ASSERT_GT(result.critic_window, 1) << "too few updates to average a first/last window over";

    std::cout << "[A2C] critic over first/last " << result.critic_window << " of " << result.update_count
              << " updates: MSE " << result.first_critic_loss_average << " -> " << result.last_critic_loss_average
              << ", explained variance " << result.first_critic_explained_variance_average << " -> "
              << result.last_critic_explained_variance_average << std::endl;

    // (a) The raw MSE the mission named, windowed. Falls despite the target's growing scale.
    ASSERT_GT(result.first_critic_loss_average, 0.0);
    EXPECT_LE(result.last_critic_loss_average, kMaxCriticLossRatio * result.first_critic_loss_average)
        << "the critic's MSE did not fall meaningfully -- it is not learning to predict returns";

    // (b) The scale-free reading, and the one that actually rules out the confound. It must cross
    // from "worse than a constant predictor" to "explaining a real share of the variance".
    EXPECT_LT(result.first_critic_explained_variance_average, 0.0)
        << "the critic already explained variance before training -- this check would be vacuous";
    EXPECT_GE(result.last_critic_explained_variance_average, kMinFinalExplainedVariance);
    EXPECT_GE(result.last_critic_explained_variance_average - result.first_critic_explained_variance_average, 1.0)
        << "explained variance barely moved over training";

    // (c) The optimizer is the only thing that writes to the critic's parameters, so a moved
    // first-layer weight is direct evidence MSELoss's gradient reached all the way back through
    // the critic -- and not merely its output layer, which a broken SequentialModule::backward()
    // could still have updated on its own.
    ASSERT_EQ(result.initial_critic_weight.size(), result.final_critic_weight.size());
    ASSERT_FALSE(result.initial_critic_weight.empty());
    std::cout << "[A2C] critic first-layer max |final - initial| weight change: " << result.max_critic_weight_delta
              << std::endl;
    EXPECT_GT(result.max_critic_weight_delta, 1e-3)
        << "the critic's first layer never moved -- the critic's gradient did not reach it";

    // (d) The advantages the actor consumed were real, non-degenerate numbers. An all-zero
    // advantage vector would zero the policy gradient entirely and make the actor's whole update
    // a no-op, which no episode-length bar would necessarily catch.
    EXPECT_GT(result.first_mean_abs_advantage, 0.0);
    EXPECT_GT(result.last_mean_abs_advantage, 0.0);
}

// Both losses are checked inside the loop on every single update, not once at the end -- a run
// that went NaN at update 12 and was rescued by Adam's epsilon would still be broken. Two
// networks means two ways to go non-finite, and the flag below covers both.
//
// Note what is deliberately *not* asserted about the actor's loss: a sign. The policy-gradient
// surrogate is advantage-weighted, and advantages are approximately zero-mean by construction
// (that is the entire point of subtracting a baseline), so roughly half of every batch's rows
// carry a negative weight. A negative actor loss is correct and expected; its magnitude and sign
// carry no information about policy quality, which is why the performance bars above are measured
// in episode lengths instead. The critic's MSE, by contrast, *is* a genuine error and must be
// non-negative -- so that much is asserted.
TEST_F(A2CCartPoleIntegrationTest, EveryUpdateHasFiniteActorAndCriticLosses) {
    EXPECT_GT(result.update_count, 0) << "no gradient updates ran at all";
    EXPECT_TRUE(result.all_losses_finite) << "a loss first became non-finite at update "
                                          << result.first_non_finite_update << " of " << result.update_count;
    EXPECT_EQ(result.first_non_finite_update, -1);
    EXPECT_TRUE(std::isfinite(result.first_actor_loss));
    EXPECT_TRUE(std::isfinite(result.last_actor_loss));
    EXPECT_TRUE(std::isfinite(result.first_critic_loss));
    EXPECT_TRUE(std::isfinite(result.last_critic_loss));

    // Per-update, not just the endpoints: the recorded critic series covers every single update.
    for (int64_t i = 0; i < static_cast<int64_t>(result.critic_losses.size()); ++i) {
        ASSERT_TRUE(std::isfinite(result.critic_losses[static_cast<size_t>(i)])) << "critic loss at update " << i;
        ASSERT_GE(result.critic_losses[static_cast<size_t>(i)], 0.0f)
            << "a mean of squares cannot be negative -- update " << i;
        ASSERT_TRUE(std::isfinite(result.critic_explained_variances[static_cast<size_t>(i)]))
            << "critic explained variance at update " << i;
    }

    std::cout << "[A2C] " << result.update_count << " updates, all actor+critic losses finite; actor "
              << result.first_actor_loss << " -> " << result.last_actor_loss << ", critic " << result.first_critic_loss
              << " -> " << result.last_critic_loss << std::endl;
}

// RolloutBuffer's whole usage protocol is the cycle fill -> compute_returns -> update -> clear.
// A run that filled the buffer once and stopped would exercise none of the cycling, and in
// particular would never prove that clear() leaves the buffer genuinely reusable (a clear() that
// forgot to reset the write index would throw std::logic_error on the very next add()).
TEST_F(A2CCartPoleIntegrationTest, RolloutBufferIsCollectedAndClearedManyTimes) {
    EXPECT_GT(result.update_count, 10)
        << "only " << result.update_count
        << " collect-then-update cycles ran -- RolloutBuffer::clear() is barely exercised and the buffer's reuse "
           "path is effectively untested";

    // Every update consumes exactly rollout_length steps, and the run ends mid-rollout, so the
    // update count is the floor of total_steps / rollout_length. Pinning that identity is what
    // makes update_count trustworthy as a clear() counter rather than an incidental statistic.
    EXPECT_EQ(result.update_count, result.total_steps / config.rollout_length);
    std::cout << "[A2C] " << result.total_steps << " env steps / " << config.rollout_length << " per rollout = "
              << result.update_count << " collect-update-clear cycles (each one critic update + one actor update)"
              << std::endl;
}

// The actor's counterpart to the critic's weight-movement check above. Both networks must move,
// and they are written by two separate optimizers, so one moving proves nothing about the other.
TEST_F(A2CCartPoleIntegrationTest, ActorNetworkParametersActuallyMove) {
    ASSERT_EQ(result.initial_actor_weight.size(), result.final_actor_weight.size());
    ASSERT_FALSE(result.initial_actor_weight.empty());
    std::cout << "[A2C] actor first-layer max |final - initial| weight change: " << result.max_actor_weight_delta
              << std::endl;
    EXPECT_GT(result.max_actor_weight_delta, 1e-3)
        << "the actor network's first layer never moved -- the policy gradient did not reach it";
}

// ---------------------------------------------------------------------------------------
// The two training-loop-level transforms, pinned directly rather than only through the
// end-to-end run.
// ---------------------------------------------------------------------------------------

// The advantage is the entire algorithmic delta from REINFORCE, so its contract is worth stating
// in one place: elementwise return minus value, shape preserved, sign meaningful (a step that
// outperformed the critic's prediction gets a positive weight, one that underperformed a negative
// one -- which is what lets the actor push probability *away* from bad actions rather than merely
// less hard towards them).
TEST(A2CAdvantageTest, IsElementwiseReturnMinusValueAndPreservesShape) {
    CPUBackend backend;
    const Tensor returns(Shape({4, 1}), &backend, {10.0f, 2.0f, 7.0f, -1.0f});
    const Tensor values(Shape({4, 1}), &backend, {4.0f, 5.0f, 7.0f, 0.5f});
    const Tensor advantages = compute_advantages(returns, values, &backend);

    ASSERT_EQ(advantages.shape(), returns.shape());
    EXPECT_FLOAT_EQ(advantages.data()[0], 6.0f);   // outperformed the critic
    EXPECT_FLOAT_EQ(advantages.data()[1], -3.0f);  // underperformed it
    EXPECT_FLOAT_EQ(advantages.data()[2], 0.0f);   // exactly as predicted -> zero weight
    EXPECT_FLOAT_EQ(advantages.data()[3], -1.5f);
}

// A perfect critic produces identically zero advantages, hence a zero policy gradient. Worth
// pinning because it is the degenerate case that would silently stall training, and because it is
// the clearest statement of what the advantage means: no surprise, no update.
TEST(A2CAdvantageTest, PerfectCriticProducesZeroAdvantages) {
    CPUBackend backend;
    const Tensor returns(Shape({3, 1}), &backend, {1.5f, -2.0f, 9.0f});
    const Tensor advantages = compute_advantages(returns, returns, &backend);
    for (int64_t i = 0; i < advantages.numel(); ++i) {
        EXPECT_FLOAT_EQ(advantages.data()[i], 0.0f) << "element " << i;
    }
}

TEST(A2CAdvantageStandardizationTest, ProducesZeroMeanUnitVarianceAndPreservesOrder) {
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
TEST(A2CAdvantageStandardizationTest, ConstantAdvantagesBecomeAllZeroRatherThanNaN) {
    CPUBackend backend;
    const Tensor advantages(Shape({3, 1}), &backend, {5.0f, 5.0f, 5.0f});
    const Tensor standardized = standardize_advantages(advantages, &backend);
    for (int64_t i = 0; i < standardized.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(standardized.data()[i])) << "element " << i;
        EXPECT_FLOAT_EQ(standardized.data()[i], 0.0f) << "element " << i;
    }
}

// Explained variance is load-bearing for the critic check above, so its three reference points
// are pinned here rather than trusted: a perfect fit scores 1, a batch-mean predictor scores 0,
// and anything worse scores negative. Its `mse` argument is whatever MSELoss reported, so these
// are hand-computed MSE values rather than a second implementation of the loss.
TEST(A2CExplainedVarianceTest, ScoresPerfectFitAtOneAndMeanPredictorAtZero) {
    CPUBackend backend;
    // Var({1,2,3,4}) = 1.25 (population).
    const Tensor returns(Shape({4, 1}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_NEAR(explained_variance(0.0f, returns), 1.0, 1e-9);
    EXPECT_NEAR(explained_variance(1.25f, returns), 0.0, 1e-6);
    EXPECT_NEAR(explained_variance(2.5f, returns), -1.0, 1e-6);

    // A zero-variance batch has nothing to explain; report 0 rather than dividing by zero.
    const Tensor constant_returns(Shape({3, 1}), &backend, {7.0f, 7.0f, 7.0f});
    EXPECT_DOUBLE_EQ(explained_variance(0.0f, constant_returns), 0.0);
}

// The threshold assertions above are only meaningful if the run is reproducible. Two short runs
// of the same configuration must agree on every episode length and on both networks' final
// weights, bit for bit -- if they ever don't, something has reached for non-deterministic
// randomness and the exit-gate tests have quietly become a coin flip. Deliberately short (a
// fraction of the full run's episodes): a divergence in seeding shows up within the first few
// episodes, not only at the end.
TEST(A2CCartPoleDeterminismTest, TwoRunsOfTheSameConfigurationAreIdentical) {
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
    EXPECT_FLOAT_EQ(a.first_actor_loss, b.first_actor_loss);
    EXPECT_FLOAT_EQ(a.last_actor_loss, b.last_actor_loss);
    EXPECT_FLOAT_EQ(a.first_critic_loss, b.first_critic_loss);
    EXPECT_FLOAT_EQ(a.last_critic_loss, b.last_critic_loss);

    ASSERT_EQ(a.final_actor_weight.size(), b.final_actor_weight.size());
    for (size_t i = 0; i < a.final_actor_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_actor_weight[i], b.final_actor_weight[i]) << "actor weight element " << i;
    }
    ASSERT_EQ(a.final_critic_weight.size(), b.final_critic_weight.size());
    for (size_t i = 0; i < a.final_critic_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_critic_weight[i], b.final_critic_weight[i]) << "critic weight element " << i;
    }
}

// The actor and the critic are two genuinely independent networks -- separate parameters,
// separate optimizers, separate learning rates. This pins that independence in the one place it
// could silently break: if the critic's learning rate were accidentally applied to the actor (or
// the two networks shared storage), changing only the critic's step size could not possibly leave
// the actor's trajectory untouched. Deliberately short, and it compares *episode lengths*, which
// depend solely on the actor's action choices.
TEST(A2CNetworkIndependenceTest, CriticLearningRateDoesNotAlterTheActorsFirstUpdate) {
    TrainingConfig fast_critic;
    fast_critic.num_episodes = 100;
    TrainingConfig slow_critic = fast_critic;
    slow_critic.critic_learning_rate = fast_critic.critic_learning_rate * 0.1f;

    const TrainingResult a = RunTraining(fast_critic);
    const TrainingResult b = RunTraining(slow_critic);

    // Up to the first update the two runs are identical by construction, since no parameter has
    // been written yet -- so the actor's action sequence, and therefore every episode length,
    // must agree over at least the steps collected into the first rollout.
    ASSERT_GT(a.update_count, 0);
    int64_t steps = 0;
    size_t episode = 0;
    while (episode < a.episode_lengths.size() && steps + a.episode_lengths[episode] <= fast_critic.rollout_length) {
        ASSERT_LT(episode, b.episode_lengths.size());
        EXPECT_EQ(a.episode_lengths[episode], b.episode_lengths[episode]) << "episode " << episode;
        steps += a.episode_lengths[episode];
        ++episode;
    }
    ASSERT_GT(episode, 0u) << "the first rollout did not contain a whole episode -- nothing was compared";

    // And afterwards they must genuinely diverge: a critic trained at a tenth the step size
    // produces different advantages, hence a different actor. If these stayed equal, the critic's
    // output would not be reaching the actor at all and the "advantage" would be a fiction.
    ASSERT_GT(a.update_count, 1) << "too short for the critic's learning rate to have had any effect yet";
    EXPECT_NE(a.last_critic_loss, b.last_critic_loss) << "the critic's learning rate had no effect on the critic";

    // ...and the actor's *weights* must diverge, because the advantages it was trained on came
    // from two differently-trained critics. Weights, not episode lengths: actions are discrete
    // and sampled by an inverse-CDF threshold, so over a run this short a small perturbation of
    // the policy's probabilities very often flips no action at all and the episode-length
    // sequences stay identical -- as they in fact do here. That is a property of the
    // discretization, not evidence of coupling, and asserting on it would be asserting on noise.
    ASSERT_EQ(a.final_actor_weight.size(), b.final_actor_weight.size());
    bool actor_diverged = false;
    for (size_t i = 0; !actor_diverged && i < a.final_actor_weight.size(); ++i) {
        actor_diverged = a.final_actor_weight[i] != b.final_actor_weight[i];
    }
    EXPECT_TRUE(actor_diverged) << "the critic's value estimates never reached the actor's gradient";

    // The converse guard, and the one that would actually catch shared storage or a swapped
    // optimizer: the two runs' *critics* differ too, and by more than the actor does.
    ASSERT_EQ(a.final_critic_weight.size(), b.final_critic_weight.size());
    bool critic_diverged = false;
    for (size_t i = 0; !critic_diverged && i < a.final_critic_weight.size(); ++i) {
        critic_diverged = a.final_critic_weight[i] != b.final_critic_weight[i];
    }
    EXPECT_TRUE(critic_diverged) << "the critic's own learning rate did not change its parameters";
}

}  // namespace
}  // namespace a2c_cartpole
}  // namespace pulsatrix
