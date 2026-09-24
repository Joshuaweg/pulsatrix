/** @file sac_continuous_cartpole_integration_test.cpp
 *  @brief Phase 4 Mission 2's acceptance criterion -- Phase 4's exit gate, and the exit gate of
 *         the entire Reinforcement Learning campaign: end-to-end proof that a Soft Actor-Critic
 *         agent assembled out of ContinuousCartPoleEnv, ReplayBuffer, TanhGaussianPolicy,
 *         PolyakUpdate, MSELoss, SyncTargetNetwork and AdamOptimizer learns to balance a pole
 *         with a *continuous* action to thresholds fixed in the mission before the run -- and,
 *         separately, that the two mechanisms this mission is specifically about are real.
 *
 *  The per-class unit tests each prove one piece in isolation (tanh_gaussian_policy_test.cpp pins
 *  the reparameterized sample and its log-density against hand-derived values and finite
 *  differences; polyak_update_test.cpp pins the soft blend, including its `tau == 1` degeneracy to
 *  SyncTargetNetwork; continuous_cartpole_env_test.cpp pins the physics against the already-
 *  verified discrete environment), and the four prior training integrations prove that the
 *  collect-update-report shape composes. What is new here, and what only an integration test can
 *  reach, is the pair of things that make SAC SAC:
 *
 *    (a) **The entropy term is genuinely additive in the actor's loss** -- the phase's own
 *        exit-gate wording. Asserted at unit level against a hand-computable scenario
 *        (`actor_loss(alpha = 0.5) - actor_loss(alpha = 0.0) == 0.5 * mean(log_prob)`), not
 *        merely inferred from the fact that training worked. A run whose entropy term were
 *        multiplied by zero somewhere would still train and would still clear both bars.
 *    (b) **The gradient-contamination fix is load-bearing.** The actor update borrows both
 *        critics' `backward()` for their *input*-gradients, which unavoidably deposits the
 *        actor's objective into the critics' *parameter*-gradient buffers. The regression test at
 *        the bottom runs two identically-initialized critic pairs, one with the `zero_grad()` fix
 *        and one deliberately without, and shows their next real critic gradient differs by
 *        exactly the leftover contamination -- the same proof shape gan_integration_test.cpp's
 *        `SkippingDiscriminatorZeroGradAfterGeneratorStepCorruptsIt` used for the identical
 *        hazard.
 *
 *  @note This mission made **no** production change whatsoever -- no new `include/exai/` class and
 *        not even an additive accessor (contrast the PPO mission's two RolloutBuffer accessors).
 *        Everything under test here was committed before the mission started; the only new code
 *        is the training loop in examples/sac_continuous_cartpole_training.hpp.
 *  @note The loop, hyperparameters and seeds all come from that header, which
 *        examples/sac_continuous_cartpole_demo.cpp includes too -- the mission requires the demo
 *        and the test to be the same run, not two configurations that can drift apart.
 *  @note Deterministic end to end. Every random draw is one of this codebase's LCGs: the four
 *        networks' weight init, the environment's reset, the replay buffer's sampling, and -- new
 *        to this mission -- the reparameterization noise, via the header's Box-Muller NormalNoise
 *        on top of the same LCG. No `<random>` anywhere, so the threshold assertions below are
 *        reproducible rather than "usually passing".
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../examples/sac_continuous_cartpole_training.hpp"

namespace exai {
namespace sac_continuous_cartpole {
namespace {

/** @brief The relative-improvement bar, fixed by the mission before any run happened. */
constexpr double kMinImprovementRatio = 3.0;
/**
 * @brief The absolute-competence bar, as a fraction of max_steps -- the same 30% REINFORCE, A2C
 *        and PPO were held to. The mission states explicitly that this is a floor, not a ceiling;
 *        it is deliberately not ratcheted up to match what the tuned run actually achieves,
 *        because the bar this mission is held to was set before the run.
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
 *       *same* run. Re-training per test would multiply the suite's cost (this is the campaign's
 *       most expensive single loop -- four trainable networks stepped on every environment step)
 *       and prove nothing extra, precisely because the run is deterministic.
 */
const TrainingResult& shared_result() {
    static const TrainingResult result = RunTraining(shared_config());
    return result;
}

class SACContinuousCartPoleIntegrationTest : public ::testing::Test {
protected:
    const TrainingConfig& config = shared_config();
    const TrainingResult& result = shared_result();
};

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (1/3): the two performance bars, stated in the mission before this ran.
// ---------------------------------------------------------------------------------------
TEST_F(SACContinuousCartPoleIntegrationTest, TrainedActorClearsBothPerformanceBars) {
    ASSERT_EQ(static_cast<int64_t>(result.episode_lengths.size()), config.num_episodes);
    ASSERT_EQ(result.window, kWindow);

    const double ratio = result.last_window_average / result.first_window_average;
    std::cout << "[SAC] first " << result.window << " episodes mean length " << result.first_window_average
              << " -> last " << result.window << " episodes mean length " << result.last_window_average << " ("
              << ratio << "x; max_steps " << config.max_steps << ")" << std::endl;

    // Non-vacuity: a first window that was already competent would make the ratio bar unreachable
    // and the absolute bar free. It must genuinely start out bad.
    ASSERT_GT(result.first_window_average, 0.0);
    EXPECT_LT(result.first_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "an untrained actor that already clears the absolute bar would make this test vacuous";

    // (a) Relative improvement.
    EXPECT_GE(ratio, kMinImprovementRatio)
        << "final-" << kWindow << " average episode length must be at least " << kMinImprovementRatio
        << "x the first-" << kWindow << " average";

    // (b) Absolute competence -- so "improved from awful to slightly-less-awful" cannot pass.
    EXPECT_GE(result.last_window_average, kMinFinalFractionOfMaxSteps * static_cast<double>(config.max_steps))
        << "final-" << kWindow << " average episode length must be at least "
        << (100.0 * kMinFinalFractionOfMaxSteps) << "% of max_steps (" << config.max_steps << ")";
}

// All THREE losses are checked on every single update, not one representative of them: SAC trains
// two critics and an actor per step, and a run in which critic2 alone went NaN would leave the
// `min` permanently selecting critic1 while critic1's own loss stayed perfectly finite and the
// episode-length bars noticed nothing.
//
// Note what is deliberately *not* asserted about the actor's loss: a sign. `alpha*log_prob -
// min(Q1, Q2)` is the negation of a maximized objective and is dominated by the Q term, so a
// large negative actor loss is correct and expected. The critics' MSE, by contrast, *is* a genuine
// error and must be non-negative, so that much is asserted.
TEST_F(SACContinuousCartPoleIntegrationTest, EveryUpdateHasFiniteCritic1Critic2AndActorLosses) {
    EXPECT_GT(result.update_count, 0) << "no gradient updates ran at all -- the warm-up was never cleared";
    ASSERT_EQ(static_cast<int64_t>(result.critic1_losses.size()), result.update_count);
    ASSERT_EQ(static_cast<int64_t>(result.critic2_losses.size()), result.update_count);
    ASSERT_EQ(static_cast<int64_t>(result.actor_losses.size()), result.update_count);

    EXPECT_TRUE(result.all_losses_finite) << "a loss first became non-finite at update "
                                          << result.first_non_finite_update << " of " << result.update_count;
    EXPECT_EQ(result.first_non_finite_update, -1);

    for (int64_t i = 0; i < result.update_count; ++i) {
        ASSERT_TRUE(std::isfinite(result.critic1_losses[static_cast<size_t>(i)])) << "critic1 loss at update " << i;
        ASSERT_TRUE(std::isfinite(result.critic2_losses[static_cast<size_t>(i)])) << "critic2 loss at update " << i;
        ASSERT_TRUE(std::isfinite(result.actor_losses[static_cast<size_t>(i)])) << "actor loss at update " << i;
        ASSERT_GE(result.critic1_losses[static_cast<size_t>(i)], 0.0f)
            << "a mean of squares cannot be negative -- critic1, update " << i;
        ASSERT_GE(result.critic2_losses[static_cast<size_t>(i)], 0.0f)
            << "a mean of squares cannot be negative -- critic2, update " << i;
        ASSERT_TRUE(std::isfinite(result.mean_log_probs[static_cast<size_t>(i)])) << "log_prob at update " << i;
        ASSERT_TRUE(std::isfinite(result.mean_stds[static_cast<size_t>(i)])) << "policy std at update " << i;
        // The policy's own exploration scale is exp() of a network output; a collapse to exactly
        // zero would make every action deterministic and the log-density infinite.
        ASSERT_GT(result.mean_stds[static_cast<size_t>(i)], 0.0f) << "policy std collapsed at update " << i;
    }

    std::cout << "[SAC] " << result.update_count << " updates, all critic1+critic2+actor losses finite; critic1 "
              << result.first_critic1_loss << " -> " << result.last_critic1_loss << ", critic2 "
              << result.first_critic2_loss << " -> " << result.last_critic2_loss << ", actor "
              << result.first_actor_loss << " -> " << result.last_actor_loss << std::endl;
}

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (2/3): PolyakUpdate is genuinely exercised.
//
// A soft update has to do *two* things, and only asserting one of them is how a broken tau slips
// through: the target must have moved away from its initial hard-copied snapshot (or tau is a
// no-op and the targets are frozen DQN-style), and it must still lag the online critic (or tau
// behaved like a hard copy and the bootstrapping target is not slowly-drifting at all).
// ---------------------------------------------------------------------------------------
TEST_F(SACContinuousCartPoleIntegrationTest, PolyakUpdateMovesBothTargetsAndLeavesThemLagging) {
    EXPECT_EQ(result.polyak_count, result.update_count)
        << "SAC blends the targets on EVERY update, not periodically -- that is the whole difference "
           "from DQN's SyncTargetNetwork";
    ASSERT_GT(result.polyak_count, 0);

    std::cout << "[SAC] Q1 online first-layer moved " << result.max_q1_weight_delta << ", Q1 target moved "
              << result.max_q1_target_weight_delta << ", Q2 target moved " << result.max_q2_target_weight_delta
              << "; max |Q1_target - Q1| over every parameter at end: " << result.max_target_lag << std::endl;

    EXPECT_GT(result.max_q1_target_weight_delta, 1e-3)
        << "Q1_target's weights are indistinguishable from the initial snapshot -- PolyakUpdate never moved it";
    EXPECT_GT(result.max_q2_target_weight_delta, 1e-3)
        << "Q2_target's weights never moved -- the second target is dead weight";
    EXPECT_GT(result.max_q1_weight_delta, 1e-3) << "the online critic itself never moved, so nothing could be blended";

    EXPECT_GT(result.max_target_lag, 1e-4)
        << "the target has caught the online critic exactly -- tau is behaving like a hard copy, not an "
           "exponential moving average";
    // ...and the lag must be a lag, not a divergence: an exponential average of a bounded-step
    // process stays within the same order of magnitude as the thing it averages.
    EXPECT_LT(result.max_target_lag, 10.0 * result.max_q1_weight_delta + 1.0)
        << "the target has drifted away from the online critic rather than trailing it";
}

// Four trainable networks, three separate loss terms, two optimizers. Each network's *first-layer*
// weight is the interesting one: the optimizer is the only thing that writes parameters, so a
// moved first layer is direct evidence the corresponding gradient reached all the way back through
// its network rather than stopping at its output layer.
//
// log_std_network deserves its own assertion rather than being lumped in with mean_network: it is
// reached only through TanhGaussianPolicy::backward()'s `grad_u*std*epsilon - grad_log_prob`
// branch, a path no other network in this campaign has. If that branch returned zeros, the policy
// would still train (through the mean alone) and every performance bar would still pass.
TEST_F(SACContinuousCartPoleIntegrationTest, AllFourTrainableNetworksMove) {
    ASSERT_FALSE(result.final_q1_weight.empty());
    ASSERT_FALSE(result.final_mean_weight.empty());
    ASSERT_FALSE(result.final_log_std_weight.empty());

    std::cout << "[SAC] first-layer max |final - initial|: Q1 " << result.max_q1_weight_delta << ", actor mean "
              << result.max_mean_weight_delta << ", actor log_std " << result.max_log_std_weight_delta << std::endl;

    EXPECT_GT(result.max_q1_weight_delta, 1e-3) << "critic 1's first layer never moved -- MSELoss's gradient did not "
                                                   "reach it";
    EXPECT_GT(result.max_mean_weight_delta, 1e-3)
        << "the actor's mean network never moved -- the critics' action-gradient did not reach it";
    EXPECT_GT(result.max_log_std_weight_delta, 1e-3)
        << "the actor's log_std network never moved -- TanhGaussianPolicy::backward()'s grad_log_std branch is "
           "delivering nothing, and the policy's exploration scale is effectively frozen";
}

// The twin critics exist to decorrelate overestimation error, which they can only do if the `min`
// genuinely selects between them. If one critic won essentially every row, `min(Q1, Q2)` would be
// that critic alone and the second would be pure cost -- and, worse, the *other* critic would
// receive a structurally zero action-gradient in every row, so the mission's per-row masking would
// be untested by this run.
TEST_F(SACContinuousCartPoleIntegrationTest, BothTwinCriticsWinRowsOfTheMinSelection) {
    ASSERT_EQ(static_cast<int64_t>(result.q1_selected_fractions.size()), result.update_count);
    double sum = 0.0;
    double min_fraction = 1.0;
    double max_fraction = 0.0;
    for (double f : result.q1_selected_fractions) {
        sum += f;
        min_fraction = std::fmin(min_fraction, f);
        max_fraction = std::fmax(max_fraction, f);
    }
    const double average = sum / static_cast<double>(result.update_count);
    std::cout << "[SAC] Q1-selected fraction: average " << average << " (per-update range " << min_fraction << " .. "
              << max_fraction << ")" << std::endl;

    EXPECT_GT(average, 0.05) << "Q2 is the min almost everywhere -- Q1 is dead weight and the twin-critic min is a no-op";
    EXPECT_LT(average, 0.95) << "Q1 is the min almost everywhere -- Q2 is dead weight and the twin-critic min is a no-op";
}

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (3/3a): the entropy term, verified as a *term*.
//
// This is the phase exit gate's own wording ("entropy-regularization term verified by test"), and
// the mission is specific about the shape it wants: a small fixed scenario in which the difference
// between the loss at two temperatures is hand-computable. A code path that computed the entropy
// term and then discarded it -- or multiplied it by zero, or added it to a local that was never
// returned -- would leave every other test in this file passing.
//
// The scenario is chosen so the Q side of the loss is exact in binary floating point: min(Q1, Q2)
// = {1.0, 1.5, 3.0, 0.5}, whose mean is exactly 1.5. So actor_loss(alpha = 0) is exactly -1.5, and
// the whole of the difference is attributable to the entropy term.
// ---------------------------------------------------------------------------------------
TEST(SACEntropyTermTest, AlphaScalesMeanLogProbAdditivelyInTheActorLoss) {
    CPUBackend backend;

    // A real TanhGaussianPolicy sample, not hand-written log-probabilities: the point is that the
    // *loss* adds alpha * (whatever the policy reports), so the log-probabilities must come from
    // the same class the training loop uses.
    const Tensor mean(Shape({4, 1}), &backend, {0.0f, 0.5f, -0.25f, 1.0f});
    const Tensor log_std(Shape({4, 1}), &backend, {0.0f, -0.5f, 0.25f, -1.0f});
    const Tensor epsilon(Shape({4, 1}), &backend, {0.5f, -1.0f, 0.25f, 2.0f});
    TanhGaussianPolicy policy(&backend);
    const TanhGaussianSample sample = policy.forward(mean, log_std, epsilon);

    // min per row = {1.0, 1.5, 3.0, 0.5}; mean = 6.0 / 4 = 1.5, exactly representable.
    const Tensor q1(Shape({4, 1}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor q2(Shape({4, 1}), &backend, {2.5f, 1.5f, 5.0f, 0.5f});

    const float loss_alpha_zero = actor_loss_value(q1, q2, sample.log_prob, 0.0f);
    const float loss_alpha_half = actor_loss_value(q1, q2, sample.log_prob, 0.5f);

    double log_prob_sum = 0.0;
    for (int64_t i = 0; i < sample.log_prob.numel(); ++i) {
        log_prob_sum += static_cast<double>(sample.log_prob.data()[i]);
    }
    const double mean_log_prob = log_prob_sum / static_cast<double>(sample.log_prob.numel());
    const double expected_difference = 0.5 * mean_log_prob;

    std::cout << "[SAC entropy] mean log_prob = " << mean_log_prob << "; actor_loss(alpha=0.0) = " << loss_alpha_zero
              << ", actor_loss(alpha=0.5) = " << loss_alpha_half
              << "; difference = " << (static_cast<double>(loss_alpha_half) - static_cast<double>(loss_alpha_zero))
              << ", hand-computed 0.5*mean(log_prob) = " << expected_difference << std::endl;

    // The Q side is exact by construction, so this pins it outright.
    EXPECT_FLOAT_EQ(loss_alpha_zero, -1.5f)
        << "at alpha = 0 the loss is exactly -mean(min(Q1, Q2)), and that mean is exactly 1.5 here";

    // THE ASSERTION THE PHASE'S EXIT GATE NAMES: the entropy term is additive, with coefficient
    // exactly alpha. The tolerance is float-narrowing only (the loss is returned as a float; the
    // expectation is computed in double), not slack in the claim.
    EXPECT_NEAR(static_cast<double>(loss_alpha_half) - static_cast<double>(loss_alpha_zero), expected_difference,
                1e-6);

    // Non-vacuity: if mean(log_prob) happened to be ~0 the assertion above would hold for a loss
    // that ignored alpha entirely.
    EXPECT_GT(std::fabs(mean_log_prob), 0.1) << "this scenario's log-probabilities are too close to zero to "
                                                "distinguish an additive entropy term from no entropy term at all";

    // And it is linear in alpha, not merely nonzero at one value -- the difference at alpha = 1.0
    // must be exactly twice the difference at alpha = 0.5.
    const float loss_alpha_one = actor_loss_value(q1, q2, sample.log_prob, 1.0f);
    EXPECT_NEAR(static_cast<double>(loss_alpha_one) - static_cast<double>(loss_alpha_zero), 2.0 * expected_difference,
                1e-6);
}

// ---------------------------------------------------------------------------------------
// THE EXIT GATE (3/3b): THE GOTCHA, as a regression test rather than prose.
//
// The actor update must call Q1.backward()/Q2.backward() to route its gradient back through the
// critics to the action -- which also accumulates into the critics' *own* parameter gradients.
// If the critic optimizer is not zeroed after that, those actor-objective gradients survive into
// the critics' next step and are applied to their weights. Both critic pairs below hold
// bit-identical parameters at the comparison point (the actor's borrowed backward never updates a
// critic's *values*, and zero_grad touches only gradient buffers), so the only possible source of
// any difference is the leftover contamination.
//
// This is deliberately the same proof shape as gan_integration_test.cpp's
// SkippingDiscriminatorZeroGradAfterGeneratorStepCorruptsIt, for the same hazard, found twice in
// two different phases of this codebase by the same "borrow a network's backward for its
// input-gradient" pattern.
// ---------------------------------------------------------------------------------------
constexpr int64_t kRegressionBatch = 8;
constexpr int64_t kRegressionObsDim = 4;
constexpr int64_t kRegressionActionDim = 1;
constexpr int64_t kRegressionHidden = 6;

/** @brief One twin-critic pair, built identically every time -- the regression test needs two that
 *         are numerically indistinguishable so any later difference is attributable. */
struct CriticPair {
    explicit CriticPair(DeviceBackend* backend)
        : q1_in(kRegressionObsDim + kRegressionActionDim, kRegressionHidden, backend),
          q1_relu(backend),
          q1_out(kRegressionHidden, 1, backend),
          q1({&q1_in, &q1_relu, &q1_out}),
          q2_in(kRegressionObsDim + kRegressionActionDim, kRegressionHidden, backend),
          q2_relu(backend),
          q2_out(kRegressionHidden, 1, backend),
          q2({&q2_in, &q2_relu, &q2_out}) {
        // These four seeds are not arbitrary: they were selected so that the per-row `min` splits
        // the batch exactly 4-4 between the two critics (asserted below). A split is required, not
        // cosmetic -- if one critic won every row, the other's borrowed backward would receive an
        // all-zero gradient and the contamination this test measures would be one critic's alone,
        // leaving the per-row masking half-unexercised.
        q1_in.set_weight(lcg_weights(
            static_cast<size_t>((kRegressionObsDim + kRegressionActionDim) * kRegressionHidden), 2u));
        q1_out.set_weight(lcg_weights(static_cast<size_t>(kRegressionHidden), 3u));
        q2_in.set_weight(lcg_weights(
            static_cast<size_t>((kRegressionObsDim + kRegressionActionDim) * kRegressionHidden), 4u));
        q2_out.set_weight(lcg_weights(static_cast<size_t>(kRegressionHidden), 5u));
    }

    LinearModule q1_in;
    ReluModule q1_relu;
    LinearModule q1_out;
    SequentialModule q1;
    LinearModule q2_in;
    ReluModule q2_relu;
    LinearModule q2_out;
    SequentialModule q2;
};

TEST(SACGradientContaminationRegressionTest, SkippingCriticZeroGradAfterTheActorStepCorruptsTheNextCriticGradient) {
    CPUBackend backend;

    // Fixed, shared inputs -- the two runs must differ in nothing but the fix.
    const std::vector<float> obs_values = lcg_weights(static_cast<size_t>(kRegressionBatch * kRegressionObsDim), 1u);
    const Tensor observations(Shape({kRegressionBatch, kRegressionObsDim}), &backend, obs_values);
    const Tensor sampled_actions(Shape({kRegressionBatch, kRegressionActionDim}), &backend,
                                 {0.3f, -0.7f, 0.1f, 0.9f, -0.2f, 0.5f, -0.4f, 0.8f});
    const Tensor stored_actions(Shape({kRegressionBatch, kRegressionActionDim}), &backend,
                                {-0.6f, 0.2f, 0.75f, -0.15f, 0.45f, -0.95f, 0.05f, -0.35f});
    const Tensor bellman_target(Shape({kRegressionBatch, 1}), &backend,
                                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    CriticPair fixed(&backend);
    CriticPair buggy(&backend);
    AdamOptimizer fixed_critic_optimizer(1e-3f, &backend);
    AdamOptimizer buggy_critic_optimizer(1e-3f, &backend);

    // --- the actor update's borrowed backward, run identically on both pairs ---
    const Tensor actor_input = concat_state_action(observations, sampled_actions, &backend);
    const auto borrow = [&](CriticPair& pair) {
        const Tensor q1_values = pair.q1.forward(actor_input);
        const Tensor q2_values = pair.q2.forward(actor_input);
        return min_selected_action_gradient(pair.q1, pair.q2, q1_values, q2_values, kRegressionObsDim, &backend);
    };
    const MinSelectedActionGradient fixed_grad = borrow(fixed);
    const MinSelectedActionGradient buggy_grad = borrow(buggy);

    // Precondition: the two runs agree exactly on what the actor update itself produced. The fix
    // is about what happens *after* this point, so if these already differed the comparison below
    // would be meaningless.
    ASSERT_EQ(fixed_grad.grad_action.numel(), buggy_grad.grad_action.numel());
    for (int64_t i = 0; i < fixed_grad.grad_action.numel(); ++i) {
        ASSERT_FLOAT_EQ(fixed_grad.grad_action.data()[i], buggy_grad.grad_action.data()[i]) << "grad_action row " << i;
    }
    // Non-vacuity of the whole exercise: the action-gradient the actor update came here for is
    // genuinely non-zero, so the borrowed backward really did run.
    double max_action_grad = 0.0;
    for (int64_t i = 0; i < fixed_grad.grad_action.numel(); ++i) {
        max_action_grad = std::fmax(max_action_grad, std::fabs(static_cast<double>(fixed_grad.grad_action.data()[i])));
    }
    ASSERT_GT(max_action_grad, 1e-6) << "the borrowed backward produced no action-gradient at all";
    // Both critics won rows, so both of their backward passes carried nonzero gradient -- without
    // this the contamination below could be one critic's alone.
    ASSERT_DOUBLE_EQ(fixed_grad.q1_selected_fraction, 0.5)
        << "the scenario's weights were chosen for an exact 4-4 split; if this changed, the two critics no longer "
           "both carry gradient and the contamination measured below is only one of theirs";

    // *** THE ONLY DIFFERENCE BETWEEN THE TWO RUNS ***
    fixed_critic_optimizer.zero_grad(fixed.q1);
    fixed_critic_optimizer.zero_grad(fixed.q2);
    // (buggy deliberately skips exactly these two lines)

    // The fixed run starts its next critic step from a zeroed gradient buffer...
    for (const float v : parameter_grads(fixed.q1)) {
        EXPECT_FLOAT_EQ(v, 0.0f) << "fixed run has leftover gradient in Q1";
    }
    for (const float v : parameter_grads(fixed.q2)) {
        EXPECT_FLOAT_EQ(v, 0.0f) << "fixed run has leftover gradient in Q2";
    }

    // ...while the buggy run is carrying the actor step's gradient. That leftover must be
    // genuinely non-zero, or this whole test would be vacuous -- it is what proves
    // Module::backward() during the actor step really does write into the critics' parameter
    // gradients, which is the hazard the fix exists for.
    const std::vector<float> contamination_q1 = parameter_grads(buggy.q1);
    const std::vector<float> contamination_q2 = parameter_grads(buggy.q2);
    double max_contamination = 0.0;
    for (const float v : contamination_q1) {
        max_contamination = std::fmax(max_contamination, std::fabs(static_cast<double>(v)));
    }
    double max_contamination_q2 = 0.0;
    for (const float v : contamination_q2) {
        max_contamination_q2 = std::fmax(max_contamination_q2, std::fabs(static_cast<double>(v)));
    }
    std::cout << "[SAC] leftover contamination after an unfixed actor step: max |grad| in Q1 = " << max_contamination
              << ", in Q2 = " << max_contamination_q2 << std::endl;
    ASSERT_GT(max_contamination, 1e-6);
    ASSERT_GT(max_contamination_q2, 1e-6) << "only one critic was contaminated -- the per-row min mask zeroed the "
                                             "other one's backward entirely, which this scenario is built to avoid";

    // Precondition of the comparison: the two pairs' *parameters* are still bit-identical. A
    // borrowed backward writes gradients, never values, and zero_grad touches only gradients.
    const std::vector<float> fixed_params = parameter_values(fixed.q1);
    const std::vector<float> buggy_params = parameter_values(buggy.q1);
    ASSERT_EQ(fixed_params.size(), buggy_params.size());
    for (size_t i = 0; i < fixed_params.size(); ++i) {
        ASSERT_FLOAT_EQ(fixed_params[i], buggy_params[i]) << "Q1 parameter " << i;
    }

    // --- now each pair's NEXT real critic-update gradient accumulation. Same inputs, same
    // parameters -- the gradients must nevertheless differ, by exactly the contamination. ---
    const Tensor critic_input = concat_state_action(observations, stored_actions, &backend);
    const auto next_critic_grads = [&](CriticPair& pair) {
        MSELoss loss(&backend);
        const Tensor prediction = pair.q1.forward(critic_input);
        (void)loss.forward(prediction, bellman_target);
        (void)pair.q1.backward(loss.backward());
        return parameter_grads(pair.q1);
    };
    const std::vector<float> fixed_next = next_critic_grads(fixed);
    const std::vector<float> buggy_next = next_critic_grads(buggy);

    ASSERT_EQ(fixed_next.size(), buggy_next.size());
    ASSERT_EQ(fixed_next.size(), contamination_q1.size());
    double max_difference = 0.0;
    for (size_t i = 0; i < fixed_next.size(); ++i) {
        EXPECT_NEAR(buggy_next[i], fixed_next[i] + contamination_q1[i], 1e-5f)
            << "Q1 parameter element " << i << ": the difference is exactly the leftover gradient";
        max_difference = std::fmax(max_difference, std::fabs(static_cast<double>(buggy_next[i] - fixed_next[i])));
    }
    std::cout << "[SAC] max |buggy next-step critic grad - clean next-step critic grad| = " << max_difference
              << std::endl;

    // The headline assertion: skipping critic_optimizer.zero_grad() after the actor update makes
    // the critics' next gradient differ from the clean-start case. The fix is load-bearing.
    EXPECT_GT(max_difference, 1e-6);
}

// ---------------------------------------------------------------------------------------
// The training-loop-level transforms, pinned directly rather than only through the end-to-end run.
// ---------------------------------------------------------------------------------------

// concat_state_action() and action_columns() are the two halves of one layout decision (observation
// columns first, action columns last). Pinning them against each other is what makes the
// action-gradient slice provably the slice belonging to the action, not to the last observation
// component.
TEST(SACConcatTest, PacksObservationsThenActionsAndSlicesTheActionColumnsBack) {
    CPUBackend backend;
    const Tensor observations(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Tensor actions(Shape({2, 2}), &backend, {-1.0f, -2.0f, -3.0f, -4.0f});

    const Tensor packed = concat_state_action(observations, actions, &backend);
    ASSERT_EQ(packed.shape().dim(0), 2);
    ASSERT_EQ(packed.shape().dim(1), 5);
    const std::vector<float> expected = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 4.0f, 5.0f, 6.0f, -3.0f, -4.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(packed.data()[i], expected[i]) << "packed element " << i;
    }

    const Tensor sliced = action_columns(packed, /*obs_dim=*/3, &backend);
    ASSERT_EQ(sliced.shape().dim(0), 2);
    ASSERT_EQ(sliced.shape().dim(1), 2);
    for (int64_t i = 0; i < actions.numel(); ++i) {
        EXPECT_FLOAT_EQ(sliced.data()[i], actions.data()[i]) << "sliced element " << i;
    }
}

// entropy_gradient() has to produce the *broadcast* (N, action_dim) block TanhGaussianPolicy's
// backward() documents, not log_prob's own (N, 1) -- passing the latter throws. Both the value and
// the shape are pinned, and the throw is pinned too, because that contract is easy to get wrong in
// exactly the direction that silently looks right for action_dim == 1.
TEST(SACEntropyGradientTest, IsAlphaOverBatchSizeBroadcastAcrossTheActionRow) {
    CPUBackend backend;
    const Tensor grad = entropy_gradient(/*n=*/4, /*action_dim=*/3, /*alpha=*/0.2f, &backend);
    ASSERT_EQ(grad.shape().dim(0), 4);
    ASSERT_EQ(grad.shape().dim(1), 3);
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FLOAT_EQ(grad.data()[i], 0.2f / 4.0f) << "element " << i;
    }

    // The shape contract, from the other side: TanhGaussianPolicy::backward() rejects the (N, 1)
    // form outright, which is what makes the broadcast a requirement rather than a convenience.
    const Tensor mean(Shape({4, 3}), &backend, std::vector<float>(12, 0.0f));
    const Tensor log_std(Shape({4, 3}), &backend, std::vector<float>(12, 0.0f));
    const Tensor epsilon(Shape({4, 3}), &backend, std::vector<float>(12, 0.5f));
    TanhGaussianPolicy policy(&backend);
    (void)policy.forward(mean, log_std, epsilon);
    const Tensor grad_action(Shape({4, 3}), &backend, std::vector<float>(12, 1.0f));
    const Tensor wrong_shape(Shape({4, 1}), &backend, {0.05f, 0.05f, 0.05f, 0.05f});
    EXPECT_THROW((void)policy.backward(grad_action, wrong_shape), std::invalid_argument);
    EXPECT_NO_THROW((void)policy.backward(grad_action, grad));
}

// min_selected_action_gradient()'s per-row mask, against a scenario whose gradient is exactly
// hand-computable. Each "critic" here is a bare LinearModule with no hidden layer, so
// Q_i(s, a) = s*w_s + a*w_a + b and dQ_i/da is the constant w_a. The action-gradient of
// `-mean(min(Q1, Q2))` is therefore exactly -w_a/N in the rows that critic won and 0 elsewhere,
// and the sum of the two slices is exactly that per row.
TEST(SACMinSelectionTest, RoutesMinusOneOverNToTheWinningCriticPerRow) {
    CPUBackend backend;
    constexpr int64_t kN = 4;

    // Q1 = 0.5*s + 2.0*a ; Q2 = 0.5*s + (-3.0)*a. LinearModule's weight layout is
    // (in_features, out_features), so the two values are the s- and a-coefficients in that order.
    LinearModule q1(2, 1, &backend);
    q1.set_weight({0.5f, 2.0f});
    LinearModule q2(2, 1, &backend);
    q2.set_weight({0.5f, -3.0f});

    // s = 0 everywhere, so Q1 = 2a and Q2 = -3a and the winner is decided by the sign of a.
    const Tensor input(Shape({kN, 2}), &backend, {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 2.0f, 0.0f, -0.5f});
    const Tensor q1_values = q1.forward(input);
    const Tensor q2_values = q2.forward(input);
    // a =  1.0 -> Q1 =  2.0, Q2 = -3.0 -> Q2 wins
    // a = -1.0 -> Q1 = -2.0, Q2 =  3.0 -> Q1 wins
    // a =  2.0 -> Q1 =  4.0, Q2 = -6.0 -> Q2 wins
    // a = -0.5 -> Q1 = -1.0, Q2 =  1.5 -> Q1 wins
    const MinSelectedActionGradient selected =
        min_selected_action_gradient(q1, q2, q1_values, q2_values, /*obs_dim=*/1, &backend);

    EXPECT_DOUBLE_EQ(selected.q1_selected_fraction, 0.5);
    ASSERT_EQ(selected.grad_action.shape().dim(0), kN);
    ASSERT_EQ(selected.grad_action.shape().dim(1), 1);

    const float scale = -1.0f / static_cast<float>(kN);
    EXPECT_FLOAT_EQ(selected.grad_action.data()[0], scale * -3.0f);  // Q2 won -> dQ2/da = -3
    EXPECT_FLOAT_EQ(selected.grad_action.data()[1], scale * 2.0f);   // Q1 won -> dQ1/da =  2
    EXPECT_FLOAT_EQ(selected.grad_action.data()[2], scale * -3.0f);
    EXPECT_FLOAT_EQ(selected.grad_action.data()[3], scale * 2.0f);
}

// The reparameterization noise is new to this mission (nothing in the library produces a standard
// normal; ToyVAE used the plain *uniform* LCG draw, which would silently mis-scale SAC's
// log-density). Its two properties that matter are that it is actually standard-normal-shaped and
// that it is deterministic given a seed.
TEST(SACNormalNoiseTest, IsApproximatelyStandardNormalAndFullyDeterministic) {
    CPUBackend backend;
    NormalNoise noise(12345u);
    constexpr int64_t kSamples = 20000;
    const Tensor values = noise.sample(Shape({kSamples, 1}), &backend);

    double sum = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (int64_t i = 0; i < kSamples; ++i) {
        const double v = static_cast<double>(values.data()[i]);
        ASSERT_TRUE(std::isfinite(v)) << "draw " << i;
        sum += v;
        sum_sq += v * v;
        max_abs = std::fmax(max_abs, std::fabs(v));
    }
    const double mean = sum / static_cast<double>(kSamples);
    const double stddev = std::sqrt(sum_sq / static_cast<double>(kSamples) - mean * mean);
    std::cout << "[SAC noise] " << kSamples << " draws: mean " << mean << ", stddev " << stddev << ", max |z| "
              << max_abs << std::endl;
    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(stddev, 1.0, 0.05);
    // A uniform draw mistaken for a normal one would never exceed its own bound; a genuine
    // standard normal reaches past 3 sigma within 20000 draws with overwhelming probability.
    EXPECT_GT(max_abs, 3.0);

    NormalNoise again(12345u);
    const Tensor repeat = again.sample(Shape({kSamples, 1}), &backend);
    for (int64_t i = 0; i < kSamples; ++i) {
        ASSERT_FLOAT_EQ(repeat.data()[i], values.data()[i]) << "draw " << i;
    }
    // ...and a different seed produces a different stream, so the seed is not being ignored.
    NormalNoise other(999u);
    const Tensor different = other.sample(Shape({kSamples, 1}), &backend);
    bool any_difference = false;
    for (int64_t i = 0; i < kSamples && !any_difference; ++i) {
        any_difference = different.data()[i] != values.data()[i];
    }
    EXPECT_TRUE(any_difference);
}

// The threshold assertions above are only meaningful if the run is reproducible. Two short runs of
// the same configuration must agree on every episode length, on all three loss series and on every
// final weight, bit for bit -- if they ever don't, something has reached for non-deterministic
// randomness and the exit-gate tests have quietly become a coin flip. Deliberately short (a third
// of the full run's episodes): a divergence in seeding shows up within the first few episodes, not
// only at the end.
TEST(SACContinuousCartPoleDeterminismTest, TwoRunsOfTheSameConfigurationAreIdentical) {
    TrainingConfig config;
    config.num_episodes = 50;

    const TrainingResult a = RunTraining(config);
    const TrainingResult b = RunTraining(config);

    ASSERT_EQ(a.episode_lengths.size(), b.episode_lengths.size());
    for (size_t i = 0; i < a.episode_lengths.size(); ++i) {
        EXPECT_EQ(a.episode_lengths[i], b.episode_lengths[i]) << "episode " << i;
    }
    EXPECT_EQ(a.total_steps, b.total_steps);
    EXPECT_EQ(a.update_count, b.update_count);
    ASSERT_GT(a.update_count, 0) << "too short to have run any gradient update at all";

    ASSERT_EQ(a.critic1_losses.size(), b.critic1_losses.size());
    for (size_t i = 0; i < a.critic1_losses.size(); ++i) {
        EXPECT_FLOAT_EQ(a.critic1_losses[i], b.critic1_losses[i]) << "critic1 loss at update " << i;
        EXPECT_FLOAT_EQ(a.critic2_losses[i], b.critic2_losses[i]) << "critic2 loss at update " << i;
        EXPECT_FLOAT_EQ(a.actor_losses[i], b.actor_losses[i]) << "actor loss at update " << i;
        EXPECT_FLOAT_EQ(a.mean_log_probs[i], b.mean_log_probs[i]) << "mean log_prob at update " << i;
    }

    ASSERT_EQ(a.final_q1_weight.size(), b.final_q1_weight.size());
    for (size_t i = 0; i < a.final_q1_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_q1_weight[i], b.final_q1_weight[i]) << "Q1 weight element " << i;
    }
    ASSERT_EQ(a.final_q1_target_weight.size(), b.final_q1_target_weight.size());
    for (size_t i = 0; i < a.final_q1_target_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_q1_target_weight[i], b.final_q1_target_weight[i]) << "Q1 target weight element " << i;
    }
    ASSERT_EQ(a.final_mean_weight.size(), b.final_mean_weight.size());
    for (size_t i = 0; i < a.final_mean_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_mean_weight[i], b.final_mean_weight[i]) << "actor mean weight element " << i;
    }
    ASSERT_EQ(a.final_log_std_weight.size(), b.final_log_std_weight.size());
    for (size_t i = 0; i < a.final_log_std_weight.size(); ++i) {
        EXPECT_FLOAT_EQ(a.final_log_std_weight[i], b.final_log_std_weight[i]) << "actor log_std weight element " << i;
    }
}

}  // namespace
}  // namespace sac_continuous_cartpole
}  // namespace exai
