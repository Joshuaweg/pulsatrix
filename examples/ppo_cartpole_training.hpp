/** @file ppo_cartpole_training.hpp
 *  @brief The PPO (proximal policy optimization) on-CartPole training loop itself, shared
 *         verbatim by examples/ppo_cartpole_demo.cpp and tests/ppo_cartpole_integration_test.cpp.
 *
 *  @note Header-only, and deliberately *not* under include/pulsatrix/, for exactly the reason
 *        examples/dqn_cartpole_training.hpp, examples/reinforce_cartpole_training.hpp and
 *        examples/a2c_cartpole_training.hpp are not: mission_ppo_cartpole_training.md's
 *        deliverable is a test/demo integration assembled out of pieces Phase 1 and Phase 3
 *        Missions 0 and 3 already shipped (CartPoleEnv, RolloutBuffer, CategoricalPolicyAgent,
 *        ComputeGAE, PPOClippedLoss, MSELoss, AdamOptimizer), and nothing here belongs in the
 *        library's public surface. The one production change this mission did make is the two
 *        additive RolloutBuffer accessors -- rewards() and dones() -- which are genuinely
 *        library surface because ComputeGAE consumes raw per-step rewards/dones and
 *        RolloutBuffer::compute_returns() reduces them away.
 *  @note It lives in one file purely because the mission requires the demo and the test to run
 *        *the same* loop with *the same* hyperparameters and seeds: two hand-copied loops would
 *        drift, and the thresholds the test asserts would then no longer be the thresholds the
 *        demo prints. The test includes it by relative path.
 *  @note Every source of randomness -- both networks' weight init, categorical action sampling,
 *        environment reset -- is one of this codebase's deterministic LCGs with a fixed seed,
 *        never `<random>`. The whole run is therefore bit-reproducible, which is what makes a
 *        performance-threshold assertion a test rather than a coin flip.
 *
 *  ## What is structurally new here, relative to A2C
 *
 *  A2C consumes each rollout with exactly one gradient step per network and throws it away.
 *  PPO takes `num_epochs` full-batch gradient steps over the *same* rollout, which is only sound
 *  because the clipped surrogate objective measures -- and refuses to pay for -- how far the
 *  policy has drifted from the one that actually collected the data. That makes the *ordering*
 *  below load-bearing in a way A2C's was not:
 *
 *    - `old_log_probs` come from RolloutBatch, i.e. from the policy at data-collection time, and
 *      are held **fixed** across every epoch. Recomputing them against the current policy each
 *      epoch would make every ratio identically 1 and silently reduce PPO to `num_epochs`
 *      unclipped policy-gradient steps on stale data -- the exact failure the ratio-movement
 *      diagnostic below exists to catch.
 *    - GAE's advantages *and* its critic-regression returns are computed **once**, from the
 *      *pre-update* critic, and are likewise held fixed across every epoch. Recomputing GAE
 *      mid-update against a partially-updated critic is not standard PPO and is not what this
 *      builds.
 *
 *  Only the two forward passes inside the epoch loop see updated parameters.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cartpole_env.hpp"
#include "pulsatrix/categorical_policy_agent.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gae.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/rollout_buffer.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/shape.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace ppo_cartpole {

/**
 * @brief Fraction of the run's episodes averaged at each end to form the performance comparison
 *        -- a *fraction*, not DQN's fixed kWindow=20, carried over from the REINFORCE and A2C
 *        missions for the same reason: a collect-then-update cadence makes the rollout, not the
 *        episode, the natural unit of progress, so a window pinned to a percentage of the total
 *        episode count stays meaningful whatever episode budget the tuned run needs.
 */
inline constexpr double kWindowFraction = 0.10;

/** @brief Window size (in episodes) the mission's two bars are evaluated over. */
inline int64_t window_size(int64_t total_episodes) {
    const int64_t window = static_cast<int64_t>(static_cast<double>(total_episodes) * kWindowFraction);
    return window < 1 ? 1 : window;
}

/**
 * @brief Every hyperparameter and seed of the training run, in one place.
 * @note The defaults are the tuned configuration that actually clears both of the mission's
 *       fixed bars (last-10% average >= 3x first-10% average, and >= 30% of max_steps). Demo and
 *       test both take them as-is; neither overrides anything. See the TUNING NOTE at the bottom
 *       of this file for what was measured to arrive at them.
 */
struct TrainingConfig {
    /** @brief Episode length limit. The absolute bar is expressed relative to this. */
    int64_t max_steps = 200;
    /** @brief Width of the actor network's single hidden layer. */
    int64_t actor_hidden_size = 64;
    /**
     * @brief Width of the critic network's single hidden layer -- a *separate* knob from the
     *        actor's, since the two networks are independent. Tuned separately; see the
     *        TUNING NOTE.
     */
    int64_t critic_hidden_size = 64;
    /** @brief Total completed training episodes. */
    int64_t num_episodes = 1000;
    /** @brief Environment steps collected per rollout, i.e. per *cycle* of num_epochs updates. */
    int64_t rollout_length = 512;
    /**
     * @brief Full-batch gradient steps taken over each collected rollout, for both networks.
     *        This is the knob that makes PPO PPO: `num_epochs == 1` degenerates to a clipped
     *        (and therefore, at ratio == 1 throughout, entirely unclipped) single-step
     *        actor-critic, which is what the ratio-movement diagnostic below rules out.
     * @note No minibatching within an epoch -- a deliberate scope cut stated in the mission.
     *       A 512-step rollout is small enough that full-batch epochs train fine, and minibatch
     *       plumbing would be infrastructure this demo does not need.
     */
    int64_t num_epochs = 4;
    /** @brief Adam step size for the actor (policy) network. */
    float actor_learning_rate = 3e-3f;
    /**
     * @brief Adam step size for the critic (value) network -- a separate knob from the actor's,
     *        and tuned separately, for A2C's reason: the critic solves a plain supervised
     *        regression against a moving target and tolerates a larger step than the actor.
     */
    float critic_learning_rate = 1e-2f;
    /** @brief Discount factor, used for the GAE recursion (and nothing else -- PPO's critic
     *         target is GAE's, not RolloutBuffer::compute_returns()'s Monte-Carlo return). */
    float gamma = 0.99f;
    /**
     * @brief GAE trace-decay parameter. 0 collapses to the one-step TD residual (low variance,
     *        critic-biased); 1 telescopes to the Monte-Carlo advantage (unbiased, high variance).
     */
    float gae_lambda = 0.95f;
    /** @brief PPO's trust-region half-width. The paper's value is 0.2. */
    float clip_epsilon = 0.2f;
    /**
     * @brief Whether to standardize the rollout's advantages to zero mean / unit variance once,
     *        before the epoch loop. See standardize_advantages() and the TUNING NOTE for the
     *        measured effect -- on because it was measured to matter, not assumed.
     */
    bool standardize_advantages = true;

    uint32_t actor_weight_seed = 12345u;
    uint32_t critic_weight_seed = 54321u;
    uint32_t env_seed = 7u;
    uint32_t agent_seed = 99u;
};

/**
 * @brief Per-epoch probability-ratio diagnostics for one epoch of one rollout's update cycle.
 * @note Recomputed by the training loop from `new_logits`/`old_log_probs` rather than read out of
 *       PPOClippedLoss (whose ratio cache is private, deliberately). That independence is a
 *       feature for a regression check: the diagnostic and the loss agree only if both compute
 *       the same stable log-softmax, so a divergence between them is itself informative.
 */
struct RatioStats {
    /** @brief mean_b |r_b - 1| -- how far the policy has drifted from the data-collecting one. */
    double mean_abs_deviation = 0.0;
    /** @brief max_b |r_b - 1| -- the single most-drifted step in the rollout. */
    double max_abs_deviation = 0.0;
    /** @brief mean_b r_b. Near 1 at epoch 0 by construction. */
    double mean_ratio = 0.0;
    /**
     * @brief Fraction of rows whose gradient PPOClippedLoss zeroes: the row has run past the
     *        trust region *in the direction that further improves the surrogate*. Exactly the
     *        `mask == 0` condition documented on PPOClippedLoss::backward().
     */
    double clipped_fraction = 0.0;
};

/** @brief Everything a caller (demo or test) needs to report on or assert against. */
struct TrainingResult {
    /** @brief Length of every completed episode, in order. */
    std::vector<int64_t> episode_lengths;
    /** @brief Episodes averaged at each end -- window_size(episode_lengths.size()). */
    int64_t window = 0;
    /** @brief Mean length of the first `window` episodes. */
    double first_window_average = 0.0;
    /** @brief Mean length of the last `window` episodes. */
    double last_window_average = 0.0;

    /** @brief Environment steps taken over the whole run. */
    int64_t total_steps = 0;
    /**
     * @brief Completed collect-then-update cycles. One per RolloutBuffer::clear(), so this is
     *        also exactly the number of times clear() ran. Distinct from update_count, which
     *        counts *gradient* steps: one rollout yields num_epochs of them per network.
     */
    int64_t rollout_count = 0;
    /** @brief Gradient steps per network -- rollout_count * num_epochs. */
    int64_t update_count = 0;

    /** @brief False if any single update's actor *or* critic loss was ever NaN or infinite. */
    bool all_losses_finite = true;
    /** @brief Flat index of the first update at which either loss was non-finite, or -1. */
    int64_t first_non_finite_update = -1;

    /** @brief PPOClippedLoss value at every update (every epoch of every rollout), in order. */
    std::vector<float> actor_losses;
    /** @brief MSELoss value at every update (every epoch of every rollout), in order. */
    std::vector<float> critic_losses;
    /** @brief Ratio diagnostics at every update, in order -- rollout r epoch e is at
     *         index r * num_epochs + e. */
    std::vector<RatioStats> ratio_stats;

    /**
     * @brief Ratio diagnostics averaged over rollouts, one entry per epoch index -- the
     *        summary form of the mission's PPO-specific regression check. Entry 0 must be ~0
     *        (the policy has not moved yet), and later entries must be meaningfully larger.
     */
    std::vector<double> mean_abs_deviation_by_epoch;
    /** @brief Clipped fraction averaged over rollouts, one entry per epoch index. */
    std::vector<double> clipped_fraction_by_epoch;

    /** @brief Actor/critic loss at the very first and very last update. */
    float first_actor_loss = 0.0f;
    float last_actor_loss = 0.0f;
    float first_critic_loss = 0.0f;
    float last_critic_loss = 0.0f;

    /** @brief The critic's explained variance against GAE's returns, one per *rollout*
     *         (measured on the pre-update critic, before that rollout's epoch loop). */
    std::vector<double> critic_explained_variances;
    /** @brief Rollouts averaged at each end for the critic check -- window_size(rollout_count). */
    int64_t critic_window = 0;
    double first_critic_explained_variance_average = 0.0;
    double last_critic_explained_variance_average = 0.0;

    /** @brief Mean |advantage| (pre-standardization) on the first and last rollout. */
    double first_mean_abs_advantage = 0.0;
    double last_mean_abs_advantage = 0.0;

    /** @brief Actor first-layer weight buffer, before and after training. */
    std::vector<float> initial_actor_weight;
    std::vector<float> final_actor_weight;
    /** @brief max |final - initial| over the actor's first-layer weights. */
    double max_actor_weight_delta = 0.0;
    /** @brief Critic first-layer weight buffer, before and after training. */
    std::vector<float> initial_critic_weight;
    std::vector<float> final_critic_weight;
    /** @brief max |final - initial| over the critic's first-layer weights. */
    double max_critic_weight_delta = 0.0;
};

/**
 * @brief Deterministic small-random values in [-0.1, 0.1] from this codebase's standard
 *        Numerical-Recipes LCG -- the same initialization pattern the DQN, REINFORCE and A2C
 *        training missions, ToyGAN and ToyVAE use. Required for the *actor* because LinearModule
 *        zero-initializes and an all-zero policy network is a fixed point of its own gradient (a
 *        uniform categorical whose logits are identically zero stays that way). Used for the
 *        critic too, for symmetry and because a zero-initialized ReLU network has a dead hidden
 *        layer whose first-layer weights would receive exactly zero gradient forever.
 */
inline std::vector<float> lcg_weights(size_t count, uint32_t seed) {
    std::vector<float> values;
    values.reserve(count);
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;  // [0, 1]
        values.push_back((unit - 0.5f) * 0.2f);                                    // [-0.1, 0.1]
    }
    return values;
}

/** @brief Flattened copy of a Tensor's values. */
inline std::vector<float> tensor_values(const Tensor& t) {
    return std::vector<float>(t.data(), t.data() + t.numel());
}

/**
 * @brief Returns `(advantages - mean) / (stddev + 1e-8)` over the whole rollout batch.
 * @param advantages ComputeGAE()'s (N, 1) advantage tensor.
 * @param backend Backend to allocate the result through.
 * @note Byte-for-byte the transform a2c_cartpole::standardize_advantages() applies, deliberately
 *       re-stated here rather than #included from the A2C header: these are independent example
 *       programs, and making one depend on another's header would couple two missions' tuned runs
 *       together (a change to A2C's helper would silently perturb PPO's asserted thresholds).
 * @note Applied **once per rollout, before the epoch loop**, not once per epoch. The advantages
 *       are a fixed property of the data this rollout collected; rescaling them per epoch would
 *       be rescaling the same numbers by the same constants repeatedly, and recomputing them per
 *       epoch is ruled out by the mission's design decisions outright.
 * @note The population (1/N) standard deviation, not the sample (1/(N-1)) one: the batch *is* the
 *       population being rescaled.
 * @note The epsilon is inside the denominator rather than guarding a branch, so a degenerate batch
 *       in which every advantage is identical produces all-zero weights (no update) rather than a
 *       division by zero.
 */
inline Tensor standardize_advantages(const Tensor& advantages, DeviceBackend* backend) {
    const int64_t n = advantages.numel();
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(advantages.data()[i]);
    }
    const double mean = n > 0 ? sum / static_cast<double>(n) : 0.0;

    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double centered = static_cast<double>(advantages.data()[i]) - mean;
        sum_sq += centered * centered;
    }
    const double stddev = n > 0 ? std::sqrt(sum_sq / static_cast<double>(n)) : 0.0;
    const double scale = 1.0 / (stddev + 1e-8);

    std::vector<float> standardized(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        standardized[static_cast<size_t>(i)] =
            static_cast<float>((static_cast<double>(advantages.data()[i]) - mean) * scale);
    }
    return Tensor(advantages.shape(), backend, standardized);
}

/**
 * @brief Fraction of a batch's target variance the critic's predictions explain,
 *        `1 - MSE / Var(targets)`.
 * @param mse The MSELoss value already computed for this batch (not recomputed, so the reported
 *        number is provably about the same prediction the critic was trained on).
 * @param targets The batch's (N, 1) regression targets -- here GAE's returns, not
 *        RolloutBuffer::compute_returns()'s Monte-Carlo ones.
 * @note A constant predictor equal to the batch mean scores exactly 0; a perfect critic scores 1.
 *       The scale-free companion to the raw MSE, for the reason A2C's own header records: as the
 *       actor improves, episodes lengthen and the returns themselves grow, so a critic getting
 *       strictly better can still show a rising raw MSE.
 */
inline double explained_variance(float mse, const Tensor& targets) {
    const int64_t n = targets.numel();
    if (n <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(targets.data()[i]);
    }
    const double mean = sum / static_cast<double>(n);
    double variance = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double centered = static_cast<double>(targets.data()[i]) - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(n);
    if (variance <= 0.0) {
        return 0.0;
    }
    return 1.0 - static_cast<double>(mse) / variance;
}

/**
 * @brief The per-step probability ratio `r_b = exp(log pi_new(a_b|s_b) - log pi_old(a_b|s_b))`,
 *        summarized -- the mission's PPO-specific regression check, computed by the training loop
 *        itself rather than read out of PPOClippedLoss's private cache.
 * @param new_logits The current policy's (N, action_dim) raw logits over the rollout.
 * @param actions Float-encoded action indices, (N, 1).
 * @param old_log_probs The data-collecting policy's log-probabilities, (N, 1).
 * @param advantages The (N, 1) advantages the loss is weighting by -- needed only to decide which
 *        side of the trust region a row's clip would bite on.
 * @param clip_epsilon The same trust-region half-width passed to PPOClippedLoss::forward().
 * @note Computes the same numerically stable row-wise log-softmax PPOClippedLoss and
 *       CategoricalPolicyAgent both fuse (subtract the row max before exponentiating), so the
 *       ratio reported here is the ratio the loss actually used, to within float reassociation.
 * @note At epoch 0 of any rollout every ratio is 1 *by construction*: nothing has written a
 *       parameter since the data was collected. It is not exactly 1.0f in floating point, because
 *       `old_log_probs` came from the agent's single-row forward passes and these come from one
 *       batched forward over the whole rollout -- two different summation orders through the same
 *       matmul. The residual is ~1e-7, which is why the test asserts "near 0", not "exactly 0".
 */
inline RatioStats probability_ratio_stats(const Tensor& new_logits, const Tensor& actions,
                                          const Tensor& old_log_probs, const Tensor& advantages,
                                          float clip_epsilon) {
    RatioStats stats;
    const int64_t n = new_logits.shape().dim(0);
    const int64_t action_dim = new_logits.shape().dim(1);
    if (n <= 0) {
        return stats;
    }

    double abs_deviation_sum = 0.0;
    double ratio_sum = 0.0;
    int64_t clipped = 0;
    for (int64_t b = 0; b < n; ++b) {
        const float* row = new_logits.data() + b * action_dim;
        float row_max = row[0];
        for (int64_t k = 1; k < action_dim; ++k) {
            row_max = row[k] > row_max ? row[k] : row_max;
        }
        double exp_sum = 0.0;
        for (int64_t k = 0; k < action_dim; ++k) {
            exp_sum += std::exp(static_cast<double>(row[k] - row_max));
        }
        const int64_t action = static_cast<int64_t>(actions.data()[b] + 0.5f);
        const double new_log_prob =
            static_cast<double>(row[action] - row_max) - std::log(exp_sum);
        const double ratio = std::exp(new_log_prob - static_cast<double>(old_log_probs.data()[b]));

        const double deviation = std::fabs(ratio - 1.0);
        abs_deviation_sum += deviation;
        ratio_sum += ratio;
        if (deviation > stats.max_abs_deviation) {
            stats.max_abs_deviation = deviation;
        }
        // Exactly PPOClippedLoss::backward()'s documented `mask == 0` condition: the row has run
        // past the trust region in the direction that would improve the surrogate further, so the
        // `min` selects the constant clipped branch and the row's gradient is structurally zero.
        const double advantage = static_cast<double>(advantages.data()[b]);
        const bool clipped_high = advantage >= 0.0 && ratio > 1.0 + static_cast<double>(clip_epsilon);
        const bool clipped_low = advantage < 0.0 && ratio < 1.0 - static_cast<double>(clip_epsilon);
        if (clipped_high || clipped_low) {
            ++clipped;
        }
    }

    const double denominator = static_cast<double>(n);
    stats.mean_abs_deviation = abs_deviation_sum / denominator;
    stats.mean_ratio = ratio_sum / denominator;
    stats.clipped_fraction = static_cast<double>(clipped) / denominator;
    return stats;
}

/**
 * @brief Runs the full PPO training loop on CartPoleEnv and returns its statistics.
 * @param config Hyperparameters and seeds.
 * @param on_episode Optional per-episode callback (episode index, episode length) -- the demo's
 *        progress printing, nothing the test needs.
 * @note Algorithm is exactly mission_ppo_cartpole_training.md's stated loop: collect
 *       `rollout_length` steps into the RolloutBuffer exactly as REINFORCE/A2C do; then reduce
 *       the buffer once (for its observations/actions/log_probs -- its Monte-Carlo `returns` are
 *       deliberately unused), read the raw rewards/dones back out through the two accessors this
 *       mission added, run the *pre-update* critic over the rollout, bootstrap from the
 *       observation the loop is currently holding, compute GAE once, optionally standardize its
 *       advantages once, then take `num_epochs` full-batch gradient steps per network against
 *       those fixed targets, and clear().
 * @note **The bootstrap** is the critic's value of whatever observation the loop holds at the
 *       rollout boundary -- the true successor state. No special-casing when the last stored step
 *       was terminal: ComputeGAE's `(1 - dones[N-1])` factor zeroes the bootstrap term exactly in
 *       that case, whatever value was passed. (Note the observation at that point is the *reset*
 *       state, since the collection loop resets on done -- and that is precisely the value whose
 *       contribution must be, and is, discarded.)
 * @note **Ordering inside the epoch loop.** Each network's `.backward()` is immediately preceded
 *       by that network's own `.forward()`, because Module::backward() consumes the cache the last
 *       forward() left behind. Three forward paths run through the actor across a cycle
 *       (agent.act()'s single-row passes during collection, and the batched pass in each epoch),
 *       and two through the critic (the pre-update values pass, the bootstrap pass, then one per
 *       epoch) -- so the safe discipline is two complete, non-interleaved forward->loss->backward
 *       ->step cycles per epoch, which is what this does.
 * @note **What is held fixed across epochs, and why it matters.** `gae.advantages`, `gae.returns`
 *       and `batch.log_probs` are computed once, before the epoch loop, and never recomputed
 *       inside it. `batch.log_probs` in particular is the data-collecting policy's snapshot: it is
 *       what makes PPOClippedLoss's ratio a measure of drift rather than the constant 1. Only
 *       `critic.forward()` and `actor.forward()` see updated parameters each epoch.
 * @note Adam for both networks, for the reasons this codebase has already recorded: the actor's
 *       gradient is a Monte-Carlo estimate whose raw magnitude swings with the episode lengths a
 *       rollout happened to contain, and the critic regresses against a target that moves as the
 *       policy changes.
 */
inline TrainingResult RunTraining(const TrainingConfig& config,
                                  const std::function<void(int64_t, int64_t)>& on_episode = {}) {
    CPUBackend backend;
    CartPoleEnv env(&backend, config.max_steps, config.env_seed);

    const int64_t obs_dim = env.observation_dim();
    const int64_t action_dim = env.action_dim();

    // Actor: raw logits. CategoricalPolicyAgent and PPOClippedLoss both fuse their own
    // numerically stable softmax, so there is deliberately no softmax layer here.
    LinearModule actor_in(obs_dim, config.actor_hidden_size, &backend);
    ReluModule actor_relu(&backend);
    LinearModule actor_out(config.actor_hidden_size, action_dim, &backend);
    SequentialModule actor({&actor_in, &actor_relu, &actor_out});

    // Critic: a single scalar value estimate per observation. No activation on the output --
    // a state value is an unbounded real number.
    LinearModule critic_in(obs_dim, config.critic_hidden_size, &backend);
    ReluModule critic_relu(&backend);
    LinearModule critic_out(config.critic_hidden_size, 1, &backend);
    SequentialModule critic({&critic_in, &critic_relu, &critic_out});

    actor_in.set_weight(lcg_weights(static_cast<size_t>(obs_dim * config.actor_hidden_size), config.actor_weight_seed));
    actor_out.set_weight(
        lcg_weights(static_cast<size_t>(config.actor_hidden_size * action_dim), config.actor_weight_seed + 1u));
    critic_in.set_weight(
        lcg_weights(static_cast<size_t>(obs_dim * config.critic_hidden_size), config.critic_weight_seed));
    critic_out.set_weight(
        lcg_weights(static_cast<size_t>(config.critic_hidden_size * 1), config.critic_weight_seed + 1u));

    CategoricalPolicyAgent agent(&actor, action_dim, &backend, config.agent_seed);
    RolloutBuffer rollout(config.rollout_length, obs_dim, /*action_dim=*/1, &backend);
    PPOClippedLoss actor_loss_fn(&backend);
    MSELoss critic_loss_fn(&backend);
    AdamOptimizer actor_optimizer(config.actor_learning_rate, &backend);
    AdamOptimizer critic_optimizer(config.critic_learning_rate, &backend);

    TrainingResult result;
    result.initial_actor_weight = tensor_values(actor_in.weight());
    result.initial_critic_weight = tensor_values(critic_in.weight());

    Tensor observation = env.reset();
    int64_t episode_length = 0;

    while (static_cast<int64_t>(result.episode_lengths.size()) < config.num_episodes) {
        const Tensor action = agent.act(observation);
        const StepResult step_result = env.step(action);
        rollout.add(observation, action, step_result.reward, agent.log_prob(), step_result.done);

        observation = step_result.observation;
        ++episode_length;
        ++result.total_steps;

        if (step_result.done) {
            result.episode_lengths.push_back(episode_length);
            if (on_episode) {
                on_episode(static_cast<int64_t>(result.episode_lengths.size()) - 1, episode_length);
            }
            episode_length = 0;
            observation = env.reset();
        }

        if (rollout.size() < config.rollout_length) {
            continue;
        }

        // ---- one rollout consumed by num_epochs updates per network, then clear() ----

        // (1) The buffer's reduction, for its observations/actions/log_probs. Its own `returns`
        // field is deliberately NOT PPO's critic target -- GAE computes a differently-weighted
        // one below, and mixing the two would train the critic at a different lambda than the
        // actor's advantages assume.
        const RolloutBatch batch = rollout.compute_returns(config.gamma);

        // (2) The raw per-step signals ComputeGAE consumes. These two accessors are this
        // mission's one additive change to a shipped Phase 1 class, and this is the call site
        // that needed them: compute_returns() has already reduced the rewards away.
        const Tensor rewards = rollout.rewards();
        const Tensor dones = rollout.dones();

        // (3) The PRE-update critic's value of every stored step. Computed once; GAE is built
        // from this and never rebuilt inside the epoch loop.
        const Tensor values = critic.forward(batch.observations);

        // (4) Bootstrap from the observation the loop is currently holding -- the true successor
        // of the last stored step. No special case for a terminal last step; see the note above.
        const Tensor bootstrap_tensor = critic.forward(observation);
        const float bootstrap_value = bootstrap_tensor.data()[0];

        // (5) GAE, once. Both outputs are fixed for every epoch below.
        const GAEResult gae =
            ComputeGAE(rewards, dones, values, bootstrap_value, config.gamma, config.gae_lambda, &backend);

        double abs_advantage_sum = 0.0;
        for (int64_t i = 0; i < gae.advantages.numel(); ++i) {
            abs_advantage_sum += std::fabs(static_cast<double>(gae.advantages.data()[i]));
        }
        const double mean_abs_advantage =
            gae.advantages.numel() > 0 ? abs_advantage_sum / static_cast<double>(gae.advantages.numel()) : 0.0;

        // (6) Standardization, also once: a fixed rescaling of fixed numbers.
        const Tensor advantages = config.standardize_advantages
                                      ? standardize_advantages(gae.advantages, &backend)
                                      : gae.advantages;

        // The pre-update critic's fit against the targets it is about to be trained on -- read
        // off the same `values` forward that GAE consumed, so the metric is provably about the
        // critic that produced the advantages.
        {
            MSELoss probe(&backend);
            const float pre_update_mse = probe.forward(values, gae.returns);
            result.critic_explained_variances.push_back(explained_variance(pre_update_mse, gae.returns));
        }

        for (int64_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            // (7a) Critic: complete forward->loss->backward->step, target = GAE's returns.
            const Tensor critic_values = critic.forward(batch.observations);
            const float critic_loss = critic_loss_fn.forward(critic_values, gae.returns);
            (void)critic.backward(critic_loss_fn.backward());
            critic_optimizer.step(critic);
            critic_optimizer.zero_grad(critic);

            // (7b) Actor: complete forward->loss->backward->step against the FIXED old
            // log-probabilities and the FIXED advantages.
            const Tensor new_logits = actor.forward(batch.observations);
            const float actor_loss = actor_loss_fn.forward(new_logits, batch.actions, batch.log_probs, advantages,
                                                           config.clip_epsilon);

            // Measured before the step, so it describes the policy that produced this epoch's
            // loss rather than the one the step is about to create.
            result.ratio_stats.push_back(probability_ratio_stats(new_logits, batch.actions, batch.log_probs,
                                                                 advantages, config.clip_epsilon));

            (void)actor.backward(actor_loss_fn.backward());
            actor_optimizer.step(actor);
            actor_optimizer.zero_grad(actor);

            if (!std::isfinite(actor_loss) || !std::isfinite(critic_loss)) {
                result.all_losses_finite = false;
                if (result.first_non_finite_update < 0) {
                    result.first_non_finite_update = result.update_count;
                }
            }
            if (result.update_count == 0) {
                result.first_actor_loss = actor_loss;
                result.first_critic_loss = critic_loss;
            }
            result.last_actor_loss = actor_loss;
            result.last_critic_loss = critic_loss;
            result.actor_losses.push_back(actor_loss);
            result.critic_losses.push_back(critic_loss);
            ++result.update_count;
        }

        if (result.rollout_count == 0) {
            result.first_mean_abs_advantage = mean_abs_advantage;
        }
        result.last_mean_abs_advantage = mean_abs_advantage;
        ++result.rollout_count;

        // (8) The rollout has been consumed. On-policy data is never reused past its epochs.
        rollout.clear();
    }

    result.final_actor_weight = tensor_values(actor_in.weight());
    result.final_critic_weight = tensor_values(critic_in.weight());
    for (size_t i = 0; i < result.final_actor_weight.size(); ++i) {
        const double delta = std::fabs(static_cast<double>(result.final_actor_weight[i]) -
                                       static_cast<double>(result.initial_actor_weight[i]));
        if (delta > result.max_actor_weight_delta) {
            result.max_actor_weight_delta = delta;
        }
    }
    for (size_t i = 0; i < result.final_critic_weight.size(); ++i) {
        const double delta = std::fabs(static_cast<double>(result.final_critic_weight[i]) -
                                       static_cast<double>(result.initial_critic_weight[i]));
        if (delta > result.max_critic_weight_delta) {
            result.max_critic_weight_delta = delta;
        }
    }

    const int64_t total = static_cast<int64_t>(result.episode_lengths.size());
    result.window = window_size(total);
    double head = 0.0;
    double tail = 0.0;
    for (int64_t i = 0; i < result.window; ++i) {
        head += static_cast<double>(result.episode_lengths[static_cast<size_t>(i)]);
        tail += static_cast<double>(result.episode_lengths[static_cast<size_t>(total - 1 - i)]);
    }
    result.first_window_average = result.window > 0 ? head / static_cast<double>(result.window) : 0.0;
    result.last_window_average = result.window > 0 ? tail / static_cast<double>(result.window) : 0.0;

    // Per-epoch-index aggregation of the ratio diagnostics: entry e averages epoch e over every
    // rollout. This is the summary the PPO-specific regression check reads, and averaging over
    // rollouts is what makes it a statement about the *mechanism* rather than about one noisy
    // rollout's drift.
    result.mean_abs_deviation_by_epoch.assign(static_cast<size_t>(config.num_epochs), 0.0);
    result.clipped_fraction_by_epoch.assign(static_cast<size_t>(config.num_epochs), 0.0);
    if (result.rollout_count > 0) {
        for (int64_t i = 0; i < static_cast<int64_t>(result.ratio_stats.size()); ++i) {
            const size_t epoch = static_cast<size_t>(i % config.num_epochs);
            result.mean_abs_deviation_by_epoch[epoch] += result.ratio_stats[static_cast<size_t>(i)].mean_abs_deviation;
            result.clipped_fraction_by_epoch[epoch] += result.ratio_stats[static_cast<size_t>(i)].clipped_fraction;
        }
        const double denominator = static_cast<double>(result.rollout_count);
        for (size_t e = 0; e < result.mean_abs_deviation_by_epoch.size(); ++e) {
            result.mean_abs_deviation_by_epoch[e] /= denominator;
            result.clipped_fraction_by_epoch[e] /= denominator;
        }
    }

    // Same 10%-of-the-run windowing the episode bars use, applied to the critic's own metric.
    const int64_t rollouts = static_cast<int64_t>(result.critic_explained_variances.size());
    result.critic_window = window_size(rollouts);
    if (rollouts > 0) {
        double ev_head = 0.0;
        double ev_tail = 0.0;
        for (int64_t i = 0; i < result.critic_window; ++i) {
            ev_head += result.critic_explained_variances[static_cast<size_t>(i)];
            ev_tail += result.critic_explained_variances[static_cast<size_t>(rollouts - 1 - i)];
        }
        const double denominator = static_cast<double>(result.critic_window);
        result.first_critic_explained_variance_average = ev_head / denominator;
        result.last_critic_explained_variance_average = ev_tail / denominator;
    }

    return result;
}

// ---------------------------------------------------------------------------------------
// TUNING NOTE
// ---------------------------------------------------------------------------------------
// What the tuning that produced TrainingConfig's defaults actually measured.
//
// Recorded here rather than in a commit message because several of the design notes above cite
// it, and a claim about an experiment is only worth as much as the numbers behind it -- the same
// standard reinforce_cartpole_training.hpp's and a2c_cartpole_training.hpp's own TUNING NOTEs
// set. Unless stated otherwise all figures are from Release builds over the full 1000-episode
// budget, max_steps=200, gamma=0.99. "ratio" is last-10%/first-10% mean episode length (bar:
// >= 3.00x); "dev/epoch" is mean_b |r_b - 1| at each epoch index, averaged over every rollout;
// "clip/epoch" is the fraction of rows whose gradient the clip zeroes, same averaging; "EV" is
// the critic's explained variance against GAE's returns, windowed over the first/last 10% of
// rollouts.
//
// (1) num_epochs x actor lr x critic lr, at hidden 64/64, rollout 512, lambda 0.95, eps 0.2.
//     All 27 cells kept every loss finite and all but one cleared both bars; what separates them
//     is where the *first* window lands (a fast early learner depresses the ratio bar by raising
//     its own denominator) and how much ratio drift the epoch count actually produces.
//       ep=2 alr=1e-3: clr 3e-3 -> 3.77x (21.84->82.24)  | 1e-2 -> 3.24x (->70.85)  | 3e-2 -> 3.91x (->85.40)
//       ep=2 alr=3e-3: clr 3e-3 -> 4.64x (23.78->110.40) | 1e-2 -> 8.49x (->200.00) | 3e-2 -> 7.75x (->182.41)
//       ep=2 alr=1e-2: clr 3e-3 -> 3.44x (27.25->93.75)  | 1e-2 -> 5.93x (->161.72) | 3e-2 -> 6.22x (->179.77)
//       ep=4 alr=1e-3: clr 3e-3 -> 5.94x (23.85->141.71) | 1e-2 -> 8.30x (->198.06) | 3e-2 -> 8.87x (->198.95)
//       ep=4 alr=3e-3: clr 3e-3 -> 7.59x (24.69->187.38) | 1e-2 -> 8.10x (->200.00) | 3e-2 -> 8.37x (->200.00)
//       ep=4 alr=1e-2: clr 3e-3 -> 2.77x (44.72->123.77) | 1e-2 -> 4.88x (->200.00) | 3e-2 -> 4.90x (->200.00)
//       ep=8 alr=1e-3: clr 3e-3 -> 8.79x (22.75->200.00) | 1e-2 -> 8.75x (->200.00) | 3e-2 -> 8.79x (->200.00)
//       ep=8 alr=3e-3: clr 3e-3 -> 6.81x (28.29->192.61) | 1e-2 -> 6.89x (->200.00) | 3e-2 -> 6.86x (->198.78)
//       ep=8 alr=1e-2: clr 3e-3 -> 3.44x (57.62->198.02) | 1e-2 -> 3.04x (->199.06) | 3e-2 -> 3.26x (->200.00)
//     Two readings worth keeping:
//       - The single failing cell is ep=4/alr=1e-2/clr=3e-3 at 2.77x -- and it *fails by learning
//         too fast*, reaching a 44.72-step first window before the measurement window closes. That
//         is the ratio bar's known weakness, not a training failure, and it is the reason the
//         mission holds the run to the absolute bar as well.
//       - The *product* alr x num_epochs is what actually moves the policy per rollout, and it
//         shows in dev/epoch: at alr=1e-2, ep=8 the drift saturates (0.089 at epoch 6, then
//         *falls* to 0.087 at epoch 7 as the clip starts refusing to pay for it, 6% of rows
//         clipped). That saturation is the clip doing exactly its job, and it is visible in the
//         data rather than asserted from theory.
//     ep=4 / alr=3e-3 / clr=1e-2 chosen: interior on both lr axes, clears both bars with large
//     margin (8.10x, 200.00), and produces unambiguous but non-saturated drift.
//
// (2) Rollout length x hidden size (both networks moved together), at the chosen lrs:
//       roll=256:  h=32 -> 7.47x (25.10->187.46) | h=64 -> 6.14x (32.57->200.00) | h=128 -> 4.03x (49.61->200.00)
//       roll=512:  h=32 -> 8.36x (22.43->187.62) | h=64 -> 8.10x (24.69->200.00) | h=128 -> 6.79x (29.45->200.00)
//       roll=1024: h=32 -> 9.90x (19.91->197.13) | h=64 -> 8.46x (22.29->188.49) | h=128 -> 7.95x (25.14->199.96)
//     Every cell clears both bars. roll=512/h=64 is the only cell that reaches a *full* 200.00
//     final window with the ratio bar still above 8x, and a full Debug run of it costs ~12s, so
//     the suite is not paying for width it cannot show a benefit from. Note the systematic trend:
//     wider nets and shorter rollouts both raise the *first* window (faster early learning), which
//     mechanically lowers the ratio without making the run worse -- a further reminder that the
//     ratio bar alone is a weak discriminator here.
//
// (3) GAE lambda x clip epsilon, at the chosen configuration:
//       lambda=0.90: eps 0.1 -> 7.98x | 0.2 -> 7.51x | 0.3 -> 7.56x
//       lambda=0.95: eps 0.1 -> 7.35x | 0.2 -> 8.10x | 0.3 -> 8.10x
//       lambda=0.99: eps 0.1 -> 8.04x | 0.2 -> 8.00x | 0.3 -> 8.10x
//       lambda=1.00: eps 0.1 -> 8.02x | 0.2 -> 8.10x | 0.3 -> 7.62x
//     CartPole is insensitive to both here (every cell 7.35x-8.10x), which is itself the useful
//     finding: the bars cannot be used to argue for a particular lambda on this task. The clip
//     epsilon *is* clearly visible in the diagnostics rather than the bars, exactly where it
//     should be -- at the final epoch, eps=0.1 clips 3.2% of rows, eps=0.2 clips 1.3%, eps=0.3
//     clips 0.9%. Defaults kept at the literature's lambda=0.95 / eps=0.2, both interior.
//
// (4) Advantage standardization, ablated at the chosen configuration across five independent
//     seed-sets (actor init, critic init, env and agent seeds all varied together):
//       on  -> 8.10x, 8.18x, 6.95x, 6.88x, 7.27x   (worst 6.88x); EV final 0.11, 0.49, 0.13, 0.53, 0.37
//       off -> 7.26x, 7.70x, 6.19x, 8.79x, 6.86x   (worst 6.19x); EV final 0.30, 0.97, 0.89, 0.07, 0.70
//     The honest reading, and it is the same genuine trade-off A2C found rather than a free win:
//       - On the bars the mission holds this to, standardization is a modest robustness gain
//         (worst seed-set 6.88x vs 6.19x). Both are comfortably clear of the 3x bar; neither
//         choice is close to failing, so this is a margin argument, not a pass/fail one.
//       - The critic fits *better* without it on four of five seed-sets, for A2C's mechanism:
//         unstandardized advantages shrink as the critic improves, the actor's effective step
//         decays, the policy moves less, and the return distribution the critic chases stays
//         narrower.
//     Kept on, for the additional reason that it makes the *actor loss* interpretable: with
//     standardization the surrogate's magnitude stays ~1e-2 (max |actor loss| over all 1224
//     updates: 0.015), whereas unstandardized it starts near -8.5 and wanders, which makes
//     "the loss did not diverge" a much weaker statement to assert.
//
// (5) The probability-ratio diagnostic at the chosen configuration -- the mission's PPO-specific
//     regression check, and the evidence that the multi-epoch update is real rather than
//     num_epochs=1 in disguise. Over 306 rollouts x 4 epochs = 1224 updates:
//       dev/epoch  = 0.0000000283 / 0.0145 / 0.0293 / 0.0443
//       clip/epoch = 0.0000 / 0.0000 / 0.0021 / 0.0131
//     Read in order:
//       - Epoch 0's drift is 2.8e-8, not 0.0f exactly, and the residual is not a defect: the
//         old log-probabilities came from CategoricalPolicyAgent's *single-row* forward passes
//         during collection, and epoch 0 recomputes them from one *batched* forward over 512
//         rows -- two summation orders through the same matmul. 2.8e-8 is float reassociation at
//         the scale one would predict, and the test asserts "near 0" rather than exact equality
//         for that reason.
//       - The drift then grows essentially linearly in the epoch index, and does so on *every
//         single rollout*: in all 306 of them, the last epoch's mean |r - 1| exceeds the first
//         epoch's by more than 1e-4, with the worst rollout still at 0.0036.
//       - It stays bounded: the largest mean |r - 1| at any of the 1224 updates is 0.135, and the
//         largest *single-row* |r - 1| is 0.834 -- rows do run past the 1 +/- 0.2 trust region,
//         and when they do in the direction that would improve the surrogate further, their
//         gradient is structurally zero (1.3% of rows by the final epoch). That is the clip
//         bounding the per-rollout policy movement, observed rather than assumed.
//
// (6) A num_epochs=1 control, which is what makes (5) a *contrast* and not just a number: with
//     one epoch per rollout the mean drift is 2.7e-8 and *no single row anywhere in the run*
//     reaches |r - 1| = 1.8e-7 (133 rollouts, 133 updates, 8.49x), and the clipped
//     fraction is exactly 0.0 everywhere -- PPOClippedLoss degenerates, as its own header
//     documents, to an advantage-weighted policy gradient and the clip never engages at all. The
//     integration test runs that control directly.
//
// (7) What the critic's explained variance does and does not show here. Windowed over the first
//     and last 10% of rollouts it moves -1.38 -> +0.11, and per-rollout it ranges -4.35 .. +0.99.
//     Crucially it is measured on the *pre-update* critic, against the very GAE targets that
//     critic produced -- so unlike A2C's it is not a clean "did the critic learn" statistic: GAE's
//     return is A_t + V(s_t), a target that moves with the critic itself and is easier to fit at
//     small lambda. It is reported and given a deliberately loose bar (see the integration test)
//     as a sanity signal, not as this mission's exit criterion; the exit criterion is the two
//     episode-length bars plus the ratio-movement check above.

}  // namespace ppo_cartpole
}  // namespace pulsatrix
