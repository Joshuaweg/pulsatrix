/** @file reinforce_cartpole_training.hpp
 *  @brief The REINFORCE-on-CartPole training loop itself, shared verbatim by
 *         examples/reinforce_cartpole_demo.cpp and
 *         tests/reinforce_cartpole_integration_test.cpp.
 *
 *  @note Header-only, and deliberately *not* under include/pulsatrix/, for exactly the reason
 *        examples/dqn_cartpole_training.hpp is not: mission_reinforce_cartpole_training.md's
 *        deliverable is a test/demo integration assembled out of pieces Phase 1 and Phase 3
 *        Mission 0 already shipped (CartPoleEnv, RolloutBuffer, CategoricalPolicyAgent,
 *        PolicyGradientLoss), and nothing here belongs in the library's public surface. It
 *        lives in one file purely because the mission requires the demo and the test to run
 *        *the same* loop with *the same* hyperparameters and seeds: two hand-copied loops
 *        would drift, and the thresholds the test asserts would then no longer be the
 *        thresholds the demo prints. The test includes it by relative path.
 *
 *  @note Every source of randomness -- weight init, categorical action sampling, environment
 *        reset -- is one of this codebase's deterministic LCGs with a fixed seed, never
 *        `<random>`. The whole run is therefore bit-reproducible, which is what makes a
 *        performance-threshold assertion a test rather than a coin flip. Note that REINFORCE
 *        has *no* exploration schedule at all, unlike DQN's epsilon-greedy: the stochastic
 *        categorical policy explores by construction, and its exploration anneals by itself as
 *        the policy sharpens.
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
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/rollout_buffer.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/shape.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace reinforce_cartpole {

/**
 * @brief Fraction of the run's episodes averaged at each end to form the performance
 *        comparison -- a *fraction*, not DQN's fixed kWindow=20, per the mission's design
 *        decision: REINFORCE's collect-then-update cadence makes the natural unit of progress
 *        the rollout rather than the episode, so a window pinned to a percentage of the total
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
 *       fixed bars (last-10% average >= 3x first-10% average, and >= 30% of max_steps). Demo
 *       and test both take them as-is; neither overrides anything.
 */
struct TrainingConfig {
    /** @brief Episode length limit. The absolute bar is expressed relative to this. */
    int64_t max_steps = 200;
    /** @brief Width of the policy network's single hidden layer. */
    int64_t hidden_size = 64;
    /** @brief Total completed training episodes. */
    int64_t num_episodes = 1000;
    /** @brief Environment steps collected per rollout, i.e. per gradient update. */
    int64_t rollout_length = 512;
    /** @brief Adam step size. */
    float learning_rate = 1e-2f;
    /** @brief Discount factor. */
    float gamma = 0.99f;
    /**
     * @brief Whether to standardize the batch's returns to zero mean / unit variance before
     *        the loss call. See standardize_returns() for why this is on (and for the measured
     *        ablation showing it is a robustness win rather than a strict necessity here).
     */
    bool standardize_returns = true;

    uint32_t weight_seed = 12345u;
    uint32_t env_seed = 7u;
    uint32_t agent_seed = 99u;
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
     *        also exactly the number of times clear() ran.
     */
    int64_t update_count = 0;

    /** @brief False if any single update's loss was ever NaN or infinite. */
    bool all_losses_finite = true;
    /** @brief Index of the first non-finite loss, or -1 if there was none. */
    int64_t first_non_finite_update = -1;
    float first_loss = 0.0f;
    float last_loss = 0.0f;

    /** @brief Policy network's first-layer weight buffer, before and after training. */
    std::vector<float> initial_policy_weight;
    std::vector<float> final_policy_weight;
    /** @brief max |final - initial| over that buffer -- evidence the optimizer moved it. */
    double max_policy_weight_delta = 0.0;
};

/**
 * @brief Deterministic small-random values in [-0.1, 0.1] from this codebase's standard
 *        Numerical-Recipes LCG -- the same initialization pattern the DQN training mission,
 *        ToyGAN and ToyVAE use, and required here because LinearModule zero-initializes and an
 *        all-zero policy network is a fixed point of its own gradient (a uniform categorical
 *        whose logits are identically zero stays that way: every row of the policy-gradient is
 *        the same for both actions).
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
 * @brief Returns `(returns - mean) / (stddev + 1e-8)` over the whole rollout batch.
 * @param returns RolloutBuffer::compute_returns()'s (N, 1) returns tensor.
 * @param backend Backend to allocate the result through.
 * @note This is the training loop's responsibility, not RolloutBuffer's and not
 *       PolicyGradientLoss's -- both of those stay algorithm-return-value-agnostic by their own
 *       documented scope cuts, and a variance-reduction trick baked into either would silently
 *       change the estimator every future caller (A2C, PPO) gets.
 * @note Why it is needed: CartPole's reward is +1 on every step, so every raw return is large
 *       and *strictly positive*. Vanilla REINFORCE then pushes up the log-probability of every
 *       action it ever took, and only the differences between returns carry the learning
 *       signal -- a signal that is a small fraction of a large common offset. Centering removes
 *       that offset (making steps that outlast their episode's average get a positive weight
 *       and the rest a negative one, which is precisely the constant-baseline variance
 *       reduction, unbiased because a state-independent baseline has zero expected gradient),
 *       and the scale division keeps the gradient magnitude comparable across rollouts whose
 *       episode lengths -- and therefore whose return scale -- grow by an order of magnitude
 *       over training. Measured effect: see the TUNING NOTE at the bottom of this file -- with Adam
 *       it is a robustness win, not a strict necessity, and that distinction is recorded there
 *       rather than asserted as folklore.
 * @note The population (1/N) standard deviation, not the sample (1/(N-1)) one. The batch *is*
 *       the population being rescaled here; there is no inference about a wider distribution,
 *       and at N in the hundreds the difference is far below the 1e-8 floor's influence anyway.
 */
inline Tensor standardize_returns(const Tensor& returns, DeviceBackend* backend) {
    const int64_t n = returns.numel();
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(returns.data()[i]);
    }
    const double mean = sum / static_cast<double>(n);

    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double centered = static_cast<double>(returns.data()[i]) - mean;
        sum_sq += centered * centered;
    }
    const double stddev = std::sqrt(sum_sq / static_cast<double>(n));

    // The epsilon is inside the denominator rather than guarding a branch: a rollout in which
    // every return happens to be identical (stddev == 0) must produce all-zero weights, i.e. no
    // update, not a division by zero -- and 0 / (0 + 1e-8) does exactly that.
    const double scale = 1.0 / (stddev + 1e-8);

    std::vector<float> standardized(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        standardized[static_cast<size_t>(i)] =
            static_cast<float>((static_cast<double>(returns.data()[i]) - mean) * scale);
    }
    return Tensor(returns.shape(), backend, standardized);
}

/**
 * @brief Runs the full REINFORCE training loop on CartPoleEnv and returns its statistics.
 * @param config Hyperparameters and seeds.
 * @param on_episode Optional per-episode callback (episode index, episode length) -- the demo's
 *        progress printing, nothing the test needs.
 * @note Algorithm is exactly mission_reinforce_cartpole_training.md's stated loop: sample an
 *       action from the categorical policy, store (observation, action, reward, log_prob, done)
 *       in the RolloutBuffer, and -- once the buffer is full -- reduce it to per-step discounted
 *       returns, standardize them, run one batched forward of the policy network over the whole
 *       rollout, take PolicyGradientLoss's gradient back through it, Adam-step, and clear().
 * @note A rollout deliberately spans whatever mixture of whole and partial episodes fits in
 *       rollout_length steps. The environment is reset only on `done`, never at a rollout
 *       boundary -- RolloutBuffer::compute_returns() already segments the return computation at
 *       episode boundaries via the stored done flags, and truncating episodes at rollout
 *       boundaries instead would bias every rollout's final episode towards looking short.
 * @note One network only, no value baseline: that is A2C's/PPO's addition (Mission 2+), not
 *       REINFORCE's. The batch-mean subtraction in standardize_returns() is a *constant*
 *       baseline, which is a property of the batch, not a learned state-dependent one.
 * @note Adam rather than SGD, the same call the DQN training mission made and for a related but
 *       distinct reason. DQN's problem was a non-stationary regression target; REINFORCE's is
 *       that the policy gradient is a Monte-Carlo estimate whose per-rollout magnitude varies
 *       wildly with the episode lengths that rollout happened to contain. Adam's per-parameter
 *       second-moment normalization makes the effective step size depend on the gradient's
 *       *direction consistency* across rollouts rather than on its raw scale, which is what
 *       keeps one fixed learning rate usable from the ~20-step episodes at initialization to
 *       the ~200-step ones at convergence. Plain SGD was measured, not assumed, and is worse at
 *       every step size tried: see the TUNING NOTE at the bottom of this file.
 */
inline TrainingResult RunTraining(const TrainingConfig& config,
                                  const std::function<void(int64_t, int64_t)>& on_episode = {}) {
    CPUBackend backend;
    CartPoleEnv env(&backend, config.max_steps, config.env_seed);

    const int64_t obs_dim = env.observation_dim();
    const int64_t action_dim = env.action_dim();

    // One policy network, emitting raw logits -- CategoricalPolicyAgent and PolicyGradientLoss
    // both fuse their own numerically stable softmax, so there is deliberately no softmax layer
    // here. Appending one would apply softmax twice.
    LinearModule policy_in(obs_dim, config.hidden_size, &backend);
    ReluModule policy_relu(&backend);
    LinearModule policy_out(config.hidden_size, action_dim, &backend);
    SequentialModule policy_network({&policy_in, &policy_relu, &policy_out});

    policy_in.set_weight(lcg_weights(static_cast<size_t>(obs_dim * config.hidden_size), config.weight_seed));
    policy_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size * action_dim), config.weight_seed + 1u));

    CategoricalPolicyAgent agent(&policy_network, action_dim, &backend, config.agent_seed);
    RolloutBuffer rollout(config.rollout_length, obs_dim, /*action_dim=*/1, &backend);
    PolicyGradientLoss policy_loss(&backend);
    AdamOptimizer optimizer(config.learning_rate, &backend);

    TrainingResult result;
    result.initial_policy_weight = tensor_values(policy_in.weight());

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

        // ---- one gradient update per full rollout, then clear() ----
        const RolloutBatch batch = rollout.compute_returns(config.gamma);
        const Tensor weights =
            config.standardize_returns ? standardize_returns(batch.returns, &backend) : batch.returns;

        // This forward must be the *last* one before backward(): Module::backward() consumes the
        // most recent forward()'s cache, and agent.act() above has been running single-row
        // forwards through the very same network.
        const Tensor logits = policy_network.forward(batch.observations);
        const float loss_value = policy_loss.forward(logits, batch.actions, weights);

        if (!std::isfinite(loss_value)) {
            result.all_losses_finite = false;
            if (result.first_non_finite_update < 0) {
                result.first_non_finite_update = result.update_count;
            }
        }
        if (result.update_count == 0) {
            result.first_loss = loss_value;
        }
        result.last_loss = loss_value;
        ++result.update_count;

        (void)policy_network.backward(policy_loss.backward());
        optimizer.step(policy_network);
        optimizer.zero_grad(policy_network);
        rollout.clear();
    }

    result.final_policy_weight = tensor_values(policy_in.weight());
    for (size_t i = 0; i < result.final_policy_weight.size(); ++i) {
        const double delta = std::fabs(static_cast<double>(result.final_policy_weight[i]) -
                                       static_cast<double>(result.initial_policy_weight[i]));
        if (delta > result.max_policy_weight_delta) {
            result.max_policy_weight_delta = delta;
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

    return result;
}

// ---------------------------------------------------------------------------------------
// TUNING NOTE
// ---------------------------------------------------------------------------------------
// What the tuning that produced TrainingConfig's defaults actually measured.
//
// Recorded here rather than in a commit message because two of the design notes above cite it,
// and a claim about an experiment is only worth as much as the numbers behind it. All figures
// are last-10%/first-10% mean episode length over 1000 episodes, max_steps=200, hidden=64,
// gamma=0.99, Release build.
//
// Rollout length x Adam learning rate (return standardization on):
//   roll=256  lr=3e-3 -> 8.51x   lr=1e-2 -> 5.77x   lr=2e-2 -> 4.59x
//   roll=512  lr=3e-3 -> 7.07x   lr=1e-2 -> 8.21x   lr=2e-2 -> 6.76x
//   roll=1024 lr=3e-3 -> 1.67x   lr=1e-2 -> 8.76x   lr=2e-2 -> 4.72x
// A long rollout at a small step size is the one clear failure mode: 1024/3e-3 gives only 27
// updates over the whole budget, far too few for the policy to move. The chosen 512/1e-2 sits
// in the interior of the passing region rather than on its edge -- deliberately, so that a
// future change that perturbs the run slightly does not tip a green test red.
//
// Return standardization, ablated at the chosen configuration across five weight seeds:
//   on  -> 8.21x, 7.06x, 7.26x, 8.42x, 6.13x   (worst 6.13x)
//   off -> 8.32x, 3.36x, 7.96x, 8.10x, 7.59x   (worst 3.36x)
// So with Adam, standardization is *not* strictly required to clear the 3x bar here -- the
// honest finding, and the opposite of what the mission anticipated. What it does buy is
// consistency: the unstandardized worst case lands at 3.36x, barely above the bar, because
// Adam's second-moment normalization already absorbs most of the scale problem but nothing
// absorbs the large positive common offset in CartPole's all-+1 returns. It also keeps the
// reported loss on a scale a human can read (~1e-2 rather than ~8-30, where the number is
// dominated by the offset and moves with return magnitude rather than with policy quality).
// Kept on for the robustness, with the ablation recorded so the next mission need not redo it.
//
// Optimizer, ablated at the chosen configuration (standardization on):
//   SGD lr=5e-3 -> 1.00x   1e-2 -> 0.95x   5e-2 -> 1.04x   0.2 -> 1.18x   1.0 -> 6.31x
//       5.0 -> 5.56x
//   Adam lr=1e-2 -> 8.21x
// SGD needs roughly a 100x larger step size before it moves at all -- exactly the signature of a
// gradient whose raw magnitude, not its direction, is what Adam is normalizing away -- and even
// at its best (lr=1.0) it is meaningfully worse than Adam at 1e-2. Adam it is.

}  // namespace reinforce_cartpole
}  // namespace pulsatrix
