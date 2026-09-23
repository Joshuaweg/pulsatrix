/** @file dqn_cartpole_training.hpp
 *  @brief The Double-DQN-on-CartPole training loop itself, shared verbatim by
 *         examples/dqn_cartpole_demo.cpp and tests/dqn_cartpole_integration_test.cpp.
 *
 *  @note Header-only, and deliberately *not* under include/exai/. This is not a new
 *        production class -- mission_dqn_cartpole_training.md's deliverable is explicitly a
 *        test/demo integration built out of the pieces Phase 1 and Phase 2 Mission 0 already
 *        shipped (CartPoleEnv, ReplayBuffer, DQNAgent, DQNLoss, ComputeDoubleDQNTarget,
 *        SyncTargetNetwork), and nothing here belongs in the library's public surface. It
 *        lives in one file purely because the mission requires the demo and the test to run
 *        *the same* loop with *the same* hyperparameters and seeds: two hand-copied loops
 *        would drift, and the threshold the test asserts would then no longer be the
 *        threshold the demo prints. The test includes it by relative path.
 *
 *  @note Every source of randomness -- weight init, epsilon-greedy exploration, replay
 *        sampling, environment reset -- is one of this codebase's deterministic LCGs with a
 *        fixed seed, never `<random>`. The whole run is therefore bit-reproducible, which is
 *        what makes a performance-threshold assertion a test rather than a coin flip.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "exai/adam_optimizer.hpp"
#include "exai/cartpole_env.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/dqn_agent.hpp"
#include "exai/dqn_loss.hpp"
#include "exai/dqn_target.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"
#include "exai/replay_buffer.hpp"
#include "exai/sequential_module.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace dqn_cartpole {

/** @brief Episodes averaged at each end of training to form the performance comparison. */
inline constexpr int64_t kWindow = 20;

/**
 * @brief Every hyperparameter and seed of the training run, in one place.
 * @note The defaults are the tuned configuration that actually clears both of the mission's
 *       fixed bars (last-20 average >= 3x first-20 average, and >= 50% of max_steps). Demo
 *       and test both take them as-is; neither overrides anything.
 */
struct TrainingConfig {
    /** @brief Episode length limit. Both threshold bars are expressed relative to this. */
    int64_t max_steps = 200;
    /** @brief Width of the single hidden layer of both Q-networks. */
    int64_t hidden_size = 32;
    /** @brief Total training episodes. */
    int64_t num_episodes = 260;
    /** @brief Minibatch size; also the replay warm-up threshold before updates begin. */
    int64_t batch_size = 32;
    /** @brief Replay capacity -- large enough that nothing is evicted on a run this short. */
    int64_t replay_capacity = 20000;
    /** @brief Adam step size. */
    float learning_rate = 2e-3f;
    /** @brief Discount factor. */
    float gamma = 0.99f;
    /** @brief Environment steps between hard SyncTargetNetwork() copies. */
    int64_t target_sync_interval = 200;
    /** @brief Exploration probability on episode 0. */
    float epsilon_start = 1.0f;
    /** @brief Exploration floor the linear schedule decays to and holds. */
    float epsilon_end = 0.05f;
    /** @brief Fraction of num_episodes over which epsilon decays from start to end. */
    float epsilon_decay_fraction = 0.5f;

    uint32_t weight_seed = 12345u;
    uint32_t env_seed = 7u;
    uint32_t agent_seed = 99u;
    uint32_t buffer_seed = 2024u;
};

/** @brief Everything a caller (demo or test) needs to report on or assert against. */
struct TrainingResult {
    /** @brief Length of every completed episode, in order. */
    std::vector<int64_t> episode_lengths;
    /** @brief Mean length of the first kWindow episodes. */
    double first_window_average = 0.0;
    /** @brief Mean length of the last kWindow episodes. */
    double last_window_average = 0.0;

    /** @brief Number of gradient updates performed. */
    int64_t update_count = 0;
    /** @brief Number of SyncTargetNetwork() calls, excluding the pre-training one. */
    int64_t sync_count = 0;

    /** @brief False if any single update's loss was ever NaN or infinite. */
    bool all_losses_finite = true;
    /** @brief Index of the first non-finite loss, or -1 if there was none. */
    int64_t first_non_finite_update = -1;
    float first_loss = 0.0f;
    float last_loss = 0.0f;

    /** @brief Target network's first-layer weight buffer, before and after training. */
    std::vector<float> initial_target_weight;
    std::vector<float> final_target_weight;
    /** @brief max |final - initial| over that buffer -- non-zero only if a sync happened. */
    double max_target_weight_delta = 0.0;
};

/**
 * @brief Deterministic small-random values in [-0.1, 0.1] from this codebase's standard
 *        Numerical-Recipes LCG -- the same initialization pattern ToyGAN/ToyVAE's own
 *        integration tests use, and required here because LinearModule zero-initializes and
 *        an all-zero network is a fixed point of its own gradient.
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
 * @brief Runs the full Double DQN training loop on CartPoleEnv and returns its statistics.
 * @param config Hyperparameters and seeds.
 * @param on_episode Optional per-episode callback (episode index, episode length, the epsilon
 *        that episode was played at) -- the demo's progress printing, nothing the test needs.
 * @note Algorithm is exactly mission_dqn_cartpole_training.md's stated loop: act
 *       epsilon-greedily, store the transition, then (once the buffer holds a batch) sample,
 *       build Double DQN targets from the online network's argmax evaluated by the target
 *       network, regress the taken action's Q-value onto them with DQNLoss, and Adam-step the
 *       online network only. The target network is never optimized -- it only ever changes via
 *       the periodic hard SyncTargetNetwork() copy.
 * @note Adam rather than SGD: DQN's gradients are notoriously noisy and non-stationary (the
 *       regression target itself moves every time the target network syncs), and Adam's
 *       per-parameter step normalization is what keeps a single fixed learning rate usable
 *       across both the near-zero initial Q-scale and the ~1/(1-gamma) scale the values grow
 *       to. Plain SGD at one fixed lr either crawls at the start or diverges at the end here.
 */
inline TrainingResult RunTraining(const TrainingConfig& config,
                                  const std::function<void(int64_t, int64_t, float)>& on_episode = {}) {
    CPUBackend backend;
    CartPoleEnv env(&backend, config.max_steps, config.env_seed);

    const int64_t obs_dim = env.observation_dim();
    const int64_t action_dim = env.action_dim();

    // Two independent, identically-shaped Q-networks with independent parameter storage --
    // the whole point of a target network is that it is a periodic *snapshot*, not an alias.
    LinearModule online_in(obs_dim, config.hidden_size, &backend);
    ReluModule online_relu(&backend);
    LinearModule online_out(config.hidden_size, action_dim, &backend);
    SequentialModule online_network({&online_in, &online_relu, &online_out});

    LinearModule target_in(obs_dim, config.hidden_size, &backend);
    ReluModule target_relu(&backend);
    LinearModule target_out(config.hidden_size, action_dim, &backend);
    SequentialModule target_network({&target_in, &target_relu, &target_out});

    online_in.set_weight(lcg_weights(static_cast<size_t>(obs_dim * config.hidden_size), config.weight_seed));
    online_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size * action_dim), config.weight_seed + 1u));
    // Target starts as an exact copy of the online network, not an independent random draw:
    // a snapshot that never matched what it snapshots would be a modeling error.
    SyncTargetNetwork(online_network, target_network);

    ReplayBuffer replay(config.replay_capacity, obs_dim, /*action_dim=*/1, &backend, config.buffer_seed);
    DQNAgent agent(&online_network, action_dim, config.epsilon_start, &backend, config.agent_seed);
    DQNLoss dqn_loss(&backend);
    AdamOptimizer optimizer(config.learning_rate, &backend);

    TrainingResult result;
    result.initial_target_weight = tensor_values(target_in.weight());

    const double decay_episodes =
        static_cast<double>(config.num_episodes) * static_cast<double>(config.epsilon_decay_fraction);

    int64_t global_step = 0;
    for (int64_t episode = 0; episode < config.num_episodes; ++episode) {
        // Linear epsilon decay, then hold the floor -- the schedule is the training loop's
        // responsibility, not DQNAgent's, per Mission 0's design decision.
        const double progress = decay_episodes > 0.0
                                    ? static_cast<double>(episode) / decay_episodes
                                    : 1.0;
        const float epsilon =
            progress >= 1.0 ? config.epsilon_end
                            : static_cast<float>(config.epsilon_start +
                                                 (config.epsilon_end - config.epsilon_start) * progress);
        agent.set_epsilon(epsilon);

        Tensor observation = env.reset();
        int64_t episode_length = 0;

        for (int64_t t = 0; t < config.max_steps; ++t) {
            Tensor action = agent.act(observation);
            StepResult step_result = env.step(action);
            replay.add(observation, action, step_result.reward, step_result.observation, step_result.done);
            observation = step_result.observation;
            ++episode_length;
            ++global_step;

            if (replay.size() >= config.batch_size) {
                ReplayBatch batch = replay.sample(config.batch_size);

                Tensor next_q_online = online_network.forward(batch.next_observations);
                Tensor next_q_target = target_network.forward(batch.next_observations);
                Tensor targets = ComputeDoubleDQNTarget(next_q_online, next_q_target, batch.rewards, batch.dones,
                                                        config.gamma, &backend);

                // This forward must be the *last* one before backward(): Module::backward()
                // consumes the most recent forward()'s cache.
                Tensor q_values = online_network.forward(batch.observations);
                const float loss_value = dqn_loss.forward(q_values, batch.actions, targets);

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

                (void)online_network.backward(dqn_loss.backward());
                optimizer.step(online_network);
                optimizer.zero_grad(online_network);
            }

            if (global_step % config.target_sync_interval == 0) {
                SyncTargetNetwork(online_network, target_network);
                ++result.sync_count;
            }

            if (step_result.done) {
                break;
            }
        }

        result.episode_lengths.push_back(episode_length);
        if (on_episode) {
            on_episode(episode, episode_length, epsilon);
        }
    }

    result.final_target_weight = tensor_values(target_in.weight());
    for (size_t i = 0; i < result.final_target_weight.size(); ++i) {
        const double delta = std::fabs(static_cast<double>(result.final_target_weight[i]) -
                                       static_cast<double>(result.initial_target_weight[i]));
        if (delta > result.max_target_weight_delta) {
            result.max_target_weight_delta = delta;
        }
    }

    const int64_t total = static_cast<int64_t>(result.episode_lengths.size());
    const int64_t window = total < kWindow ? total : kWindow;
    double head = 0.0;
    double tail = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        head += static_cast<double>(result.episode_lengths[static_cast<size_t>(i)]);
        tail += static_cast<double>(result.episode_lengths[static_cast<size_t>(total - 1 - i)]);
    }
    result.first_window_average = window > 0 ? head / static_cast<double>(window) : 0.0;
    result.last_window_average = window > 0 ? tail / static_cast<double>(window) : 0.0;

    return result;
}

}  // namespace dqn_cartpole
}  // namespace exai
