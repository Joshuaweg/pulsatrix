/** @file dqn_cartpole_demo.cpp
 *  @brief Standalone demo: trains a Double DQN agent on CartPoleEnv and prints the
 *         episode-length progression plus the first-20/last-20 average comparison.
 *  @note Not a test -- tests/dqn_cartpole_integration_test.cpp is the actual acceptance
 *        criterion for mission_dqn_cartpole_training.md (Phase 2's exit gate). This exists
 *        purely so a human can watch the same run, via the `dqn_cartpole_demo` target.
 *  @note The loop and every hyperparameter come from examples/dqn_cartpole_training.hpp,
 *        which the test includes too -- the demo and the test cannot drift apart.
 *  @note Fully deterministic (this codebase's LCG convention throughout, no `<random>`), so
 *        every run of this binary prints identical numbers.
 */
#include <cstdio>
#include <vector>

#include "dqn_cartpole_training.hpp"

int main() {
    using namespace pulsatrix::dqn_cartpole;

    const TrainingConfig config;

    std::printf("DQN CartPole demo -- Double DQN, Linear(4,%lld)->ReLU->Linear(%lld,2)\n",
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size));
    std::printf("episodes=%lld  max_steps=%lld  batch=%lld  Adam(lr=%.4f)  gamma=%.2f  target_sync=%lld steps\n",
                static_cast<long long>(config.num_episodes), static_cast<long long>(config.max_steps),
                static_cast<long long>(config.batch_size), static_cast<double>(config.learning_rate),
                static_cast<double>(config.gamma), static_cast<long long>(config.target_sync_interval));
    std::printf("epsilon: %.2f -> %.2f linearly over the first %.0f%% of episodes\n\n",
                static_cast<double>(config.epsilon_start), static_cast<double>(config.epsilon_end),
                100.0 * static_cast<double>(config.epsilon_decay_fraction));

    // Rolling mean over the last kWindow episodes -- a single episode's length is far too
    // noisy to read a trend off, in CartPole as in every RL task.
    std::vector<int64_t> lengths;
    const TrainingResult result = RunTraining(config, [&](int64_t episode, int64_t length, float epsilon) {
        lengths.push_back(length);
        if ((episode + 1) % kWindow != 0) {
            return;
        }
        const size_t window = lengths.size() < static_cast<size_t>(kWindow) ? lengths.size()
                                                                           : static_cast<size_t>(kWindow);
        double sum = 0.0;
        for (size_t i = lengths.size() - window; i < lengths.size(); ++i) {
            sum += static_cast<double>(lengths[i]);
        }
        std::printf("episode %4lld | epsilon %.3f | mean length over last %lld episodes: %.2f\n",
                    static_cast<long long>(episode + 1), static_cast<double>(epsilon),
                    static_cast<long long>(window), sum / static_cast<double>(window));
    });

    std::printf("\ngradient updates: %lld | target-network syncs: %lld | losses all finite: %s\n",
                static_cast<long long>(result.update_count), static_cast<long long>(result.sync_count),
                result.all_losses_finite ? "yes" : "NO");
    std::printf("loss: first update %.4f -> last update %.4f\n", static_cast<double>(result.first_loss),
                static_cast<double>(result.last_loss));
    std::printf("target-network first-layer max |final - initial| weight change: %.6f\n",
                result.max_target_weight_delta);

    const double ratio = result.first_window_average > 0.0 ? result.last_window_average / result.first_window_average
                                                           : 0.0;
    std::printf("\n=== performance ===\n");
    std::printf("first %lld episodes, mean length: %.2f\n", static_cast<long long>(kWindow),
                result.first_window_average);
    std::printf("last  %lld episodes, mean length: %.2f\n", static_cast<long long>(kWindow),
                result.last_window_average);
    std::printf("improvement ratio: %.2fx (bar: >= 3.00x)  -- %s\n", ratio, ratio >= 3.0 ? "PASS" : "FAIL");
    std::printf("last-%lld mean vs 50%% of max_steps (%.1f): %.2f -- %s\n", static_cast<long long>(kWindow),
                0.5 * static_cast<double>(config.max_steps), result.last_window_average,
                result.last_window_average >= 0.5 * static_cast<double>(config.max_steps) ? "PASS" : "FAIL");

    return 0;
}
