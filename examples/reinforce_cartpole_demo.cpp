/** @file reinforce_cartpole_demo.cpp
 *  @brief Standalone demo: trains a REINFORCE policy on CartPoleEnv and prints the
 *         episode-length progression plus the first-10%/last-10% average comparison.
 *  @note Not a test -- tests/reinforce_cartpole_integration_test.cpp is the actual acceptance
 *        criterion for mission_reinforce_cartpole_training.md. This exists purely so a human
 *        can watch the same run, via the `reinforce_cartpole_demo` target.
 *  @note The loop and every hyperparameter come from examples/reinforce_cartpole_training.hpp,
 *        which the test includes too -- the demo and the test cannot drift apart.
 *  @note Fully deterministic (this codebase's LCG convention throughout, no `<random>`), so
 *        every run of this binary prints identical numbers.
 */
#include <cstdio>
#include <vector>

#include "reinforce_cartpole_training.hpp"

int main() {
    using namespace exai::reinforce_cartpole;

    const TrainingConfig config;
    const int64_t window = window_size(config.num_episodes);

    std::printf("REINFORCE CartPole demo -- Linear(4,%lld)->ReLU->Linear(%lld,2) categorical policy\n",
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size));
    std::printf("episodes=%lld  max_steps=%lld  rollout=%lld steps/update  Adam(lr=%.4f)  gamma=%.2f\n",
                static_cast<long long>(config.num_episodes), static_cast<long long>(config.max_steps),
                static_cast<long long>(config.rollout_length), static_cast<double>(config.learning_rate),
                static_cast<double>(config.gamma));
    std::printf("return standardization: %s   (no exploration schedule -- the categorical policy\n",
                config.standardize_returns ? "on" : "off");
    std::printf("explores by construction and anneals itself as it sharpens)\n\n");

    // Rolling mean over the last `window` episodes -- a single episode's length is far too
    // noisy to read a trend off, in CartPole as in every RL task.
    std::vector<int64_t> lengths;
    const TrainingResult result = RunTraining(config, [&](int64_t episode, int64_t length) {
        lengths.push_back(length);
        if ((episode + 1) % window != 0) {
            return;
        }
        const size_t span = lengths.size() < static_cast<size_t>(window) ? lengths.size()
                                                                        : static_cast<size_t>(window);
        double sum = 0.0;
        for (size_t i = lengths.size() - span; i < lengths.size(); ++i) {
            sum += static_cast<double>(lengths[i]);
        }
        std::printf("episode %4lld | mean length over last %lld episodes: %.2f\n",
                    static_cast<long long>(episode + 1), static_cast<long long>(span),
                    sum / static_cast<double>(span));
    });

    std::printf("\nenvironment steps: %lld | gradient updates (= RolloutBuffer::clear() calls): %lld\n",
                static_cast<long long>(result.total_steps), static_cast<long long>(result.update_count));
    std::printf("losses all finite: %s | loss: first update %.4f -> last update %.4f\n",
                result.all_losses_finite ? "yes" : "NO", static_cast<double>(result.first_loss),
                static_cast<double>(result.last_loss));
    std::printf("policy first-layer max |final - initial| weight change: %.6f\n", result.max_policy_weight_delta);

    const double ratio = result.first_window_average > 0.0 ? result.last_window_average / result.first_window_average
                                                           : 0.0;
    std::printf("\n=== performance ===\n");
    std::printf("first %lld episodes, mean length: %.2f\n", static_cast<long long>(result.window),
                result.first_window_average);
    std::printf("last  %lld episodes, mean length: %.2f\n", static_cast<long long>(result.window),
                result.last_window_average);
    std::printf("improvement ratio: %.2fx (bar: >= 3.00x)  -- %s\n", ratio, ratio >= 3.0 ? "PASS" : "FAIL");
    std::printf("last-%lld mean vs 30%% of max_steps (%.1f): %.2f -- %s\n", static_cast<long long>(result.window),
                0.3 * static_cast<double>(config.max_steps), result.last_window_average,
                result.last_window_average >= 0.3 * static_cast<double>(config.max_steps) ? "PASS" : "FAIL");

    return 0;
}
