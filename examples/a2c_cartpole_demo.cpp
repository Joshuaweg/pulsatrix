/** @file a2c_cartpole_demo.cpp
 *  @brief Standalone demo: trains an A2C actor-critic pair on CartPoleEnv and prints the
 *         episode-length progression, the first-10%/last-10% average comparison, and the
 *         critic's own learning curve.
 *  @note Not a test -- tests/a2c_cartpole_integration_test.cpp is the actual acceptance criterion
 *        for mission_a2c_cartpole_training.md. This exists purely so a human can watch the same
 *        run, via the `a2c_cartpole_demo` target.
 *  @note The loop and every hyperparameter come from examples/a2c_cartpole_training.hpp, which
 *        the test includes too -- the demo and the test cannot drift apart.
 *  @note Fully deterministic (this codebase's LCG convention throughout, no `<random>`), so every
 *        run of this binary prints identical numbers.
 */
#include <cstdio>
#include <vector>

#include "a2c_cartpole_training.hpp"

int main() {
    using namespace exai::a2c_cartpole;

    const TrainingConfig config;
    const int64_t window = window_size(config.num_episodes);

    std::printf("A2C CartPole demo -- actor Linear(4,%lld)->ReLU->Linear(%lld,2) logits, "
                "critic Linear(4,%lld)->ReLU->Linear(%lld,1) value\n",
                static_cast<long long>(config.actor_hidden_size),
                static_cast<long long>(config.actor_hidden_size),
                static_cast<long long>(config.critic_hidden_size),
                static_cast<long long>(config.critic_hidden_size));
    std::printf("episodes=%lld  max_steps=%lld  rollout=%lld steps/update  gamma=%.2f\n",
                static_cast<long long>(config.num_episodes), static_cast<long long>(config.max_steps),
                static_cast<long long>(config.rollout_length), static_cast<double>(config.gamma));
    std::printf("Adam: actor lr=%.4f, critic lr=%.4f (separate optimizers, independent parameters)\n",
                static_cast<double>(config.actor_learning_rate), static_cast<double>(config.critic_learning_rate));
    std::printf("advantage standardization: %s   (critic target: RolloutBuffer's Monte-Carlo\n",
                config.standardize_advantages ? "on" : "off");
    std::printf("return-to-go -- GAE is PPO's job, not this mission's)\n\n");

    // Rolling mean over the last `window` episodes -- a single episode's length is far too noisy
    // to read a trend off, in CartPole as in every RL task.
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

    std::printf("\nenvironment steps: %lld | gradient updates per network "
                "(= RolloutBuffer::clear() calls): %lld\n",
                static_cast<long long>(result.total_steps), static_cast<long long>(result.update_count));
    std::printf("all actor+critic losses finite: %s\n", result.all_losses_finite ? "yes" : "NO");
    std::printf("actor  loss (PolicyGradientLoss): first update %.4f -> last update %.4f\n",
                static_cast<double>(result.first_actor_loss), static_cast<double>(result.last_actor_loss));
    std::printf("critic loss (MSELoss):            first update %.4f -> last update %.4f\n",
                static_cast<double>(result.first_critic_loss), static_cast<double>(result.last_critic_loss));
    std::printf("actor  first-layer max |final - initial| weight change: %.6f\n", result.max_actor_weight_delta);
    std::printf("critic first-layer max |final - initial| weight change: %.6f\n", result.max_critic_weight_delta);

    std::printf("\n=== critic: did the value function actually learn? ===\n");
    std::printf("averaged over the first/last %lld of %lld updates (one update is one rollout's\n",
                static_cast<long long>(result.critic_window), static_cast<long long>(result.update_count));
    std::printf("worth of Monte-Carlo noise, so a single update is not a trend):\n");
    std::printf("  critic MSE:        %.2f -> %.2f\n", result.first_critic_loss_average,
                result.last_critic_loss_average);
    std::printf("  explained variance: %.3f -> %.3f   (0 = no better than predicting the batch\n",
                result.first_critic_explained_variance_average, result.last_critic_explained_variance_average);
    std::printf("                                       mean; negative = worse than that)\n");
    std::printf("  mean |advantage|:  %.3f -> %.3f\n", result.first_mean_abs_advantage,
                result.last_mean_abs_advantage);
    std::printf("note: raw MSE is confounded -- the returns themselves grow as episodes lengthen.\n");
    std::printf("      Explained variance is the scale-free reading. See the header's TUNING NOTE\n");
    std::printf("      for the frozen-critic control that rules out 'the actor did all the work'.\n");

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
