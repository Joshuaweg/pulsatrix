/** @file sac_continuous_cartpole_demo.cpp
 *  @brief Standalone demo: trains a Soft Actor-Critic agent -- two independent actor MLPs, twin
 *         critics and twin Polyak-averaged targets -- on ContinuousCartPoleEnv, and prints the
 *         episode-length progression, the first-20/last-20 average comparison, and the
 *         SAC-specific diagnostics (the policy's own learned exploration scale, the entropy
 *         term's log-probability, the twin critics' min-selection balance, and how far the soft
 *         targets have moved and how far they still lag).
 *  @note Not a test -- tests/sac_continuous_cartpole_integration_test.cpp is the actual
 *        acceptance criterion for mission_sac_continuous_cartpole_training.md, and it is that
 *        mission's (and the whole Reinforcement Learning campaign's) exit gate. This exists
 *        purely so a human can watch the same run, via the `sac_continuous_cartpole_demo` target.
 *  @note The loop and every hyperparameter come from examples/sac_continuous_cartpole_training.hpp,
 *        which the test includes too -- the demo and the test cannot drift apart.
 *  @note Fully deterministic (this codebase's LCG convention throughout, including the Box-Muller
 *        normal draws the reparameterization needs -- no `<random>`), so every run of this binary
 *        prints identical numbers.
 */
#include <cstdio>
#include <vector>

#include "sac_continuous_cartpole_training.hpp"

int main() {
    using namespace pulsatrix::sac_continuous_cartpole;

    const TrainingConfig config;

    std::printf("SAC ContinuousCartPole demo\n");
    std::printf("  actor : mean Linear(4,%lld)->ReLU->Linear(%lld,1)  +  log_std "
                "Linear(4,%lld)->ReLU->Linear(%lld,1)   (two INDEPENDENT MLPs)\n",
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size),
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size));
    std::printf("  critic: twin Q1,Q2 Linear(4+1,%lld)->ReLU->Linear(%lld,1) over concat(obs|action), "
                "plus Polyak-averaged Q1_target,Q2_target\n",
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size));
    std::printf("episodes=%lld  max_steps=%lld  batch=%lld  replay capacity=%lld  warm-up=%lld\n",
                static_cast<long long>(config.num_episodes), static_cast<long long>(config.max_steps),
                static_cast<long long>(config.batch_size), static_cast<long long>(config.replay_capacity),
                static_cast<long long>(config.warmup_size));
    std::printf("gamma=%.3f  tau=%.4f (soft, EVERY step -- not DQN's periodic hard sync)  alpha=%.4f "
                "(fixed entropy temperature)\n",
                static_cast<double>(config.gamma), static_cast<double>(config.tau),
                static_cast<double>(config.alpha));
    std::printf("Adam: actor lr=%.5f (both actor MLPs), critic lr=%.5f (both critics)\n\n",
                static_cast<double>(config.actor_learning_rate), static_cast<double>(config.critic_learning_rate));

    // Rolling mean over the last kWindow episodes -- a single episode's length is far too noisy to
    // read a trend off, in CartPole as in every RL task.
    std::vector<int64_t> lengths;
    const TrainingResult result = RunTraining(config, [&](int64_t episode, int64_t length) {
        lengths.push_back(length);
        if ((episode + 1) % kWindow != 0) {
            return;
        }
        const size_t span = lengths.size() < static_cast<size_t>(kWindow) ? lengths.size()
                                                                         : static_cast<size_t>(kWindow);
        double sum = 0.0;
        for (size_t i = lengths.size() - span; i < lengths.size(); ++i) {
            sum += static_cast<double>(lengths[i]);
        }
        std::printf("episode %4lld | mean length over last %zu episodes: %.2f\n",
                    static_cast<long long>(episode + 1), span, sum / static_cast<double>(span));
    });

    std::printf("\nenvironment steps: %lld | gradient updates: %lld | PolyakUpdate calls per target: %lld\n",
                static_cast<long long>(result.total_steps), static_cast<long long>(result.update_count),
                static_cast<long long>(result.polyak_count));
    std::printf("all THREE losses finite (critic1, critic2, actor -- checked at EVERY update): %s\n",
                result.all_losses_finite ? "yes" : "NO");
    std::printf("critic1 loss (MSE): first %.4f -> last %.4f\n", static_cast<double>(result.first_critic1_loss),
                static_cast<double>(result.last_critic1_loss));
    std::printf("critic2 loss (MSE): first %.4f -> last %.4f\n", static_cast<double>(result.first_critic2_loss),
                static_cast<double>(result.last_critic2_loss));
    std::printf("actor   loss      : first %.4f -> last %.4f\n", static_cast<double>(result.first_actor_loss),
                static_cast<double>(result.last_actor_loss));

    std::printf("\n=== SAC-specific diagnostics ===\n");
    if (!result.mean_stds.empty()) {
        std::printf("policy exploration scale mean exp(log_std): %.4f -> %.4f\n",
                    static_cast<double>(result.mean_stds.front()), static_cast<double>(result.mean_stds.back()));
        std::printf("  (nothing external schedules this -- contrast DQN's epsilon-greedy decay. It is the\n");
        std::printf("   entropy term in the actor objective that keeps it from collapsing to zero.)\n");
    }
    if (!result.mean_log_probs.empty()) {
        std::printf("mean log_prob of the sampled batch action: %.4f -> %.4f   (alpha * this is the entropy "
                    "term's contribution to the actor loss)\n",
                    static_cast<double>(result.mean_log_probs.front()),
                    static_cast<double>(result.mean_log_probs.back()));
    }
    if (!result.q1_selected_fractions.empty()) {
        double sum = 0.0;
        for (double f : result.q1_selected_fractions) {
            sum += f;
        }
        std::printf("fraction of rows where Q1 was the min-selected critic, averaged over all updates: %.3f\n",
                    sum / static_cast<double>(result.q1_selected_fractions.size()));
        std::printf("  (near 0 or 1 would mean one of the twin critics is dead weight and the min is a no-op)\n");
    }

    std::printf("\n=== PolyakUpdate: did the soft targets actually move, and do they still lag? ===\n");
    std::printf("Q1 online first-layer max |final - initial|: %.6f\n", result.max_q1_weight_delta);
    std::printf("Q1 target first-layer max |final - initial|: %.6f\n", result.max_q1_target_weight_delta);
    std::printf("Q2 target first-layer max |final - initial|: %.6f\n", result.max_q2_target_weight_delta);
    std::printf("max |Q1_target param - Q1 param| at end, over EVERY parameter: %.6f\n", result.max_target_lag);
    std::printf("  (a soft update must do both: move -- or tau is a no-op -- and still lag -- or it is a hard copy)\n");
    std::printf("actor  mean-network first-layer max |final - initial|: %.6f\n", result.max_mean_weight_delta);
    std::printf("actor log_std-network first-layer max |final - initial|: %.6f\n", result.max_log_std_weight_delta);

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
