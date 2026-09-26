/** @file dqn_cartpole.cpp
 *  @brief Recipe: trains Double DQN on CartPoleEnv and reports the first-20/last-20 episode
 *         average comparison. Paired with docs/recipes/rl/dqn_cartpole.md.
 *  @note Reuses examples/dqn_cartpole_training.hpp verbatim -- the same tuned hyperparameters
 *        examples/dqn_cartpole_demo.cpp and tests/dqn_cartpole_integration_test.cpp share, so
 *        this recipe cannot silently drift from the validated training loop. Only the printed
 *        output is trimmed down to the summary.
 */
#include <cstdio>

#include "dqn_cartpole_training.hpp"

int main() {
    using namespace pulsatrix::dqn_cartpole;

    const TrainingConfig config;

    std::printf("DQN CartPole recipe -- Double DQN, Linear(4,%lld)->ReLU->Linear(%lld,2)\n",
                static_cast<long long>(config.hidden_size), static_cast<long long>(config.hidden_size));
    std::printf("episodes=%lld  max_steps=%lld  Adam(lr=%.4f)  gamma=%.2f\n\n",
                static_cast<long long>(config.num_episodes), static_cast<long long>(config.max_steps),
                static_cast<double>(config.learning_rate), static_cast<double>(config.gamma));

    const TrainingResult result = RunTraining(config, [](int64_t, int64_t, float) {});

    const double ratio = result.first_window_average > 0.0 ? result.last_window_average / result.first_window_average
                                                            : 0.0;
    std::printf("first %lld episodes, mean length: %.2f\n", static_cast<long long>(kWindow),
                result.first_window_average);
    std::printf("last  %lld episodes, mean length: %.2f\n", static_cast<long long>(kWindow),
                result.last_window_average);
    std::printf("improvement ratio: %.2fx\n", ratio);

    std::printf(
        "\nSyncTargetNetwork's hard periodic copy keeps the DQN target stable between\n"
        "updates -- without it, the Q-network would be regressing toward a target that moves\n"
        "every single step, which is a well-known source of divergence in DQN training.\n"
        "See PolyakUpdate (used by SAC) for the soft alternative to this hard-copy scheme.\n");

    return 0;
}
