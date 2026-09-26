/** @file gae_and_ppo_clipped.cpp
 *  @brief Recipe: Generalized Advantage Estimation over a small synthetic rollout, then
 *         PPO's clipped surrogate objective using those advantages. Paired with
 *         docs/recipes/rl/gae_and_ppo_clipped.md.
 */
#include <cmath>
#include <cstdio>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gae.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"

namespace {

// Row-wise stabilized log-softmax, matching PPOClippedLoss's own internal convention --
// used here only to build old_log_probs from a snapshot of the "data-collecting" policy.
std::vector<float> log_softmax_row(const std::vector<float>& logits) {
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    float sum_exp = 0.0f;
    for (float v : logits) sum_exp += std::exp(v - max_logit);
    float log_sum_exp = std::log(sum_exp) + max_logit;
    std::vector<float> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] - log_sum_exp;
    return out;
}

}  // namespace

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    constexpr int64_t kSteps = 6;
    constexpr int64_t kActionDim = 2;

    // A short synthetic rollout: rewards spike near the end, one terminal step.
    Tensor rewards(Shape({kSteps, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f});
    Tensor dones(Shape({kSteps, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
    Tensor values(Shape({kSteps, 1}), &backend, {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});

    GAEResult gae = ComputeGAE(rewards, dones, values, /*bootstrap_value=*/0.0f, /*gamma=*/0.99f, /*lambda=*/0.95f,
                                &backend);

    std::printf("GAE recipe -- %lld-step synthetic rollout, gamma=0.99, lambda=0.95\n\n",
                static_cast<long long>(kSteps));
    std::printf("%-5s %8s %8s %8s %10s %10s\n", "step", "reward", "done", "value", "advantage", "return");
    for (int64_t t = 0; t < kSteps; ++t) {
        std::printf("%-5lld %8.2f %8.0f %8.2f %10.4f %10.4f\n", static_cast<long long>(t), rewards.data()[t],
                    dones.data()[t], values.data()[t], gae.advantages.data()[t], gae.returns.data()[t]);
    }

    // --- PPO's clipped surrogate objective, using the advantages just computed ---
    // "Old" (data-collecting) policy logits per step, and the action each one took.
    std::vector<std::vector<float>> old_logits_rows = {
        {0.1f, -0.1f}, {0.2f, 0.0f}, {-0.1f, 0.3f}, {0.0f, 0.0f}, {0.4f, -0.2f}, {0.1f, 0.1f}};
    Tensor actions(Shape({kSteps, 1}), &backend, {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f});

    std::vector<float> old_log_probs_data(static_cast<size_t>(kSteps));
    for (int64_t t = 0; t < kSteps; ++t) {
        auto log_probs = log_softmax_row(old_logits_rows[static_cast<size_t>(t)]);
        int64_t action = static_cast<int64_t>(actions.data()[t] + 0.5f);
        old_log_probs_data[static_cast<size_t>(t)] = log_probs[static_cast<size_t>(action)];
    }
    Tensor old_log_probs(Shape({kSteps, 1}), &backend, old_log_probs_data);

    PPOClippedLoss ppo(&backend);

    std::printf("\nPPO clipped surrogate, comparing the current policy to the data-collecting one:\n");

    // Case 1: current policy is identical to the data-collecting one -- every ratio is
    // exactly 1, so the loss degenerates to an advantage-weighted negative log-likelihood.
    std::vector<float> flat_old_logits;
    for (auto& row : old_logits_rows)
        for (float v : row) flat_old_logits.push_back(v);
    Tensor new_logits_unchanged(Shape({kSteps, kActionDim}), &backend, flat_old_logits);

    float loss_unchanged =
        ppo.forward(new_logits_unchanged, actions, old_log_probs, gae.advantages, /*clip_epsilon=*/0.2f);
    std::printf("  policy unchanged (ratio == 1 everywhere): loss = %.6f\n", loss_unchanged);

    // Case 2: current policy has drifted -- the taken action's logit is pushed up relative
    // to the others (an asymmetric shift; a uniform shift would leave softmax unchanged,
    // since softmax is shift-invariant), so some ratios now exceed the trust region.
    std::vector<float> drifted_logits;
    for (int64_t t = 0; t < kSteps; ++t) {
        const auto& row = old_logits_rows[static_cast<size_t>(t)];
        int64_t action = static_cast<int64_t>(actions.data()[t] + 0.5f);
        for (int64_t k = 0; k < kActionDim; ++k) {
            drifted_logits.push_back(row[static_cast<size_t>(k)] + (k == action ? 3.0f : 0.0f));
        }
    }
    Tensor new_logits_drifted(Shape({kSteps, kActionDim}), &backend, drifted_logits);

    float loss_drifted =
        ppo.forward(new_logits_drifted, actions, old_log_probs, gae.advantages, /*clip_epsilon=*/0.2f);
    std::printf("  policy drifted (taken action's logit +3): loss = %.6f\n", loss_drifted);

    std::printf(
        "\nAt ratio == 1 the clip never bites and the loss is exactly PolicyGradientLoss's own\n"
        "advantage-weighted surrogate. Once the policy drifts, PPOClippedLoss::backward()\n"
        "zeroes the gradient for any step that has already run past the trust region in the\n"
        "direction that would over-reward it -- that's the mechanism that keeps PPO's updates\n"
        "bounded across multiple gradient epochs over the same rollout.\n");

    return 0;
}
