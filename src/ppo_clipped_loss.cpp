#include "exai/ppo_clipped_loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "exai/assert.hpp"
#include "exai/shape.hpp"

namespace exai {
namespace {

// How far a discrete action's float encoding may sit from a whole number before it is
// rejected -- byte-for-byte PolicyGradientLoss::forward()'s kActionIntegerTolerance, itself
// DQNLoss's, itself CartPoleEnv::step()'s. Same value, same reasoning; deliberately not shared
// via a common header, because each copy is that class's own private decoding detail.
constexpr float kActionIntegerTolerance = 1e-4f;

// Rejects anything that is not exactly a rank-2 (rows, expected_width) block.
void require_matrix_shape(const Tensor& tensor, int64_t rows, int64_t expected_width, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != rows || tensor.shape().dim(1) != expected_width) {
        throw std::invalid_argument(std::string("PPOClippedLoss::forward: ") + what + " must have shape (" +
                                    std::to_string(rows) + ", " + std::to_string(expected_width) + ")");
    }
}

}  // namespace

PPOClippedLoss::PPOClippedLoss(DeviceBackend* backend)
    : backend_(backend), last_probs_(Shape({0}), backend), last_advantages_(Shape({0}), backend) {}

float PPOClippedLoss::forward(const Tensor& new_logits, const Tensor& actions, const Tensor& old_log_probs,
                              const Tensor& advantages, float clip_epsilon) {
    // Dereferences Tensor::data() directly in raw host loops -- not yet backend-generic (a
    // row-wise stabilized softmax and a per-row gather at a data-dependent column have no
    // DeviceBackend primitive). See mission_host_loop_guards.md.
    EXAI_ASSERT(new_logits.device() == DeviceType::Cpu);
    EXAI_ASSERT(actions.device() == DeviceType::Cpu);
    EXAI_ASSERT(old_log_probs.device() == DeviceType::Cpu);
    EXAI_ASSERT(advantages.device() == DeviceType::Cpu);

    if (new_logits.rank() != 2) {
        throw std::invalid_argument("PPOClippedLoss::forward: new_logits must have shape (N, action_dim)");
    }
    const int64_t batch_size = new_logits.shape().dim(0);
    const int64_t action_dim = new_logits.shape().dim(1);
    if (batch_size < 1 || action_dim < 1) {
        throw std::invalid_argument("PPOClippedLoss::forward: new_logits must have N >= 1 and action_dim >= 1");
    }
    require_matrix_shape(actions, batch_size, 1, "actions");
    require_matrix_shape(old_log_probs, batch_size, 1, "old_log_probs");
    require_matrix_shape(advantages, batch_size, 1, "advantages");
    if (!(clip_epsilon > 0.0f && clip_epsilon < 1.0f)) {
        throw std::invalid_argument("PPOClippedLoss::forward: clip_epsilon must be in (0, 1)");
    }

    // Decode (and fully validate) every action index before touching any logit, so a malformed
    // rollout throws without leaving a half-populated cache behind.
    std::vector<int64_t> indices(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float encoded = actions.data()[b];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
            throw std::invalid_argument("PPOClippedLoss::forward: actions must encode whole-number action indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= action_dim) {
            throw std::invalid_argument("PPOClippedLoss::forward: action index out of range [0, action_dim)");
        }
        indices[static_cast<size_t>(b)] = index;
    }

    const float lower = 1.0f - clip_epsilon;
    const float upper = 1.0f + clip_epsilon;

    Tensor probs(new_logits.shape(), backend_);
    std::vector<float> ratios(static_cast<size_t>(batch_size));
    std::vector<float> masks(static_cast<size_t>(batch_size));
    float loss_sum = 0.0f;

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row = new_logits.data() + b * action_dim;

        // Row-wise numerically stable softmax: subtract the row max before exponentiating,
        // PolicyGradientLoss::forward()'s pattern byte for byte. The selected action's
        // log-probability is read off the stabilized expression directly, never as log(p[a]),
        // so a probability that underflowed to zero cannot produce an infinite ratio exponent.
        float max_logit = row[0];
        for (int64_t a = 1; a < action_dim; ++a) {
            max_logit = std::max(max_logit, row[a]);
        }
        float exp_sum = 0.0f;
        for (int64_t a = 0; a < action_dim; ++a) {
            exp_sum += std::exp(row[a] - max_logit);
        }
        const float log_exp_sum = std::log(exp_sum);

        for (int64_t a = 0; a < action_dim; ++a) {
            probs.data()[b * action_dim + a] = std::exp(row[a] - max_logit) / exp_sum;
        }

        const int64_t index = indices[static_cast<size_t>(b)];
        const float new_log_prob = row[index] - max_logit - log_exp_sum;

        // The probability ratio, formed in log space and exponentiated once -- never as a
        // quotient of two probabilities, which would be the same number with two extra
        // roundings and an overflow mode when the denominator underflows.
        const float ratio = std::exp(new_log_prob - old_log_probs.data()[b]);
        const float advantage = advantages.data()[b];

        const float unclipped = ratio * advantage;
        const float clipped = std::min(std::max(ratio, lower), upper) * advantage;
        // PPO's pessimistic bound: the objective is min(unclipped, clipped), and the loss is
        // its negation.
        loss_sum += -std::min(unclipped, clipped);

        // The clip-and-mask rule. The gradient is zeroed exactly where the row has already
        // exceeded the trust region *in the direction that would increase the objective
        // further* -- there the min selects the clipped branch, which is constant in the ratio.
        // Everywhere else (inside the trust region, or outside it in the direction that hurts
        // the objective and should be pulled back) the unclipped gradient stands in full.
        const bool clipped_out = (advantage >= 0.0f && ratio > upper) || (advantage < 0.0f && ratio < lower);
        ratios[static_cast<size_t>(b)] = ratio;
        masks[static_cast<size_t>(b)] = clipped_out ? 0.0f : 1.0f;
    }

    last_probs_ = probs;
    last_advantages_ = advantages;
    last_ratios_ = std::move(ratios);
    last_masks_ = std::move(masks);
    last_action_indices_ = std::move(indices);
    has_forwarded_ = true;

    // Mean over rollout *steps*, not over all N*action_dim logits: each step contributes
    // exactly one clipped surrogate term, for the action it actually took.
    return loss_sum / static_cast<float>(batch_size);
}

Tensor PPOClippedLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("PPOClippedLoss::backward called before forward");
    }

    const int64_t batch_size = last_probs_.shape().dim(0);
    const int64_t action_dim = last_probs_.shape().dim(1);
    const float scale = 1.0f / static_cast<float>(batch_size);

    // Dense across every action column, unlike DQNLoss::backward()'s masked write: pushing
    // probability onto the taken action takes it from every other action. A masked-out *row*
    // is a different thing entirely -- there the whole row is exactly 0.0f by construction,
    // because `weight` is exactly 0.0f.
    Tensor grad(last_probs_.shape(), backend_);
    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t index = last_action_indices_[static_cast<size_t>(b)];
        const float weight = -last_masks_[static_cast<size_t>(b)] * last_advantages_.data()[b] *
                             last_ratios_[static_cast<size_t>(b)] * scale;
        for (int64_t k = 0; k < action_dim; ++k) {
            const float indicator = (k == index) ? 1.0f : 0.0f;
            grad.data()[b * action_dim + k] = weight * (indicator - last_probs_.data()[b * action_dim + k]);
        }
    }
    return grad;
}

}  // namespace exai
