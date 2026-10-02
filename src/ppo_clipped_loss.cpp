#include "pulsatrix/ppo_clipped_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/shape.hpp"

namespace pulsatrix {
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
    : backend_(backend),
      last_probs_(Shape({0}), backend),
      last_advantages_(Shape({0}), backend),
      last_ratios_(Shape({0}), backend),
      last_masks_(Shape({0}), backend),
      last_action_indices_(Shape({0}), backend) {}

float PPOClippedLoss::forward(const Tensor& new_logits, const Tensor& actions, const Tensor& old_log_probs,
                              const Tensor& advantages, float clip_epsilon) {
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
    // rollout throws without leaving a half-populated cache behind. Validation can throw per
    // element, so it runs on the host: one device->host copy of the N actions.
    const std::vector<float> encoded_actions = actions.to_host_vector();
    std::vector<float> indices(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float encoded = encoded_actions[static_cast<size_t>(b)];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
            throw std::invalid_argument("PPOClippedLoss::forward: actions must encode whole-number action indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= action_dim) {
            throw std::invalid_argument("PPOClippedLoss::forward: action index out of range [0, action_dim)");
        }
        indices[static_cast<size_t>(b)] = static_cast<float>(index);
    }
    Tensor index_tensor(Shape({batch_size, 1}), backend_, indices);

    const float lower = 1.0f - clip_epsilon;
    const float upper = 1.0f + clip_epsilon;

    // Device-generic (GPU-native-kernels Mission 7): one rl_rows(PpoLoss) lane per row runs
    // PolicyGradientLoss::forward()'s row-wise stabilized softmax byte for byte, forms the
    // probability ratio in log space and exponentiates it once (never a quotient of two
    // probabilities), and writes PPO's pessimistic term -min(unclipped, clipped) together with
    // the row's ratio and clip mask. The mask is 0 exactly where the row has already exceeded
    // the trust region *in the direction that would increase the objective further* -- there
    // the min selects the clipped branch, which is constant in the ratio. column_sums adds the
    // per-row terms in increasing row order from 0.0f -- the original `loss_sum += term` order.
    Tensor probs(new_logits.shape(), backend_);
    Tensor terms(Shape({batch_size, 1}), backend_);
    Tensor ratios(Shape({batch_size, 1}), backend_);
    Tensor masks(Shape({batch_size, 1}), backend_);
    RlRowArgs args;
    args.in[0] = new_logits.data();
    args.in[1] = index_tensor.data();
    args.in[2] = old_log_probs.data();
    args.in[3] = advantages.data();
    args.out[0] = probs.data();
    args.out[1] = terms.data();
    args.out[2] = ratios.data();
    args.out[3] = masks.data();
    args.rows = batch_size;
    args.cols = action_dim;
    args.lower = lower;
    args.upper = upper;
    backend_->rl_rows(RlRowOp::PpoLoss, args);
    Tensor loss_sum(Shape({1}), backend_);
    backend_->column_sums(terms.data(), loss_sum.data(), static_cast<size_t>(batch_size), 1, 0.0f);

    last_probs_ = probs;
    last_advantages_ = advantages;
    last_ratios_ = std::move(ratios);
    last_masks_ = std::move(masks);
    last_action_indices_ = std::move(index_tensor);
    has_forwarded_ = true;

    // Mean over rollout *steps*, not over all N*action_dim logits: each step contributes
    // exactly one clipped surrogate term, for the action it actually took.
    return loss_sum.read_element(0) / static_cast<float>(batch_size);
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
    // because `weight` is exactly 0.0f. One rl_rows(PpoGrad) lane per row.
    Tensor grad(last_probs_.shape(), backend_);
    RlRowArgs args;
    args.in[0] = last_probs_.data();
    args.in[1] = last_action_indices_.data();
    args.in[2] = last_advantages_.data();
    args.in[3] = last_ratios_.data();
    args.in[4] = last_masks_.data();
    args.out[0] = grad.data();
    args.rows = batch_size;
    args.cols = action_dim;
    args.scale = scale;
    backend_->rl_rows(RlRowOp::PpoGrad, args);
    return grad;
}

}  // namespace pulsatrix
