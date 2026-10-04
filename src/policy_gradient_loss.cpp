#include "pulsatrix/policy_gradient_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// How far a discrete action's float encoding may sit from a whole number before it is
// rejected -- byte-for-byte DQNLoss::forward()'s kActionIntegerTolerance, itself
// CartPoleEnv::step()'s. Same value, same reasoning; deliberately not shared via a common
// header, because each copy is that class's own private decoding detail and a shared constant
// would couple three unrelated classes for four bytes.
constexpr float kActionIntegerTolerance = 1e-4f;

// Rejects anything that is not exactly a rank-2 (rows, expected_width) block.
void require_matrix_shape(const Tensor& tensor, int64_t rows, int64_t expected_width, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != rows || tensor.shape().dim(1) != expected_width) {
        throw std::invalid_argument(std::string("PolicyGradientLoss::forward: ") + what + " must have shape (" +
                                    std::to_string(rows) + ", " + std::to_string(expected_width) + ")");
    }
}

}  // namespace

PolicyGradientLoss::PolicyGradientLoss(DeviceBackend* backend)
    : backend_(backend),
      last_probs_(Shape({0}), backend),
      last_returns_(Shape({0}), backend),
      last_action_indices_(Shape({0}), backend) {}

float PolicyGradientLoss::forward(const Tensor& logits, const Tensor& actions, const Tensor& returns) {
    require_device(logits, backend_->device(), "PolicyGradientLoss::forward");
    require_device(actions, backend_->device(), "PolicyGradientLoss::forward");
    require_device(returns, backend_->device(), "PolicyGradientLoss::forward");
    if (logits.rank() != 2) {
        throw std::invalid_argument("PolicyGradientLoss::forward: logits must have shape (N, action_dim)");
    }
    const int64_t batch_size = logits.shape().dim(0);
    const int64_t action_dim = logits.shape().dim(1);
    if (batch_size < 1 || action_dim < 1) {
        throw std::invalid_argument("PolicyGradientLoss::forward: logits must have N >= 1 and action_dim >= 1");
    }
    require_matrix_shape(actions, batch_size, 1, "actions");
    require_matrix_shape(returns, batch_size, 1, "returns");

    // Decode (and fully validate) every action index before touching any logit, so a malformed
    // rollout throws without leaving a half-populated cache behind. Validation can throw per
    // element, so it runs on the host: one device->host copy of the N actions.
    const std::vector<float> encoded_actions = actions.to_host_vector();
    std::vector<float> indices(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float encoded = encoded_actions[static_cast<size_t>(b)];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
            throw std::invalid_argument("PolicyGradientLoss::forward: actions must encode whole-number action indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= action_dim) {
            throw std::invalid_argument("PolicyGradientLoss::forward: action index out of range [0, action_dim)");
        }
        indices[static_cast<size_t>(b)] = static_cast<float>(index);
    }
    Tensor index_tensor(Shape({batch_size, 1}), backend_, indices);

    // Device-generic (GPU-native-kernels Mission 7). Row-wise numerically stable softmax:
    // subtract the row max before exponentiating, the pattern CrossEntropyLoss::forward
    // established -- applied per row here rather than to one vector. The selected action's
    // log-probability is read off the stabilized expression directly, never as log(p[a]), so a
    // probability that underflowed to zero cannot produce an infinite loss. rl_rows(PgLoss)
    // writes probs and the per-row terms; column_sums adds the terms in increasing row order
    // from 0.0f -- the original `loss_sum += term` order.
    Tensor probs(logits.shape(), backend_);
    Tensor terms(Shape({batch_size, 1}), backend_);
    RlRowArgs args;
    args.in[0] = logits.data();
    args.in[1] = index_tensor.data();
    args.in[2] = returns.data();
    args.out[0] = probs.data();
    args.out[1] = terms.data();
    args.rows = batch_size;
    args.cols = action_dim;
    backend_->rl_rows(RlRowOp::PgLoss, args);
    Tensor loss_sum(Shape({1}), backend_);
    backend_->column_sums(terms.data(), loss_sum.data(), static_cast<size_t>(batch_size), 1, 0.0f);

    last_probs_ = probs;
    last_returns_ = returns;
    last_action_indices_ = std::move(index_tensor);
    has_forwarded_ = true;

    // Mean over rollout *steps*, not over all N*action_dim logits: each step contributes
    // exactly one return-weighted log-probability, for the action it actually took.
    return loss_sum.read_element(0) / static_cast<float>(batch_size);
}

Tensor PolicyGradientLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("PolicyGradientLoss::backward called before forward");
    }

    const int64_t batch_size = last_probs_.shape().dim(0);
    const int64_t action_dim = last_probs_.shape().dim(1);
    const float scale = 1.0f / static_cast<float>(batch_size);

    // Every element is written, unlike DQNLoss::backward()'s masked write: the policy gradient
    // is dense across all actions, because pushing probability onto the taken action takes it
    // from every other action. The `- 1` term applies only in the taken action's column; the
    // `p[b,k]` term applies everywhere. One rl_rows(PgGrad) lane per row.
    Tensor grad(last_probs_.shape(), backend_);
    RlRowArgs args;
    args.in[0] = last_probs_.data();
    args.in[1] = last_action_indices_.data();
    args.in[2] = last_returns_.data();
    args.out[0] = grad.data();
    args.rows = batch_size;
    args.cols = action_dim;
    args.scale = scale;
    backend_->rl_rows(RlRowOp::PgGrad, args);
    return grad;
}

}  // namespace pulsatrix
