#include "pulsatrix/policy_gradient_loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "pulsatrix/assert.hpp"
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
    : backend_(backend), last_probs_(Shape({0}), backend), last_returns_(Shape({0}), backend) {}

float PolicyGradientLoss::forward(const Tensor& logits, const Tensor& actions, const Tensor& returns) {
    // Dereferences Tensor::data() directly in raw host loops -- not yet backend-generic (a
    // row-wise stabilized softmax and a per-row gather at a data-dependent column have no
    // DeviceBackend primitive). See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope
    // decision and mission_host_loop_guards.md.
    PULSATRIX_ASSERT(logits.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(actions.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(returns.device() == DeviceType::Cpu);

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
    // rollout throws without leaving a half-populated cache behind.
    std::vector<int64_t> indices(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float encoded = actions.data()[b];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
            throw std::invalid_argument("PolicyGradientLoss::forward: actions must encode whole-number action indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= action_dim) {
            throw std::invalid_argument("PolicyGradientLoss::forward: action index out of range [0, action_dim)");
        }
        indices[static_cast<size_t>(b)] = index;
    }

    // Row-wise numerically stable softmax: subtract the row max before exponentiating, the
    // pattern CrossEntropyLoss::forward established -- applied per row here rather than to one
    // vector. The selected action's log-probability is read off the stabilized expression
    // directly, never as log(p[a]), so a probability that underflowed to zero cannot produce an
    // infinite loss.
    Tensor probs(logits.shape(), backend_);
    float loss_sum = 0.0f;
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* row = logits.data() + b * action_dim;

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
        const float log_softmax_selected = row[index] - max_logit - log_exp_sum;
        loss_sum += -log_softmax_selected * returns.data()[b];
    }

    last_probs_ = probs;
    last_returns_ = returns;
    last_action_indices_ = std::move(indices);
    has_forwarded_ = true;

    // Mean over rollout *steps*, not over all N*action_dim logits: each step contributes
    // exactly one return-weighted log-probability, for the action it actually took.
    return loss_sum / static_cast<float>(batch_size);
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
    // `p[b,k]` term applies everywhere.
    Tensor grad(last_probs_.shape(), backend_);
    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t index = last_action_indices_[static_cast<size_t>(b)];
        const float weight = last_returns_.data()[b] * scale;
        for (int64_t k = 0; k < action_dim; ++k) {
            const float indicator = (k == index) ? 1.0f : 0.0f;
            grad.data()[b * action_dim + k] = weight * (last_probs_.data()[b * action_dim + k] - indicator);
        }
    }
    return grad;
}

}  // namespace pulsatrix
