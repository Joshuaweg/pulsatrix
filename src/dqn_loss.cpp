#include "pulsatrix/dqn_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// How far a discrete action's float encoding may sit from a whole number before it is
// rejected -- byte-for-byte CartPoleEnv::step()'s own kActionIntegerTolerance. Deliberately
// the same constant value and the same reasoning: loose enough to tolerate a policy's float
// round-trip, tight enough that a genuinely fractional "action" (an un-argmaxed probability)
// is caught. Not shared via a common header, because CartPoleEnv's copy is an environment's
// private decoding detail and this one is a loss's; a shared constant would couple the two
// classes for no benefit beyond four bytes.
constexpr float kActionIntegerTolerance = 1e-4f;

// Rejects anything that is not exactly a rank-2 (rows, expected_width) block.
void require_matrix_shape(const Tensor& tensor, int64_t rows, int64_t expected_width, const char* what) {
    if (tensor.rank() != 2 || tensor.shape().dim(0) != rows || tensor.shape().dim(1) != expected_width) {
        throw std::invalid_argument(std::string("DQNLoss::forward: ") + what + " must have shape (" +
                                    std::to_string(rows) + ", " + std::to_string(expected_width) + ")");
    }
}

}  // namespace

DQNLoss::DQNLoss(DeviceBackend* backend)
    : backend_(backend), last_q_values_(Shape({0}), backend), last_targets_(Shape({0}), backend) {}

float DQNLoss::forward(const Tensor& q_values, const Tensor& actions, const Tensor& targets) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic (a
    // per-row gather at a data-dependent column has no DeviceBackend primitive). See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(q_values.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(actions.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(targets.device() == DeviceType::Cpu);

    if (q_values.rank() != 2) {
        throw std::invalid_argument("DQNLoss::forward: q_values must have shape (N, action_dim)");
    }
    const int64_t batch_size = q_values.shape().dim(0);
    const int64_t action_dim = q_values.shape().dim(1);
    if (batch_size < 1 || action_dim < 1) {
        throw std::invalid_argument("DQNLoss::forward: q_values must have N >= 1 and action_dim >= 1");
    }
    require_matrix_shape(actions, batch_size, 1, "actions");
    require_matrix_shape(targets, batch_size, 1, "targets");

    // Decode (and fully validate) every action index before touching any Q-value, so a
    // malformed batch throws without leaving a half-populated cache behind.
    std::vector<int64_t> indices(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const float encoded = actions.data()[b];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kActionIntegerTolerance) {
            throw std::invalid_argument("DQNLoss::forward: actions must encode whole-number action indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= action_dim) {
            throw std::invalid_argument("DQNLoss::forward: action index out of range [0, action_dim)");
        }
        indices[static_cast<size_t>(b)] = index;
    }

    last_q_values_ = q_values;
    last_targets_ = targets;
    last_action_indices_ = std::move(indices);
    has_forwarded_ = true;

    // Mean over *transitions*, not over all N*action_dim Q-values: each transition
    // contributes exactly one squared TD error, on the action it actually took.
    float sum_squared = 0.0f;
    for (int64_t b = 0; b < batch_size; ++b) {
        const float selected = q_values.data()[b * action_dim + last_action_indices_[static_cast<size_t>(b)]];
        const float diff = selected - targets.data()[b];
        sum_squared += diff * diff;
    }
    return sum_squared / static_cast<float>(batch_size);
}

Tensor DQNLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("DQNLoss::backward called before forward");
    }

    const int64_t batch_size = last_q_values_.shape().dim(0);
    const int64_t action_dim = last_q_values_.shape().dim(1);
    const float scale = 2.0f / static_cast<float>(batch_size);

    // Zero-initialized by Tensor's own contract, so the non-selected columns are already
    // exactly 0.0f -- only the taken action's column is written. That untouched-zero pattern
    // is the masking: the Q-network's backward() receives gradient signal solely for the
    // action the behaviour policy actually took.
    Tensor grad(last_q_values_.shape(), backend_);
    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t index = last_action_indices_[static_cast<size_t>(b)];
        const float selected = last_q_values_.data()[b * action_dim + index];
        grad.data()[b * action_dim + index] = scale * (selected - last_targets_.data()[b]);
    }
    return grad;
}

}  // namespace pulsatrix
