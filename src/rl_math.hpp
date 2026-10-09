// Reinforcement-learning row math (DQNLoss, PolicyGradientLoss, PPOClippedLoss, the DQN target
// helpers, PolyakUpdate) shared verbatim by CPUBackend (a loop over rows) and the GPU kernels
// (one thread per row). Each routine is the pre-campaign host loop body for one batch row,
// unchanged in expression and order:
// - a row's softmax max / exp-sum / argmax is owned by its thread and walks the columns in the
//   original order (std::max / std::min reproduced as the libstdc++ comparisons, argmax first
//   max wins with strict >);
// - a gradient row is written only by its own thread -- no atomics. DqnGrad writes only the
//   taken action's element into a caller-zeroed buffer, exactly like the original masked loop.
// Cross-row loss sums are not done here: each op writes per-row terms that the caller reduces
// with DeviceBackend::column_sums, which adds rows in increasing order from 0.0f -- the original
// `loss_sum += term` order. Slot meanings per op are documented on RlRowOp in device_backend.hpp.
// Private to src/. GPU-native-kernels Mission 7.
#pragma once

#include <cstdint>

#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace rl {

// std::max(a, b) / std::min(a, b) exactly as libstdc++ defines them (including which operand
// a NaN comparison returns), usable in device code.
PULSATRIX_HOST_DEVICE inline float max_of(float a, float b) { return (a < b) ? b : a; }
PULSATRIX_HOST_DEVICE inline float min_of(float a, float b) { return (b < a) ? b : a; }

// The validated action index of row b, stored as a whole-number float (exact below 2^24).
PULSATRIX_HOST_DEVICE inline int64_t action_index(const RlRowArgs& a, int64_t b) {
    return static_cast<int64_t>(a.in[1][b]);
}

// DQNLoss::forward: the squared TD error of the taken action.
PULSATRIX_HOST_DEVICE inline void dqn_loss(const RlRowArgs& a, int64_t b) {
    const float selected = a.in[0][b * a.cols + action_index(a, b)];
    const float diff = selected - a.in[2][b];
    a.out[0][b] = diff * diff;
}

// DQNLoss::backward: only the taken action's column is written.
PULSATRIX_HOST_DEVICE inline void dqn_grad(const RlRowArgs& a, int64_t b) {
    const int64_t index = action_index(a, b);
    const float selected = a.in[0][b * a.cols + index];
    a.out[0][b * a.cols + index] = a.scale * (selected - a.in[2][b]);
}

// Row-wise stabilized softmax of logits row b into probs, returning the taken action's
// log-probability read off the stabilized expression -- PolicyGradientLoss::forward()'s and
// PPOClippedLoss::forward()'s shared prefix, byte for byte.
PULSATRIX_HOST_DEVICE inline float softmax_row(const RlRowArgs& a, int64_t b, float* probs) {
    const float* row = a.in[0] + b * a.cols;
    float max_logit = row[0];
    for (int64_t k = 1; k < a.cols; ++k) {
        max_logit = max_of(max_logit, row[k]);
    }
    float exp_sum = 0.0f;
    for (int64_t k = 0; k < a.cols; ++k) {
        exp_sum += expf(row[k] - max_logit);
    }
    const float log_exp_sum = logf(exp_sum);
    for (int64_t k = 0; k < a.cols; ++k) {
        probs[b * a.cols + k] = expf(row[k] - max_logit) / exp_sum;
    }
    return row[action_index(a, b)] - max_logit - log_exp_sum;
}

// PolicyGradientLoss::forward: probs row plus the return-weighted negative log-probability.
PULSATRIX_HOST_DEVICE inline void pg_loss(const RlRowArgs& a, int64_t b) {
    const float log_softmax_selected = softmax_row(a, b, a.out[0]);
    a.out[1][b] = -log_softmax_selected * a.in[2][b];
}

// PolicyGradientLoss::backward: dense row, weight * (p - 1{k == a_b}).
PULSATRIX_HOST_DEVICE inline void pg_grad(const RlRowArgs& a, int64_t b) {
    const int64_t index = action_index(a, b);
    const float weight = a.in[2][b] * a.scale;
    for (int64_t k = 0; k < a.cols; ++k) {
        const float indicator = (k == index) ? 1.0f : 0.0f;
        a.out[0][b * a.cols + k] = weight * (a.in[0][b * a.cols + k] - indicator);
    }
}

// PPOClippedLoss::forward: probs row, the clipped-surrogate term, the ratio and the mask.
PULSATRIX_HOST_DEVICE inline void ppo_loss(const RlRowArgs& a, int64_t b) {
    const float new_log_prob = softmax_row(a, b, a.out[0]);
    const float ratio = expf(new_log_prob - a.in[2][b]);
    const float advantage = a.in[3][b];
    const float unclipped = ratio * advantage;
    const float clipped = min_of(max_of(ratio, a.lower), a.upper) * advantage;
    a.out[1][b] = -min_of(unclipped, clipped);
    const bool clipped_out = (advantage >= 0.0f && ratio > a.upper) || (advantage < 0.0f && ratio < a.lower);
    a.out[2][b] = ratio;
    a.out[3][b] = clipped_out ? 0.0f : 1.0f;
}

// PPOClippedLoss::backward: dense row, weight * (1{k == a_b} - p).
PULSATRIX_HOST_DEVICE inline void ppo_grad(const RlRowArgs& a, int64_t b) {
    const int64_t index = action_index(a, b);
    const float weight = -a.in[4][b] * a.in[2][b] * a.in[3][b] * a.scale;
    for (int64_t k = 0; k < a.cols; ++k) {
        const float indicator = (k == index) ? 1.0f : 0.0f;
        a.out[0][b * a.cols + k] = weight * (indicator - a.in[0][b * a.cols + k]);
    }
}

// ComputeDQNTarget / ComputeDoubleDQNTarget: argmax of the selection row (ties to the lowest
// index, strict >), evaluated in the evaluation row, bootstrapped through (1 - done).
PULSATRIX_HOST_DEVICE inline void dqn_target(const RlRowArgs& a, int64_t b) {
    const float* select = a.in[0] + b * a.cols;
    int64_t best = 0;
    for (int64_t k = 1; k < a.cols; ++k) {
        if (select[k] > select[best]) {
            best = k;
        }
    }
    const float evaluated = a.in[1][b * a.cols + best];
    const float bootstrap = a.gamma * (1.0f - a.in[3][b]) * evaluated;
    a.out[0][b] = a.in[2][b] + bootstrap;
}

// PolyakUpdate: one element of the exponential moving average.
PULSATRIX_HOST_DEVICE inline void polyak_blend(const RlRowArgs& a, int64_t i) {
    a.out[0][i] = a.tau * a.in[0][i] + (1.0f - a.tau) * a.in[1][i];
}

// TokenCrossEntropyLoss::forward (HIP-6): decode and check row b's target on the device, then
// PolicyGradientLoss's row with the decoded index and weight -- the same terms the host-validated
// path computed. Statistics row: term, counted token, not a whole number, out of range.
PULSATRIX_HOST_DEVICE inline void token_ce_loss(const RlRowArgs& a, int64_t b) {
    const float e = a.in[1][b];
    const float rounded = roundf(e);
    float index = 0.0f, weight = 0.0f, bad_integer = 0.0f, bad_range = 0.0f;
    if (!(fabsf(e - rounded) <= 1e-4f)) {  // TokenCrossEntropyLoss's kIndexIntegerTolerance
        bad_integer = 1.0f;
    } else {
        const auto i = static_cast<int64_t>(rounded);
        if (i != a.ignore_index) {
            if (i < 0 || i >= a.cols) {
                bad_range = 1.0f;
            } else {
                index = static_cast<float>(i);
                weight = 1.0f;
            }
        }
    }
    a.out[2][b] = index;
    a.out[3][b] = weight;
    RlRowArgs decoded = a;
    decoded.in[1] = a.out[2];
    const float log_softmax_selected = softmax_row(decoded, b, a.out[0]);
    float* stats = a.out[1] + b * 4;
    stats[0] = -log_softmax_selected * weight;
    stats[1] = weight;
    stats[2] = bad_integer;
    stats[3] = bad_range;
}

PULSATRIX_HOST_DEVICE inline void row(RlRowOp op, const RlRowArgs& a, int64_t b) {
    switch (op) {
        case RlRowOp::DqnLoss:
            dqn_loss(a, b);
            break;
        case RlRowOp::DqnGrad:
            dqn_grad(a, b);
            break;
        case RlRowOp::PgLoss:
            pg_loss(a, b);
            break;
        case RlRowOp::PgGrad:
            pg_grad(a, b);
            break;
        case RlRowOp::PpoLoss:
            ppo_loss(a, b);
            break;
        case RlRowOp::PpoGrad:
            ppo_grad(a, b);
            break;
        case RlRowOp::DqnTarget:
            dqn_target(a, b);
            break;
        case RlRowOp::PolyakBlend:
            polyak_blend(a, b);
            break;
        case RlRowOp::TokenCeLoss:
            token_ce_loss(a, b);
            break;
    }
}

}  // namespace rl
}  // namespace pulsatrix
