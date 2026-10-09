#include "pulsatrix/token_cross_entropy_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {

// Same tolerance PolicyGradientLoss uses for float-encoded indices.
constexpr float kIndexIntegerTolerance = 1e-4f;

}  // namespace

TokenCrossEntropyLoss::TokenCrossEntropyLoss(DeviceBackend* backend, int64_t ignore_index)
    : backend_(backend),
      ignore_index_(ignore_index),
      probs_(Shape({0}), backend),
      indices_(Shape({0}), backend),
      weights_(Shape({0}), backend) {}

float TokenCrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets, float normalizer) {
    require_device(logits, backend_->device(), "TokenCrossEntropyLoss::forward");
    require_device(targets, backend_->device(), "TokenCrossEntropyLoss::forward");
    if (logits.rank() < 2 || logits.numel() <= 0) {
        throw std::invalid_argument("TokenCrossEntropyLoss::forward: logits must be (..., classes) with rank >= 2");
    }
    const int64_t classes = logits.shape().dim(logits.rank() - 1);
    const int64_t rows = logits.numel() / classes;
    bool shape_ok = targets.rank() == logits.rank() - 1;
    for (int64_t d = 0; shape_ok && d < targets.rank(); ++d) {
        shape_ok = targets.shape().dim(d) == logits.shape().dim(d);
    }
    if (!shape_ok) {
        throw std::invalid_argument("TokenCrossEntropyLoss::forward: targets must have the logits' shape without classes");
    }
    if (!std::isfinite(normalizer) || normalizer < 0.0f) {
        throw std::invalid_argument("TokenCrossEntropyLoss::forward: normalizer must be finite and >= 0");
    }

    // HIP-6: one pass on the targets' device decodes and checks every target and computes the
    // per-row terms; one read back returns the loss, the token count and the error counts.
    // Into locals, so a bad target leaves the previous forward()'s state as it was.
    Tensor probs(Shape({rows, classes}), backend_), indices(Shape({rows, 1}), backend_), weights(Shape({rows, 1}), backend_);
    Tensor stats(Shape({rows, 4}), backend_);
    RlRowArgs args;
    args.in[0] = logits.data();
    args.in[1] = targets.data();
    args.out[0] = probs.data();
    args.out[1] = stats.data();
    args.out[2] = indices.data();
    args.out[3] = weights.data();
    args.rows = rows;
    args.cols = classes;
    args.ignore_index = ignore_index_;
    backend_->rl_rows(RlRowOp::TokenCeLoss, args);
    // Columns summed over rows in order from 0.0f: the loss terms add up as they did one by one.
    Tensor sums(Shape({4}), backend_);
    backend_->column_sums(stats.data(), sums.data(), static_cast<size_t>(rows), 4, 0.0f);
    const std::vector<float> totals = sums.to_host_vector();
    if (totals[2] > 0.0f || totals[3] > 0.0f) {
        ThrowForBadTarget(targets, classes);
    }
    const auto tokens = static_cast<int64_t>(totals[1]);
    probs_ = std::move(probs);
    indices_ = std::move(indices);
    weights_ = std::move(weights);

    logits_shape_ = logits.shape();
    num_tokens_ = tokens;
    normalizer_ = normalizer > 0.0f ? normalizer : static_cast<float>(tokens);
    has_forwarded_ = true;
    if (tokens == 0) {
        return 0.0f;
    }
    return totals[0] / normalizer_;
}

void TokenCrossEntropyLoss::ThrowForBadTarget(const Tensor& targets, int64_t classes) const {
    // The slow path, only for an error: find the first bad target on the host, for the message.
    for (const float e : targets.to_host_vector()) {
        const float rounded = std::round(e);
        if (!(std::fabs(e - rounded) <= kIndexIntegerTolerance)) {
            throw std::invalid_argument("TokenCrossEntropyLoss::forward: targets must be whole numbers");
        }
        const auto index = static_cast<int64_t>(rounded);
        if (index != ignore_index_ && (index < 0 || index >= classes)) {
            throw std::invalid_argument("TokenCrossEntropyLoss::forward: target " + std::to_string(index) +
                                        " is outside [0, " + std::to_string(classes) + ") and not the ignore index");
        }
    }
    throw std::logic_error("TokenCrossEntropyLoss::forward: the device reported a bad target the host can't find");
}

Tensor TokenCrossEntropyLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("TokenCrossEntropyLoss::backward called before forward");
    }
    const int64_t rows = probs_.shape().dim(0), classes = probs_.shape().dim(1);
    Tensor grad(logits_shape_, backend_);
    if (num_tokens_ == 0) {
        grad.fill(0.0f);
        return grad;
    }
    // weight * (p - onehot) / normalizer per row: zero rows for ignored tokens.
    RlRowArgs args;
    args.in[0] = probs_.data();
    args.in[1] = indices_.data();
    args.in[2] = weights_.data();
    args.out[0] = grad.data();
    args.rows = rows;
    args.cols = classes;
    args.scale = 1.0f / normalizer_;
    backend_->rl_rows(RlRowOp::PgGrad, args);
    return grad;
}

int64_t CountTargetTokens(const Tensor& targets, int64_t ignore_index) {
    int64_t count = 0;
    for (float e : targets.to_host_vector()) {
        if (static_cast<int64_t>(std::round(e)) != ignore_index) {
            ++count;
        }
    }
    return count;
}

}  // namespace pulsatrix
