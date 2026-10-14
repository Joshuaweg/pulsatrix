// Knowledge distillation (KD-1): T² · KL between softened teacher and student outputs, mixed with
// the hard-label cross-entropy when there are targets.

#include "pulsatrix/distillation_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

DistillationLoss::DistillationLoss(DeviceBackend* backend, DistillationOptions options)
    : backend_(backend),
      options_(options),
      hard_(backend, options.ignore_index),
      soft_grad_(Shape({0}), backend),
      kept_(Shape({0}), backend) {
    if (!(std::isfinite(options.temperature) && options.temperature > 0.0f)) {
        throw std::invalid_argument("DistillationLoss: temperature must be finite and positive");
    }
    if (!(options.alpha >= 0.0f && options.alpha <= 1.0f)) {
        throw std::invalid_argument("DistillationLoss: alpha must be in [0, 1]");
    }
}

float DistillationLoss::forward(const Tensor& student_logits, const Tensor& teacher_logits, float normalizer) {
    return Run(student_logits, teacher_logits, nullptr, normalizer);
}

float DistillationLoss::forward(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor& targets,
                                float normalizer) {
    return Run(student_logits, teacher_logits, &targets, normalizer);
}

float DistillationLoss::Run(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor* targets,
                            float normalizer) {
    require_device(student_logits, backend_->device(), "DistillationLoss::forward");
    require_device(teacher_logits, backend_->device(), "DistillationLoss::forward");
    if (student_logits.rank() < 2 || student_logits.numel() <= 0) {
        throw std::invalid_argument("DistillationLoss::forward: logits must be (..., classes) with rank >= 2");
    }
    if (!(teacher_logits.shape() == student_logits.shape())) {
        throw std::invalid_argument("DistillationLoss::forward: the teacher's logits must have the student's shape");
    }
    if (!std::isfinite(normalizer) || normalizer < 0.0f) {
        throw std::invalid_argument("DistillationLoss::forward: normalizer must be finite and >= 0");
    }
    const int64_t classes = student_logits.shape().dim(student_logits.rank() - 1);
    const int64_t rows = student_logits.numel() / classes;

    // The hard term first: TokenCrossEntropyLoss checks the targets' shape and values.
    float hard = 0.0f;
    std::vector<float> kept_rows;
    if (targets != nullptr) {
        require_device(*targets, backend_->device(), "DistillationLoss::forward");
        hard = hard_.forward(student_logits, *targets, normalizer);
        const std::vector<float> t = targets->to_host_vector();
        for (int64_t r = 0; r < rows; ++r) {
            if (static_cast<int64_t>(std::round(t[static_cast<size_t>(r)])) != options_.ignore_index) {
                kept_rows.push_back(static_cast<float>(r));
            }
        }
    }
    const bool all_rows = targets == nullptr || static_cast<int64_t>(kept_rows.size()) == rows;
    const int64_t k = all_rows ? rows : static_cast<int64_t>(kept_rows.size());
    const float norm = normalizer > 0.0f ? normalizer : static_cast<float>(k);
    const auto n = static_cast<size_t>(k * classes);
    const auto uk = static_cast<size_t>(k), uc = static_cast<size_t>(classes);

    Tensor kept(Shape({0}), backend_);
    Tensor soft_grad(Shape({k, classes}), backend_);
    float soft = 0.0f;
    if (k > 0) {
        const float t = options_.temperature;
        // zs = s / T and zt = t / T, gathered down to the kept rows when some were dropped.
        Tensor zs(Shape({k, classes}), backend_), zt(Shape({k, classes}), backend_);
        if (all_rows) {
            backend_->axpby(1.0f / t, student_logits.data(), 0.0f, student_logits.data(), zs.data(), n);
            backend_->axpby(1.0f / t, teacher_logits.data(), 0.0f, teacher_logits.data(), zt.data(), n);
        } else {
            kept = Tensor(Shape({k, 1}), backend_, kept_rows, backend_->device());
            backend_->gather_rows(student_logits.data(), kept.data(), zs.data(), uk, uc);
            backend_->gather_rows(teacher_logits.data(), kept.data(), zt.data(), uk, uc);
            backend_->axpby(1.0f / t, zs.data(), 0.0f, zs.data(), zs.data(), n);
            backend_->axpby(1.0f / t, zt.data(), 0.0f, zt.data(), zt.data(), n);
        }
        Tensor ps(Shape({k, classes}), backend_), pt(Shape({k, classes}), backend_);
        backend_->softmax_rows(zs.data(), ps.data(), uk, uc);
        backend_->softmax_rows(zt.data(), pt.data(), uk, uc);
        // Per row, KL(p_t ‖ p_s) = Σ p_t (zt − zs) + lse(zs) − lse(zt), and
        //          KL(p_s ‖ p_t) = −Σ p_s (zt − zs) − (lse(zs) − lse(zt)).
        Tensor lse_s(Shape({k}), backend_), lse_t(Shape({k}), backend_);
        backend_->logsumexp_rows(zs.data(), lse_s.data(), uk, uc);
        backend_->logsumexp_rows(zt.data(), lse_t.data(), uk, uc);
        backend_->axpby(1.0f, lse_s.data(), -1.0f, lse_t.data(), lse_s.data(), uk);
        Tensor d(Shape({k, classes}), backend_);
        backend_->axpby(1.0f, zt.data(), -1.0f, zs.data(), d.data(), n);
        const float lse_diff = backend_->sum(lse_s.data(), uk);
        const float scale = t / norm;
        double kl = 0.0;
        if (options_.divergence == DistillationDivergence::Forward) {
            kl = static_cast<double>(backend_->dot(pt.data(), d.data(), n)) + lse_diff;
            // d(T² KL)/ds = T (p_s − p_t).
            backend_->axpby(scale, ps.data(), -scale, pt.data(), soft_grad.data(), n);
        } else {
            kl = -static_cast<double>(backend_->dot(ps.data(), d.data(), n)) - lse_diff;
            // dKL/dzs = p_s ⊙ (log p_s − log p_t − KL): softmax's backward with dy = zs − zt = −d.
            backend_->softmax_rows_backward(ps.data(), d.data(), soft_grad.data(), uk, uc);
            backend_->axpby(-scale, soft_grad.data(), 0.0f, soft_grad.data(), soft_grad.data(), n);
        }
        soft = static_cast<float>(static_cast<double>(t) * t * kl / norm);
    }

    soft_grad_ = std::move(soft_grad);
    kept_ = std::move(kept);
    logits_shape_ = student_logits.shape();
    soft_loss_ = soft;
    hard_loss_ = hard;
    num_rows_ = k;
    all_rows_ = all_rows;
    has_targets_ = targets != nullptr;
    has_forwarded_ = true;
    if (!has_targets_) return soft;
    return options_.alpha * soft + (1.0f - options_.alpha) * hard;
}

Tensor DistillationLoss::backward() const {
    if (!has_forwarded_) throw std::logic_error("DistillationLoss::backward called before forward");
    const auto numel = static_cast<size_t>(logits_shape_.numel());
    const float soft_weight = has_targets_ ? options_.alpha : 1.0f;
    Tensor grad(logits_shape_, backend_);
    if (has_targets_) {
        const Tensor hard = hard_.backward();
        backend_->axpby(1.0f - options_.alpha, hard.data(), 0.0f, hard.data(), grad.data(), numel);
    } else {
        grad.fill(0.0f);
    }
    if (num_rows_ == 0) return grad;
    const auto n = static_cast<size_t>(soft_grad_.numel());
    if (all_rows_) {
        backend_->axpby(soft_weight, soft_grad_.data(), 1.0f, grad.data(), grad.data(), n);
    } else {
        Tensor weighted(soft_grad_.shape(), backend_);
        backend_->axpby(soft_weight, soft_grad_.data(), 0.0f, soft_grad_.data(), weighted.data(), n);
        backend_->scatter_add_rows(weighted.data(), kept_.data(), grad.data(), static_cast<size_t>(num_rows_),
                                   static_cast<size_t>(soft_grad_.shape().dim(1)));
    }
    return grad;
}

}  // namespace pulsatrix
