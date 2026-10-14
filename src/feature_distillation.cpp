// Feature distillation (KD-3): FitNets' hint loss and attention transfer.

#include "pulsatrix/feature_distillation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "portable_random.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

namespace {

// Tensor's copy, viewed as (rows, width).
Tensor Rows(const Tensor& t) {
    Tensor copy = t;
    const int64_t width = t.shape().dim(t.rank() - 1);
    copy.reshape(Shape({t.numel() / width, width}));
    return copy;
}

constexpr double kNormalizeEps = 1e-12;  // F.normalize's

// (N, positions) maps mean_c |x|^p, from host values of (N, C, positions).
std::vector<double> RawMap(const std::vector<float>& x, int64_t n, int64_t c, int64_t s, double p) {
    std::vector<double> q(static_cast<size_t>(n * s), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t ch = 0; ch < c; ++ch) {
            for (int64_t k = 0; k < s; ++k) {
                const double v = x[static_cast<size_t>((i * c + ch) * s + k)];
                q[static_cast<size_t>(i * s + k)] += (p == 2.0 ? v * v : std::pow(std::abs(v), p)) / static_cast<double>(c);
            }
        }
    }
    return q;
}

// Normalizes each row in place and returns the norms as F.normalize clamps them.
std::vector<double> NormalizeRows(std::vector<double>& q, int64_t n, int64_t s) {
    std::vector<double> norms(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        double sq = 0;
        for (int64_t k = 0; k < s; ++k) sq += q[static_cast<size_t>(i * s + k)] * q[static_cast<size_t>(i * s + k)];
        const double norm = std::max(std::sqrt(sq), kNormalizeEps);
        norms[static_cast<size_t>(i)] = norm;
        for (int64_t k = 0; k < s; ++k) q[static_cast<size_t>(i * s + k)] /= norm;
    }
    return norms;
}

}  // namespace

HintLoss::HintLoss(int64_t student_width, int64_t teacher_width, DeviceBackend* backend, uint64_t seed)
    : backend_(backend), mse_(backend) {
    if (student_width < 1 || teacher_width < 1) throw std::invalid_argument("HintLoss: widths must be positive");
    regressor_ = std::make_unique<LinearModule>(student_width, teacher_width, backend);
    // nn.Linear's start: kaiming_uniform(a = √5) on the weight is uniform in ±1/√fan_in, as is the bias.
    const double bound = 1.0 / std::sqrt(static_cast<double>(student_width));
    PortableRng rng{seed};
    std::vector<float> w(static_cast<size_t>(student_width * teacher_width)), b(static_cast<size_t>(teacher_width));
    for (float& v : w) v = static_cast<float>((2 * rng.uniform() - 1) * bound);
    for (float& v : b) v = static_cast<float>((2 * rng.uniform() - 1) * bound);
    regressor_->set_weight(w);
    regressor_->set_bias(b);
}

HintLoss::HintLoss(DeviceBackend* backend) : backend_(backend), mse_(backend) {}

float HintLoss::forward(const Tensor& student_features, const Tensor& teacher_features) {
    require_device(student_features, backend_->device(), "HintLoss::forward");
    require_device(teacher_features, backend_->device(), "HintLoss::forward");
    if (student_features.rank() < 2 || teacher_features.rank() != student_features.rank()) {
        throw std::invalid_argument("HintLoss::forward: features must be (..., width) with rank >= 2 and the same rank");
    }
    for (int64_t d = 0; d + 1 < student_features.rank(); ++d) {
        if (student_features.shape().dim(d) != teacher_features.shape().dim(d)) {
            throw std::invalid_argument("HintLoss::forward: the student's and teacher's leading shapes must match");
        }
    }
    const int64_t sw = student_features.shape().dim(student_features.rank() - 1);
    const int64_t tw = teacher_features.shape().dim(teacher_features.rank() - 1);
    if (regressor_ ? (sw != regressor_->in_features() || tw != regressor_->out_features()) : sw != tw) {
        throw std::invalid_argument("HintLoss::forward: the widths don't match the regressor (or each other, without one)");
    }
    const Tensor teacher = Rows(teacher_features);
    const float loss = regressor_ ? mse_.forward(regressor_->forward(Rows(student_features)), teacher)
                                  : mse_.forward(Rows(student_features), teacher);
    student_shape_ = student_features.shape();
    has_forwarded_ = true;
    return loss;
}

Tensor HintLoss::backward() {
    if (!has_forwarded_) throw std::logic_error("HintLoss::backward called before forward");
    Tensor grad = regressor_ ? regressor_->backward(mse_.backward()) : mse_.backward();
    grad.reshape(student_shape_);
    return grad;
}

AttentionTransferLoss::AttentionTransferLoss(DeviceBackend* backend, double p)
    : backend_(backend), p_(p), student_(Shape({0}), backend) {
    if (!(std::isfinite(p) && p > 0)) throw std::invalid_argument("AttentionTransferLoss: p must be finite and positive");
}

float AttentionTransferLoss::forward(const Tensor& student_features, const Tensor& teacher_features) {
    require_device(student_features, backend_->device(), "AttentionTransferLoss::forward");
    require_device(teacher_features, backend_->device(), "AttentionTransferLoss::forward");
    if (student_features.rank() < 3 || teacher_features.rank() < 3) {
        throw std::invalid_argument("AttentionTransferLoss::forward: features must be (N, C, ...) with rank >= 3");
    }
    const int64_t n = student_features.shape().dim(0), cs = student_features.shape().dim(1);
    const int64_t ct = teacher_features.shape().dim(1);
    if (teacher_features.shape().dim(0) != n || n < 1 || cs < 1 || ct < 1) {
        throw std::invalid_argument("AttentionTransferLoss::forward: the batch sizes must match and be positive");
    }
    const int64_t s = student_features.numel() / (n * cs);
    if (teacher_features.numel() / (n * ct) != s || s < 1) {
        throw std::invalid_argument("AttentionTransferLoss::forward: the student and teacher must have the same number of positions");
    }
    std::vector<double> qs = RawMap(student_features.to_host_vector(), n, cs, s, p_);
    std::vector<double> qt = RawMap(teacher_features.to_host_vector(), n, ct, s, p_);
    norms_ = NormalizeRows(qs, n, s);
    (void)NormalizeRows(qt, n, s);
    double loss = 0;
    for (size_t k = 0; k < qs.size(); ++k) loss += (qs[k] - qt[k]) * (qs[k] - qt[k]);
    q_s_ = std::move(qs);
    q_t_ = std::move(qt);
    student_ = student_features;
    batch_ = n, channels_ = cs, positions_ = s;
    has_forwarded_ = true;
    return static_cast<float>(loss / static_cast<double>(n * s));
}

Tensor AttentionTransferLoss::backward() const {
    if (!has_forwarded_) throw std::logic_error("AttentionTransferLoss::backward called before forward");
    const int64_t n = batch_, c = channels_, s = positions_;
    const std::vector<float> x = student_.to_host_vector();
    std::vector<float> grad(x.size());
    std::vector<double> g(static_cast<size_t>(s));
    for (int64_t i = 0; i < n; ++i) {
        // d loss / d q, then through q = Q / max(‖Q‖, eps).
        const auto row = static_cast<size_t>(i * s);
        double qg = 0;
        for (int64_t k = 0; k < s; ++k) {
            g[static_cast<size_t>(k)] = 2 * (q_s_[row + static_cast<size_t>(k)] - q_t_[row + static_cast<size_t>(k)]) / static_cast<double>(n * s);
            qg += q_s_[row + static_cast<size_t>(k)] * g[static_cast<size_t>(k)];
        }
        const double norm = norms_[static_cast<size_t>(i)];
        const bool clamped = norm <= kNormalizeEps;
        for (int64_t k = 0; k < s; ++k) {
            const double gq = g[static_cast<size_t>(k)];
            g[static_cast<size_t>(k)] = (clamped ? gq : gq - q_s_[row + static_cast<size_t>(k)] * qg) / norm;
        }
        // Then through Q = mean_c |x|^p.
        for (int64_t ch = 0; ch < c; ++ch) {
            for (int64_t k = 0; k < s; ++k) {
                const auto at = static_cast<size_t>((i * c + ch) * s + k);
                const double v = x[at];
                const double dv = p_ == 2.0 ? 2 * v : p_ * std::pow(std::abs(v), p_ - 1) * (v > 0 ? 1.0 : v < 0 ? -1.0 : 0.0);
                grad[at] = static_cast<float>(g[static_cast<size_t>(k)] * dv / static_cast<double>(c));
            }
        }
    }
    return Tensor(student_.shape(), backend_, grad, student_.device());
}

Tensor AttentionMap(const Tensor& features, DeviceBackend* backend, double p) {
    if (features.rank() < 3) throw std::invalid_argument("AttentionMap: features must be (N, C, ...) with rank >= 3");
    if (!(std::isfinite(p) && p > 0)) throw std::invalid_argument("AttentionMap: p must be finite and positive");
    const int64_t n = features.shape().dim(0), c = features.shape().dim(1), s = features.numel() / (n * c);
    std::vector<double> q = RawMap(features.to_host_vector(), n, c, s, p);
    (void)NormalizeRows(q, n, s);
    return Tensor(Shape({n, s}), backend, std::vector<float>(q.begin(), q.end()), features.device());
}

}  // namespace pulsatrix
