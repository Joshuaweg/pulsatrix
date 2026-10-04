#include "pulsatrix/calibration_loss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {
namespace {

// How far a discrete class index's float encoding may sit from a whole number before it is
// rejected -- byte-for-byte PolicyGradientLoss::forward()'s kActionIntegerTolerance.
constexpr float kClassIntegerTolerance = 1e-4f;

}  // namespace

CalibrationLoss::CalibrationLoss(DeviceBackend* backend)
    : backend_(backend), last_probs_(Shape({0}), backend) {}

float CalibrationLoss::forward(const Tensor& probs, const Tensor& target_class) {
    require_device(probs, backend_->device(), "CalibrationLoss::forward");
    require_device(target_class, backend_->device(), "CalibrationLoss::forward");
    if (probs.rank() != 2) {
        throw std::invalid_argument("CalibrationLoss::forward: probs must have shape (N, num_classes)");
    }
    const int64_t batch_size = probs.shape().dim(0);
    const int64_t num_classes = probs.shape().dim(1);
    if (batch_size < 1 || num_classes < 1) {
        throw std::invalid_argument("CalibrationLoss::forward: probs must have N >= 1 and num_classes >= 1");
    }
    if (target_class.rank() != 2 || target_class.shape().dim(0) != batch_size || target_class.shape().dim(1) != 1) {
        throw std::invalid_argument("CalibrationLoss::forward: target_class must have shape (N, 1)");
    }

    // Index validation needs the values on the host (it can throw per element); N floats is
    // a small transfer next to the (N, num_classes) work below.
    std::vector<float> encoded_targets(static_cast<size_t>(batch_size));
    target_class.backend()->copy(encoded_targets.data(), target_class.data(), encoded_targets.size() * sizeof(float),
                                 target_class.device() == DeviceType::Cpu ? CopyDirection::HostToHost
                                                                          : CopyDirection::DeviceToHost);
    std::vector<int64_t> indices(static_cast<size_t>(batch_size));
    std::vector<float> one_hot(static_cast<size_t>(batch_size * num_classes), 0.0f);
    for (int64_t n = 0; n < batch_size; ++n) {
        const float encoded = encoded_targets[static_cast<size_t>(n)];
        const float rounded = std::round(encoded);
        if (std::abs(encoded - rounded) > kClassIntegerTolerance) {
            throw std::invalid_argument("CalibrationLoss::forward: target_class must encode whole-number class "
                                        "indices");
        }
        const int64_t index = static_cast<int64_t>(rounded);
        if (index < 0 || index >= num_classes) {
            throw std::invalid_argument("CalibrationLoss::forward: class index out of range [0, num_classes)");
        }
        indices[static_cast<size_t>(n)] = index;
        one_hot[static_cast<size_t>(n * num_classes + index)] = 1.0f;
    }

    // Device-generic (GPU-native-kernels Mission 1b). Brier score: per example the sum over
    // classes of (p - y)^2, then the sum over examples -- the same two-level order as the
    // original host loop: gemm_ex against a ones column gives the per-row sums (accumulated
    // from 0 in class order), column_sums over that (N, 1) column gives the total.
    const DeviceType device = probs.device();
    const auto rows = static_cast<size_t>(batch_size);
    const auto cols = static_cast<size_t>(num_classes);
    last_one_hot_ = Tensor(probs.shape(), backend_, one_hot, device);
    Tensor squared(probs.shape(), backend_, device);
    backend_->axpby(1.0f, probs.data(), -1.0f, last_one_hot_.data(), squared.data(), rows * cols);
    backend_->mul(squared.data(), squared.data(), squared.data(), rows * cols);
    Tensor ones(Shape({num_classes, 1}), backend_, device);
    ones.fill(1.0f);
    Tensor example_scores(Shape({batch_size, 1}), backend_, device);
    backend_->gemm_ex(squared.data(), false, ones.data(), false, example_scores.data(), rows, cols, 1, 0.0f);
    Tensor loss_sum(Shape({1}), backend_, device);
    backend_->column_sums(example_scores.data(), loss_sum.data(), rows, 1, 0.0f);

    last_probs_ = probs;
    last_target_indices_ = std::move(indices);
    has_forwarded_ = true;
    return loss_sum.read_element(0) / static_cast<float>(batch_size);
}

Tensor CalibrationLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("CalibrationLoss::backward called before forward");
    }
    const int64_t batch_size = last_probs_.shape().dim(0);
    const float scale = 2.0f / static_cast<float>(batch_size);
    // grad = scale * (p - one_hot): difference, then scale, as the original loop rounded.
    const auto n = static_cast<size_t>(last_probs_.numel());
    Tensor grad(last_probs_.shape(), backend_, last_probs_.device());
    backend_->axpby(1.0f, last_probs_.data(), -1.0f, last_one_hot_.data(), grad.data(), n);
    backend_->axpby(scale, grad.data(), 0.0f, nullptr, grad.data(), n);
    return grad;
}

}  // namespace pulsatrix
