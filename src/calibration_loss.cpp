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
    PULSATRIX_REQUIRE_HOST(probs);
    PULSATRIX_REQUIRE_HOST(target_class);

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

    std::vector<int64_t> indices(static_cast<size_t>(batch_size));
    for (int64_t n = 0; n < batch_size; ++n) {
        const float encoded = target_class.data()[n];
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
    }

    float loss_sum = 0.0f;
    for (int64_t n = 0; n < batch_size; ++n) {
        const float* row = probs.data() + n * num_classes;
        const int64_t target = indices[static_cast<size_t>(n)];
        float example_score = 0.0f;
        for (int64_t k = 0; k < num_classes; ++k) {
            const float y = (k == target) ? 1.0f : 0.0f;
            const float diff = row[k] - y;
            example_score += diff * diff;
        }
        loss_sum += example_score;
    }

    last_probs_ = probs;
    last_target_indices_ = std::move(indices);
    has_forwarded_ = true;

    return loss_sum / static_cast<float>(batch_size);
}

Tensor CalibrationLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("CalibrationLoss::backward called before forward");
    }

    const int64_t batch_size = last_probs_.shape().dim(0);
    const int64_t num_classes = last_probs_.shape().dim(1);
    const float scale = 2.0f / static_cast<float>(batch_size);

    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(last_probs_);
    Tensor grad(last_probs_.shape(), backend_);
    PULSATRIX_REQUIRE_HOST(grad);
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t target = last_target_indices_[static_cast<size_t>(n)];
        for (int64_t k = 0; k < num_classes; ++k) {
            const float y = (k == target) ? 1.0f : 0.0f;
            grad.data()[n * num_classes + k] = scale * (last_probs_.data()[n * num_classes + k] - y);
        }
    }
    return grad;
}

}  // namespace pulsatrix
