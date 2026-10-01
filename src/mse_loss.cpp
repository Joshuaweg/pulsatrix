#include "pulsatrix/mse_loss.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

MSELoss::MSELoss(DeviceBackend* backend)
    : backend_(backend), last_prediction_(Shape({0}), backend), last_target_(Shape({0}), backend) {}

float MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    // External boundary: both tensors come from the caller. Mixed devices would hand the
    // backend a pointer it cannot address.
    if (prediction.device() != target.device()) {
        throw std::invalid_argument("MSELoss::forward: prediction and target must be on the same device");
    }
    last_prediction_ = prediction;
    last_target_ = target;

    // Device-generic (GPU-native-kernels Mission 1): diff on the device, then one dot product
    // -- the only host transfer is the scalar result.
    const auto n = static_cast<size_t>(prediction.numel());
    Tensor diff(prediction.shape(), backend_, prediction.device());
    backend_->axpby(1.0f, prediction.data(), -1.0f, target.data(), diff.data(), n);
    return backend_->dot(diff.data(), diff.data(), n) / static_cast<float>(prediction.numel());
}

Tensor MSELoss::backward() const {
    // grad = (2/n) * (prediction - target), as two exact steps (difference, then scale) so the
    // rounding matches the original scale * (p - t) host loop rather than scale*p - scale*t.
    const auto n = static_cast<size_t>(last_prediction_.numel());
    const float scale = 2.0f / static_cast<float>(last_prediction_.numel());
    Tensor grad(last_prediction_.shape(), backend_, last_prediction_.device());
    backend_->axpby(1.0f, last_prediction_.data(), -1.0f, last_target_.data(), grad.data(), n);
    backend_->axpby(scale, grad.data(), 0.0f, grad.data(), grad.data(), n);
    return grad;
}

}  // namespace pulsatrix
