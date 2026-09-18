#include "exai/mse_loss.hpp"

namespace exai {

MSELoss::MSELoss(DeviceBackend* backend)
    : backend_(backend), last_prediction_(Shape({0}), backend), last_target_(Shape({0}), backend) {}

float MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    last_prediction_ = prediction;
    last_target_ = target;

    float sum_squared = 0.0f;
    for (int64_t i = 0; i < prediction.numel(); ++i) {
        float diff = prediction.data()[i] - target.data()[i];
        sum_squared += diff * diff;
    }
    return sum_squared / static_cast<float>(prediction.numel());
}

Tensor MSELoss::backward() const {
    int64_t n = last_prediction_.numel();
    Tensor grad(last_prediction_.shape(), backend_);
    float scale = 2.0f / static_cast<float>(n);
    for (int64_t i = 0; i < n; ++i) {
        grad.data()[i] = scale * (last_prediction_.data()[i] - last_target_.data()[i]);
    }
    return grad;
}

}  // namespace exai
