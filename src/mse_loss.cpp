#include "pulsatrix/mse_loss.hpp"

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

MSELoss::MSELoss(DeviceBackend* backend)
    : backend_(backend), last_prediction_(Shape({0}), backend), last_target_(Shape({0}), backend) {}

float MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(prediction.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(target.device() == DeviceType::Cpu);

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

}  // namespace pulsatrix
