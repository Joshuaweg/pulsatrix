#include "pulsatrix/bce_with_logits_loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
BCEWithLogitsLoss::BCEWithLogitsLoss(DeviceBackend* backend)
    : backend_(backend), last_logits_(Shape({0}), backend), last_target_(Shape({0}), backend) {}

float BCEWithLogitsLoss::forward(const Tensor& logits, const Tensor& target) {
    require_device(logits, backend_->device(), "BCEWithLogitsLoss::forward");
    require_device(target, backend_->device(), "BCEWithLogitsLoss::forward");
    if (!(logits.shape() == target.shape())) {
        throw std::invalid_argument("BCEWithLogitsLoss::forward: logits and target must have the same shape");
    }
    if (logits.device() != target.device()) {
        throw std::invalid_argument("BCEWithLogitsLoss::forward: logits and target must be on the same device");
    }
    last_logits_ = logits;
    last_target_ = target;
    has_forwarded_ = true;

    // Device-generic (GPU-native-kernels Mission 1b): per-element terms by the fused kernel,
    // then one reduction; only the scalar crosses to the host.
    const auto n = static_cast<size_t>(logits.numel());
    Tensor terms(logits.shape(), backend_, logits.device());
    backend_->bce_with_logits(logits.data(), target.data(), terms.data(), n);
    return backend_->sum(terms.data(), n) / static_cast<float>(logits.numel());
}

Tensor BCEWithLogitsLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("BCEWithLogitsLoss::backward called before forward");
    }
    const int64_t n = last_logits_.numel();
    Tensor grad(last_logits_.shape(), backend_, last_logits_.device());
    backend_->bce_with_logits_grad(last_logits_.data(), last_target_.data(), grad.data(), static_cast<size_t>(n),
                                   1.0f / static_cast<float>(n));
    return grad;
}

}  // namespace pulsatrix
