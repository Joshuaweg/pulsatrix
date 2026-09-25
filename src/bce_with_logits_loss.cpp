#include "pulsatrix/bce_with_logits_loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief Overflow-free logistic sigmoid: the exponent argument is never positive.
 * @note The naive 1/(1+exp(-x)) overflows exp() for large negative x; the mirrored
 *       exp(x)/(1+exp(x)) branch keeps -|x| in the exponent either way.
 */
float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

}  // namespace

BCEWithLogitsLoss::BCEWithLogitsLoss(DeviceBackend* backend)
    : backend_(backend), last_logits_(Shape({0}), backend), last_target_(Shape({0}), backend) {}

float BCEWithLogitsLoss::forward(const Tensor& logits, const Tensor& target) {
    // Dereferences Tensor::data() directly in a raw host loop (exp()/log() have no backend
    // primitive) -- not yet backend-generic. See
    // campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(logits.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(target.device() == DeviceType::Cpu);

    if (!(logits.shape() == target.shape())) {
        throw std::invalid_argument("BCEWithLogitsLoss::forward: logits and target must have the same shape");
    }

    last_logits_ = logits;
    last_target_ = target;
    has_forwarded_ = true;

    // max(x,0) - x*y + log(1 + exp(-|x|)): the numerically stable rewrite of
    // -y*log(sigmoid(x)) - (1-y)*log(1 - sigmoid(x)). log1p keeps the small-|x| end accurate
    // too, where exp(-|x|) is near 1 and 1 + it loses no significance, but 1 + tiny would.
    float total = 0.0f;
    for (int64_t i = 0; i < logits.numel(); ++i) {
        const float x = logits.data()[i];
        const float y = target.data()[i];
        total += std::max(x, 0.0f) - x * y + std::log1p(std::exp(-std::fabs(x)));
    }
    return total / static_cast<float>(logits.numel());
}

Tensor BCEWithLogitsLoss::backward() const {
    if (!has_forwarded_) {
        throw std::logic_error("BCEWithLogitsLoss::backward called before forward");
    }

    const int64_t n = last_logits_.numel();
    const float scale = 1.0f / static_cast<float>(n);
    Tensor grad(last_logits_.shape(), backend_);
    for (int64_t i = 0; i < n; ++i) {
        grad.data()[i] = (stable_sigmoid(last_logits_.data()[i]) - last_target_.data()[i]) * scale;
    }
    return grad;
}

}  // namespace pulsatrix
