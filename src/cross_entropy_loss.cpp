#include "pulsatrix/cross_entropy_loss.hpp"

#include <cmath>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

CrossEntropyLoss::CrossEntropyLoss(DeviceBackend* backend) : backend_(backend), softmax_probs_(Shape({0}), backend) {}

float CrossEntropyLoss::forward(const Tensor& logits, int64_t target_class) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic,
    // same as MSELoss. See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope
    // decision and mission_host_loop_guards.md.
    PULSATRIX_ASSERT(logits.device() == DeviceType::Cpu);
    PULSATRIX_ASSERT(target_class >= 0 && target_class < logits.numel());

    int64_t n = logits.numel();

    // Numerically stable softmax: subtract the max logit before exponentiating.
    float max_logit = logits.data()[0];
    for (int64_t i = 1; i < n; ++i) {
        max_logit = std::max(max_logit, logits.data()[i]);
    }

    float exp_sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        exp_sum += std::exp(logits.data()[i] - max_logit);
    }

    softmax_probs_ = Tensor(logits.shape(), backend_);
    for (int64_t i = 0; i < n; ++i) {
        softmax_probs_.data()[i] = std::exp(logits.data()[i] - max_logit) / exp_sum;
    }
    target_class_ = target_class;

    float log_sum_exp = max_logit + std::log(exp_sum);
    return log_sum_exp - logits.data()[target_class];
}

Tensor CrossEntropyLoss::backward() const {
    int64_t n = softmax_probs_.numel();
    Tensor grad(softmax_probs_.shape(), backend_);
    for (int64_t i = 0; i < n; ++i) {
        grad.data()[i] = softmax_probs_.data()[i] - ((i == target_class_) ? 1.0f : 0.0f);
    }
    return grad;
}

}  // namespace pulsatrix
