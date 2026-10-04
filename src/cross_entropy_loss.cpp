#include "pulsatrix/cross_entropy_loss.hpp"

#include <cmath>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

CrossEntropyLoss::CrossEntropyLoss(DeviceBackend* backend) : backend_(backend), softmax_probs_(Shape({0}), backend) {}

float CrossEntropyLoss::forward(const Tensor& logits, int64_t target_class) {
    require_device(logits, backend_->device(), "CrossEntropyLoss::forward");
    PULSATRIX_ASSERT(target_class >= 0 && target_class < logits.numel());

    // Device-generic (GPU-native-kernels Mission 1): softmax and log-sum-exp on the logits'
    // device; only two scalars (log-sum-exp, the target logit) come back to the host.
    const auto n = static_cast<size_t>(logits.numel());
    softmax_probs_ = Tensor(logits.shape(), backend_, logits.device());
    backend_->softmax_rows(logits.data(), softmax_probs_.data(), 1, n);
    target_class_ = target_class;

    Tensor log_sum_exp(Shape({1}), backend_, logits.device());
    backend_->logsumexp_rows(logits.data(), log_sum_exp.data(), 1, n);
    return log_sum_exp.read_element(0) - logits.read_element(target_class);
}

Tensor CrossEntropyLoss::backward() const {
    // grad = softmax - one_hot(target): copy the probabilities, then fix up one element.
    Tensor grad(softmax_probs_);
    grad.write_element(target_class_, softmax_probs_.read_element(target_class_) - 1.0f);
    return grad;
}

}  // namespace pulsatrix
