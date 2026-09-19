#include "exai/adam_optimizer.hpp"

#include <cmath>

namespace exai {

AdamOptimizer::AdamOptimizer(float learning_rate, DeviceBackend* backend, float beta1, float beta2, float eps)
    : learning_rate_(learning_rate), backend_(backend), beta1_(beta1), beta2_(beta2), eps_(eps) {}

void AdamOptimizer::step(Module& module) {
    for (ParamRef p : module.parameters()) {
        auto it = state_.find(p.value);
        if (it == state_.end()) {
            AdamState fresh{Tensor(p.value->shape(), backend_), Tensor(p.value->shape(), backend_), 0};
            fresh.m.fill(0.0f);
            fresh.v.fill(0.0f);
            it = state_.emplace(p.value, std::move(fresh)).first;
        }
        AdamState& s = it->second;
        ++s.t;

        for (int64_t i = 0; i < p.value->numel(); ++i) {
            float g = p.grad->data()[i];
            s.m.data()[i] = beta1_ * s.m.data()[i] + (1.0f - beta1_) * g;
            s.v.data()[i] = beta2_ * s.v.data()[i] + (1.0f - beta2_) * g * g;

            float m_hat = s.m.data()[i] / (1.0f - std::pow(beta1_, static_cast<float>(s.t)));
            float v_hat = s.v.data()[i] / (1.0f - std::pow(beta2_, static_cast<float>(s.t)));

            p.value->data()[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void AdamOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace exai
