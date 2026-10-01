#include "pulsatrix/adam_optimizer.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

AdamOptimizer::AdamOptimizer(float learning_rate, DeviceBackend* backend, float beta1, float beta2, float eps)
    : learning_rate_(learning_rate), backend_(backend), beta1_(beta1), beta2_(beta2), eps_(eps) {}

void AdamOptimizer::step(Module& module) {
    for (ParamRef p : module.parameters()) {
        // Moments are allocated through the optimizer's own backend_ -- whose lifetime the
        // caller already guarantees -- so they must live where the parameter lives. A
        // mismatch would hand adam_step pointers from two devices (GPU-native-kernels
        // Mission 1).
        if (p.value->device() != backend_->device()) {
            throw std::invalid_argument(
                "AdamOptimizer::step: parameter is on a different device than the optimizer's backend");
        }
        auto it = state_.find(p.value);
        if (it == state_.end()) {
            AdamState fresh{Tensor(p.value->shape(), backend_), Tensor(p.value->shape(), backend_), 0};
            fresh.m.fill(0.0f);
            fresh.v.fill(0.0f);
            it = state_.emplace(p.value, std::move(fresh)).first;
        }
        AdamState& s = it->second;
        ++s.t;

        // Bias corrections once per parameter per step on the host -- the same float
        // expressions the original per-element host loop evaluated.
        const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(s.t));
        const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(s.t));
        backend_->adam_step(p.value->data(), p.grad->data(), s.m.data(), s.v.data(),
                            static_cast<size_t>(p.value->numel()), learning_rate_, beta1_, beta2_, eps_,
                            bias_correction1, bias_correction2);
    }
}

void AdamOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace pulsatrix
