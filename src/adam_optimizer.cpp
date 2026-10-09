#include "pulsatrix/adam_optimizer.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

AdamOptimizer::AdamOptimizer(float learning_rate, DeviceBackend* backend, float beta1, float beta2, float eps)
    : learning_rate_(learning_rate), backend_(backend), beta1_(beta1), beta2_(beta2), eps_(eps) {}

void AdamOptimizer::step(Module& module) {
    std::vector<AdamTensorStep> steps;
    for (const ParamGroupSet::Assignment& a : groups_.resolve(module, learning_rate_, weight_decay_)) {
        const ParamRef p = a.ref;
        if (!p.value->requires_grad()) {
            continue;  // frozen (FND-2): never moved, and no moment state allocated or advanced
        }
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
        AdamTensorStep t;
        t.param = p.value->data();
        t.grad = p.grad->data();
        t.m = s.m.data();
        t.v = s.v.data();
        t.n = static_cast<size_t>(p.value->numel());
        t.lr = a.learning_rate;
        t.bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(s.t));
        t.bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(s.t));
        if (decoupled_weight_decay_) {
            // AdamW (TRN-2): shrink the weights first, w <- (1 - lr*wd) * w; the Adam step then
            // uses the raw gradient. Same order as torch.optim.AdamW.
            if (a.weight_decay != 0.0f) t.decay = 1.0f - a.learning_rate * a.weight_decay;
        } else if (a.weight_decay != 0.0f) {
            // Weight decay as L2 (TRN-1, PyTorch's Adam): the step sees grad + wd * value, and the
            // stored gradient is left as backward() wrote it.
            t.l2 = a.weight_decay;
        }
        steps.push_back(t);
    }
    // HIP-6: every parameter in one call, so one launch on a GPU.
    backend_->adam_step_multi(steps.data(), steps.size(), beta1_, beta2_, eps_);
}

void AdamOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace pulsatrix
