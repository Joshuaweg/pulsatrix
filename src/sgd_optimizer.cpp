#include "pulsatrix/sgd_optimizer.hpp"

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

void SGDOptimizer::step(Module& module) {
    for (const ParamGroupSet::Assignment& a : groups_.resolve(module, learning_rate_, weight_decay_)) {
        const ParamRef p = a.ref;
        if (!p.value->requires_grad()) {
            continue;  // frozen (FND-2): never moved, whatever its gradient holds
        }
        DeviceBackend* be = p.value->backend();
        const auto n = static_cast<size_t>(p.value->numel());
        if (momentum_ == 0.0f) {
            // Device-generic (GPU-native-kernels Mission 1): value = (1 - lr*wd) * value - lr * grad,
            // which is value - lr * (grad + wd * value) in one pass, computed by the parameter's own
            // backend. With wd = 0 the factor is exactly 1, the original plain-SGD update.
            be->axpby(-a.learning_rate, p.grad->data(), 1.0f - a.learning_rate * a.weight_decay, p.value->data(),
                      p.value->data(), n);
            continue;
        }
        // Momentum (TRN-2), torch.optim.SGD's rule. d = grad + wd * value, in a temporary so the
        // stored gradient is untouched.
        Tensor d(p.value->shape(), be, p.value->device());
        be->axpby(a.weight_decay, p.value->data(), 1.0f, p.grad->data(), d.data(), n);
        auto it = buffers_.find(p.value);
        if (it == buffers_.end()) {
            Tensor buf(p.value->shape(), be, p.value->device());
            buf.fill(0.0f);
            be->axpby(1.0f, d.data(), 0.0f, buf.data(), buf.data(), n);  // buf = d
            it = buffers_.emplace(p.value, std::move(buf)).first;
        } else {
            be->axpby(1.0f, d.data(), momentum_, it->second.data(), it->second.data(), n);  // buf = m*buf + d
        }
        const float* direction = it->second.data();
        if (nesterov_) {
            be->axpby(momentum_, it->second.data(), 1.0f, d.data(), d.data(), n);  // d + m*buf
            direction = d.data();
        }
        be->axpby(-a.learning_rate, direction, 1.0f, p.value->data(), p.value->data(), n);
    }
}

void SGDOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace pulsatrix
