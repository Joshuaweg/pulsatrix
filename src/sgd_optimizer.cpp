#include "pulsatrix/sgd_optimizer.hpp"

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

void SGDOptimizer::step(Module& module) {
    for (const ParamGroupSet::Assignment& a : groups_.resolve(module, learning_rate_, weight_decay_)) {
        const ParamRef p = a.ref;
        if (!p.value->requires_grad()) {
            continue;  // frozen (FND-2): never moved, whatever its gradient holds
        }
        // Device-generic (GPU-native-kernels Mission 1): value = (1 - lr*wd) * value - lr * grad,
        // which is value - lr * (grad + wd * value) in one pass, computed by the parameter's own
        // backend. With wd = 0 the factor is exactly 1, the original plain-SGD update.
        p.value->backend()->axpby(-a.learning_rate, p.grad->data(), 1.0f - a.learning_rate * a.weight_decay,
                                  p.value->data(), p.value->data(), static_cast<size_t>(p.value->numel()));
    }
}

void SGDOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace pulsatrix
