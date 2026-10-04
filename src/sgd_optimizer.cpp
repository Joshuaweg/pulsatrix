#include "pulsatrix/sgd_optimizer.hpp"

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

void SGDOptimizer::step(Module& module) {
    for (ParamRef p : module.parameters()) {
        if (!p.value->requires_grad()) {
            continue;  // frozen (FND-2): never moved, whatever its gradient holds
        }
        // Device-generic (GPU-native-kernels Mission 1): value += (-lr) * grad, computed by
        // the parameter's own backend on whichever device it lives.
        p.value->backend()->axpby(-learning_rate_, p.grad->data(), 1.0f, p.value->data(), p.value->data(),
                                  static_cast<size_t>(p.value->numel()));
    }
}

void SGDOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace pulsatrix
