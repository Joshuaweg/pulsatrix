#include "exai/sgd_optimizer.hpp"

namespace exai {

void SGDOptimizer::step(Module& module) {
    for (ParamRef p : module.parameters()) {
        for (int64_t i = 0; i < p.value->numel(); ++i) {
            p.value->data()[i] -= learning_rate_ * p.grad->data()[i];
        }
    }
}

void SGDOptimizer::zero_grad(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

}  // namespace exai
