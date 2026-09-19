#include "exai/sgd_optimizer.hpp"

#include "exai/assert.hpp"

namespace exai {

void SGDOptimizer::step(Module& module) {
    for (ParamRef p : module.parameters()) {
        // Dereferences Tensor::data() directly in a raw host loop -- not yet
        // backend-generic. See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope
        // decision and mission_host_loop_guards.md.
        EXAI_ASSERT(p.value->device() == DeviceType::Cpu);
        EXAI_ASSERT(p.grad->device() == DeviceType::Cpu);

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
