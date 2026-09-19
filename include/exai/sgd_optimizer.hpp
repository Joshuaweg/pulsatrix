/** @file sgd_optimizer.hpp
 *  @brief Stochastic gradient descent -- operates uniformly across any Module's parameters().
 */
#pragma once

#include "exai/module.hpp"

namespace exai {

/** @brief param -= learning_rate * grad, per parameter, for every parameter a Module exposes. */
class SGDOptimizer {
public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param learning_rate Step size.
     */
    explicit SGDOptimizer(float learning_rate) : learning_rate_(learning_rate) {}

    /**
     * @brief Applies one SGD update to every parameter the module exposes.
     * @param module Module to update. Safe no-op if it has no parameters.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. EXAI_ASSERT(device() == DeviceType::Cpu) on each parameter guards against
     *       silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    void step(Module& module);

    /**
     * @brief Resets every parameter's gradient to zero.
     * @param module Module whose gradients to reset. Safe no-op if it has no parameters.
     */
    void zero_grad(Module& module);

private:
    float learning_rate_;
};

}  // namespace exai
