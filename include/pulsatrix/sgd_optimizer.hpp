/** @file sgd_optimizer.hpp
 *  @brief Stochastic gradient descent -- operates uniformly across any Module's parameters().
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

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
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1).
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

}  // namespace pulsatrix
