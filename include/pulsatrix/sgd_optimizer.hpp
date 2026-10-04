/** @file sgd_optimizer.hpp
 *  @brief Stochastic gradient descent -- operates uniformly across any Module's parameters().
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/module.hpp"
#include "pulsatrix/param_groups.hpp"

namespace pulsatrix {

/**
 * @brief param -= learning_rate * (grad + weight_decay * param), per parameter, for every
 *        parameter a Module exposes. Learning rate and weight decay can differ per parameter
 *        group (TRN-1). With no weight decay this is plain SGD.
 */
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

    /** @brief The default group's learning rate (parameters no group selects). */
    [[nodiscard]] float learning_rate() const { return learning_rate_; }
    void set_learning_rate(float learning_rate) { learning_rate_ = learning_rate; }

    /** @brief The default group's weight decay. Defaults to 0. @throws std::invalid_argument if
     *         negative or not finite. */
    [[nodiscard]] float weight_decay() const { return weight_decay_; }
    void set_weight_decay(float weight_decay) {
        check_optimizer_setting(weight_decay, "SGDOptimizer weight_decay");
        weight_decay_ = weight_decay;
    }

    /** @brief Replaces the parameter groups; see ParamGroup. @throws std::invalid_argument on a
     *         group without a selector or with an invalid setting. */
    void set_param_groups(std::vector<ParamGroup> groups) { groups_.set(std::move(groups)); }
    [[nodiscard]] std::vector<ParamGroup>& param_groups() { return groups_.groups(); }

private:
    float learning_rate_;
    float weight_decay_ = 0.0f;
    ParamGroupSet groups_;
};

}  // namespace pulsatrix
