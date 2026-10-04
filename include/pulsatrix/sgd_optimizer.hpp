/** @file sgd_optimizer.hpp
 *  @brief Stochastic gradient descent -- operates uniformly across any Module's parameters().
 *  @ingroup dl_modules
 */
#pragma once

#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "pulsatrix/module.hpp"
#include "pulsatrix/param_groups.hpp"

namespace pulsatrix {

/**
 * @brief param -= learning_rate * (grad + weight_decay * param), per parameter, for every
 *        parameter a Module exposes. Learning rate and weight decay can differ per parameter
 *        group (TRN-1). With no weight decay this is plain SGD.
 * @note Optional momentum and Nesterov momentum follow torch.optim.SGD (dampening 0): with
 *       d = grad + weight_decay * param, the buffer is buf = d on a parameter's first step and
 *       buf = momentum * buf + d after; the step is -lr * buf, or -lr * (d + momentum * buf) with
 *       Nesterov (TRN-2).
 */
class SGDOptimizer {
public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param learning_rate Step size.
     */
    /**
     * @param momentum In [0, 1). 0 (the default) is plain SGD, with no per-parameter state.
     * @param nesterov Use Nesterov momentum. Needs momentum > 0.
     * @throws std::invalid_argument if momentum is outside [0, 1), or nesterov without momentum.
     */
    explicit SGDOptimizer(float learning_rate, float momentum = 0.0f, bool nesterov = false)
        : learning_rate_(learning_rate), momentum_(momentum), nesterov_(nesterov) {
        if (!(momentum >= 0.0f && momentum < 1.0f)) {
            throw std::invalid_argument("SGDOptimizer: momentum must be in [0, 1)");
        }
        if (nesterov && momentum == 0.0f) {
            throw std::invalid_argument("SGDOptimizer: Nesterov momentum needs momentum > 0");
        }
    }

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

    [[nodiscard]] float momentum() const { return momentum_; }
    [[nodiscard]] bool nesterov() const { return nesterov_; }

    /** @brief The momentum buffer for `parameter`, or nullptr before its first step (or with no
     *         momentum). */
    [[nodiscard]] const Tensor* momentum_buffer(const Tensor* parameter) const {
        auto it = buffers_.find(parameter);
        return it == buffers_.end() ? nullptr : &it->second;
    }

private:
    float learning_rate_;
    float momentum_;
    bool nesterov_;
    std::unordered_map<const Tensor*, Tensor> buffers_;
    float weight_decay_ = 0.0f;
    ParamGroupSet groups_;
};

}  // namespace pulsatrix
