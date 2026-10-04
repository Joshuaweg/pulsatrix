/** @file adam_optimizer.hpp
 *  @brief Adam optimizer -- operates uniformly across any Module's parameters().
 *  @ingroup dl_modules
 */
#pragma once

#include <stdexcept>
#include <unordered_map>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Adam (Kingma & Ba, 2015): per-parameter moving averages of gradient (m) and
 *        squared gradient (v), with bias correction.
 * @note Per-parameter state is keyed by the parameter Tensor's pointer identity
 *       (`ParamRef::value`), which is stable for as long as the owning Module exists --
 *       LinearModule/Conv2DModule's weight_/bias_ members never move or get reallocated.
 */
class AdamOptimizer {
public:
    /**
     * @brief Constructs an Adam optimizer.
     * @param learning_rate Step size.
     * @param backend Backend used to allocate per-parameter moment-tracking tensors.
     * @param beta1 First moment decay rate.
     * @param beta2 Second moment decay rate.
     * @param eps Denominator stabilizer.
     */
    explicit AdamOptimizer(float learning_rate, DeviceBackend* backend, float beta1 = 0.9f, float beta2 = 0.999f,
                            float eps = 1e-8f);

    /**
     * @brief Applies one Adam update to every parameter the module exposes.
     * @param module Module to update. Safe no-op if it has no parameters.
     * @note Device-generic: one fused DeviceBackend::adam_step per parameter. Moments are
     *       allocated through this optimizer's backend, so every parameter must live on that
     *       backend's device.
     * @throws std::invalid_argument if a parameter's device differs from the backend's.
     */
    void step(Module& module);

    /**
     * @brief Resets every parameter's gradient to zero. Does not reset Adam's moment state.
     * @param module Module whose gradients to reset. Safe no-op if it has no parameters.
     */
    void zero_grad(Module& module);

    /** @brief Current step size. */
    [[nodiscard]] float learning_rate() const { return learning_rate_; }

    /**
     * @brief Overwrites the step size used by every subsequent step() call -- necessary
     *        infrastructure for any mid-training hyperparameter schedule (e.g. Population
     *        Based Training's own explore step), found necessary by
     *        campaign_exai_dl_library_evolutionary_deep_learning's Phase 4 Mission 0, logged
     *        as a small addition beyond this class's original fixed-at-construction scope.
     *        Does not reset Adam's own moment state (m/v), matching zero_grad()'s own
     *        precedent that state and gradient are independent concerns.
     */
    void set_learning_rate(float learning_rate) { learning_rate_ = learning_rate; }

    /** @brief One parameter's Adam state: first and second moments, and its step count. */
    struct AdamState {
        Tensor m;
        Tensor v;
        int64_t t = 0;
    };

    /**
     * @brief The state for `parameter` (a value tensor from Module::parameters()), or nullptr if
     *        step() hasn't updated it yet. For checkpointing (IO-2).
     */
    [[nodiscard]] const AdamState* state(const Tensor* parameter) const {
        auto it = state_.find(parameter);
        return it == state_.end() ? nullptr : &it->second;
    }

    /**
     * @brief Replaces the state for `parameter`, e.g. when resuming from a checkpoint.
     * @throws std::invalid_argument if the moments' shapes differ from the parameter's, they are
     *         not on the parameter's device, or t is negative.
     */
    void set_state(const Tensor* parameter, AdamState state) {
        if (state.m.shape() != parameter->shape() || state.v.shape() != parameter->shape()) {
            throw std::invalid_argument("AdamOptimizer::set_state: moments must have the parameter's shape");
        }
        if (state.m.device() != parameter->device() || state.v.device() != parameter->device()) {
            throw std::invalid_argument("AdamOptimizer::set_state: moments must be on the parameter's device");
        }
        if (state.t < 0) {
            throw std::invalid_argument("AdamOptimizer::set_state: step count must be non-negative");
        }
        state_.insert_or_assign(parameter, std::move(state));
    }

private:

    float learning_rate_;
    DeviceBackend* backend_;
    float beta1_;
    float beta2_;
    float eps_;
    std::unordered_map<const Tensor*, AdamState> state_;
};

}  // namespace pulsatrix
