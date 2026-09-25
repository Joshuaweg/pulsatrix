/** @file adam_optimizer.hpp
 *  @brief Adam optimizer -- operates uniformly across any Module's parameters().
 */
#pragma once

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
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. PULSATRIX_ASSERT(device() == DeviceType::Cpu) on each parameter guards against
     *       silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    void step(Module& module);

    /**
     * @brief Resets every parameter's gradient to zero. Does not reset Adam's moment state.
     * @param module Module whose gradients to reset. Safe no-op if it has no parameters.
     */
    void zero_grad(Module& module);

private:
    struct AdamState {
        Tensor m;
        Tensor v;
        int64_t t = 0;
    };

    float learning_rate_;
    DeviceBackend* backend_;
    float beta1_;
    float beta2_;
    float eps_;
    std::unordered_map<const Tensor*, AdamState> state_;
};

}  // namespace pulsatrix
