/** @file mse_loss.hpp
 *  @brief Mean squared error loss.
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief MSE = mean((prediction - target)^2).
 * @note Not a Module subclass. Losses are the seed point relevance/gradient propagation
 *       starts from, not something a `propagate_relevance` rule is defined for -- LRP
 *       explains a model's prediction, not the loss function used to train it. Deliberate
 *       scope decision (see mission_conv2d_losses.md's Objective 1), not an oversight.
 */
class MSELoss {
public:
    /**
     * @brief Constructs an MSE loss.
     * @param backend Backend to compute through. Not owned; must outlive this loss.
     */
    explicit MSELoss(DeviceBackend* backend);

    /**
     * @brief Computes the loss value and caches prediction/target for backward().
     * @param prediction Model output.
     * @param target Ground truth. Must match prediction's shape.
     * @return The scalar MSE value.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. PULSATRIX_ASSERT(device() == DeviceType::Cpu) on both prediction and target
     *       guards against silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not
     *       remove this guard without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] float forward(const Tensor& prediction, const Tensor& target);

    /**
     * @brief Computes the gradient w.r.t. the prediction: (2/n) * (prediction - target).
     * @return Gradient tensor, same shape as the prediction passed to forward().
     * @note Must be called after forward() -- uses the cached prediction/target.
     * @note Also dereferences Tensor::data() directly, but deliberately not independently
     *       guarded: it only reads state forward() already validated before caching, and
     *       forward()'s own guard is the only way a non-Cpu tensor could ever reach that
     *       cache -- a second guard here would be untestable dead code, not a real safety
     *       net. See mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_prediction_;
    Tensor last_target_;
};

}  // namespace pulsatrix
