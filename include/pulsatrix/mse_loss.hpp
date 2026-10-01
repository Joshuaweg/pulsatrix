/** @file mse_loss.hpp
 *  @brief Mean squared error loss.
 *  @ingroup dl_modules
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
     * @throws std::invalid_argument if prediction and target are on different devices.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1).
     */
    [[nodiscard]] float forward(const Tensor& prediction, const Tensor& target);

    /**
     * @brief Computes the gradient w.r.t. the prediction: (2/n) * (prediction - target).
     * @return Gradient tensor, same shape as the prediction passed to forward().
     * @note Must be called after forward() -- uses the cached prediction/target.
     * @note Device-generic; the gradient is on the prediction's device.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_prediction_;
    Tensor last_target_;
};

}  // namespace pulsatrix
