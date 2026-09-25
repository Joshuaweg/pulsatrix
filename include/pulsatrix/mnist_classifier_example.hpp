/** @file mnist_classifier_example.hpp
 *  @brief Real MNIST classifier: Conv2DModule -> ReluModule -> FlattenModule ->
 *         LinearModule -> CrossEntropyLoss, mirroring XorNetwork's training-loop pattern
 *         and grad_cam_mnist_demo.cpp's network shape exactly.
 */
#pragma once

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {

/**
 * @brief Conv2D(1,8,5,5) -> ReLU -> Flatten -> Linear(4608,10), trained via
 *        CrossEntropyLoss + Adam, one real MNIST image at a time (this library has no
 *        batch dimension anywhere, same constraint XorNetwork already works under).
 * @note Reuses grad_cam_mnist_demo.cpp's exact network shape (kernel size 5, 8 output
 *       channels) deliberately -- once this network is trained on real data, a Grad-CAM
 *       call against it produces a meaningful heatmap for the first time in this project,
 *       not just pipeline mechanics on random weights.
 * @note Random (not hand-picked) initial weights -- unlike XorNetwork's 12-weight network,
 *       hand-picking ~4600+ asymmetric values isn't practical. A fixed RNG seed keeps this
 *       reproducible; see mission Recon for why zero-init specifically doesn't work
 *       (identical rationale to XorNetwork's own note).
 */
class MnistConvNet {
public:
    /**
     * @brief Constructs the network with randomly initialized weights.
     * @param backend Backend to compute through. Not owned; must outlive this network.
     * @param seed RNG seed for weight initialization.
     */
    explicit MnistConvNet(DeviceBackend* backend, unsigned seed = 42);

    /**
     * @brief Runs the network forward.
     * @param image Shape (1, 28, 28), pixel values normalized to [0,1].
     * @return Shape (10,) raw logits (pre-softmax).
     */
    [[nodiscard]] Tensor forward(const Tensor& image);

    /**
     * @brief forward() plus argmax -- the predicted class index.
     * @param image Shape (1, 28, 28).
     * @return Predicted class, 0-9.
     */
    [[nodiscard]] int64_t predict(const Tensor& image);

    /**
     * @brief Runs one training step: forward, cross-entropy loss, backward through every
     *        layer, one Adam update per layer's parameters, and logs the loss.
     * @param image Shape (1, 28, 28).
     * @param target_class Ground-truth class, 0-9.
     * @param optimizer Optimizer to update this network's parameters with.
     * @param sink Where the loss value is logged (tag "loss").
     * @param step Training step number, passed through to sink.
     * @return The loss value for this example, before the update.
     */
    float train_step(const Tensor& image, int64_t target_class, AdamOptimizer& optimizer, MetricsSink& sink,
                      int step);

    /** @brief Test/inspection accessor. */
    [[nodiscard]] const Tensor& classifier_weight() const { return classifier_.weight(); }

private:
    DeviceBackend* backend_;
    Conv2DModule conv_;
    ReluModule relu_;
    FlattenModule flatten_;
    LinearModule classifier_;
    CrossEntropyLoss loss_;
};

}  // namespace pulsatrix
