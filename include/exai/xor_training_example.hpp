/** @file xor_training_example.hpp
 *  @brief Phase 1's training-loop proof: Linear(2,4)->ReLU->Linear(4,1) learning XOR.
 */
#pragma once

#include "exai/adam_optimizer.hpp"
#include "exai/linear_module.hpp"
#include "exai/metrics_sink.hpp"
#include "exai/mse_loss.hpp"
#include "exai/relu_module.hpp"

namespace exai {

/**
 * @brief A tiny MLP (Linear(2,4) -> ReLU -> Linear(4,1)) trained on XOR -- the canonical
 *        not-linearly-separable case, exactly representable by a small MLP, giving an
 *        unambiguous convergence target for Phase 1's "prove the training loop works" goal.
 * @note Modules are chained directly (no ComputationGraph/Autograd involvement) -- see
 *       mission_training_loop.md's design rationale. Batch size 1 (operator decision) --
 *       trains on one example at a time, no batching added to any module.
 * @note Deterministic, hand-picked non-zero initial weights, NOT a general Xavier/He
 *       scheme. All-zero initialization was tried first during this mission's design and
 *       found to break symmetry completely: with W=0 everywhere, every layer's forward
 *       output is 0 regardless of input, and every gradient that depends on multiplying by
 *       a zero weight or a zero (pre-activation) ReLU input is itself zero -- linear1's
 *       weights and linear2's weights never receive a gradient at all, only linear2's bias
 *       does. Non-zero initial weights are required to break this symmetry; a small,
 *       fixed, sign-varied initialization is sufficient for a network this tiny. A general
 *       initialization scheme (Xavier/He) remains deferred until a real dataset needs one.
 */
class XorNetwork {
public:
    /**
     * @brief Constructs the network with fixed, non-zero initial weights.
     * @param backend Backend to compute through. Not owned; must outlive this network.
     */
    explicit XorNetwork(DeviceBackend* backend);

    /**
     * @brief Runs the network forward.
     * @param input Shape (2,).
     * @return Shape (1,).
     */
    [[nodiscard]] Tensor forward(const Tensor& input);

    /**
     * @brief Runs one training step: forward, loss, backward through every layer,
     *        one Adam update per layer's parameters, and logs the loss.
     * @param input Shape (2,).
     * @param target Shape (1,).
     * @param optimizer Optimizer to update this network's parameters with.
     * @param sink Where the loss value is logged (tag "loss").
     * @param step Training step number, passed through to sink.
     * @return The loss value for this example, before the update.
     */
    float train_step(const Tensor& input, const Tensor& target, AdamOptimizer& optimizer, MetricsSink& sink,
                      int step);

    /** @brief Test/inspection accessor. */
    [[nodiscard]] const Tensor& linear1_weight() const { return linear1_.weight(); }

private:
    DeviceBackend* backend_;
    LinearModule linear1_;
    ReluModule relu_;
    LinearModule linear2_;
    MSELoss loss_;
};

}  // namespace exai
