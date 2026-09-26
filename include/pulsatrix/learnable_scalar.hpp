/** @file learnable_scalar.hpp
 *  @brief A single trainable float, updated by plain SGD -- not a Tensor/Module parameter.
 */
#pragma once

namespace pulsatrix {

/**
 * @brief A bare learnable scalar (e.g. a GFlowNet loss's `log Z`), outside the Module/LRP
 *        hierarchy entirely.
 * @note Deliberately not a Module: `AdamOptimizer`/`SGDOptimizer` only operate over
 *       `Module::parameters()`, and forcing a training-only scalar into the `Module`
 *       contract would also force a `propagate_relevance` definition for something with no
 *       forward pass to explain -- the same reasoning that keeps `MSELoss` outside the
 *       `Module` hierarchy (see `mission_shared_gflownet_machinery.md`'s Recon). The
 *       GFlowNet losses this class supports compute `d(loss)/d(log Z)` analytically
 *       themselves; no autograd machinery is needed here at all.
 * @note Not a Tensor either -- a Tensor's whole reason to exist (device placement, batched
 *       storage, DeviceBackend dispatch) is irrelevant to one scalar that never participates
 *       in a tensor op.
 */
class LearnableScalar {
public:
    /** @brief Constructs a scalar with the given initial value and zero gradient. */
    explicit LearnableScalar(float initial_value = 0.0f);

    /** @brief Current value. */
    [[nodiscard]] float value() const { return value_; }

    /** @brief Accumulated gradient since the last zero_grad(). */
    [[nodiscard]] float grad() const { return grad_; }

    /** @brief Adds to the accumulated gradient -- mirrors Tensor::accumulate()'s
     *         across-multiple-contributions convention. */
    void accumulate_grad(float grad);

    /** @brief Resets the accumulated gradient to zero. Does not change value(). */
    void zero_grad();

    /**
     * @brief Plain SGD update: `value_ -= learning_rate * grad_`.
     * @param learning_rate Step size. Unvalidated -- same convention as
     *        `SGDOptimizer`'s own constructor, which does not validate its learning rate
     *        either.
     * @note Does not reset grad() -- mirrors AdamOptimizer::step()'s own convention
     *       (zero_grad() is a separate, explicit call).
     */
    void step(float learning_rate);

private:
    float value_;
    float grad_ = 0.0f;
};

}  // namespace pulsatrix
