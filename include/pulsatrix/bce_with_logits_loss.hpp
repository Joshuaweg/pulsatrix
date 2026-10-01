/** @file bce_with_logits_loss.hpp
 *  @brief Binary cross-entropy on raw logits -- combined sigmoid + BCE, numerically stable.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief loss = mean( max(x,0) - x*y + log(1 + exp(-|x|)) ), over all N*k elements of a
 *        (N, k) logit tensor `x` against a target tensor `y` of the same shape.
 *
 * This is the numerically stable algebraic rewrite of
 * `-y*log(sigmoid(x)) - (1-y)*log(1 - sigmoid(x))`. The naive form computes `sigmoid(x)`
 * first and then takes its log: for large positive `x`, `sigmoid(x)` rounds to exactly 1.0f
 * and `log(1 - 1) = -inf`; for large negative `x` it rounds to 0.0f and `log(0) = -inf`. The
 * combined form above never evaluates a log of a rounded-to-boundary probability and never
 * exponentiates a positive number (`exp(-|x|)` is always in `(0, 1]`), so it is finite for
 * every finite logit. Combining the squashing nonlinearity into the loss for stability is
 * exactly the pattern CrossEntropyLoss already establishes in this codebase (softmax + NLL
 * fused, max-logit subtracted), which is also why no standalone `SigmoidModule` is needed to
 * train a binary classifier here.
 *
 * @note Not a Module subclass, for exactly MSELoss/CrossEntropyLoss's reason: losses are the
 *       seed point relevance/gradient propagation starts *from*, not something a
 *       `propagate_relevance` rule is defined for -- LRP explains a model's prediction, not
 *       the loss function used to train it.
 * @note backward() takes no incoming gradient: like MSELoss, this loss is a graph root.
 * @note The single loss both GAN objectives are built from (Goodfellow et al. 2014):
 *       `loss_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)` and, in Goodfellow's non-saturating
 *       generator form, `loss_G = BCE(D(G(z)), 1)`. No GAN-specific loss class exists or is
 *       needed -- see tests/gan_integration_test.cpp.
 *
 * @warning **Gradient-accumulation hazard in GAN training loops.** This codebase's modules
 *       accumulate parameter gradients across `forward()`/`backward()` call pairs until
 *       something calls `optimizer.zero_grad()`. For the discriminator step that is a
 *       *feature*: `D.backward(grad_real)` then `D.backward(grad_fake)` sum the two terms of
 *       `loss_D` straight into `D`'s gradient buffers, no manual addition needed. For the
 *       generator step it is a *hazard*: computing `loss_G` requires calling `D.backward()`
 *       to route the gradient *through* `D` back to `G`'s output, and that call also
 *       accumulates into `D`'s own parameter gradients -- gradients that belong to `G`'s
 *       objective and must never feed `D`'s next update. So a correct loop calls
 *       `D_optimizer.zero_grad()` *after the generator step too*, not only after the
 *       discriminator step. This is regression-tested, not merely documented:
 *       `GANIntegrationTest.SkippingDiscriminatorZeroGradAfterGeneratorStepCorruptsIt`.
 */
class BCEWithLogitsLoss {
public:
    /**
     * @brief Constructs a BCE-with-logits loss.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     */
    explicit BCEWithLogitsLoss(DeviceBackend* backend);

    /**
     * @brief Computes the loss value and caches logits/target for backward().
     * @param logits Raw, pre-sigmoid model output.
     * @param target Ground-truth labels in {0, 1}. Must match logits' shape.
     * @return The scalar mean binary cross-entropy.
     * @throws std::invalid_argument if logits and target shapes differ -- external boundary,
     *         same classification as MSELoss::forward's shape check.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
     * @note Target values are not range-checked against {0, 1}: the formula is well-defined
     *       (and standard practice, cf. label smoothing / soft targets) for any real y, and
     *       the finite-difference check exercises it as a smooth function of the logits.
     */
    [[nodiscard]] float forward(const Tensor& logits, const Tensor& target);

    /**
     * @brief Gradient w.r.t. the logits: grad[i] = (sigmoid(x[i]) - y[i]) / numel.
     * @return Gradient tensor, same shape as the logits passed to forward().
     * @throws std::logic_error if forward() has never been called -- uses the cached
     *         logits/target.
     * @note The `1/numel` factor is the mean-reduction scaling, the same convention MSELoss
     *       already folds into its own `2/n`.
     * @note sigmoid() is evaluated here in a branch-stable form (`exp(x)/(1+exp(x))` for
     *       x < 0, `1/(1+exp(-x))` otherwise) so the exponent argument is never positive.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 1b);
     *       inputs must share one device.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_logits_;
    Tensor last_target_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
