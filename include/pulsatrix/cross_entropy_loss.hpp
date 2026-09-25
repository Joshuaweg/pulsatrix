/** @file cross_entropy_loss.hpp
 *  @brief Softmax + negative log-likelihood classification loss.
 */
#pragma once

#include <cstdint>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief loss = -log(softmax(logits)[target_class]), combined for numerical stability
 *        (subtract the max logit before exponentiating) rather than computing softmax and
 *        log separately.
 * @note Not a Module subclass, same as MSELoss -- losses are the seed point relevance/
 *       gradient propagation starts from, not something propagate_relevance is defined
 *       for.
 * @note Takes an integer class index, not a one-hot Tensor -- the standard classification-
 *       loss convention (mirrors torch.nn.CrossEntropyLoss's (logits, target) signature),
 *       and avoids inventing a one-hot Tensor construction step every caller would
 *       otherwise need.
 */
class CrossEntropyLoss {
public:
    /**
     * @brief Constructs a cross-entropy loss.
     * @param backend Backend to compute through. Not owned; must outlive this loss.
     */
    explicit CrossEntropyLoss(DeviceBackend* backend);

    /**
     * @brief Computes the loss value and caches softmax probabilities/target for
     *        backward().
     * @param logits Raw (pre-softmax) model output, shape (num_classes,).
     * @param target_class Ground-truth class index, 0-based. Must be in
     *        [0, logits.numel()).
     * @return The scalar cross-entropy loss.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop, same as MSELoss. PULSATRIX_ASSERT(device() == DeviceType::Cpu) guards against
     *       silent UB on a CUDA-backed Tensor. target_class range is also an
     *       PULSATRIX_ASSERT -- an internal invariant for this loss's current (non-Python-
     *       bound) call sites, not yet a Python-reachable external boundary.
     */
    [[nodiscard]] float forward(const Tensor& logits, int64_t target_class);

    /**
     * @brief Computes the gradient w.r.t. the logits: softmax(logits) - one_hot(target_class).
     * @return Gradient tensor, same shape as the logits passed to forward().
     * @note Must be called after forward() -- uses the cached softmax probabilities.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor softmax_probs_;
    int64_t target_class_ = 0;
};

}  // namespace pulsatrix
