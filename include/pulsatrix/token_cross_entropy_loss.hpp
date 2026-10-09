/** @file token_cross_entropy_loss.hpp
 *  @brief Cross-entropy over a batch of tokens, with padding ignored and a caller-chosen
 *         normalizer for exact gradient accumulation (roadmap TRN-5).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief sum over tokens of -log softmax(logits)[target], divided by a normalizer; tokens whose
 *        target is the ignore index (padding) contribute nothing.
 * @note Logits are (..., classes): every row along the last dimension is one token. Targets have
 *       the leading shape (...), as whole-number floats (the library's index convention).
 * @note Gradient accumulation: averaging each micro-batch's own mean is wrong when micro-batches
 *       hold different numbers of real tokens (the bug Hugging Face Trainer fixed in v4.46). Count
 *       the tokens of the whole accumulation window with CountTargetTokens() and pass that total as
 *       every micro-batch's normalizer; the accumulated gradients then equal one big batch's.
 * @note Device-generic: the per-row softmax and gradient run through DeviceBackend::rl_rows, the
 *       same row kernels as PolicyGradientLoss with a weight of 1 per real token and 0 per ignored
 *       one. The targets are checked on their own device (HIP-6): forward() reads back four
 *       numbers, the loss, the token count and two error counts, and nothing else.
 */
class TokenCrossEntropyLoss {
public:
    /** @brief PyTorch's default ignore_index. */
    static constexpr int64_t kIgnoreIndex = -100;

    explicit TokenCrossEntropyLoss(DeviceBackend* backend, int64_t ignore_index = kIgnoreIndex);

    /**
     * @brief Computes the loss and caches what backward() needs.
     * @param normalizer What the summed loss is divided by. 0 (the default) means this batch's own
     *        number of real tokens, i.e. the mean.
     * @return The loss; 0 when every token is ignored.
     * @throws std::invalid_argument if logits are not rank >= 2 with at least one row and class,
     *         targets don't have logits' leading shape, a target is neither a class index nor the
     *         ignore index, or the normalizer is negative or not finite.
     */
    [[nodiscard]] float forward(const Tensor& logits, const Tensor& targets, float normalizer = 0.0f);

    /** @brief Gradient w.r.t. the logits, same shape. @throws std::logic_error before forward(). */
    [[nodiscard]] Tensor backward() const;

    /** @brief Real (not ignored) tokens in the last forward(). */
    [[nodiscard]] int64_t num_tokens() const { return num_tokens_; }

private:
    /** @brief Rereads the targets on the host to throw the right message for the first bad one. */
    [[noreturn]] void ThrowForBadTarget(const Tensor& targets, int64_t classes) const;

    DeviceBackend* backend_;
    int64_t ignore_index_;
    Tensor probs_;
    Tensor indices_;
    Tensor weights_;
    Shape logits_shape_ = Shape({0});
    float normalizer_ = 1.0f;
    int64_t num_tokens_ = 0;
    bool has_forwarded_ = false;
};

/**
 * @brief Number of entries of `targets` that aren't `ignore_index`: the normalizer for one
 *        accumulation window when summed over its micro-batches' targets.
 */
[[nodiscard]] int64_t CountTargetTokens(const Tensor& targets,
                                        int64_t ignore_index = TokenCrossEntropyLoss::kIgnoreIndex);

}  // namespace pulsatrix
