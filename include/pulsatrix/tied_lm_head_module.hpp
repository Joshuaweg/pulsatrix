/** @file tied_lm_head_module.hpp
 *  @brief Language-model head that reuses an EmbeddingModule's table as its weight (LLM-2).
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>
#include <vector>

#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief `logits = x @ E^T`, where E is a token embedding table `(vocab, d_model)` that the
 *        head shares rather than copies -- Hugging Face's `tie_word_embeddings` (SmolLM2,
 *        Qwen3, Gemma 3). Shape `(..., d_model) -> (..., vocab)`.
 *
 * @note **One parameter, two uses.** The head holds no weight of its own: it reads the
 *       embedding's table and accumulates its gradient into the embedding's gradient buffer,
 *       so the table's gradient is the sum of both uses, as in PyTorch. named_parameters() is
 *       therefore empty -- the optimizer, freezing and checkpoints see the table once, under
 *       the embedding's name. Freezing the embedding freezes the head too.
 * @note **No bias**, as in every tied model. An untied head is a bias-free LinearModule.
 * @note **LRP** is the affine layer's rule with the transposed table as its weight: every
 *       LRPRule LinearModule supports, with the same Zennit semantics. With no bias, the
 *       epsilon rule's two bias conventions agree, and it conserves relevance up to the
 *       stabilizer.
 * @note The embedding must outlive this module, and stay on the same backend.
 */
class TiedLMHeadModule : public Module {
public:
    /**
     * @param embedding The table to share. Not owned; must outlive this module.
     * @param backend The embedding's backend. Not owned; must outlive this module.
     */
    TiedLMHeadModule(EmbeddingModule& embedding, DeviceBackend* backend);

    /**
     * @brief Input gradient `grad @ E`; adds `grad^T @ x` into the embedding's weight gradient
     *        unless the embedding is frozen.
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if grad_output's shape differs from the cached output shape.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /**
     * @brief LRP through `x @ E^T`, by config.rule (Epsilon, Gamma, AlphaBeta or ZBox).
     * @throws std::logic_error if called before any forward().
     * @throws std::invalid_argument if relevance_out's shape differs from the cached output
     *         shape, or config's rule parameters are invalid.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }
    [[nodiscard]] OpType op_type() const override { return OpType::Linear; }

    /** @brief Empty: the shared table is the embedding's parameter, not the head's. */
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override { return {}; }

    [[nodiscard]] int64_t vocab_size() const { return weight_->shape().dim(0); }
    [[nodiscard]] int64_t d_model() const { return weight_->shape().dim(1); }

    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    void release_activations() override;

protected:
    /**
     * @param input `(..., d_model)`, rank >= 2.
     * @return `(..., vocab)`.
     * @throws std::invalid_argument if input's rank is below 2 or its last dimension isn't d_model.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    // The embedding's own weight and gradient members. Their addresses never change (set_weight
    // assigns into them), so the head always sees the current table.
    Tensor* weight_;
    Tensor* weight_grad_;
    Tensor zero_bias_;    ///< (vocab,) zeros, for the LRP rules' bias argument
    Tensor last_input_;   ///< (M, d_model), the input flattened over its leading dimensions
    Tensor last_logits_;  ///< (M, vocab)
    Shape last_input_shape_ = Shape({0});
    Shape last_output_shape_ = Shape({0});
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
