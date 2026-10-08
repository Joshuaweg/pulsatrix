/** @file encoder_block.hpp
 *  @brief A configurable transformer encoder layer (PLM-1): LayerNorm or RMSNorm, before or after
 *         each sublayer, and a plain or gated MLP. ESM-2's layer, BERT's layer and the
 *         Llama-style pre-norm layer are all settings of it.
 *  @ingroup dl_modules
 */
#pragma once

#include <memory>
#include <optional>

#include "pulsatrix/module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/swiglu_module.hpp"

namespace pulsatrix {

/** @brief The layer's normalization. */
enum class NormType { LayerNorm, RMSNorm };

/** @brief Where the norms sit. */
enum class NormPosition {
    /** @brief `y = x + f(norm(x))` (ESM, GPT-2, Llama). */
    Pre,
    /** @brief `y = norm(x + f(x))` (BERT and the original Transformer). */
    Post,
};

/** @brief The MLP's form. */
enum class MlpType {
    /** @brief `Linear -> activation -> Linear` (BERT, ESM): FeedForwardModule. */
    Plain,
    /** @brief `down(act(gate(x)) * up(x))` (Llama's SwiGLU, Gemma's GeGLU): SwiGLUModule. */
    Gated,
};

/** @brief Everything about an EncoderBlock beyond its attention. The defaults are ESM-2's. */
struct EncoderBlockOptions {
    NormType norm = NormType::LayerNorm;
    /** @brief The norms' stabilizer (Hugging Face `layer_norm_eps`; ESM-2 uses 1e-5). */
    float norm_eps = 1e-5f;
    NormPosition norm_position = NormPosition::Pre;
    MlpType mlp = MlpType::Plain;
    /** @brief The plain MLP's activation (Gelu for BERT and ESM). Ignored for a gated MLP. */
    ElementwiseOp activation = ElementwiseOp::Gelu;
    /** @brief The gated MLP's gate activation. Ignored for a plain MLP. */
    GatedActivation gate_activation = GatedActivation::Silu;
    /** @brief Biases on the MLP's projections. */
    bool mlp_bias = true;
};

/**
 * @brief One encoder layer. Shape `(N, L, d_model) -> (N, L, d_model)`.
 *
 * - Pre-norm: `y1 = x + attn(norm1(x))`, `y2 = y1 + mlp(norm2(y1))`.
 * - Post-norm: `y1 = norm1(x + attn(x))`, `y2 = norm2(y1 + mlp(y1))`.
 *
 * Attention is configured in full by AttentionConfig. An encoder leaves `causal` off; ESM-2 uses
 * rotate-half RoPE with biases.
 *
 * Parameters are named `norm1.*`, `mha.*`, `norm2.*` and `mlp.*`. A plain MLP's are `mlp.fc1.*`
 * and `mlp.fc2.*`; a gated MLP's are `mlp.gate_proj.*`, `mlp.up_proj.*` and `mlp.down_proj.*`.
 *
 * @note LRP follows TransformerBlock's rules:
 *       - each residual add is split by the two-term epsilon rule;
 *       - norms and activations pass relevance through unchanged, as AttnLRP prescribes;
 *       - attention and linear layers use their own rules.
 *       Pre-norm with RMSNorm and a gated MLP gives the same output, gradient and relevance as
 *       TransformerBlock; the tests check this.
 */
class EncoderBlock : public Module {
public:
    /** @throws std::invalid_argument from the sub-modules' constructors, or if d_ff isn't positive. */
    EncoderBlock(const AttentionConfig& attention, int64_t d_ff, DeviceBackend* backend, const EncoderBlockOptions& options = {});

    /** @throws std::logic_error before any forward(); std::invalid_argument for a shape mismatch. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @throws std::logic_error before any forward(); std::invalid_argument for a shape mismatch. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    /** @brief Owns the residual adds, as TransformerBlock does. */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    void set_training(bool training) override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    [[nodiscard]] int64_t d_model() const { return d_model_; }
    [[nodiscard]] const EncoderBlockOptions& options() const { return options_; }
    /** @name Sub-modules, for loading weights and reading attention maps. */
    ///@{
    [[nodiscard]] Module& norm1() { return *norm1_; }
    [[nodiscard]] MultiHeadAttentionModule& mha() { return mha_; }
    [[nodiscard]] Module& norm2() { return *norm2_; }
    [[nodiscard]] Module& mlp() { return *mlp_; }
    ///@}

    void release_activations() override;

protected:
    /** @throws std::invalid_argument unless the input is rank 3 with last dimension d_model. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    [[nodiscard]] Tensor Norm(Module& norm, const Tensor& x) const;
    [[nodiscard]] Tensor NormBackward(Module& norm, const Tensor& g) const;
    [[nodiscard]] Tensor NormRelevance(Module& norm, const Tensor& r, const LRPRuleConfig& config) const;

    int64_t d_model_;
    DeviceBackend* backend_;
    EncoderBlockOptions options_;
    std::unique_ptr<Module> norm1_;
    MultiHeadAttentionModule mha_;
    std::unique_ptr<Module> norm2_;
    std::unique_ptr<Module> mlp_;

    // Forward caches: the operands of the two residual splits.
    Shape last_input_shape_ = Shape({0});
    Tensor last_x_;         ///< The input.
    Tensor last_attn_out_;  ///< The attention's output.
    Tensor last_y1_;        ///< The first sublayer's result (after norm1 when post-norm).
    Tensor last_mlp_out_;   ///< The MLP's output.
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
