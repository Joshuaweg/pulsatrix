/** @file encoder_lm.hpp
 *  @brief Masked-language-model encoders, starting with ESM-2 protein language models (PLM-2):
 *         Hugging Face `EsmForMaskedLM` checkpoints loaded, run, trained and explained natively.
 *  @ingroup dl_modules
 */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/activation_module.hpp"
#include "pulsatrix/causal_lm.hpp"  // WeightMapping, LoadWeights
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/encoder_block.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"

namespace pulsatrix {

/** @brief The settings of an ESM-family masked LM, from its Hugging Face `config.json`. */
struct EncoderLMConfig {
    std::string architecture;  ///< `EsmForMaskedLM`
    std::string model_type;    ///< `esm`
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    float layer_norm_eps = 1e-12f;
    float rope_theta = 10000.0f;
    int64_t max_position_embeddings = 0;
    /** @brief ESM's "mask dropout": masked tokens' embeddings are zeroed and the rest rescaled. */
    bool token_dropout = false;
    int64_t mask_token_id = -1;
    int64_t pad_token_id = -1;
    std::string dtype;
    /**
     * @brief RoPE frequencies to use instead of `rope_theta`'s, one per pair. LoadEncoderLM fills
     *        them from the checkpoint's `inv_freq` buffer: ESM-2's are stored rounded to fp16
     *        (0.316162 rather than 0.316228), the model was trained with those, and transformers
     *        uses them, so recomputing them exactly shifts attention by about 2e-4.
     */
    std::vector<double> rope_inverse_frequencies;
};

/**
 * @brief Reads an ESM `config.json`.
 * @throws std::invalid_argument for malformed JSON, a missing size, or a model this class doesn't
 *         run yet: anything but `EsmForMaskedLM`/`EsmModel`, absolute position embeddings
 *         (ESM-1b), `emb_layer_norm_before`, ESMFold, or an activation other than GELU.
 */
[[nodiscard]] EncoderLMConfig ParseEsmConfig(std::string_view json);
/** @brief ParseEsmConfig on a file. @throws std::runtime_error if it can't be read. */
[[nodiscard]] EncoderLMConfig ReadEsmConfig(const std::string& path);

/**
 * @brief An ESM-2 masked language model: token embeddings (with ESM's token-dropout scaling),
 *        pre-LayerNorm encoder layers with rotary attention (EncoderBlock), a final LayerNorm,
 *        and the LM head: dense, GELU, LayerNorm, then the embedding table transposed plus a bias.
 *        Shape: token ids `(N, L)` to logits `(N, L, vocab)`.
 *
 * Parameters are named `embed_tokens.weight`, `layers.<i>.*` (EncoderBlock's names),
 * `norm.*`, and `lm_head.dense.*`, `lm_head.norm.*` and `lm_head.bias`. The head's decoder shares
 * the embedding table.
 *
 * @note **Token dropout.** ESM-2 was trained with masked tokens' embeddings zeroed. To match at
 *       inference, every embedding is scaled by `(1 - 0.15 * 0.8) / (1 - masked fraction)`,
 *       where the fraction counts the sequence's real (unpadded) tokens. This is done even
 *       when nothing is masked, and the model's logits depend on it.
 * @note **Padding.** set_padding_mask() masks padding keys in every layer and zeroes padded
 *       embeddings, as Hugging Face does with `attention_mask`.
 * @note **LRP.**
 *       - The token-dropout scale is a constant, so relevance passes through it.
 *       - The head's bias takes its epsilon-rule share.
 *       - Everything else uses its module's own rule.
 */
class EncoderLM : public Module {
public:
    /** @throws std::invalid_argument for invalid sizes. */
    EncoderLM(const EncoderLMConfig& config, DeviceBackend* backend);

    /** @throws std::logic_error before any forward(); std::invalid_argument for a shape mismatch. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Relevance per input token, `(N, L)`. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    /**
     * @brief propagate_relevance, stopping at every layer boundary.
     * @return One tensor per boundary, each shaped `(N, L, hidden_size)`: the relevance entering
     *         layer 0 (the scaled embeddings), then the relevance leaving each layer.
     */
    [[nodiscard]] std::vector<Tensor> propagate_relevance_by_layer(const Tensor& relevance_out, const LRPRuleConfig& config);
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    void set_training(bool training) override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    /**
     * @brief Masks padding for the following forward passes: @p keep is `(N, L)`, 1 for a real
     *        token and 0 for padding, as Hugging Face's `attention_mask`.
     * @throws std::invalid_argument unless every value is 0 or 1.
     */
    void set_padding_mask(const Tensor& keep);
    /** @brief Treats every token as real again. */
    void clear_padding_mask();

    [[nodiscard]] const EncoderLMConfig& config() const { return config_; }
    [[nodiscard]] int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }
    [[nodiscard]] EncoderBlock& layer(int64_t i) { return *layers_.at(static_cast<size_t>(i)); }
    [[nodiscard]] EmbeddingModule& embed_tokens() { return embed_; }
    /**
     * @brief The last forward pass's hidden states, each `(N, L, hidden_size)`: the scaled
     *        embeddings, then every layer's output. The last entry is before the final LayerNorm.
     *        Hugging Face's `hidden_states` match these, except that its last entry is after it.
     */
    [[nodiscard]] const std::vector<Tensor>& hidden_states() const { return hidden_; }
    /** @brief The last forward pass's final-LayerNorm output, `(N, L, hidden_size)`: the per-residue
     *         representations ("embeddings") usually taken from ESM. */
    [[nodiscard]] const Tensor& last_hidden_state() const { return last_hidden_; }

protected:
    /** @throws std::invalid_argument unless the input is `(N, L)` ids in the vocabulary, or
     *          the padding mask doesn't match it. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    [[nodiscard]] Tensor Rows(Module& m, const Tensor& x, int64_t width) const;
    [[nodiscard]] Tensor RowsBackward(Module& m, const Tensor& g, int64_t width) const;
    [[nodiscard]] Tensor RowsRelevance(Module& m, const Tensor& r, int64_t width, const LRPRuleConfig& config) const;
    /** @brief Relevance at the final LayerNorm's input, from relevance at the logits. */
    [[nodiscard]] Tensor HeadRelevance(const Tensor& relevance_out, const LRPRuleConfig& config);

    EncoderLMConfig config_;
    DeviceBackend* backend_;
    EmbeddingModule embed_;
    std::vector<std::unique_ptr<EncoderBlock>> layers_;
    LayerNormModule norm_;
    LinearModule lm_dense_;
    ActivationModule lm_gelu_;
    LayerNormModule lm_norm_;
    TiedLMHeadModule lm_decoder_;
    Tensor lm_bias_;
    Tensor lm_bias_grad_;

    std::vector<float> padding_keep_;  ///< (N * L); empty for no padding
    Shape padding_shape_ = Shape({0});
    Tensor embed_scale_;  ///< (N, L, hidden): token dropout and padding, per element
    Tensor last_logits_nobias_;
    std::vector<Tensor> hidden_;
    Tensor last_hidden_;
    Shape last_ids_shape_ = Shape({0});
    bool has_forwarded_ = false;
};

/**
 * @brief The name mapping from an `EsmForMaskedLM` checkpoint to EncoderLM. Q/K/V, output, MLP
 *        and LM-head dense weights are transposed.
 */
[[nodiscard]] std::vector<WeightMapping> EsmMapping(const EncoderLMConfig& config);
/** @brief Checkpoint tensors an ESM-2 EncoderLM doesn't use: the absolute-position table and ids
 *         (unused with rotary), each layer's rotary `inv_freq` (recomputed), and the contact head
 *         (PLM-4). */
[[nodiscard]] std::vector<std::string> EsmIgnoredTensors(const EncoderLMConfig& config);

/**
 * @brief Loads an ESM-2 model directory: `config.json` and safetensors weights (one file or
 *        sharded), such as `facebook/esm2_t6_8M_UR50D`. LayerNorm parameters may be named
 *        `weight`/`bias` or, as transformers 5 saves them, `gamma`/`beta`. RoPE frequencies come
 *        from the checkpoint's `inv_freq` buffer when it has one (see
 *        EncoderLMConfig::rope_inverse_frequencies).
 * @throws std::invalid_argument for an unsupported config or a weight that doesn't fit;
 *         std::runtime_error for a missing file.
 */
[[nodiscard]] std::unique_ptr<EncoderLM> LoadEncoderLM(const std::string& directory, DeviceBackend* backend);

}  // namespace pulsatrix
