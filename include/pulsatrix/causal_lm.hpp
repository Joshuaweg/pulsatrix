/** @file causal_lm.hpp
 *  @brief A Llama-family decoder-only language model built from a Hugging Face config, and
 *         loading its Hugging Face weights by name (IO-4).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/generation.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {

/**
 * @brief Token embedding, `num_hidden_layers` pre-norm transformer blocks, a final RMSNorm and a
 *        language-model head (tied to the embedding or its own bias-free Linear): the architecture
 *        of Llama, SmolLM2, Qwen2/2.5 and Qwen3. Maps `(N, L)` token ids to `(N, L, vocab)` logits.
 *
 * Parameter names follow Hugging Face's with pulsatrix's module names inside a layer:
 * `embed_tokens.weight`, `layers.<i>.norm1.weight`, `layers.<i>.mha.q_proj.weight`, ...,
 * `layers.<i>.swiglu.down_proj.weight`, `norm.weight`, and `lm_head.weight` when untied.
 * HuggingFaceMapping() lists how each Hugging Face tensor maps onto them.
 *
 * @note backward() and propagate_relevance() run through every layer, so the model trains and
 *       explains like any other module.
 */
class CausalLM : public Module {
public:
    /**
     * @param config Its sizes, head counts, RoPE, biases, QK-Norm, epsilon and tying.
     * @param allow_unsupported false (the default) refuses a config whose `unsupported` list isn't
     *        empty, since the model would run but compute something else.
     * @throws std::invalid_argument for such a config, or invalid sizes.
     */
    CausalLM(const HfModelConfig& config, DeviceBackend* backend, bool allow_unsupported = false);

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    /** @brief A chain of other modules, as SequentialModule. */
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    void set_training(bool training) override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }

    /**
     * @brief A cached next-token function for Generate (LLM-5): one pass over the prompt, then one
     *        position per new token.
     * @note Holds pointers into this model, which must outlive it.
     */
    [[nodiscard]] NextTokenLogitsFn next_token_logits(int64_t max_length);

    [[nodiscard]] const HfModelConfig& config() const { return config_; }
    [[nodiscard]] EmbeddingModule& embed_tokens() { return embed_; }
    [[nodiscard]] TransformerBlock& layer(int64_t i) { return *layers_.at(static_cast<size_t>(i)); }
    [[nodiscard]] int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }
    [[nodiscard]] RMSNormModule& norm() { return norm_; }

protected:
    /** @throws std::invalid_argument if input isn't `(N, L)` token ids in the vocabulary. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    /** @brief Token embeddings times config.embed_scale (Gemma 3's sqrt(hidden_size)). */
    [[nodiscard]] Tensor ScaleEmbeddings(const Tensor& x) const;
    HfModelConfig config_;
    DeviceBackend* backend_;
    EmbeddingModule embed_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    RMSNormModule norm_;
    std::unique_ptr<TiedLMHeadModule> tied_head_;
    std::unique_ptr<LinearModule> lm_head_;
    Shape last_hidden_shape_ = Shape({0});
};

/** @brief How a source tensor becomes a target parameter. */
enum class WeightTransform {
    Identity,
    /** Swap the two axes of a matrix: PyTorch's `nn.Linear` stores `(out, in)`, pulsatrix's
     *  LinearModule `(in, out)`. */
    Transpose,
};

/** @brief One entry of a name-mapping manifest. */
struct WeightMapping {
    /** @brief The checkpoint tensor's name. */
    std::string source;
    /** @brief The model parameter's (or buffer's) name. */
    std::string target;
    WeightTransform transform = WeightTransform::Identity;
    /** @brief Take only these rows of the source's first axis, before the transform: one part of
     *         a fused tensor such as a `qkv_proj`. row_count < 0 takes them all. */
    int64_t row_begin = 0;
    int64_t row_count = -1;
};

/** @brief Loading settings. */
struct WeightLoadOptions {
    /** @brief Every target must be loaded and every source used (or ignored); otherwise nothing
     *         is changed and the call throws. */
    bool strict = true;
    /** @brief Source tensors that may go unused, for example a duplicated `lm_head.weight` in a
     *         tied checkpoint. */
    std::vector<std::string> ignore;
};

/** @brief What a load did. */
struct WeightLoadReport {
    std::vector<std::string> loaded;
    /** @brief Model parameters and buffers nothing loaded (empty when strict). */
    std::vector<std::string> missing;
    /** @brief Checkpoint tensors nothing used, ignored ones included. */
    std::vector<std::string> unused;
};

/**
 * @brief Loads checkpoint tensors into @p model's named parameters and buffers by @p mapping. Every
 *        tensor is read, converted to fp32 (IO-6), sliced and transformed, and checked against its
 *        target's shape before anything is written, so a failed load leaves the model unchanged.
 * @throws std::invalid_argument if a source or target doesn't exist, a target is mapped twice, a
 *         shape doesn't match, a slice is out of range, a Transpose source isn't a matrix, or (strict)
 *         a target is left unloaded or a source unused and not ignored.
 */
WeightLoadReport LoadWeights(Module& model, const HfCheckpoint& checkpoint, const std::vector<WeightMapping>& mapping,
                             const WeightLoadOptions& options = {});

/**
 * @brief The manifest for a Llama-family Hugging Face checkpoint (`model.embed_tokens.weight`,
 *        `model.layers.<i>.self_attn.q_proj.weight`, ..., `model.norm.weight`, `lm_head.weight`)
 *        onto CausalLM's parameters: Linear weights transposed, biases and norms as they are.
 */
[[nodiscard]] std::vector<WeightMapping> HuggingFaceMapping(const HfModelConfig& config);

/**
 * @brief Reads `config.json` and the weights in a downloaded Hugging Face model directory into a
 *        new CausalLM, strictly: a tied checkpoint's `lm_head.weight` is the only tensor allowed
 *        to go unused.
 * @throws std::invalid_argument for a config CausalLM refuses or weights that don't match it;
 *         std::runtime_error if files can't be read.
 */
[[nodiscard]] std::unique_ptr<CausalLM> LoadCausalLM(const std::string& directory, DeviceBackend* backend);

}  // namespace pulsatrix
