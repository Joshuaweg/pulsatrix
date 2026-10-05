/** @file hf_model.hpp
 *  @brief Hugging Face checkpoints: `config.json` and single or sharded safetensors (IO-5).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {

/** @brief A `rope_scaling` (or `rope_parameters`) entry other than the default. */
struct HfRopeScaling {
    /** @brief `rope_type` (older files: `type`), for example `"llama3"`, `"linear"`, `"yarn"`. */
    std::string type;
    float factor = 1.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 1.0f;
    int64_t original_max_position_embeddings = 0;
};

/**
 * @brief A decoder-only language model's `config.json`, with the values Hugging Face leaves to
 *        the model class filled in: every field is what the model actually uses.
 */
struct HfModelConfig {
    /** @brief `architectures[0]`, for example `"LlamaForCausalLM"`. */
    std::string architecture;
    /** @brief For example `"llama"`, `"qwen2"`, `"qwen3"`, `"gemma3_text"`. */
    std::string model_type;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    /** @brief `num_key_value_heads`, or num_attention_heads when absent. */
    int64_t num_key_value_heads = 0;
    /** @brief `head_dim`, or hidden_size / num_attention_heads when absent. */
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    /** @brief `rope_theta` (or `rope_parameters.rope_theta`), 10000 when absent. */
    float rope_theta = 10000.0f;
    std::optional<HfRopeScaling> rope_scaling;
    bool tie_word_embeddings = true;
    /** @brief Q, K and V projection biases: `attention_bias`, or the architecture's own rule (Qwen2
     *         always has them). */
    bool qkv_bias = false;
    /** @brief Output projection bias (Llama's `attention_bias` covers it; Qwen2 never has one). */
    bool out_bias = false;
    bool mlp_bias = false;
    /** @brief RMSNorm on each head's queries and keys (Qwen3, Gemma 3). */
    bool qk_norm = false;
    /** @brief `hidden_act` (Gemma: `hidden_activation`), for example `"silu"`. */
    std::string hidden_act = "silu";
    int64_t max_position_embeddings = 0;
    std::optional<int64_t> bos_token_id;
    /** @brief `eos_token_id`, which may be one id or a list. */
    std::vector<int64_t> eos_token_ids;
    /** @brief `torch_dtype` (newer files: `dtype`), for example `"bfloat16"`. */
    std::string dtype;
    /** @brief The attention window when sliding-window attention is actually used. */
    std::optional<int64_t> sliding_window;
    std::optional<float> query_pre_attn_scalar;
    /** @brief `layer_types`, for example `"sliding_attention"` / `"full_attention"` per layer. */
    std::vector<std::string> layer_types;
    /**
     * @brief Everything in the config that pulsatrix can't run yet, each naming the roadmap item
     *        that adds it (for example `"rope_scaling \"llama3\" (LLM-6)"`), including an
     *        architecture that isn't a `...ForCausalLM` decoder. Empty means the model
     *        maps onto pulsatrix's modules exactly; a loader should refuse otherwise.
     */
    std::vector<std::string> unsupported;
};

/**
 * @brief Parses `config.json` text. A multimodal config's `text_config` is read for the language
 *        model.
 * @throws std::invalid_argument if the text isn't a JSON object, a required size is missing or not
 *         a positive integer (hidden_size, intermediate_size, num_hidden_layers,
 *         num_attention_heads, vocab_size), a field has the wrong type, or the head counts don't
 *         fit together.
 */
[[nodiscard]] HfModelConfig ParseHfConfig(std::string_view json);

/** @brief ParseHfConfig on a file. @throws std::runtime_error if it can't be read. */
[[nodiscard]] HfModelConfig ReadHfConfig(const std::string& path);

/** @brief The attention layer the config describes: rotate-half RoPE, causal, its head counts,
 *         biases, QK-Norm and epsilon. */
[[nodiscard]] AttentionConfig ToAttentionConfig(const HfModelConfig& config);

/**
 * @brief A Hugging Face checkpoint directory's weights: `model.safetensors`, or the shards listed
 *        by `model.safetensors.index.json` (IO-5). Shards are memory-mapped, so opening a
 *        checkpoint reads only headers, and a tensor's bytes are paged in when it is read.
 */
class HfCheckpoint {
public:
    /**
     * @brief Opens @p directory: the index if there is one, otherwise the single file.
     * @throws std::runtime_error if neither file exists, or a file can't be read.
     * @throws std::invalid_argument if the index is malformed, names a shard outside the directory
     *         (any path separator or `..`), lists a tensor its shard doesn't hold, or a shard holds a
     *         tensor the index doesn't list; or a shard isn't valid safetensors.
     */
    [[nodiscard]] static HfCheckpoint Open(const std::string& directory);

    /** @brief Every tensor name, sorted. */
    [[nodiscard]] const std::vector<std::string>& names() const { return names_; }
    [[nodiscard]] bool contains(const std::string& name) const { return shard_of_.count(name) != 0; }
    /** @brief The shard file holding @p name. @throws std::invalid_argument if there's no such tensor. */
    [[nodiscard]] const std::string& shard_of(const std::string& name) const;
    /** @brief @p name's header entry (dtype, shape). @throws std::invalid_argument if there's none. */
    [[nodiscard]] const SafetensorsTensorInfo& info(const std::string& name) const;
    /** @brief Reads @p name into a Tensor; see SafetensorsFile::tensor for the dtypes it converts. */
    [[nodiscard]] Tensor tensor(const std::string& name, DeviceBackend* backend) const;
    /** @brief The shard files, in the order first listed. */
    [[nodiscard]] std::vector<std::string> shards() const;

private:
    HfCheckpoint() = default;
    std::map<std::string, SafetensorsFile> files_;
    std::map<std::string, std::string> shard_of_;
    std::vector<std::string> names_;
    std::vector<std::string> shard_order_;
};

}  // namespace pulsatrix
