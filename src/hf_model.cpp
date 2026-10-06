#include "pulsatrix/hf_model.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/json.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Bad(const std::string& what) { throw std::invalid_argument("config.json: " + what); }

const JsonValue* Member(const JsonValue& o, const char* key) {
    const JsonValue* v = o.find(key);
    return (v == nullptr || v->is_null()) ? nullptr : v;
}

int64_t Integer(const JsonValue& v, const char* key) {
    if (v.type() != JsonValue::Type::Number) Bad(std::string(key) + " must be a number");
    try {
        return v.as_int64();
    } catch (const std::invalid_argument&) {
        Bad(std::string(key) + " must be an integer");
    }
}

int64_t RequiredPositive(const JsonValue& o, const char* key) {
    const JsonValue* v = Member(o, key);
    if (v == nullptr) Bad(std::string(key) + " is missing");
    const int64_t i = Integer(*v, key);
    if (i <= 0) Bad(std::string(key) + " must be positive");
    return i;
}

std::optional<int64_t> OptionalInteger(const JsonValue& o, const char* key) {
    const JsonValue* v = Member(o, key);
    if (v == nullptr) return std::nullopt;
    return Integer(*v, key);
}

std::optional<double> OptionalNumber(const JsonValue& o, const char* key) {
    const JsonValue* v = Member(o, key);
    if (v == nullptr) return std::nullopt;
    if (v->type() != JsonValue::Type::Number) Bad(std::string(key) + " must be a number");
    return v->as_double();
}

std::optional<bool> OptionalBool(const JsonValue& o, const char* key) {
    const JsonValue* v = Member(o, key);
    if (v == nullptr) return std::nullopt;
    if (v->type() != JsonValue::Type::Bool) Bad(std::string(key) + " must be true or false");
    return v->as_bool();
}

std::optional<std::string> OptionalString(const JsonValue& o, const char* key) {
    const JsonValue* v = Member(o, key);
    if (v == nullptr) return std::nullopt;
    if (v->type() != JsonValue::Type::String) Bad(std::string(key) + " must be a string");
    return v->as_string();
}

std::optional<HfRopeScaling> RopeScaling(const JsonValue& o) {
    const JsonValue* v = Member(o, "rope_scaling");
    if (v == nullptr) v = Member(o, "rope_parameters");
    if (v == nullptr) return std::nullopt;
    if (v->type() != JsonValue::Type::Object) Bad("rope_scaling must be an object");
    HfRopeScaling r;
    r.type = OptionalString(*v, "rope_type").value_or(OptionalString(*v, "type").value_or("default"));
    if (r.type == "default") return std::nullopt;
    r.factor = static_cast<float>(OptionalNumber(*v, "factor").value_or(1.0));
    r.low_freq_factor = static_cast<float>(OptionalNumber(*v, "low_freq_factor").value_or(1.0));
    r.high_freq_factor = static_cast<float>(OptionalNumber(*v, "high_freq_factor").value_or(1.0));
    r.original_max_position_embeddings = OptionalInteger(*v, "original_max_position_embeddings").value_or(0);
    return r;
}

std::string ReadText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    if (in.bad()) throw std::runtime_error("cannot read " + path);
    return ss.str();
}

}  // namespace

HfModelConfig ParseHfConfig(std::string_view json) {
    JsonValue root;
    try {
        root = ParseJson(json);
    } catch (const std::exception& e) {
        Bad(std::string("not valid JSON: ") + e.what());
    }
    if (root.type() != JsonValue::Type::Object) Bad("must be a JSON object");
    HfModelConfig c;
    if (const JsonValue* arch = Member(root, "architectures")) {
        if (arch->type() != JsonValue::Type::Array) Bad("architectures must be a list");
        if (!arch->as_array().empty()) {
            if (arch->as_array()[0].type() != JsonValue::Type::String) Bad("architectures must hold strings");
            c.architecture = arch->as_array()[0].as_string();
        }
    }
    // A multimodal model keeps its language model's settings in text_config.
    const JsonValue* text = Member(root, "text_config");
    if (text != nullptr && text->type() != JsonValue::Type::Object) Bad("text_config must be an object");
    const JsonValue& t = text != nullptr ? *text : root;

    c.model_type = OptionalString(t, "model_type").value_or(OptionalString(root, "model_type").value_or(""));
    c.hidden_size = RequiredPositive(t, "hidden_size");
    c.intermediate_size = RequiredPositive(t, "intermediate_size");
    c.num_hidden_layers = RequiredPositive(t, "num_hidden_layers");
    c.num_attention_heads = RequiredPositive(t, "num_attention_heads");
    c.vocab_size = RequiredPositive(t, "vocab_size");
    c.num_key_value_heads = OptionalInteger(t, "num_key_value_heads").value_or(c.num_attention_heads);
    if (c.num_key_value_heads <= 0 || c.num_attention_heads % c.num_key_value_heads != 0) {
        Bad("num_attention_heads must be a positive multiple of num_key_value_heads");
    }
    if (auto hd = OptionalInteger(t, "head_dim")) {
        if (*hd <= 0) Bad("head_dim must be positive");
        c.head_dim = *hd;
    } else {
        if (c.hidden_size % c.num_attention_heads != 0) Bad("hidden_size must be divisible by num_attention_heads");
        c.head_dim = c.hidden_size / c.num_attention_heads;
    }
    c.rms_norm_eps = static_cast<float>(OptionalNumber(t, "rms_norm_eps").value_or(1e-6));
    if (auto theta = OptionalNumber(t, "rope_theta")) {
        c.rope_theta = static_cast<float>(*theta);
    } else if (const JsonValue* rp = Member(t, "rope_parameters"); rp != nullptr && rp->type() == JsonValue::Type::Object) {
        c.rope_theta = static_cast<float>(OptionalNumber(*rp, "rope_theta").value_or(10000.0));
    }
    c.rope_scaling = RopeScaling(t);
    c.tie_word_embeddings =
        OptionalBool(t, "tie_word_embeddings").value_or(OptionalBool(root, "tie_word_embeddings").value_or(true));

    const std::string& type = c.model_type;
    const bool qwen2 = type == "qwen2" || type == "qwen2_moe";
    const std::optional<bool> attention_bias = OptionalBool(t, "attention_bias");
    c.qkv_bias = attention_bias.value_or(qwen2);
    c.out_bias = qwen2 ? false : attention_bias.value_or(false);
    c.mlp_bias = OptionalBool(t, "mlp_bias").value_or(false);
    c.qk_norm = type == "qwen3" || type == "qwen3_moe" || type == "gemma3" || type == "gemma3_text" || type == "olmo2";
    c.hidden_act = OptionalString(t, "hidden_act").value_or(OptionalString(t, "hidden_activation").value_or("silu"));
    c.max_position_embeddings = OptionalInteger(t, "max_position_embeddings").value_or(0);
    c.bos_token_id = OptionalInteger(t, "bos_token_id");
    if (!c.bos_token_id) c.bos_token_id = OptionalInteger(root, "bos_token_id");
    const JsonValue* eos = Member(t, "eos_token_id");
    if (eos == nullptr) eos = Member(root, "eos_token_id");
    if (eos != nullptr) {
        if (eos->type() == JsonValue::Type::Array) {
            for (const JsonValue& e : eos->as_array()) c.eos_token_ids.push_back(Integer(e, "eos_token_id"));
        } else {
            c.eos_token_ids.push_back(Integer(*eos, "eos_token_id"));
        }
    }
    c.dtype = OptionalString(t, "torch_dtype").value_or(OptionalString(t, "dtype").value_or(
        OptionalString(root, "torch_dtype").value_or(OptionalString(root, "dtype").value_or(""))));
    // Qwen2 configs carry a sliding_window that use_sliding_window switches off.
    if (OptionalBool(t, "use_sliding_window").value_or(true)) {
        c.sliding_window = OptionalInteger(t, "sliding_window");
    }
    if (auto s = OptionalNumber(t, "query_pre_attn_scalar")) c.query_pre_attn_scalar = static_cast<float>(*s);
    if (const JsonValue* lt = Member(t, "layer_types")) {
        if (lt->type() != JsonValue::Type::Array) Bad("layer_types must be a list");
        for (const JsonValue& v : lt->as_array()) {
            if (v.type() != JsonValue::Type::String) Bad("layer_types must hold strings");
            c.layer_types.push_back(v.as_string());
        }
    }

    // What pulsatrix can't run yet, with the roadmap item that adds it.
    const std::string suffix = "ForCausalLM";
    const bool causal_lm = c.architecture.size() >= suffix.size() &&
                           c.architecture.compare(c.architecture.size() - suffix.size(), suffix.size(), suffix) == 0;
    if (!causal_lm) {
        c.unsupported.push_back("architecture \"" + c.architecture + "\" is not a decoder-only causal language model");
    }
    if (c.rope_scaling) c.unsupported.push_back("rope_scaling \"" + c.rope_scaling->type + "\" (LLM-6)");
    if (c.sliding_window && *c.sliding_window < c.max_position_embeddings) {
        c.unsupported.push_back("sliding-window attention (LLM-9)");
    }
    // Gemma scales scores by query_pre_attn_scalar^-0.5; only a value other than head_dim differs
    // from the usual scale.
    if (c.query_pre_attn_scalar && *c.query_pre_attn_scalar != static_cast<float>(c.head_dim)) {
        c.unsupported.push_back("query_pre_attn_scalar (LLM-9)");
    }
    if (type.rfind("gemma", 0) == 0) {
        c.unsupported.push_back("Gemma's (1 + w) RMSNorm and embedding scaling (LLM-9)");
    }
    if (c.hidden_act != "silu") c.unsupported.push_back("activation \"" + c.hidden_act + "\" in the MLP");
    if (c.mlp_bias) c.unsupported.push_back("MLP biases (IO-4)");
    if (Member(t, "attn_logit_softcapping") || Member(t, "final_logit_softcapping")) {
        c.unsupported.push_back("logit softcapping");
    }
    return c;
}

HfModelConfig ReadHfConfig(const std::string& path) { return ParseHfConfig(ReadText(path)); }

AttentionConfig ToAttentionConfig(const HfModelConfig& config) {
    AttentionConfig a;
    a.d_model = config.hidden_size;
    a.num_heads = config.num_attention_heads;
    a.num_kv_heads = config.num_key_value_heads;
    a.head_dim = config.head_dim;
    a.use_rope = true;
    a.rope_layout = RoPELayout::RotateHalf;
    a.rope_base = config.rope_theta;
    a.use_qk_norm = config.qk_norm;
    a.qk_norm_eps = config.rms_norm_eps;
    a.qkv_bias = config.qkv_bias;
    a.out_bias = config.out_bias;
    a.causal = true;
    return a;
}

// ---- checkpoint -------------------------------------------------------------------------------

HfCheckpoint HfCheckpoint::Open(const std::string& directory) {
    namespace fs = std::filesystem;
    const fs::path dir(directory);
    const fs::path index_path = dir / "model.safetensors.index.json";
    const fs::path single_path = dir / "model.safetensors";
    HfCheckpoint ck;
    auto open_shard = [&](const std::string& file) -> const SafetensorsFile& {
        auto it = ck.files_.find(file);
        if (it == ck.files_.end()) {
            it = ck.files_.emplace(file, SafetensorsFile::Map((dir / file).string())).first;
            ck.shard_order_.push_back(file);
        }
        return it->second;
    };
    if (fs::exists(index_path)) {
        JsonValue index;
        try {
            index = ParseJson(ReadText(index_path.string()));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("model.safetensors.index.json: not valid JSON: ") + e.what());
        }
        const JsonValue* map = index.type() == JsonValue::Type::Object ? index.find("weight_map") : nullptr;
        if (map == nullptr || map->type() != JsonValue::Type::Object) {
            throw std::invalid_argument("model.safetensors.index.json: needs a \"weight_map\" object");
        }
        for (const auto& [name, file] : map->as_object()) {
            if (file.type() != JsonValue::Type::String) {
                throw std::invalid_argument("model.safetensors.index.json: \"" + name + "\" must map to a file name");
            }
            const std::string& f = file.as_string();
            // A shard must be a plain file in this directory: never follow a path out of it.
            if (f.empty() || f.find('/') != std::string::npos || f.find('\\') != std::string::npos || f == "." ||
                f == ".." || f.find(':') != std::string::npos) {
                throw std::invalid_argument("model.safetensors.index.json: shard \"" + f +
                                            "\" is not a plain file name in the checkpoint directory");
            }
            if (!ck.shard_of_.emplace(name, f).second) {
                throw std::invalid_argument("model.safetensors.index.json: \"" + name + "\" is listed twice");
            }
        }
        for (const auto& [name, f] : ck.shard_of_) {
            if (!open_shard(f).contains(name)) {
                throw std::invalid_argument("model.safetensors.index.json: shard \"" + f + "\" holds no tensor \"" +
                                            name + "\"");
            }
        }
        for (const auto& [f, file] : ck.files_) {
            for (const std::string& name : file.names()) {
                auto it = ck.shard_of_.find(name);
                if (it == ck.shard_of_.end() || it->second != f) {
                    throw std::invalid_argument("shard \"" + f + "\" holds \"" + name + "\", which the index doesn't list there");
                }
            }
        }
    } else if (fs::exists(single_path)) {
        const SafetensorsFile& file = open_shard("model.safetensors");
        for (const std::string& name : file.names()) ck.shard_of_.emplace(name, "model.safetensors");
    } else {
        throw std::runtime_error("HfCheckpoint: " + directory + " has neither model.safetensors nor model.safetensors.index.json");
    }
    for (const auto& [name, f] : ck.shard_of_) ck.names_.push_back(name);
    return ck;
}

const std::string& HfCheckpoint::shard_of(const std::string& name) const {
    auto it = shard_of_.find(name);
    if (it == shard_of_.end()) throw std::invalid_argument("HfCheckpoint: no tensor named \"" + name + "\"");
    return it->second;
}

const SafetensorsTensorInfo& HfCheckpoint::info(const std::string& name) const {
    return files_.at(shard_of(name)).info(name);
}

Tensor HfCheckpoint::tensor(const std::string& name, DeviceBackend* backend) const {
    return files_.at(shard_of(name)).tensor(name, backend);
}

std::vector<std::string> HfCheckpoint::shards() const { return shard_order_; }

}  // namespace pulsatrix
