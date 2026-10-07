#include "pulsatrix/causal_lm.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

const HfModelConfig& Checked(const HfModelConfig& c, bool allow_unsupported) {
    if (!allow_unsupported && !c.unsupported.empty()) {
        std::string list;
        for (const auto& u : c.unsupported) list += (list.empty() ? "" : "; ") + u;
        throw std::invalid_argument("CausalLM: the config needs features pulsatrix can't run yet: " + list);
    }
    return c;
}

Tensor Reshaped(const Tensor& t, Shape shape) {
    Tensor out(t);
    out.reshape(std::move(shape));
    return out;
}

}  // namespace

CausalLM::CausalLM(const HfModelConfig& config, DeviceBackend* backend, bool allow_unsupported)
    : config_(Checked(config, allow_unsupported)),
      backend_(backend),
      embed_(config.vocab_size, config.hidden_size, backend),
      norm_(config.hidden_size, backend, backend->device(), config.rms_norm_eps) {
    norm_.set_weight_offset(config.norm_weight_offset);
    TransformerBlockOptions options;
    options.norm_eps = config.rms_norm_eps;
    options.mlp_bias = config.mlp_bias;
    options.mlp_activation = config.hidden_act == "silu" ? GatedActivation::Silu : GatedActivation::GeluTanh;
    options.post_norms = config.post_norms;
    options.norm_weight_offset = config.norm_weight_offset;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        // Each layer's own attention: Gemma 3 alternates sliding-window and full layers, each with
        // its own RoPE base (LLM-9).
        layers_.push_back(std::make_unique<TransformerBlock>(ToAttentionConfig(config, i), config.intermediate_size,
                                                             backend, options));
    }
    if (config.tie_word_embeddings) {
        tied_head_ = std::make_unique<TiedLMHeadModule>(embed_, backend);
    } else {
        lm_head_ = std::make_unique<LinearModule>(config.hidden_size, config.vocab_size, backend, /*use_bias=*/false);
    }
}

Tensor CausalLM::ScaleEmbeddings(const Tensor& x) const {
    if (config_.embed_scale == 1.0f) return x;
    Tensor out(x.shape(), backend_, x.device());
    backend_->axpby(config_.embed_scale, x.data(), 0.0f, nullptr, out.data(), static_cast<size_t>(x.numel()));
    return out;
}

Tensor CausalLM::forward_impl(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("CausalLM::forward: input must be (N, L) token ids");
    }
    Tensor x = ScaleEmbeddings(embed_.forward(input));
    for (auto& layer : layers_) x = layer->forward(x);
    last_hidden_shape_ = x.shape();
    const int64_t d = config_.hidden_size;
    const int64_t rows = x.numel() / d;
    Tensor normed = norm_.forward(Reshaped(x, Shape({rows, d})));
    Tensor logits = tied_head_ ? tied_head_->forward(normed) : lm_head_->forward(normed);
    return Reshaped(logits, Shape({input.shape().dim(0), input.shape().dim(1), config_.vocab_size}));
}

Tensor CausalLM::backward(const Tensor& grad_output) {
    const int64_t rows = grad_output.numel() / config_.vocab_size;
    Tensor g = Reshaped(grad_output, Shape({rows, config_.vocab_size}));
    g = tied_head_ ? tied_head_->backward(g) : lm_head_->backward(g);
    g = Reshaped(norm_.backward(g), last_hidden_shape_);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) g = (*it)->backward(g);
    return embed_.backward(ScaleEmbeddings(g));  // d(s e)/de = s
}

Tensor CausalLM::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    const int64_t rows = relevance_out.numel() / config_.vocab_size;
    Tensor r = Reshaped(relevance_out, Shape({rows, config_.vocab_size}));
    r = tied_head_ ? tied_head_->propagate_relevance(r, config) : lm_head_->propagate_relevance(r, config);
    r = Reshaped(norm_.propagate_relevance(r, config), last_hidden_shape_);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) r = (*it)->propagate_relevance(r, config);
    // Scaling the embeddings by a constant passes relevance through unchanged.
    return embed_.propagate_relevance(r, config);
}

std::vector<NamedParamRef> CausalLM::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "embed_tokens", embed_);
    for (size_t i = 0; i < layers_.size(); ++i) append_named_parameters(out, "layers." + std::to_string(i), *layers_[i]);
    append_named_parameters(out, "norm", norm_);
    if (lm_head_) append_named_parameters(out, "lm_head", *lm_head_);
    return out;
}

std::vector<NamedBufferRef> CausalLM::named_buffers() {
    std::vector<NamedBufferRef> out;
    for (size_t i = 0; i < layers_.size(); ++i) append_named_buffers(out, "layers." + std::to_string(i), *layers_[i]);
    return out;
}

void CausalLM::set_training(bool training) {
    Module::set_training(training);
    embed_.set_training(training);
    for (auto& layer : layers_) layer->set_training(training);
    norm_.set_training(training);
    if (tied_head_) tied_head_->set_training(training);
    if (lm_head_) lm_head_->set_training(training);
}

NextTokenLogitsFn CausalLM::next_token_logits(int64_t max_length) {
    std::vector<TransformerBlock*> blocks;
    for (auto& layer : layers_) blocks.push_back(layer.get());
    Module* head = tied_head_ ? static_cast<Module*>(tied_head_.get()) : lm_head_.get();
    return MakeCachedNextTokenLogits(embed_, blocks, {&norm_, head}, backend_, max_length, config_.embed_scale);
}

// ---- loading --------------------------------------------------------------------------------

WeightLoadReport LoadWeights(Module& model, const HfCheckpoint& checkpoint, const std::vector<WeightMapping>& mapping,
                             const WeightLoadOptions& options) {
    std::map<std::string, Tensor*> targets;
    for (NamedParamRef& p : model.named_parameters()) targets.emplace(p.name, p.ref.value);
    for (NamedBufferRef& b : model.named_buffers()) targets.emplace(b.name, b.value);

    // Build every converted tensor first; write only when all of them check out.
    std::vector<std::pair<Tensor*, Tensor>> writes;
    std::set<std::string> loaded_targets, used_sources;
    for (const WeightMapping& m : mapping) {
        auto t = targets.find(m.target);
        if (t == targets.end()) {
            throw std::invalid_argument("LoadWeights: the model has no parameter \"" + m.target + "\"");
        }
        if (!loaded_targets.insert(m.target).second) {
            throw std::invalid_argument("LoadWeights: \"" + m.target + "\" is mapped twice");
        }
        if (!checkpoint.contains(m.source)) {
            throw std::invalid_argument("LoadWeights: the checkpoint has no tensor \"" + m.source + "\"");
        }
        used_sources.insert(m.source);
        Tensor* target = t->second;
        Tensor source = checkpoint.tensor(m.source, target->backend());
        std::vector<int64_t> dims;
        for (int64_t d = 0; d < source.rank(); ++d) dims.push_back(source.shape().dim(static_cast<size_t>(d)));
        std::vector<float> values = source.to_host_vector();
        if (m.row_count >= 0 || m.row_begin != 0) {
            if (dims.empty()) throw std::invalid_argument("LoadWeights: can't take rows of scalar \"" + m.source + "\"");
            const int64_t count = m.row_count >= 0 ? m.row_count : dims[0] - m.row_begin;
            if (m.row_begin < 0 || count < 0 || m.row_begin + count > dims[0]) {
                throw std::invalid_argument("LoadWeights: rows out of range for \"" + m.source + "\"");
            }
            const int64_t row = source.numel() / std::max<int64_t>(dims[0], 1);
            values = std::vector<float>(values.begin() + m.row_begin * row, values.begin() + (m.row_begin + count) * row);
            dims[0] = count;
        }
        if (m.transform == WeightTransform::Transpose) {
            if (dims.size() != 2) throw std::invalid_argument("LoadWeights: \"" + m.source + "\" is not a matrix");
            std::vector<float> t2(values.size());
            for (int64_t r = 0; r < dims[0]; ++r) {
                for (int64_t c = 0; c < dims[1]; ++c) t2[static_cast<size_t>(c * dims[0] + r)] = values[static_cast<size_t>(r * dims[1] + c)];
            }
            values = std::move(t2);
            std::swap(dims[0], dims[1]);
        }
        if (Shape(dims) != target->shape()) {
            throw std::invalid_argument("LoadWeights: \"" + m.source + "\" doesn't fit \"" + m.target + "\"'s shape");
        }
        writes.emplace_back(target, Tensor(target->shape(), target->backend(), values, target->device()));
    }

    WeightLoadReport report;
    for (const auto& [name, t] : targets) {
        if (!loaded_targets.count(name)) report.missing.push_back(name);
    }
    std::set<std::string> ignore(options.ignore.begin(), options.ignore.end());
    std::vector<std::string> unexpected;
    for (const std::string& name : checkpoint.names()) {
        if (used_sources.count(name)) continue;
        report.unused.push_back(name);
        if (!ignore.count(name)) unexpected.push_back(name);
    }
    if (options.strict && (!report.missing.empty() || !unexpected.empty())) {
        std::string what = "LoadWeights: strict load failed:";
        if (!report.missing.empty()) what += " " + std::to_string(report.missing.size()) + " parameter(s) not loaded, first \"" + report.missing.front() + "\";";
        if (!unexpected.empty()) what += " " + std::to_string(unexpected.size()) + " checkpoint tensor(s) unused, first \"" + unexpected.front() + "\"";
        throw std::invalid_argument(what);
    }
    for (auto& [target, value] : writes) {
        const bool requires_grad = target->requires_grad();
        *target = std::move(value);
        target->set_requires_grad(requires_grad);
    }
    report.loaded.assign(loaded_targets.begin(), loaded_targets.end());
    return report;
}

std::vector<WeightMapping> HuggingFaceMapping(const HfModelConfig& c) {
    using T = WeightTransform;
    std::vector<WeightMapping> m{{"model.embed_tokens.weight", "embed_tokens.weight", T::Identity},
                                 {"model.norm.weight", "norm.weight", T::Identity}};
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string hf = "model.layers." + std::to_string(i) + ".";
        const std::string px = "layers." + std::to_string(i) + ".";
        m.push_back({hf + "input_layernorm.weight", px + "norm1.weight", T::Identity});
        if (c.post_norms) {
            // Gemma 3: post_attention_layernorm is the sandwich norm after attention, and
            // pre_feedforward_layernorm the one before the MLP.
            m.push_back({hf + "post_attention_layernorm.weight", px + "post_attn_norm.weight", T::Identity});
            m.push_back({hf + "pre_feedforward_layernorm.weight", px + "norm2.weight", T::Identity});
            m.push_back({hf + "post_feedforward_layernorm.weight", px + "post_mlp_norm.weight", T::Identity});
        } else {
            m.push_back({hf + "post_attention_layernorm.weight", px + "norm2.weight", T::Identity});
        }
        for (const char* p : {"q_proj", "k_proj", "v_proj"}) {
            m.push_back({hf + "self_attn." + p + ".weight", px + "mha." + p + ".weight", T::Transpose});
            if (c.qkv_bias) m.push_back({hf + "self_attn." + p + ".bias", px + "mha." + p + ".bias", T::Identity});
        }
        m.push_back({hf + "self_attn.o_proj.weight", px + "mha.out_proj.weight", T::Transpose});
        if (c.out_bias) m.push_back({hf + "self_attn.o_proj.bias", px + "mha.out_proj.bias", T::Identity});
        if (c.qk_norm) {
            m.push_back({hf + "self_attn.q_norm.weight", px + "mha.q_norm.weight", T::Identity});
            m.push_back({hf + "self_attn.k_norm.weight", px + "mha.k_norm.weight", T::Identity});
        }
        for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
            m.push_back({hf + "mlp." + p + ".weight", px + "swiglu." + p + ".weight", T::Transpose});
            if (c.mlp_bias) m.push_back({hf + "mlp." + p + ".bias", px + "swiglu." + p + ".bias", T::Identity});
        }
    }
    if (!c.tie_word_embeddings) m.push_back({"lm_head.weight", "lm_head.weight", T::Transpose});
    return m;
}

std::unique_ptr<CausalLM> LoadCausalLM(const std::string& directory, DeviceBackend* backend) {
    const HfModelConfig config = ReadHfConfig((std::filesystem::path(directory) / "config.json").string());
    auto model = std::make_unique<CausalLM>(config, backend);
    const HfCheckpoint checkpoint = HfCheckpoint::Open(directory);
    WeightLoadOptions options;
    if (config.tie_word_embeddings) options.ignore.push_back("lm_head.weight");  // a copy of the embedding
    (void)LoadWeights(*model, checkpoint, HuggingFaceMapping(config), options);
    return model;
}

}  // namespace pulsatrix
