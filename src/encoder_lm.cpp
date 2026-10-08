#include "pulsatrix/encoder_lm.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Bad(const std::string& what) { throw std::invalid_argument("ESM config: " + what); }

int64_t Positive(const JsonValue& o, const char* key) {
    const JsonValue* v = o.find(key);
    if (v == nullptr || v->type() != JsonValue::Type::Number) Bad(std::string("needs a number \"") + key + "\"");
    const int64_t n = v->as_int64();
    if (n <= 0) Bad(std::string(key) + " must be positive");
    return n;
}

std::string String(const JsonValue& o, const char* key, const std::string& fallback = "") {
    const JsonValue* v = o.find(key);
    return v == nullptr || v->is_null() ? fallback : v->as_string();
}

bool Bool(const JsonValue& o, const char* key, bool fallback) {
    const JsonValue* v = o.find(key);
    return v == nullptr || v->is_null() ? fallback : v->as_bool();
}

Tensor Reshaped(const Tensor& t, Shape shape) {
    Tensor out(t);
    out.reshape(std::move(shape));
    return out;
}

AttentionConfig EsmAttention(const EncoderLMConfig& c) {
    AttentionConfig a;
    a.d_model = c.hidden_size;
    a.num_heads = c.num_attention_heads;
    a.use_rope = true;
    a.rope_layout = RoPELayout::RotateHalf;
    a.rope_base = c.rope_theta;
    a.rope_inverse_frequencies = c.rope_inverse_frequencies;
    a.qkv_bias = true;
    a.out_bias = true;
    a.causal = false;
    return a;
}

EncoderBlockOptions EsmLayerOptions(const EncoderLMConfig& c) {
    EncoderBlockOptions o;
    o.norm = NormType::LayerNorm;
    o.norm_eps = c.layer_norm_eps;
    o.norm_position = NormPosition::Pre;
    o.mlp = MlpType::Plain;
    o.activation = ElementwiseOp::Gelu;
    o.mlp_bias = true;
    return o;
}

int64_t Checked(int64_t v, const char* what) {
    if (v <= 0) throw std::invalid_argument(std::string("EncoderLM: ") + what + " must be positive");
    return v;
}

}  // namespace

EncoderLMConfig ParseEsmConfig(std::string_view json) {
    const JsonValue o = ParseJson(json);
    if (o.type() != JsonValue::Type::Object) Bad("not a JSON object");
    EncoderLMConfig c;
    if (const JsonValue* a = o.find("architectures"); a != nullptr && !a->is_null() && !a->as_array().empty()) {
        c.architecture = a->as_array().front().as_string();
    }
    c.model_type = String(o, "model_type");
    if (c.model_type != "esm") Bad("model_type is \"" + c.model_type + "\", not \"esm\"");
    if (!c.architecture.empty() && c.architecture != "EsmForMaskedLM" && c.architecture != "EsmModel") {
        Bad("architecture " + c.architecture + " isn't supported (EsmForMaskedLM is)");
    }
    if (Bool(o, "is_folding_model", false)) Bad("ESMFold isn't supported");
    const std::string positions = String(o, "position_embedding_type", "absolute");
    if (positions != "rotary") Bad("position_embedding_type \"" + positions + "\" isn't supported yet (ESM-2's \"rotary\" is)");
    if (Bool(o, "emb_layer_norm_before", false)) Bad("emb_layer_norm_before isn't supported yet (ESM-2 has none)");
    const std::string act = String(o, "hidden_act", "gelu");
    if (act != "gelu") Bad("hidden_act \"" + act + "\" isn't supported (ESM's \"gelu\" is)");
    c.vocab_size = Positive(o, "vocab_size");
    c.hidden_size = Positive(o, "hidden_size");
    c.intermediate_size = Positive(o, "intermediate_size");
    c.num_hidden_layers = Positive(o, "num_hidden_layers");
    c.num_attention_heads = Positive(o, "num_attention_heads");
    if (c.hidden_size % c.num_attention_heads != 0) Bad("hidden_size isn't a multiple of num_attention_heads");
    if (const JsonValue* v = o.find("layer_norm_eps"); v != nullptr && !v->is_null()) c.layer_norm_eps = v->as_float();
    if (const JsonValue* v = o.find("rope_theta"); v != nullptr && !v->is_null()) c.rope_theta = v->as_float();
    if (const JsonValue* v = o.find("max_position_embeddings"); v != nullptr && !v->is_null()) c.max_position_embeddings = v->as_int64();
    c.token_dropout = Bool(o, "token_dropout", false);
    if (const JsonValue* v = o.find("mask_token_id"); v != nullptr && !v->is_null()) c.mask_token_id = v->as_int64();
    if (const JsonValue* v = o.find("pad_token_id"); v != nullptr && !v->is_null()) c.pad_token_id = v->as_int64();
    if (c.token_dropout && (c.mask_token_id < 0 || c.mask_token_id >= c.vocab_size)) Bad("token_dropout needs a mask_token_id in the vocabulary");
    c.dtype = String(o, "torch_dtype", String(o, "dtype"));
    return c;
}

EncoderLMConfig ReadEsmConfig(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadEsmConfig: can't read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ParseEsmConfig(ss.str());
}

EncoderLM::EncoderLM(const EncoderLMConfig& config, DeviceBackend* backend)
    : config_(config),
      backend_(backend),
      embed_(Checked(config.vocab_size, "vocab_size"), Checked(config.hidden_size, "hidden_size"), backend),
      norm_(config.hidden_size, backend, backend->device(), config.layer_norm_eps),
      lm_dense_(config.hidden_size, config.hidden_size, backend),
      lm_gelu_(ElementwiseOp::Gelu, backend),
      lm_norm_(config.hidden_size, backend, backend->device(), config.layer_norm_eps),
      lm_decoder_(embed_, backend),
      lm_bias_(Shape({config.vocab_size}), backend),
      lm_bias_grad_(Shape({config.vocab_size}), backend),
      embed_scale_(Shape({0}), backend),
      last_logits_nobias_(Shape({0}), backend),
      last_hidden_(Shape({0}), backend) {
    for (int64_t i = 0; i < Checked(config.num_hidden_layers, "num_hidden_layers"); ++i) {
        layers_.push_back(std::make_unique<EncoderBlock>(EsmAttention(config), Checked(config.intermediate_size, "intermediate_size"), backend,
                                                         EsmLayerOptions(config)));
    }
}

void EncoderLM::set_padding_mask(const Tensor& keep) {
    if (keep.rank() != 2) throw std::invalid_argument("EncoderLM::set_padding_mask: the mask must be (N, L)");
    std::vector<float> v = keep.to_host_vector();
    for (float x : v) {
        if (x != 0.0f && x != 1.0f) throw std::invalid_argument("EncoderLM::set_padding_mask: values must be 0 or 1");
    }
    padding_keep_ = std::move(v);
    padding_shape_ = keep.shape();
    for (auto& l : layers_) l->mha().set_key_padding_mask(keep);
}

void EncoderLM::clear_padding_mask() {
    padding_keep_.clear();
    padding_shape_ = Shape({0});
    for (auto& l : layers_) l->mha().clear_key_padding_mask();
}

// LinearModule and the norms take (rows, width); the model's tensors are (N, L, width).
Tensor EncoderLM::Rows(Module& m, const Tensor& x, int64_t width) const {
    Shape out_shape = x.shape();
    Tensor y = m.forward(Reshaped(x, Shape({x.numel() / x.shape().dim(x.rank() - 1), x.shape().dim(x.rank() - 1)})));
    std::vector<int64_t> dims;
    for (int64_t d = 0; d + 1 < x.rank(); ++d) dims.push_back(x.shape().dim(static_cast<size_t>(d)));
    dims.push_back(width);
    return Reshaped(y, Shape(dims));
}

Tensor EncoderLM::RowsBackward(Module& m, const Tensor& g, int64_t width) const {
    Tensor r = m.backward(Reshaped(g, Shape({g.numel() / g.shape().dim(g.rank() - 1), g.shape().dim(g.rank() - 1)})));
    std::vector<int64_t> dims;
    for (int64_t d = 0; d + 1 < g.rank(); ++d) dims.push_back(g.shape().dim(static_cast<size_t>(d)));
    dims.push_back(width);
    return Reshaped(r, Shape(dims));
}

Tensor EncoderLM::RowsRelevance(Module& m, const Tensor& rel, int64_t width, const LRPRuleConfig& config) const {
    Tensor r = m.propagate_relevance(Reshaped(rel, Shape({rel.numel() / rel.shape().dim(rel.rank() - 1), rel.shape().dim(rel.rank() - 1)})), config);
    std::vector<int64_t> dims;
    for (int64_t d = 0; d + 1 < rel.rank(); ++d) dims.push_back(rel.shape().dim(static_cast<size_t>(d)));
    dims.push_back(width);
    return Reshaped(r, Shape(dims));
}

Tensor EncoderLM::forward_impl(const Tensor& input) {
    if (input.rank() != 2) throw std::invalid_argument("EncoderLM::forward: input must be (N, L) token ids");
    const int64_t N = input.shape().dim(0), L = input.shape().dim(1), h = config_.hidden_size, V = config_.vocab_size;
    const std::vector<float> ids = input.to_host_vector();
    for (float id : ids) {
        if (id < 0 || id >= static_cast<float>(V) || id != std::floor(id)) throw std::invalid_argument("EncoderLM::forward: a token id is outside the vocabulary");
    }
    if (!padding_keep_.empty() && padding_shape_ != input.shape()) {
        throw std::invalid_argument("EncoderLM::forward: the padding mask doesn't match the input");
    }
    // Token dropout and padding: one multiplier per (n, l), as EsmEmbeddings applies them.
    std::vector<float> scale(static_cast<size_t>(N * L * h));
    for (int64_t n = 0; n < N; ++n) {
        double length = 0, masked = 0;
        for (int64_t l = 0; l < L; ++l) {
            const size_t k = static_cast<size_t>(n * L + l);
            length += padding_keep_.empty() ? 1.0 : padding_keep_[k];
            masked += config_.token_dropout && ids[k] == static_cast<float>(config_.mask_token_id) ? 1.0 : 0.0;
        }
        const double row = config_.token_dropout ? (1.0 - 0.15 * 0.8) / (1.0 - masked / length) : 1.0;
        for (int64_t l = 0; l < L; ++l) {
            const size_t k = static_cast<size_t>(n * L + l);
            const bool dropped = config_.token_dropout && ids[k] == static_cast<float>(config_.mask_token_id);
            const float m = static_cast<float>((dropped ? 0.0 : row) * (padding_keep_.empty() ? 1.0 : padding_keep_[k]));
            std::fill_n(scale.begin() + static_cast<std::ptrdiff_t>(k * h), h, m);
        }
    }
    embed_scale_ = Tensor(Shape({N, L, h}), backend_, scale, backend_->device());
    const Tensor raw = embed_.forward(input);
    Tensor x(Shape({N, L, h}), backend_);
    backend_->mul(raw.data(), embed_scale_.data(), x.data(), static_cast<size_t>(x.numel()));

    hidden_.clear();
    if (keep_activations_) hidden_.push_back(x);
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        x = layer->forward(x);
        if (attention_observer_) attention_observer_(static_cast<int64_t>(i), layer->mha().last_attention_weights());
        if (keep_activations_) {
            hidden_.push_back(x);
        } else {
            layer->release_activations();
        }
    }
    last_hidden_ = Rows(norm_, x, h);
    const Tensor head = Rows(lm_norm_, lm_gelu_.forward(Rows(lm_dense_, last_hidden_, h)), h);
    last_logits_nobias_ = Reshaped(lm_decoder_.forward(Reshaped(head, Shape({N * L, h}))), Shape({N * L, V}));
    Tensor logits(Shape({N * L, V}), backend_);
    backend_->add_row_vector(last_logits_nobias_.data(), lm_bias_.data(), logits.data(), static_cast<size_t>(N * L), static_cast<size_t>(V));
    last_ids_shape_ = input.shape();
    has_forwarded_ = true;
    if (!keep_activations_) {
        const Tensor kept = last_hidden_;  // (N, L, hidden): small next to a layer's activations
        release_activations();
        last_hidden_ = kept;
    }
    return Reshaped(logits, Shape({N, L, V}));
}

Tensor EncoderLM::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "EncoderLM::backward");
    if (!has_forwarded_) throw std::logic_error("EncoderLM::backward: called before any forward()");
    const int64_t N = last_ids_shape_.dim(0), L = last_ids_shape_.dim(1), h = config_.hidden_size, V = config_.vocab_size;
    if (grad_output.shape() != Shape({N, L, V})) throw std::invalid_argument("EncoderLM::backward: grad_output must be (N, L, vocab)");
    const Tensor g_flat = Reshaped(grad_output, Shape({N * L, V}));
    if (lm_bias_.requires_grad()) {
        backend_->column_sums(g_flat.data(), lm_bias_grad_.data(), static_cast<size_t>(N * L), static_cast<size_t>(V), 1.0f);
    }
    Tensor g = Reshaped(lm_decoder_.backward(g_flat), Shape({N, L, h}));
    g = RowsBackward(lm_dense_, lm_gelu_.backward(RowsBackward(lm_norm_, g, h)), h);
    g = RowsBackward(norm_, g, h);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) g = (*it)->backward(g);
    Tensor g_embed(g.shape(), backend_);
    backend_->mul(g.data(), embed_scale_.data(), g_embed.data(), static_cast<size_t>(g.numel()));
    return embed_.backward(g_embed);
}

Tensor EncoderLM::HeadRelevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) throw std::logic_error("EncoderLM::propagate_relevance: called before any forward()");
    const int64_t N = last_ids_shape_.dim(0), L = last_ids_shape_.dim(1), h = config_.hidden_size, V = config_.vocab_size;
    if (relevance_out.numel() != N * L * V) throw std::invalid_argument("EncoderLM::propagate_relevance: relevance must be (N, L, vocab)");
    // logits = decoder(x) + bias: the bias takes its epsilon-rule share, as in a biased linear layer.
    const Tensor r = Reshaped(relevance_out, Shape({N * L, V}));
    Tensor bias_rows(Shape({N * L, V}), backend_), zeros(Shape({N * L, V}), backend_);
    backend_->add_row_vector(zeros.data(), lm_bias_.data(), bias_rows.data(), static_cast<size_t>(N * L), static_cast<size_t>(V));
    Tensor r_decoder(r.shape(), backend_), r_bias(r.shape(), backend_);
    backend_->lrp_residual_split(last_logits_nobias_.data(), bias_rows.data(), r.data(), r_decoder.data(), r_bias.data(),
                                 static_cast<size_t>(r.numel()), config.epsilon);
    Tensor rel = Reshaped(lm_decoder_.propagate_relevance(r_decoder, config), Shape({N, L, h}));
    rel = RowsRelevance(lm_dense_, lm_gelu_.propagate_relevance(RowsRelevance(lm_norm_, rel, h, config), config), h, config);
    return RowsRelevance(norm_, rel, h, config);
}

std::vector<Tensor> EncoderLM::propagate_relevance_by_layer(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "EncoderLM::propagate_relevance_by_layer");
    Tensor r = HeadRelevance(relevance_out, config);
    std::vector<Tensor> boundaries(layers_.size() + 1, r);
    for (size_t i = layers_.size(); i-- > 0;) {
        boundaries[i + 1] = r;
        r = layers_[i]->propagate_relevance(r, config);
    }
    boundaries[0] = r;
    return boundaries;
}

Tensor EncoderLM::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    // The token-dropout scale is a constant: relevance passes through it to the embeddings.
    return embed_.propagate_relevance(propagate_relevance_by_layer(relevance_out, config).front(), config);
}

std::vector<NamedParamRef> EncoderLM::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "embed_tokens", embed_);
    for (size_t i = 0; i < layers_.size(); ++i) append_named_parameters(out, "layers." + std::to_string(i), *layers_[i]);
    append_named_parameters(out, "norm", norm_);
    append_named_parameters(out, "lm_head.dense", lm_dense_);
    append_named_parameters(out, "lm_head.norm", lm_norm_);
    out.push_back({"lm_head.bias", {&lm_bias_, &lm_bias_grad_}});
    return out;
}

std::vector<NamedBufferRef> EncoderLM::named_buffers() {
    std::vector<NamedBufferRef> out;
    for (size_t i = 0; i < layers_.size(); ++i) append_named_buffers(out, "layers." + std::to_string(i), *layers_[i]);
    return out;
}

void EncoderLM::set_training(bool training) {
    Module::set_training(training);
    embed_.set_training(training);
    for (auto& l : layers_) l->set_training(training);
    norm_.set_training(training);
    lm_dense_.set_training(training);
    lm_gelu_.set_training(training);
    lm_norm_.set_training(training);
    lm_decoder_.set_training(training);
}

std::vector<WeightMapping> EsmMapping(const EncoderLMConfig& c) {
    using T = WeightTransform;
    std::vector<WeightMapping> m;
    auto linear = [&](const std::string& hf, const std::string& ours) {
        m.push_back({hf + ".weight", ours + ".weight", T::Transpose});
        m.push_back({hf + ".bias", ours + ".bias"});
    };
    auto norm = [&](const std::string& hf, const std::string& ours) {
        m.push_back({hf + ".weight", ours + ".weight"});
        m.push_back({hf + ".bias", ours + ".bias"});
    };
    m.push_back({"esm.embeddings.word_embeddings.weight", "embed_tokens.weight"});
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string hf = "esm.encoder.layer." + std::to_string(i) + ".", px = "layers." + std::to_string(i) + ".";
        linear(hf + "attention.self.query", px + "mha.q_proj");
        linear(hf + "attention.self.key", px + "mha.k_proj");
        linear(hf + "attention.self.value", px + "mha.v_proj");
        linear(hf + "attention.output.dense", px + "mha.out_proj");
        norm(hf + "attention.LayerNorm", px + "norm1");
        linear(hf + "intermediate.dense", px + "mlp.fc1");
        linear(hf + "output.dense", px + "mlp.fc2");
        norm(hf + "LayerNorm", px + "norm2");
    }
    norm("esm.encoder.emb_layer_norm_after", "norm");
    linear("lm_head.dense", "lm_head.dense");
    norm("lm_head.layer_norm", "lm_head.norm");
    m.push_back({"lm_head.bias", "lm_head.bias"});
    return m;
}

std::vector<std::string> EsmIgnoredTensors(const EncoderLMConfig& c) {
    std::vector<std::string> ignore = {"esm.embeddings.position_embeddings.weight", "esm.embeddings.position_ids",
                                       "esm.contact_head.regression.weight", "esm.contact_head.regression.bias", "lm_head.decoder.weight",
                                       "esm.rotary_embeddings.inv_freq",
                                       // transformers 5 saves the layers' shared buffer once, under a wildcard name
                                       "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq"};
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
        ignore.push_back("esm.encoder.layer." + std::to_string(i) + ".attention.self.rotary_embeddings.inv_freq");
    }
    return ignore;
}

std::unique_ptr<EncoderLM> LoadEncoderLM(const std::string& directory, DeviceBackend* backend) {
    EncoderLMConfig config = ReadEsmConfig((std::filesystem::path(directory) / "config.json").string());
    const HfCheckpoint checkpoint = HfCheckpoint::Open(directory);
    // The stored RoPE frequencies, wherever this checkpoint keeps them; every layer's are the same.
    CPUBackend host;
    for (const std::string& name : {std::string("esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq"),
                                    std::string("esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq"),
                                    std::string("esm.rotary_embeddings.inv_freq")}) {
        if (!checkpoint.contains(name)) continue;
        const std::vector<float> f = checkpoint.tensor(name, &host).to_host_vector();
        if (static_cast<int64_t>(f.size()) * 2 != config.hidden_size / config.num_attention_heads) {
            throw std::invalid_argument("LoadEncoderLM: " + name + " doesn't hold one frequency per pair of a head's dimensions");
        }
        config.rope_inverse_frequencies.assign(f.begin(), f.end());
        break;
    }
    auto model = std::make_unique<EncoderLM>(config, backend);
    WeightLoadOptions options;
    options.ignore = EsmIgnoredTensors(config);
    std::vector<WeightMapping> mapping = EsmMapping(config);
    // transformers 5 saves modules named "LayerNorm" with the legacy names gamma and beta (and
    // reads either); older checkpoints, such as facebook/esm2_*, use weight and bias.
    for (WeightMapping& m : mapping) {
        if (checkpoint.contains(m.source) || m.source.find("LayerNorm.") == std::string::npos) continue;
        const bool weight = m.source.size() >= 7 && m.source.compare(m.source.size() - 7, 7, ".weight") == 0;
        const std::string legacy = m.source.substr(0, m.source.rfind('.')) + (weight ? ".gamma" : ".beta");
        if (checkpoint.contains(legacy)) m.source = legacy;
    }
    (void)LoadWeights(*model, checkpoint, mapping, options);
    return model;
}

void EncoderLM::release_activations() {
    embed_.release_activations();
    for (auto& layer : layers_) layer->release_activations();
    for (Module* m : std::initializer_list<Module*>{&norm_, &lm_dense_, &lm_gelu_, &lm_norm_, &lm_decoder_}) m->release_activations();
    for (Tensor* t : {&embed_scale_, &last_logits_nobias_, &last_hidden_}) release_tensor(*t);
    hidden_.clear();
    has_forwarded_ = false;
}

}  // namespace pulsatrix
