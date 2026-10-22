#include "pulsatrix/encoder_lm.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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

[[noreturn]] void BadMdlm(const std::string& what) { throw std::invalid_argument("MDLM config: " + what); }

AttentionConfig EncoderAttention(const EncoderLMConfig& c) {
    AttentionConfig a;
    a.d_model = c.hidden_size;
    a.num_heads = c.num_attention_heads;
    a.use_rope = true;
    a.rope_layout = RoPELayout::RotateHalf;
    a.rope_base = c.rope_theta;
    a.rope_inverse_frequencies = c.rope_inverse_frequencies;
    a.qkv_bias = c.attention_bias;
    a.out_bias = c.attention_bias;
    a.causal = false;
    return a;
}

EncoderBlockOptions EncoderLayerOptions(const EncoderLMConfig& c) {
    EncoderBlockOptions o;
    o.norm = NormType::LayerNorm;
    o.norm_eps = c.layer_norm_eps;
    o.norm_position = NormPosition::Pre;
    o.mlp = MlpType::Plain;
    o.activation = c.activation;
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
    // DPLM (DIFF-4) is EsmForMaskedLM with a diffusion objective: same weights, same forward.
    if (!c.architecture.empty() && c.architecture != "EsmForMaskedLM" && c.architecture != "EsmModel" && c.architecture != "EsmForDPLM") {
        Bad("architecture " + c.architecture + " isn't supported (EsmForMaskedLM and EsmForDPLM are)");
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

EncoderLMConfig ParseMdlmConfig(std::string_view json) {
    const JsonValue o = ParseJson(json);
    if (o.type() != JsonValue::Type::Object) BadMdlm("not a JSON object");
    EncoderLMConfig c;
    if (const JsonValue* a = o.find("architectures"); a != nullptr && !a->is_null() && !a->as_array().empty()) {
        c.architecture = a->as_array().front().as_string();
    }
    c.model_type = String(o, "model_type");
    if (c.model_type != "mdlm") BadMdlm("model_type is \"" + c.model_type + "\", not \"mdlm\"");
    if (!c.architecture.empty() && c.architecture != "MDLM") BadMdlm("architecture " + c.architecture + " isn't supported (MDLM is)");
    if (Bool(o, "time_conditioning", false)) {
        BadMdlm("time_conditioning makes the adaLN modulation depend on the noise level, so it can't be folded into the weights");
    }
    const auto positive = [&](const char* key) {
        const JsonValue* v = o.find(key);
        if (v == nullptr || v->type() != JsonValue::Type::Number) BadMdlm(std::string("needs a number \"") + key + "\"");
        const int64_t n = v->as_int64();
        if (n <= 0) BadMdlm(std::string(key) + " must be positive");
        return n;
    };
    c.vocab_size = positive("vocab_size");
    c.hidden_size = positive("hidden_dim");
    c.num_hidden_layers = positive("n_blocks");
    c.num_attention_heads = positive("n_heads");
    (void)positive("cond_dim");
    if (c.hidden_size % c.num_attention_heads != 0) BadMdlm("hidden_dim isn't a multiple of n_heads");
    if ((c.hidden_size / c.num_attention_heads) % 2 != 0) BadMdlm("a head's width must be even for rotary embeddings");
    c.intermediate_size = 4 * c.hidden_size;  // DDiTBlock's fixed mlp_ratio
    c.max_position_embeddings = positive("model_length");
    c.layer_norm_eps = 1e-5f;  // F.layer_norm's default
    c.rope_theta = 10000.0f;
    c.mask_token_id = c.vocab_size - 1;
    c.token_dropout = false;
    c.head = EncoderLMHead::Linear;
    c.attention_bias = false;
    c.activation = ElementwiseOp::GeluTanh;
    c.dtype = String(o, "torch_dtype", String(o, "dtype"));
    return c;
}

EncoderLM::EncoderLM(const EncoderLMConfig& config, DeviceBackend* backend)
    : config_(config),
      backend_(backend),
      embed_(Checked(config.vocab_size, "vocab_size"), Checked(config.hidden_size, "hidden_size"), backend),
      norm_(config.hidden_size, backend, backend->device(), config.layer_norm_eps),
      lm_bias_(Shape({config.head == EncoderLMHead::Esm ? config.vocab_size : 0}), backend),
      lm_bias_grad_(Shape({config.head == EncoderLMHead::Esm ? config.vocab_size : 0}), backend),
      embed_scale_(Shape({0}), backend),
      last_logits_nobias_(Shape({0}), backend),
      last_hidden_(Shape({0}), backend) {
    for (int64_t i = 0; i < Checked(config.num_hidden_layers, "num_hidden_layers"); ++i) {
        layers_.push_back(std::make_unique<EncoderBlock>(EncoderAttention(config), Checked(config.intermediate_size, "intermediate_size"), backend,
                                                         EncoderLayerOptions(config)));
    }
    if (config.head == EncoderLMHead::Esm) {
        lm_dense_ = std::make_unique<LinearModule>(config.hidden_size, config.hidden_size, backend);
        lm_gelu_ = std::make_unique<ActivationModule>(ElementwiseOp::Gelu, backend);
        lm_norm_ = std::make_unique<LayerNormModule>(config.hidden_size, backend, backend->device(), config.layer_norm_eps);
        lm_decoder_ = std::make_unique<TiedLMHeadModule>(embed_, backend);
    } else {
        lm_linear_ = std::make_unique<LinearModule>(config.hidden_size, config.vocab_size, backend);
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
    if (hidden_state_hook_) x = detail::ApplyHiddenStateHook(hidden_state_hook_, 0, x);
    if (keep_activations_) hidden_.push_back(x);
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        x = layer->forward(x);
        if (attention_observer_) attention_observer_(static_cast<int64_t>(i), layer->mha().last_attention_weights());
        if (hidden_state_hook_) x = detail::ApplyHiddenStateHook(hidden_state_hook_, static_cast<int64_t>(i) + 1, x);
        if (keep_activations_) {
            hidden_.push_back(x);
        } else {
            layer->release_activations();
        }
    }
    last_hidden_ = Rows(norm_, x, h);
    Tensor logits(Shape({0}), backend_);
    if (lm_linear_) {
        logits = Reshaped(Rows(*lm_linear_, last_hidden_, V), Shape({N * L, V}));
    } else {
        logits = Tensor(Shape({N * L, V}), backend_);
        const Tensor head = Rows(*lm_norm_, lm_gelu_->forward(Rows(*lm_dense_, last_hidden_, h)), h);
        last_logits_nobias_ = Reshaped(lm_decoder_->forward(Reshaped(head, Shape({N * L, h}))), Shape({N * L, V}));
        backend_->add_row_vector(last_logits_nobias_.data(), lm_bias_.data(), logits.data(), static_cast<size_t>(N * L), static_cast<size_t>(V));
    }
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
    if (lm_linear_) return backward_hidden(RowsBackward(*lm_linear_, grad_output, h));
    const Tensor g_flat = Reshaped(grad_output, Shape({N * L, V}));
    if (lm_bias_.requires_grad()) {
        backend_->column_sums(g_flat.data(), lm_bias_grad_.data(), static_cast<size_t>(N * L), static_cast<size_t>(V), 1.0f);
    }
    Tensor g = Reshaped(lm_decoder_->backward(g_flat), Shape({N, L, h}));
    return backward_hidden(RowsBackward(*lm_dense_, lm_gelu_->backward(RowsBackward(*lm_norm_, g, h)), h));
}

Tensor EncoderLM::backward_hidden(const Tensor& grad_hidden) {
    require_device(grad_hidden, *compute_device(), "EncoderLM::backward_hidden");
    if (!has_forwarded_) throw std::logic_error("EncoderLM::backward_hidden: called before any forward()");
    const int64_t N = last_ids_shape_.dim(0), L = last_ids_shape_.dim(1), h = config_.hidden_size;
    if (grad_hidden.numel() != N * L * h) throw std::invalid_argument("EncoderLM::backward_hidden: the gradient must be (N, L, hidden_size)");
    Tensor g = RowsBackward(norm_, Reshaped(grad_hidden, Shape({N, L, h})), h);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) g = (*it)->backward(g);
    Tensor g_embed(g.shape(), backend_);
    backend_->mul(g.data(), embed_scale_.data(), g_embed.data(), static_cast<size_t>(g.numel()));
    return embed_.backward(g_embed);
}

Tensor EncoderLM::HeadRelevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) throw std::logic_error("EncoderLM::propagate_relevance: called before any forward()");
    const int64_t N = last_ids_shape_.dim(0), L = last_ids_shape_.dim(1), h = config_.hidden_size, V = config_.vocab_size;
    if (relevance_out.numel() != N * L * V) throw std::invalid_argument("EncoderLM::propagate_relevance: relevance must be (N, L, vocab)");
    if (lm_linear_) return RowsRelevance(*lm_linear_, Reshaped(relevance_out, Shape({N, L, V})), h, config);
    // logits = decoder(x) + bias: the bias takes its epsilon-rule share, as in a biased linear layer.
    const Tensor r = Reshaped(relevance_out, Shape({N * L, V}));
    Tensor bias_rows(Shape({N * L, V}), backend_), zeros(Shape({N * L, V}), backend_);
    backend_->add_row_vector(zeros.data(), lm_bias_.data(), bias_rows.data(), static_cast<size_t>(N * L), static_cast<size_t>(V));
    Tensor r_decoder(r.shape(), backend_), r_bias(r.shape(), backend_);
    backend_->lrp_residual_split(last_logits_nobias_.data(), bias_rows.data(), r.data(), r_decoder.data(), r_bias.data(),
                                 static_cast<size_t>(r.numel()), config.epsilon);
    Tensor rel = Reshaped(lm_decoder_->propagate_relevance(r_decoder, config), Shape({N, L, h}));
    return RowsRelevance(*lm_dense_, lm_gelu_->propagate_relevance(RowsRelevance(*lm_norm_, rel, h, config), config), h, config);
}

std::vector<Tensor> EncoderLM::propagate_relevance_by_layer(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "EncoderLM::propagate_relevance_by_layer");
    return propagate_hidden_relevance_by_layer(HeadRelevance(relevance_out, config), config);
}

std::vector<Tensor> EncoderLM::propagate_hidden_relevance_by_layer(const Tensor& relevance_hidden, const LRPRuleConfig& config) {
    require_device(relevance_hidden, *compute_device(), "EncoderLM::propagate_hidden_relevance_by_layer");
    if (!has_forwarded_) throw std::logic_error("EncoderLM::propagate_hidden_relevance_by_layer: called before any forward()");
    const int64_t N = last_ids_shape_.dim(0), L = last_ids_shape_.dim(1), h = config_.hidden_size;
    if (relevance_hidden.numel() != N * L * h) {
        throw std::invalid_argument("EncoderLM::propagate_hidden_relevance_by_layer: relevance must be (N, L, hidden_size)");
    }
    Tensor r = RowsRelevance(norm_, Reshaped(relevance_hidden, Shape({N, L, h})), h, config);
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

std::vector<Module*> EncoderLM::HeadModules() {
    if (lm_linear_) return {lm_linear_.get()};
    return {lm_dense_.get(), lm_gelu_.get(), lm_norm_.get(), lm_decoder_.get()};
}

std::vector<NamedParamRef> EncoderLM::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "embed_tokens", embed_);
    for (size_t i = 0; i < layers_.size(); ++i) append_named_parameters(out, "layers." + std::to_string(i), *layers_[i]);
    append_named_parameters(out, "norm", norm_);
    if (lm_linear_) {
        append_named_parameters(out, "lm_head", *lm_linear_);
        return out;
    }
    append_named_parameters(out, "lm_head.dense", *lm_dense_);
    append_named_parameters(out, "lm_head.norm", *lm_norm_);
    out.push_back({"lm_head.bias", {&lm_bias_, &lm_bias_grad_}});
    return out;
}

std::vector<NamedModuleRef> EncoderLM::named_children() {
    std::vector<NamedModuleRef> out{{"embed_tokens", &embed_}};
    for (size_t i = 0; i < layers_.size(); ++i) out.push_back({"layers." + std::to_string(i), layers_[i].get()});
    out.push_back({"norm", &norm_});
    if (lm_linear_) {
        out.push_back({"lm_head", lm_linear_.get()});
        return out;
    }
    out.push_back({"lm_head.dense", lm_dense_.get()});
    out.push_back({"lm_head.norm", lm_norm_.get()});
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
    for (Module* m : HeadModules()) m->set_training(training);
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

std::vector<WeightMapping> MdlmMapping(const EncoderLMConfig& c) {
    using T = WeightTransform;
    const int64_t h = c.hidden_size;
    std::vector<WeightMapping> m;
    m.push_back({"backbone.vocab_embed.embedding", "embed_tokens.weight"});
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string src = "backbone.blocks." + std::to_string(i) + ".", px = "layers." + std::to_string(i) + ".";
        // attn_qkv's rows are q, then k, then v, each (heads x head width).
        m.push_back({src + "attn_qkv.weight", px + "mha.q_proj.weight", T::Transpose, 0, h});
        m.push_back({src + "attn_qkv.weight", px + "mha.k_proj.weight", T::Transpose, h, h});
        m.push_back({src + "attn_qkv.weight", px + "mha.v_proj.weight", T::Transpose, 2 * h, h});
        m.push_back({src + "attn_out.weight", px + "mha.out_proj.weight", T::Transpose});
        m.push_back({src + "norm1.weight", px + "norm1.weight"});
        m.push_back({src + "norm2.weight", px + "norm2.weight"});
        m.push_back({src + "mlp.0.weight", px + "mlp.fc1.weight", T::Transpose});
        m.push_back({src + "mlp.0.bias", px + "mlp.fc1.bias"});
        m.push_back({src + "mlp.2.weight", px + "mlp.fc2.weight", T::Transpose});
        m.push_back({src + "mlp.2.bias", px + "mlp.fc2.bias"});
    }
    m.push_back({"backbone.output_layer.norm_final.weight", "norm.weight"});
    m.push_back({"backbone.output_layer.linear.weight", "lm_head.weight", T::Transpose});
    m.push_back({"backbone.output_layer.linear.bias", "lm_head.bias"});
    return m;
}

namespace {

/** @brief y = W x + b for a PyTorch `nn.Linear` stored (out, in), in double. */
std::vector<double> Affine(const std::vector<float>& w, const std::vector<float>& b, const std::vector<double>& x) {
    const size_t in = x.size(), out = b.size();
    if (w.size() != in * out) throw std::invalid_argument("LoadEncoderLM: an MDLM conditioning layer has the wrong shape");
    std::vector<double> y(out);
    for (size_t o = 0; o < out; ++o) {
        double acc = b[o];
        for (size_t i = 0; i < in; ++i) acc += static_cast<double>(w[o * in + i]) * x[i];
        y[o] = acc;
    }
    return y;
}

double Silu(double v) { return v / (1.0 + std::exp(-v)); }

/**
 * @brief Folds MDLM's constant adaLN-Zero modulation into @p model's weights (see LoadEncoderLM).
 *        The norms' weights must already hold the checkpoint's, and their biases are set here.
 */
void FoldMdlmModulation(EncoderLM& model, const HfCheckpoint& checkpoint) {
    CPUBackend host;
    const auto read = [&](const std::string& name) { return checkpoint.tensor(name, &host).to_host_vector(); };
    // c = silu(sigma_map(0)): the frequency embedding of t = 0 is cos(0) = 1 then sin(0) = 0.
    const std::vector<float> w0 = read("backbone.sigma_map.mlp.0.weight"), b0 = read("backbone.sigma_map.mlp.0.bias");
    const size_t frequencies = w0.size() / b0.size();
    std::vector<double> t_freq(frequencies, 0.0);
    std::fill_n(t_freq.begin(), frequencies / 2, 1.0);
    std::vector<double> c = Affine(w0, b0, t_freq);
    for (double& v : c) v = Silu(v);
    c = Affine(read("backbone.sigma_map.mlp.2.weight"), read("backbone.sigma_map.mlp.2.bias"), c);
    for (double& v : c) v = Silu(v);

    std::vector<NamedParamRef> params = model.named_parameters();
    const auto param = [&](const std::string& name) -> Tensor& {
        for (NamedParamRef& p : params) {
            if (p.name == name) return *p.ref.value;
        }
        throw std::logic_error("LoadEncoderLM: no parameter " + name);
    };
    const auto set = [&](Tensor& t, const std::vector<double>& v) {
        t = Tensor(t.shape(), t.backend(), std::vector<float>(v.begin(), v.end()), t.device());
    };
    const int64_t h = model.config().hidden_size;
    const auto chunk = [&](const std::vector<double>& mod, int64_t k) {
        return std::vector<double>(mod.begin() + k * h, mod.begin() + (k + 1) * h);
    };
    // LN(x)·w·(1 + scale) + shift: a LayerNorm with weight w·(1 + scale) and bias shift.
    const auto fold_norm = [&](const std::string& norm, const std::vector<double>& shift, const std::vector<double>& scale) {
        Tensor& weight = param(norm + ".weight");
        const std::vector<float> w = weight.to_host_vector();
        std::vector<double> folded(w.size());
        for (size_t j = 0; j < w.size(); ++j) folded[j] = w[j] * (1.0 + scale[j]);
        set(weight, folded);
        set(param(norm + ".bias"), shift);
    };
    // gate · (h W + b): LinearModule stores W as (in, out), so the gate scales each column.
    const auto fold_gate = [&](const std::string& linear, const std::vector<double>& gate, bool has_bias) {
        Tensor& weight = param(linear + ".weight");
        const std::vector<float> w = weight.to_host_vector();
        std::vector<double> folded(w.size());
        for (size_t k = 0; k < w.size(); ++k) folded[k] = w[k] * gate[k % gate.size()];
        set(weight, folded);
        if (!has_bias) return;
        Tensor& bias = param(linear + ".bias");
        const std::vector<float> b = bias.to_host_vector();
        std::vector<double> scaled(b.size());
        for (size_t j = 0; j < b.size(); ++j) scaled[j] = b[j] * gate[j];
        set(bias, scaled);
    };
    for (int64_t i = 0; i < model.num_layers(); ++i) {
        const std::string src = "backbone.blocks." + std::to_string(i) + ".adaLN_modulation.", px = "layers." + std::to_string(i) + ".";
        const std::vector<double> mod = Affine(read(src + "weight"), read(src + "bias"), c);
        if (static_cast<int64_t>(mod.size()) != 6 * h) throw std::invalid_argument("LoadEncoderLM: an MDLM block's adaLN modulation isn't 6 x hidden");
        // DDiTBlock's order: shift, scale, gate for attention, then for the MLP.
        fold_norm(px + "norm1", chunk(mod, 0), chunk(mod, 1));
        fold_gate(px + "mha.out_proj", chunk(mod, 2), false);
        fold_norm(px + "norm2", chunk(mod, 3), chunk(mod, 4));
        fold_gate(px + "mlp.fc2", chunk(mod, 5), true);
    }
    const std::vector<double> mod = Affine(read("backbone.output_layer.adaLN_modulation.weight"), read("backbone.output_layer.adaLN_modulation.bias"), c);
    if (static_cast<int64_t>(mod.size()) != 2 * h) throw std::invalid_argument("LoadEncoderLM: MDLM's final adaLN modulation isn't 2 x hidden");
    fold_norm("norm", chunk(mod, 0), chunk(mod, 1));
}

std::unique_ptr<EncoderLM> LoadMdlm(const std::string& directory, const std::string& config_json, DeviceBackend* backend) {
    EncoderLMConfig config = ParseMdlmConfig(config_json);
    const HfCheckpoint checkpoint = HfCheckpoint::Open(directory);
    CPUBackend host;
    if (const std::string name = "backbone.rotary_emb.inv_freq"; checkpoint.contains(name)) {
        const std::vector<float> f = checkpoint.tensor(name, &host).to_host_vector();
        if (static_cast<int64_t>(f.size()) * 2 != config.hidden_size / config.num_attention_heads) {
            throw std::invalid_argument("LoadEncoderLM: " + name + " doesn't hold one frequency per pair of a head's dimensions");
        }
        config.rope_inverse_frequencies.assign(f.begin(), f.end());
    }
    auto model = std::make_unique<EncoderLM>(config, backend);
    WeightLoadOptions options;
    options.strict = false;  // the norms' biases come from the modulation, below
    options.ignore = {"backbone.rotary_emb.inv_freq", "backbone.sigma_map.mlp.0.weight", "backbone.sigma_map.mlp.0.bias",
                      "backbone.sigma_map.mlp.2.weight", "backbone.sigma_map.mlp.2.bias", "backbone.output_layer.adaLN_modulation.weight",
                      "backbone.output_layer.adaLN_modulation.bias"};
    std::vector<std::string> folded = {"norm.bias"};
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string src = "backbone.blocks." + std::to_string(i) + ".", px = "layers." + std::to_string(i) + ".";
        options.ignore.push_back(src + "adaLN_modulation.weight");
        options.ignore.push_back(src + "adaLN_modulation.bias");
        folded.push_back(px + "norm1.bias");
        folded.push_back(px + "norm2.bias");
    }
    const WeightLoadReport report = LoadWeights(*model, checkpoint, MdlmMapping(config), options);
    std::vector<std::string> missing = report.missing, expected = folded;
    std::sort(missing.begin(), missing.end());
    std::sort(expected.begin(), expected.end());
    if (missing != expected) {
        throw std::invalid_argument("LoadEncoderLM: the MDLM checkpoint " + directory + " doesn't hold every weight the model needs");
    }
    for (const std::string& name : report.unused) {
        if (std::find(options.ignore.begin(), options.ignore.end(), name) == options.ignore.end()) {
            throw std::invalid_argument("LoadEncoderLM: the MDLM checkpoint has a tensor nothing uses: " + name);
        }
    }
    FoldMdlmModulation(*model, checkpoint);
    return model;
}

}  // namespace

std::unique_ptr<EncoderLM> LoadEncoderLM(const std::string& directory, DeviceBackend* backend) {
    const std::string config_path = (std::filesystem::path(directory) / "config.json").string();
    std::ifstream in(config_path, std::ios::binary);
    if (!in) throw std::runtime_error("LoadEncoderLM: can't read " + config_path);
    std::stringstream config_json;
    config_json << in.rdbuf();
    if (const JsonValue o = ParseJson(config_json.str()); o.type() == JsonValue::Type::Object && String(o, "model_type") == "mdlm") {
        return LoadMdlm(directory, config_json.str(), backend);
    }
    EncoderLMConfig config = ParseEsmConfig(config_json.str());
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
    norm_.release_activations();
    for (Module* m : HeadModules()) m->release_activations();
    for (Tensor* t : {&embed_scale_, &last_logits_nobias_, &last_hidden_}) release_tensor(*t);
    hidden_.clear();
    has_forwarded_ = false;
}

}  // namespace pulsatrix
