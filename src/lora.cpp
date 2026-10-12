#include "pulsatrix/lora.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/safetensors.hpp"
#include "peft_util.hpp"

namespace pulsatrix {
namespace {

using namespace peft_util;

constexpr const char* kSuffixA = ".lora_A.weight";
constexpr const char* kSuffixB = ".lora_B.weight";

// A per-layer seed from the model-wide one, so layers don't share A.
uint64_t LayerSeed(uint64_t seed, uint64_t index) { return seed * 0x9E3779B97F4A7C15ULL + index + 1; }

}  // namespace

LoRAAdapter::LoRAAdapter(int64_t in_features, int64_t out_features, const LoRAConfig& config, DeviceBackend* backend,
                         DeviceType device)
    : in_features_(in_features),
      out_features_(out_features),
      rank_(config.rank),
      alpha_(config.alpha),
      dropout_p_(config.dropout),
      backend_(backend),
      a_(Shape({in_features, std::max<int64_t>(config.rank, 1)}), backend, device),
      b_(Shape({std::max<int64_t>(config.rank, 1), out_features}), backend, device),
      a_grad_(Shape({in_features, std::max<int64_t>(config.rank, 1)}), backend, device),
      b_grad_(Shape({std::max<int64_t>(config.rank, 1), out_features}), backend, device),
      dropout_(config.dropout >= 0 && config.dropout < 1 ? config.dropout : 0.0f, backend, config.seed ^ 0xD20F0u),
      last_input_(Shape({0}), backend, device),
      last_hidden_(Shape({0}), backend, device),
      last_base_relevance_(Shape({0}), backend, device),
      last_adapter_relevance_(Shape({0}), backend, device) {
    if (config.rank < 1) throw std::invalid_argument("LoRAAdapter: rank must be at least 1");
    if (!std::isfinite(config.alpha)) throw std::invalid_argument("LoRAAdapter: alpha must be finite");
    if (!(config.dropout >= 0 && config.dropout < 1)) throw std::invalid_argument("LoRAAdapter: dropout must be in [0, 1)");
    // PEFT: kaiming_uniform_(lora_A, a=sqrt(5)) on its (r, in) weight, whose fan-in is in, gives
    // bound gain * sqrt(3 / in) with gain sqrt(2 / (1 + 5)): 1 / sqrt(in). Raw mt19937_64 output keeps
    // the values the same on every platform.
    std::mt19937_64 rng(config.seed);
    const double bound = 1.0 / std::sqrt(static_cast<double>(in_features));
    std::vector<float> a(static_cast<std::size_t>(in_features * rank_));
    for (float& v : a) {
        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;  // [0, 1)
        v = static_cast<float>((2 * u - 1) * bound);
    }
    a_ = Tensor(a_.shape(), backend, a, device);
}

void LoRAAdapter::set_a(const std::vector<float>& values) {
    if (static_cast<int64_t>(values.size()) != a_.numel()) throw std::invalid_argument("LoRAAdapter::set_a: expected in x r values");
    const bool trainable = a_.requires_grad();
    a_ = Tensor(a_.shape(), backend_, values, a_.device());
    a_.set_requires_grad(trainable);
}

void LoRAAdapter::set_b(const std::vector<float>& values) {
    if (static_cast<int64_t>(values.size()) != b_.numel()) throw std::invalid_argument("LoRAAdapter::set_b: expected r x out values");
    const bool trainable = b_.requires_grad();
    b_ = Tensor(b_.shape(), backend_, values, b_.device());
    b_.set_requires_grad(trainable);
}

Tensor LoRAAdapter::update() const {
    Tensor u(Shape({in_features_, out_features_}), backend_, a_.device());
    backend_->fill(u.data(), 0.0f, static_cast<std::size_t>(u.numel()));
    add_update(u, 1.0f);
    return u;
}

const Tensor& LoRAAdapter::last_base_relevance() const {
    if (!has_relevance()) throw std::logic_error("LoRAAdapter::last_base_relevance: no split relevance pass since the last forward()");
    return last_base_relevance_;
}

const Tensor& LoRAAdapter::last_adapter_relevance() const {
    if (!has_relevance()) throw std::logic_error("LoRAAdapter::last_adapter_relevance: no split relevance pass since the last forward()");
    return last_adapter_relevance_;
}

void LoRAAdapter::clear_relevance() {
    release_tensor(last_base_relevance_);
    release_tensor(last_adapter_relevance_);
}

Tensor LoRAAdapter::effective_weight(const Tensor& weight) const {
    Tensor w(weight.shape(), backend_, weight.device());
    backend_->copy(w.data(), weight.data(), static_cast<std::size_t>(weight.numel()) * sizeof(float), CopyDirection::DeviceToDevice);
    add_update(w, 1.0f);
    return w;
}

void LoRAAdapter::add_update(Tensor& weight, float sign) const {
    Tensor ab(weight.shape(), backend_, weight.device());
    backend_->gemm(a_.data(), b_.data(), ab.data(), static_cast<std::size_t>(in_features_), static_cast<std::size_t>(rank_),
                   static_cast<std::size_t>(out_features_));
    backend_->axpby(sign * scale(), ab.data(), 1.0f, weight.data(), weight.data(), static_cast<std::size_t>(weight.numel()));
}

void LoRAAdapter::add_forward(const Tensor& input, Tensor& pre_bias, bool training) {
    const int64_t n = input.shape().dim(0);
    used_dropout_ = training && dropout_p_ > 0;
    if (used_dropout_) {
        dropout_.set_training(true);
        last_input_ = dropout_.forward(input);
    } else {
        last_input_ = input;
    }
    last_hidden_ = Tensor(Shape({n, rank_}), backend_, a_.device());
    backend_->gemm(last_input_.data(), a_.data(), last_hidden_.data(), static_cast<std::size_t>(n),
                   static_cast<std::size_t>(in_features_), static_cast<std::size_t>(rank_));
    backend_->axpby(scale(), last_hidden_.data(), 0.0f, last_hidden_.data(), last_hidden_.data(),
                    static_cast<std::size_t>(last_hidden_.numel()));
    backend_->gemm_ex(last_hidden_.data(), false, b_.data(), false, pre_bias.data(), static_cast<std::size_t>(n),
                      static_cast<std::size_t>(rank_), static_cast<std::size_t>(out_features_), 1.0f);
}

void LoRAAdapter::backward(const Tensor& grad_output, Tensor& grad_input) {
    const auto n = static_cast<std::size_t>(grad_output.shape().dim(0));
    const auto in = static_cast<std::size_t>(in_features_), r = static_cast<std::size_t>(rank_),
               out = static_cast<std::size_t>(out_features_);
    // last_hidden_ already carries the scale: dL/dB = (s x A)^T g.
    if (b_.requires_grad()) backend_->gemm_ex(last_hidden_.data(), true, grad_output.data(), false, b_grad_.data(), r, n, out, 1.0f);
    Tensor grad_hidden(Shape({static_cast<int64_t>(n), rank_}), backend_, a_.device());
    backend_->gemm_ex(grad_output.data(), false, b_.data(), true, grad_hidden.data(), n, out, r, 0.0f);
    backend_->axpby(scale(), grad_hidden.data(), 0.0f, grad_hidden.data(), grad_hidden.data(), n * r);
    if (a_.requires_grad()) backend_->gemm_ex(last_input_.data(), true, grad_hidden.data(), false, a_grad_.data(), in, n, r, 1.0f);
    Tensor grad_adapter(Shape({static_cast<int64_t>(n), in_features_}), backend_, a_.device());
    backend_->gemm_ex(grad_hidden.data(), false, a_.data(), true, grad_adapter.data(), n, r, in, 0.0f);
    if (used_dropout_) grad_adapter = dropout_.backward(grad_adapter);
    backend_->add(grad_input.data(), grad_adapter.data(), grad_input.data(), n * in);
}

void LoRAAdapter::release_activations() {
    release_tensor(last_input_);
    release_tensor(last_hidden_);
    dropout_.release_activations();
    clear_relevance();
}

std::vector<std::string> ApplyLora(Module& model, const LoRAConfig& config) {
    const std::vector<std::pair<std::string, LinearModule*>> linear = LinearLayers(model);
    std::vector<std::pair<std::string, LinearModule*>> selected;
    if (config.target_modules.empty()) {
        for (const auto& l : linear) {
            if (l.first != "lm_head") selected.push_back(l);
        }
    } else {
        const std::vector<NamedModuleRef> all = NamedModules(model);
        for (const std::string& target : config.target_modules) {
            bool found = false;
            for (const NamedModuleRef& m : all) {
                if (!MatchesTarget(m.name, target)) continue;
                found = true;
                if (dynamic_cast<LinearModule*>(m.module) == nullptr) {
                    throw std::invalid_argument("ApplyLora: '" + m.name + "' matches target '" + target +
                                                "' but isn't a LinearModule");
                }
            }
            if (!found) throw std::invalid_argument("ApplyLora: target '" + target + "' matches no module");
        }
        for (const auto& l : linear) {
            for (const std::string& target : config.target_modules) {
                if (MatchesTarget(l.first, target)) {
                    selected.push_back(l);
                    break;
                }
            }
        }
    }
    if (selected.empty()) throw std::invalid_argument("ApplyLora: the model has no LinearModule to adapt");
    for (const auto& l : selected) {
        if (l.second->lora() != nullptr) throw std::invalid_argument("ApplyLora: '" + l.first + "' already has an adapter");
    }
    // Validates the settings before anything changes.
    (void)LoRAAdapter(1, 1, LoRAConfig{config.rank, config.alpha, config.dropout, config.seed, {}}, selected[0].second->weight().backend(),
                      selected[0].second->weight().device());

    std::vector<std::string> names;
    model.set_requires_grad(false);
    for (std::size_t i = 0; i < selected.size(); ++i) {
        LoRAConfig layer = config;
        layer.seed = LayerSeed(config.seed, i);
        selected[i].second->attach_lora(layer);
        names.push_back(selected[i].first);
    }
    return names;
}

std::vector<std::pair<std::string, LinearModule*>> LoraLayers(Module& model) {
    std::vector<std::pair<std::string, LinearModule*>> out;
    for (const auto& l : LinearLayers(model)) {
        if (l.second->lora() != nullptr) out.push_back(l);
    }
    return out;
}

std::vector<LoraLayerRelevance> LoraRelevanceSplit(Module& model) {
    std::vector<LoraLayerRelevance> out;
    for (const auto& [name, layer] : LoraLayers(model)) {
        const LoRAAdapter& a = *layer->lora();
        if (!a.has_relevance()) continue;
        LoraLayerRelevance r{name, a.last_base_relevance(), a.last_adapter_relevance()};
        for (float v : r.base.to_host_vector()) r.base_total += v;
        for (float v : r.adapter.to_host_vector()) r.adapter_total += v;
        out.push_back(std::move(r));
    }
    if (out.empty()) throw std::logic_error("LoraRelevanceSplit: no adapted layer was split in the last relevance pass");
    return out;
}

void MergeLora(Module& model) {
    for (const auto& l : LoraLayers(model)) {
        if (!l.second->lora()->merged()) l.second->merge_lora();
    }
}

void UnmergeLora(Module& model) {
    for (const auto& l : LoraLayers(model)) {
        if (l.second->lora()->merged()) l.second->unmerge_lora();
    }
}

void SetLoraEnabled(Module& model, bool enabled) {
    const auto layers = LoraLayers(model);
    for (const auto& l : layers) {
        if (l.second->lora()->merged()) throw std::logic_error("SetLoraEnabled: '" + l.first + "' is merged; unmerge it first");
    }
    for (const auto& l : layers) l.second->set_lora_enabled(enabled);
}

void RemoveLora(Module& model) {
    for (const auto& l : LoraLayers(model)) l.second->detach_lora();
}

void SaveLoraAdapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping) {
    const auto layers = LoraLayers(model);
    if (layers.empty()) throw std::invalid_argument("SaveLoraAdapter: the model has no adapters");
    const LoRAAdapter& first = *layers[0].second->lora();
    const std::map<std::string, std::string> names = ModuleNames(mapping);
    std::vector<std::string> modules;
    for (const auto& l : layers) {
        const LoRAAdapter& a = *l.second->lora();
        if (a.rank() != first.rank() || a.alpha() != first.alpha() || a.dropout() != first.dropout()) {
            throw std::invalid_argument("SaveLoraAdapter: the adapters' rank, alpha or dropout differ ('" + l.first + "')");
        }
        if (a.merged()) throw std::invalid_argument("SaveLoraAdapter: '" + l.first + "' is merged; unmerge it first");
        if (mapping.empty()) {
            modules.push_back(l.first);
        } else {
            const auto it = names.find(l.first);
            if (it == names.end()) throw std::invalid_argument("SaveLoraAdapter: the mapping doesn't load '" + l.first + ".weight' whole");
            modules.push_back(it->second);
        }
    }

    std::vector<Tensor> owned;
    owned.reserve(2 * layers.size());
    std::vector<std::pair<std::string, const Tensor*>> tensors;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const LoRAAdapter& a = *layers[i].second->lora();
        const int64_t in = a.a().shape().dim(0), r = a.rank(), out = a.b().shape().dim(1);
        DeviceBackend* backend = a.a().backend();
        owned.emplace_back(Shape({r, in}), backend, Transposed(a.a().to_host_vector(), in, r));
        tensors.emplace_back(kKeyPrefix + modules[i] + kSuffixA, &owned.back());
        owned.emplace_back(Shape({out, r}), backend, Transposed(a.b().to_host_vector(), r, out));
        tensors.emplace_back(kKeyPrefix + modules[i] + kSuffixB, &owned.back());
    }

    JsonValue targets{JsonValue::Array{}};
    for (const std::string& m : modules) targets.push_back(m);
    JsonValue config{JsonValue::Object{}};
    config.add("peft_type", "LORA");
    config.add("base_model_name_or_path", nullptr);
    config.add("task_type", nullptr);
    config.add("r", first.rank());
    config.add("lora_alpha", JsonValue::Float(first.alpha()));
    config.add("lora_dropout", JsonValue::Float(first.dropout()));
    config.add("target_modules", targets);
    config.add("bias", "none");
    config.add("fan_in_fan_out", false);
    config.add("use_rslora", false);
    config.add("use_dora", false);
    config.add("init_lora_weights", true);
    config.add("inference_mode", true);
    config.add("modules_to_save", nullptr);
    config.add("rank_pattern", JsonValue::Object{});
    config.add("alpha_pattern", JsonValue::Object{});

    std::filesystem::create_directories(directory);
    WriteSafetensors(directory + "/adapter_model.safetensors", tensors, {{"format", "pt"}});
    std::ofstream out(directory + "/adapter_config.json", std::ios::binary);
    out << WriteJson(config);
    if (!out) throw std::runtime_error("SaveLoraAdapter: can't write " + directory + "/adapter_config.json");
}

void LoadLoraAdapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping) {
    const JsonValue config = ParseJson(ReadFile(directory + "/adapter_config.json"));
    auto get = [&](const char* key) { return config.find(key); };
    auto is_true = [&](const char* key) {
        const JsonValue* v = get(key);
        return v != nullptr && v->type() == JsonValue::Type::Bool && v->as_bool();
    };
    auto non_empty = [&](const char* key) {
        const JsonValue* v = get(key);
        if (v == nullptr || v->is_null()) return false;
        if (v->type() == JsonValue::Type::Object) return !v->as_object().empty();
        if (v->type() == JsonValue::Type::Array) return !v->as_array().empty();
        return true;
    };
    const JsonValue* type = get("peft_type");
    if (type == nullptr || type->type() != JsonValue::Type::String || type->as_string() != "LORA") {
        throw std::invalid_argument("LoadLoraAdapter: adapter_config.json's peft_type isn't LORA");
    }
    for (const char* unsupported : {"use_dora", "use_rslora", "fan_in_fan_out", "lora_bias", "use_qalora"}) {
        if (is_true(unsupported)) throw std::invalid_argument(std::string("LoadLoraAdapter: ") + unsupported + " isn't supported");
    }
    for (const char* unsupported : {"modules_to_save", "rank_pattern", "alpha_pattern", "layers_to_transform", "trainable_token_indices"}) {
        if (non_empty(unsupported)) throw std::invalid_argument(std::string("LoadLoraAdapter: ") + unsupported + " isn't supported");
    }
    if (const JsonValue* bias = get("bias"); bias != nullptr && !(bias->type() == JsonValue::Type::String && bias->as_string() == "none")) {
        throw std::invalid_argument("LoadLoraAdapter: only bias \"none\" is supported");
    }
    const JsonValue* alpha = get("lora_alpha");
    if (alpha == nullptr) throw std::invalid_argument("LoadLoraAdapter: adapter_config.json has no lora_alpha");
    const JsonValue* dropout = get("lora_dropout");

    const SafetensorsFile file = SafetensorsFile::Read(directory + "/adapter_model.safetensors");
    std::map<std::string, std::string> to_module;  // checkpoint module name to ours
    for (const auto& [ours, theirs] : ModuleNames(mapping)) to_module[theirs] = ours;
    std::map<std::string, LinearModule*> linear;
    for (const auto& l : LinearLayers(model)) linear[l.first] = l.second;

    struct Pending {
        LinearModule* layer;
        std::string name;
        std::string key_a, key_b;
        int64_t rank;
    };
    std::map<std::string, Pending> pending;  // by checkpoint module name
    const std::string prefix = kKeyPrefix;
    for (const std::string& key : file.names()) {
        const bool is_a = EndsWith(key, kSuffixA), is_b = EndsWith(key, kSuffixB);
        if (key.rfind(prefix, 0) != 0 || !(is_a || is_b)) {
            throw std::invalid_argument("LoadLoraAdapter: '" + key + "' isn't a LoRA weight (base_model.model.<module>.lora_A/B.weight)");
        }
        const std::size_t suffix = is_a ? std::string(kSuffixA).size() : std::string(kSuffixB).size();
        const std::string module = key.substr(prefix.size(), key.size() - prefix.size() - suffix);
        Pending& p = pending[module];
        (is_a ? p.key_a : p.key_b) = key;
    }
    for (auto& [module, p] : pending) {
        if (p.key_a.empty() || p.key_b.empty()) throw std::invalid_argument("LoadLoraAdapter: '" + module + "' lacks lora_A or lora_B");
        p.name = mapping.empty() ? module : (to_module.count(module) ? to_module.at(module) : std::string());
        if (p.name.empty() || linear.count(p.name) == 0) {
            throw std::invalid_argument("LoadLoraAdapter: no LinearModule for '" + module + "'");
        }
        p.layer = linear.at(p.name);
        if (p.layer->lora() != nullptr) throw std::invalid_argument("LoadLoraAdapter: '" + p.name + "' already has an adapter");
        const std::vector<int64_t>& a = file.info(p.key_a).shape;
        const std::vector<int64_t>& b = file.info(p.key_b).shape;
        if (a.size() != 2 || b.size() != 2 || a[1] != p.layer->in_features() || b[0] != p.layer->out_features() || a[0] != b[1] || a[0] < 1) {
            throw std::invalid_argument("LoadLoraAdapter: '" + module + "' has shapes that don't fit a " +
                                        std::to_string(p.layer->in_features()) + " -> " +
                                        std::to_string(p.layer->out_features()) + " layer");
        }
        p.rank = a[0];
    }
    if (pending.empty()) throw std::invalid_argument("LoadLoraAdapter: the adapter has no weights");

    LoRAConfig layer_config;
    layer_config.alpha = alpha->as_float();
    layer_config.dropout = dropout == nullptr || dropout->is_null() ? 0.0f : dropout->as_float();
    // Validates the settings before anything changes.
    (void)LoRAAdapter(1, 1, LoRAConfig{1, layer_config.alpha, layer_config.dropout, 0, {}}, pending.begin()->second.layer->weight().backend(),
                      pending.begin()->second.layer->weight().device());
    model.set_requires_grad(false);
    for (auto& [module, p] : pending) {
        layer_config.rank = p.rank;
        p.layer->attach_lora(layer_config);
        LoRAAdapter& adapter = *p.layer->lora();
        DeviceBackend* backend = p.layer->weight().backend();
        adapter.set_a(Transposed(file.tensor(p.key_a, backend).to_host_vector(), p.rank, p.layer->in_features()));
        adapter.set_b(Transposed(file.tensor(p.key_b, backend).to_host_vector(), p.layer->out_features(), p.rank));
    }
}

}  // namespace pulsatrix
