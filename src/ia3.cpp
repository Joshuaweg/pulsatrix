#include "pulsatrix/ia3.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

#include "peft_util.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

using namespace peft_util;

constexpr const char* kSuffix = ".ia3_l";

bool MatchesAny(const std::string& name, const std::vector<std::string>& targets) {
    for (const std::string& t : targets) {
        if (MatchesTarget(name, t)) return true;
    }
    return false;
}

}  // namespace

IA3Adapter::IA3Adapter(int64_t size, bool feedforward, DeviceBackend* backend, DeviceType device)
    : feedforward_(feedforward),
      backend_(backend),
      scale_(Shape({size}), backend, std::vector<float>(static_cast<std::size_t>(size), 1.0f), device),
      grad_(Shape({size}), backend, device),
      last_input_(Shape({0}), backend, device),
      last_output_(Shape({0}), backend, device) {}

void IA3Adapter::set_scale(const std::vector<float>& values) {
    if (static_cast<int64_t>(values.size()) != scale_.numel()) throw std::invalid_argument("IA3Adapter::set_scale: wrong number of values");
    const bool trainable = scale_.requires_grad();
    scale_ = Tensor(scale_.shape(), backend_, values, scale_.device());
    scale_.set_requires_grad(trainable);
}

std::vector<std::string> ApplyIA3(Module& model, const IA3Config& config) {
    const std::vector<std::pair<std::string, LinearModule*>> linear = LinearLayers(model);
    std::vector<std::string> targets = config.target_modules, feedforward = config.feedforward_modules;
    if (targets.empty()) {
        if (!feedforward.empty()) throw std::invalid_argument("ApplyIA3: feedforward_modules needs target_modules");
        // The paper's keys, values and MLP inner activations, under whichever MLP names the model uses.
        targets = {"k_proj", "v_proj"};
        for (const char* mlp : {"down_proj", "fc2"}) {
            for (const auto& l : linear) {
                if (MatchesTarget(l.first, mlp)) {
                    targets.push_back(mlp);
                    feedforward.push_back(mlp);
                    break;
                }
            }
        }
    }
    const std::vector<NamedModuleRef> all = NamedModules(model);
    for (const std::string& target : targets) {
        bool found = false;
        for (const NamedModuleRef& m : all) {
            if (!MatchesTarget(m.name, target)) continue;
            found = true;
            if (dynamic_cast<LinearModule*>(m.module) == nullptr) {
                throw std::invalid_argument("ApplyIA3: '" + m.name + "' matches target '" + target + "' but isn't a LinearModule");
            }
        }
        if (!found) throw std::invalid_argument("ApplyIA3: target '" + target + "' matches no module");
    }
    std::vector<std::pair<std::string, LinearModule*>> selected;
    for (const auto& l : linear) {
        if (MatchesAny(l.first, targets)) selected.push_back(l);
    }
    for (const std::string& f : feedforward) {
        bool among = false;
        for (const auto& l : selected) among = among || MatchesTarget(l.first, f);
        if (!among) throw std::invalid_argument("ApplyIA3: feedforward module '" + f + "' isn't among the targets");
    }
    if (selected.empty()) throw std::invalid_argument("ApplyIA3: nothing to adapt");
    for (const auto& l : selected) {
        if (l.second->ia3() != nullptr || l.second->lora() != nullptr) {
            throw std::invalid_argument("ApplyIA3: '" + l.first + "' already has an adapter");
        }
    }
    model.set_requires_grad(false);
    std::vector<std::string> names;
    for (const auto& l : selected) {
        l.second->attach_ia3(MatchesAny(l.first, feedforward));
        names.push_back(l.first);
    }
    return names;
}

std::vector<std::pair<std::string, LinearModule*>> IA3Layers(Module& model) {
    std::vector<std::pair<std::string, LinearModule*>> out;
    for (const auto& l : LinearLayers(model)) {
        if (l.second->ia3() != nullptr) out.push_back(l);
    }
    return out;
}

void MergeIA3(Module& model) {
    for (const auto& l : IA3Layers(model)) {
        if (!l.second->ia3()->merged()) l.second->merge_ia3();
    }
}

void UnmergeIA3(Module& model) {
    for (const auto& l : IA3Layers(model)) {
        if (l.second->ia3()->merged()) l.second->unmerge_ia3();
    }
}

void SetIA3Enabled(Module& model, bool enabled) {
    const auto layers = IA3Layers(model);
    for (const auto& l : layers) {
        if (l.second->ia3()->merged()) throw std::logic_error("SetIA3Enabled: '" + l.first + "' is merged; unmerge it first");
    }
    for (const auto& l : layers) l.second->set_ia3_enabled(enabled);
}

void RemoveIA3(Module& model) {
    for (const auto& l : IA3Layers(model)) l.second->detach_ia3();
}

void SaveIA3Adapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping) {
    const auto layers = IA3Layers(model);
    if (layers.empty()) throw std::invalid_argument("SaveIA3Adapter: the model has no IA3 vectors");
    const std::map<std::string, std::string> names = ModuleNames(mapping);
    std::vector<std::string> modules;
    for (const auto& l : layers) {
        if (l.second->ia3()->merged()) throw std::invalid_argument("SaveIA3Adapter: '" + l.first + "' is merged; unmerge it first");
        if (mapping.empty()) {
            modules.push_back(l.first);
        } else {
            const auto it = names.find(l.first);
            if (it == names.end()) throw std::invalid_argument("SaveIA3Adapter: the mapping doesn't load '" + l.first + ".weight' whole");
            modules.push_back(it->second);
        }
    }
    std::vector<Tensor> owned;
    owned.reserve(layers.size());
    std::vector<std::pair<std::string, const Tensor*>> tensors;
    JsonValue targets{JsonValue::Array{}}, feedforward{JsonValue::Array{}};
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const IA3Adapter& a = *layers[i].second->ia3();
        const int64_t size = a.scale().numel();
        owned.emplace_back(a.feedforward() ? Shape({1, size}) : Shape({size, 1}), a.scale().backend(), a.scale().to_host_vector());
        tensors.emplace_back(kKeyPrefix + modules[i] + kSuffix, &owned.back());
        targets.push_back(modules[i]);
        if (a.feedforward()) feedforward.push_back(modules[i]);
    }
    JsonValue config{JsonValue::Object{}};
    config.add("peft_type", "IA3");
    config.add("base_model_name_or_path", nullptr);
    config.add("task_type", nullptr);
    config.add("target_modules", targets);
    config.add("feedforward_modules", feedforward);
    config.add("fan_in_fan_out", false);
    config.add("init_ia3_weights", true);
    config.add("inference_mode", true);
    config.add("modules_to_save", nullptr);
    std::filesystem::create_directories(directory);
    WriteSafetensors(directory + "/adapter_model.safetensors", tensors, {{"format", "pt"}});
    std::ofstream out(directory + "/adapter_config.json", std::ios::binary);
    out << WriteJson(config);
    if (!out) throw std::runtime_error("SaveIA3Adapter: can't write " + directory + "/adapter_config.json");
}

void LoadIA3Adapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping) {
    const JsonValue config = ParseJson(ReadFile(directory + "/adapter_config.json"));
    const JsonValue* type = config.find("peft_type");
    if (type == nullptr || type->type() != JsonValue::Type::String || type->as_string() != "IA3") {
        throw std::invalid_argument("LoadIA3Adapter: adapter_config.json's peft_type isn't IA3");
    }
    if (const JsonValue* f = config.find("fan_in_fan_out"); f != nullptr && f->type() == JsonValue::Type::Bool && f->as_bool()) {
        throw std::invalid_argument("LoadIA3Adapter: fan_in_fan_out isn't supported");
    }
    if (const JsonValue* m = config.find("modules_to_save");
        m != nullptr && !m->is_null() && !(m->type() == JsonValue::Type::Array && m->as_array().empty())) {
        throw std::invalid_argument("LoadIA3Adapter: modules_to_save isn't supported");
    }
    std::vector<std::string> feedforward;
    if (const JsonValue* f = config.find("feedforward_modules"); f != nullptr && !f->is_null()) {
        if (f->type() == JsonValue::Type::String) {
            feedforward.push_back(f->as_string());
        } else {
            for (const JsonValue& v : f->as_array()) feedforward.push_back(v.as_string());
        }
    }

    const SafetensorsFile file = SafetensorsFile::Read(directory + "/adapter_model.safetensors");
    std::map<std::string, std::string> to_module;
    for (const auto& [ours, theirs] : ModuleNames(mapping)) to_module[theirs] = ours;
    std::map<std::string, LinearModule*> linear;
    for (const auto& l : LinearLayers(model)) linear[l.first] = l.second;
    struct Pending {
        LinearModule* layer;
        std::string key;
        bool feedforward;
    };
    std::vector<Pending> pending;
    const std::string prefix = kKeyPrefix;
    for (const std::string& key : file.names()) {
        if (key.rfind(prefix, 0) != 0 || !EndsWith(key, kSuffix)) {
            throw std::invalid_argument("LoadIA3Adapter: '" + key + "' isn't an IA3 vector (base_model.model.<module>.ia3_l)");
        }
        const std::string module = key.substr(prefix.size(), key.size() - prefix.size() - std::string(kSuffix).size());
        const std::string name = mapping.empty() ? module : (to_module.count(module) ? to_module.at(module) : std::string());
        if (name.empty() || linear.count(name) == 0) throw std::invalid_argument("LoadIA3Adapter: no LinearModule for '" + module + "'");
        LinearModule* layer = linear.at(name);
        if (layer->ia3() != nullptr || layer->lora() != nullptr) throw std::invalid_argument("LoadIA3Adapter: '" + name + "' already has an adapter");
        const std::vector<int64_t>& shape = file.info(key).shape;
        const bool is_ff = shape == std::vector<int64_t>{1, layer->in_features()};
        if (!is_ff && shape != std::vector<int64_t>{layer->out_features(), 1}) {
            throw std::invalid_argument("LoadIA3Adapter: '" + module + "''s vector fits neither side of the layer");
        }
        if (is_ff != MatchesAny(module, feedforward)) {
            throw std::invalid_argument("LoadIA3Adapter: '" + module + "''s shape disagrees with feedforward_modules");
        }
        pending.push_back({layer, key, is_ff});
    }
    if (pending.empty()) throw std::invalid_argument("LoadIA3Adapter: the adapter has no vectors");
    model.set_requires_grad(false);
    for (const Pending& p : pending) {
        p.layer->attach_ia3(p.feedforward);
        p.layer->ia3()->set_scale(file.tensor(p.key, p.layer->weight().backend()).to_host_vector());
    }
}

}  // namespace pulsatrix
