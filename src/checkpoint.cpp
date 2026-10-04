#include "pulsatrix/checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

constexpr const char* kFormat = "pulsatrix";
constexpr const char* kOptimizerFormat = "pulsatrix-optimizer";

// Renames entries from one format version to the next: the map takes each name in the file to
// the name the newer version uses. migrations[v] turns version v into v + 1.
using Migration = std::function<void(std::map<std::string, std::string>& names)>;

const std::vector<Migration>& migrations() {
    static const std::vector<Migration> table = {
        // 0 -> 1: version 0 is a plain safetensors file with no format_version. Version 1 uses
        // the same named_parameters()/named_buffers() names, so nothing is renamed.
        [](std::map<std::string, std::string>&) {},
    };
    return table;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("checkpoint: cannot open " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) {
        throw std::runtime_error("checkpoint: cannot read " + path);
    }
    return bytes;
}

// FNV-1a: ties an optimizer file to the exact model file it was saved with.
std::string checksum(const std::vector<uint8_t>& bytes) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : bytes) {
        h = (h ^ b) * 1099511628211ULL;
    }
    return std::to_string(h);
}

int parse_version(const std::string& text) {
    if (text.empty() || text.size() > 9 || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("checkpoint: format_version \"" + text + "\" is not a version number");
    }
    return std::stoi(text);
}

// Which version a file is, after checking it's a pulsatrix file at all.
int file_version(const SafetensorsFile& file, const char* expected_format) {
    const auto& md = file.metadata();
    auto fmt = md.find("format");
    if (fmt != md.end() && fmt->second != expected_format) {
        throw std::invalid_argument("checkpoint: file has format \"" + fmt->second + "\", not \"" + expected_format +
                                    "\"; loading another framework's weights needs name mapping (roadmap IO-4)");
    }
    auto v = md.find("format_version");
    const int version = v == md.end() ? 0 : parse_version(v->second);
    if (version > kCheckpointFormatVersion) {
        throw std::invalid_argument("checkpoint: file is format version " + std::to_string(version) +
                                    ", newer than this build's " + std::to_string(kCheckpointFormatVersion));
    }
    return version;
}

// Model name -> name in the file, after migrating the file to the current version.
std::map<std::string, std::string> migrated_names(const SafetensorsFile& file, int version) {
    std::map<std::string, std::string> names;
    for (const std::string& n : file.names()) {
        names[n] = n;
    }
    for (int v = version; v < kCheckpointFormatVersion; ++v) {
        migrations()[static_cast<size_t>(v)](names);
    }
    return names;
}

// Every parameter and buffer, by name.
std::map<std::string, Tensor*> model_tensors(Module& model) {
    std::map<std::string, Tensor*> out;
    for (const NamedParamRef& p : model.named_parameters()) {
        out[p.name] = p.ref.value;
    }
    for (const NamedBufferRef& b : model.named_buffers()) {
        if (!out.emplace(b.name, b.value).second) {
            throw std::invalid_argument("checkpoint: \"" + b.name + "\" names both a parameter and a buffer");
        }
    }
    return out;
}

std::string list(const std::vector<std::string>& names) {
    std::string out;
    for (size_t i = 0; i < names.size() && i < 5; ++i) {
        out += (i ? ", " : "") + names[i];
    }
    if (names.size() > 5) {
        out += ", ... (" + std::to_string(names.size()) + " in all)";
    }
    return out;
}

// A file entry as a tensor on `like`'s backend and device, after checking its shape matches.
Tensor read_like(const SafetensorsFile& file, const std::string& file_name, const Tensor& like,
                 const std::string& model_name) {
    const SafetensorsTensorInfo& info = file.info(file_name);
    if (info.dtype != SafetensorsDtype::F32) {
        throw std::invalid_argument("checkpoint: \"" + file_name + "\" is not F32");
    }
    if (Shape(info.shape) != like.shape()) {
        throw std::invalid_argument("checkpoint: \"" + model_name + "\" has a different shape in the file");
    }
    auto [ptr, size] = file.bytes(file_name);
    std::vector<float> values(size / sizeof(float));
    if (size > 0) {
        std::memcpy(values.data(), ptr, size);
    }
    return Tensor(like.shape(), like.backend(), values, like.device());
}

void load_model(const SafetensorsFile& file, Module& model, const CheckpointLoadOptions& options) {
    const std::map<std::string, std::string> names = migrated_names(file, file_version(file, kFormat));
    const std::map<std::string, Tensor*> targets = model_tensors(model);

    std::vector<std::string> missing, unexpected;
    for (const auto& [name, t] : targets) {
        if (names.count(name) == 0) {
            missing.push_back(name);
        }
    }
    for (const auto& [name, file_name] : names) {
        if (targets.count(name) == 0) {
            unexpected.push_back(name);
        }
    }
    if (options.strict && !missing.empty()) {
        throw std::invalid_argument("checkpoint: missing from the file: " + list(missing));
    }
    if (options.strict && !unexpected.empty()) {
        throw std::invalid_argument("checkpoint: not in the model: " + list(unexpected));
    }

    // Read and check everything first, so a failure leaves the model untouched.
    std::vector<std::pair<Tensor*, Tensor>> updates;
    for (const auto& [name, target] : targets) {
        auto it = names.find(name);
        if (it != names.end()) {
            updates.emplace_back(target, read_like(file, it->second, *target, name));
        }
    }
    for (auto& [target, value] : updates) {
        *target = std::move(value);  // assignment keeps the target's requires_grad flag
    }
}

std::map<std::string, std::string> with_format(const std::map<std::string, std::string>& metadata,
                                               const char* format) {
    if (metadata.count("format") || metadata.count("format_version")) {
        throw std::invalid_argument("checkpoint: \"format\" and \"format_version\" metadata are reserved");
    }
    std::map<std::string, std::string> out = metadata;
    out["format"] = format;
    out["format_version"] = std::to_string(kCheckpointFormatVersion);
    return out;
}

std::vector<uint8_t> serialize_model(Module& model, const std::map<std::string, std::string>& metadata) {
    std::vector<std::pair<std::string, const Tensor*>> tensors;
    for (const auto& [name, t] : model_tensors(model)) {
        tensors.emplace_back(name, t);
    }
    return SerializeSafetensors(tensors, with_format(metadata, kFormat));
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("checkpoint: cannot open " + path + " for writing");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("checkpoint: cannot write " + path);
    }
}

}  // namespace

std::string OptimizerStatePath(const std::string& checkpoint_path) {
    const std::string ext = ".safetensors";
    if (checkpoint_path.size() > ext.size() &&
        checkpoint_path.compare(checkpoint_path.size() - ext.size(), ext.size(), ext) == 0) {
        return checkpoint_path.substr(0, checkpoint_path.size() - ext.size()) + ".optim" + ext;
    }
    return checkpoint_path + ".optim" + ext;
}

void SaveCheckpoint(const std::string& path, Module& model, const std::map<std::string, std::string>& metadata) {
    write_bytes(path, serialize_model(model, metadata));
}

void SaveCheckpoint(const std::string& path, Module& model, const AdamOptimizer& optimizer,
                    const std::map<std::string, std::string>& metadata) {
    const std::vector<uint8_t> model_bytes = serialize_model(model, metadata);

    // Per parameter with state: its two moments as tensors, its step count as metadata.
    std::vector<std::pair<std::string, const Tensor*>> tensors;
    std::map<std::string, std::string> optimizer_metadata = {{"optimizer", "adam"},
                                                             {"model_checksum", checksum(model_bytes)}};
    for (const NamedParamRef& p : model.named_parameters()) {
        if (const AdamOptimizer::AdamState* s = optimizer.state(p.ref.value)) {
            tensors.emplace_back(p.name + ".exp_avg", &s->m);
            tensors.emplace_back(p.name + ".exp_avg_sq", &s->v);
            optimizer_metadata[p.name + ".step"] = std::to_string(s->t);
        }
    }
    const std::vector<uint8_t> optimizer_bytes =
        SerializeSafetensors(tensors, with_format(optimizer_metadata, kOptimizerFormat));
    write_bytes(path, model_bytes);
    write_bytes(OptimizerStatePath(path), optimizer_bytes);
}

void LoadCheckpoint(const std::string& path, Module& model, CheckpointLoadOptions options) {
    load_model(SafetensorsFile::Read(path), model, options);
}

void LoadCheckpoint(const std::string& path, Module& model, AdamOptimizer& optimizer, CheckpointLoadOptions options) {
    std::vector<uint8_t> model_bytes = read_bytes(path);
    const std::string model_sum = checksum(model_bytes);
    const SafetensorsFile model_file = SafetensorsFile::Parse(std::move(model_bytes));
    const SafetensorsFile opt = SafetensorsFile::Read(OptimizerStatePath(path));

    (void)file_version(opt, kOptimizerFormat);
    const auto& md = opt.metadata();
    auto kind = md.find("optimizer");
    if (kind == md.end() || kind->second != "adam") {
        throw std::invalid_argument("checkpoint: optimizer file isn't Adam state");
    }
    auto sum = md.find("model_checksum");
    if (sum == md.end() || sum->second != model_sum) {
        throw std::invalid_argument("checkpoint: optimizer file belongs to a different save of the model");
    }

    // Check the optimizer state against the model before loading either.
    std::vector<std::pair<const Tensor*, AdamOptimizer::AdamState>> states;
    std::set<std::string> used;
    for (const NamedParamRef& p : model.named_parameters()) {
        const std::string m = p.name + ".exp_avg", v = p.name + ".exp_avg_sq", t = p.name + ".step";
        if (!opt.contains(m)) {
            continue;  // step() never updated this parameter (e.g. it was frozen)
        }
        auto step = md.find(t);
        if (!opt.contains(v) || step == md.end()) {
            throw std::invalid_argument("checkpoint: incomplete optimizer state for \"" + p.name + "\"");
        }
        states.push_back({p.ref.value,
                          {read_like(opt, m, *p.ref.value, p.name), read_like(opt, v, *p.ref.value, p.name),
                           parse_version(step->second)}});
        used.insert(m);
        used.insert(v);
    }
    if (options.strict) {
        for (const std::string& n : opt.names()) {
            if (used.count(n) == 0) {
                throw std::invalid_argument("checkpoint: optimizer state for a parameter not in the model: " + n);
            }
        }
    }

    load_model(model_file, model, options);
    for (auto& [param, state] : states) {
        optimizer.set_state(param, std::move(state));
    }
}

}  // namespace pulsatrix
