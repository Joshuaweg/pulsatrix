// Helpers shared by the PEFT adapter formats (lora.cpp, ia3.cpp): PEFT's target matching, module-name
// mapping through a weight manifest, and file reading. Private to src/.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {
namespace peft_util {

constexpr const char* kKeyPrefix = "base_model.model.";

// A row-major (rows, cols) matrix's transpose.
inline std::vector<float> Transposed(const std::vector<float>& m, int64_t rows, int64_t cols) {
    std::vector<float> t(m.size());
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            t[static_cast<std::size_t>(j * rows + i)] = m[static_cast<std::size_t>(i * cols + j)];
        }
    }
    return t;
}

inline bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// PEFT's target matching: the name itself, or a last part of it after a dot.
inline bool MatchesTarget(const std::string& name, const std::string& target) {
    return name == target || EndsWith(name, "." + target);
}

inline std::vector<std::pair<std::string, LinearModule*>> LinearLayers(Module& model) {
    std::vector<std::pair<std::string, LinearModule*>> out;
    for (const NamedModuleRef& m : NamedModules(model)) {
        if (auto* linear = dynamic_cast<LinearModule*>(m.module)) out.emplace_back(m.name, linear);
    }
    return out;
}

// Module names (`layers.0.mha.q_proj`) to checkpoint module names (`model.layers.0.self_attn.q_proj`),
// from the mappings that load a whole `.weight`.
inline std::map<std::string, std::string> ModuleNames(const std::vector<WeightMapping>& mapping) {
    std::map<std::string, std::string> out;
    const std::string weight = ".weight";
    for (const WeightMapping& m : mapping) {
        if (m.row_begin != 0 || m.row_count >= 0 || !EndsWith(m.target, weight) || !EndsWith(m.source, weight)) continue;
        out[m.target.substr(0, m.target.size() - weight.size())] = m.source.substr(0, m.source.size() - weight.size());
    }
    return out;
}

inline std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("can't read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

}  // namespace peft_util
}  // namespace pulsatrix
