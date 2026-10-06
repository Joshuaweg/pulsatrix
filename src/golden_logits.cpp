#include "pulsatrix/golden_logits.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pulsatrix {

GoldenSequenceResult CompareLogits(const std::vector<float>& got, const std::vector<float>& expected,
                                   int64_t num_tokens, int64_t vocab) {
    const size_t n = static_cast<size_t>(num_tokens * vocab);
    if (got.size() != n || expected.size() != n) {
        throw std::invalid_argument("CompareLogits: both logit buffers must hold num_tokens * vocab values");
    }
    GoldenSequenceResult r;
    r.num_tokens = num_tokens;
    double sum = 0.0;
    for (int64_t p = 0; p < num_tokens; ++p) {
        const float* g = got.data() + p * vocab;
        const float* e = expected.data() + p * vocab;
        float scale = 1.0f;
        for (int64_t v = 0; v < vocab; ++v) scale = std::max(scale, std::fabs(e[v]));
        for (int64_t v = 0; v < vocab; ++v) {
            float d = std::fabs(g[v] - e[v]);
            if (std::isnan(d)) d = INFINITY;
            r.max_abs_diff = std::max(r.max_abs_diff, d);
            r.max_scaled_diff = std::max(r.max_scaled_diff, d / scale);
            sum += d;
        }
        if (std::max_element(g, g + vocab) - g != std::max_element(e, e + vocab) - e) ++r.argmax_disagreements;
    }
    r.mean_abs_diff = static_cast<float>(sum / static_cast<double>(n));
    return r;
}

GoldenReport CompareToGolden(CausalLM& model, const SafetensorsFile& golden, float threshold) {
    GoldenReport report;
    report.threshold = threshold;
    report.metadata = golden.metadata();
    const int64_t vocab = model.config().vocab_size;
    DeviceBackend* backend = model.embed_tokens().weight().backend();
    for (int64_t k = 0;; ++k) {
        const std::string ids_name = "ids." + std::to_string(k);
        const std::string logits_name = "logits." + std::to_string(k);
        if (!golden.contains(ids_name)) break;
        const SafetensorsTensorInfo& info = golden.info(ids_name);
        if (info.dtype != SafetensorsDtype::I64 || info.shape.size() != 1) {
            throw std::invalid_argument("CompareToGolden: " + ids_name + " must be a 1-D I64 tensor");
        }
        auto [ptr, size] = golden.bytes(ids_name);
        const int64_t L = info.shape[0];
        std::vector<float> ids(static_cast<size_t>(L));
        for (int64_t i = 0; i < L; ++i) {
            int64_t id = 0;
            std::memcpy(&id, ptr + 8 * i, 8);
            if (id < 0 || id >= vocab) {
                throw std::invalid_argument("CompareToGolden: " + ids_name + " holds an id outside the vocabulary");
            }
            ids[static_cast<size_t>(i)] = static_cast<float>(id);
        }
        if (!golden.contains(logits_name)) {
            throw std::invalid_argument("CompareToGolden: " + logits_name + " is missing");
        }
        const std::vector<float> expected = golden.tensor(logits_name, backend).to_host_vector();
        if (static_cast<int64_t>(expected.size()) != L * vocab) {
            throw std::invalid_argument("CompareToGolden: " + logits_name + " is not (L, vocab)");
        }
        const std::vector<float> got = model.forward(Tensor(Shape({1, L}), backend, ids)).to_host_vector();
        GoldenSequenceResult r = CompareLogits(got, expected, L, vocab);
        report.max_abs_diff = std::max(report.max_abs_diff, r.max_abs_diff);
        report.max_scaled_diff = std::max(report.max_scaled_diff, r.max_scaled_diff);
        report.sequences.push_back(r);
    }
    if (report.sequences.empty()) {
        throw std::invalid_argument("CompareToGolden: the golden file holds no ids.0");
    }
    report.passed = report.max_scaled_diff < threshold;
    return report;
}

}  // namespace pulsatrix
