#include "pulsatrix/attnlrp_parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace pulsatrix {

namespace {

std::vector<int64_t> ReadIds(const SafetensorsFile& file, const std::string& name) {
    const SafetensorsTensorInfo& info = file.info(name);
    if (info.dtype != SafetensorsDtype::I64 || info.shape.size() != 1) {
        throw std::invalid_argument("CompareToAttnLrp: " + name + " must be a 1-D I64 tensor");
    }
    auto [ptr, size] = file.bytes(name);
    std::vector<int64_t> ids(static_cast<size_t>(info.shape[0]));
    std::memcpy(ids.data(), ptr, ids.size() * sizeof(int64_t));
    return ids;
}

std::vector<float> ReadFloats(const SafetensorsFile& file, const std::string& name, DeviceBackend* backend,
                              size_t expected_size) {
    if (!file.contains(name)) throw std::invalid_argument("CompareToAttnLrp: " + name + " is missing");
    std::vector<float> v = file.tensor(name, backend).to_host_vector();
    if (v.size() != expected_size) throw std::invalid_argument("CompareToAttnLrp: " + name + " has the wrong size");
    return v;
}

/** @brief Per-(KV head, token) totals, (Hkv, L), of a (L, Hkv * D) projection relevance. */
std::vector<float> PerHead(const Tensor& relevance, int64_t L, int64_t kv_heads) {
    const std::vector<float> r = relevance.to_host_vector();
    const int64_t D = static_cast<int64_t>(r.size()) / (L * kv_heads);
    std::vector<float> out(static_cast<size_t>(kv_heads * L), 0.0f);
    for (int64_t t = 0; t < L; ++t) {
        for (int64_t h = 0; h < kv_heads; ++h) {
            double sum = 0.0;
            for (int64_t d = 0; d < D; ++d) sum += r[static_cast<size_t>((t * kv_heads + h) * D + d)];
            out[static_cast<size_t>(h * L + t)] = static_cast<float>(sum);
        }
    }
    return out;
}

}  // namespace

double PearsonCorrelation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) throw std::invalid_argument("PearsonCorrelation: needs two equal, non-empty vectors");
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= static_cast<double>(x.size());
    my /= static_cast<double>(y.size());
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    return (sxx == 0.0 || syy == 0.0) ? 0.0 : sxy / std::sqrt(sxx * syy);
}

LRPRuleConfig LxtAttnLrpConfig() {
    LRPRuleConfig config{1e-9f};
    config.epsilon_bias_in_denominator = true;
    config.layer_norm_detach_std = true;
    return config;
}

AttnLrpReport CompareToAttnLrp(CausalLM& model, const SafetensorsFile& reference, const LRPRuleConfig& config,
                               float threshold) {
    AttnLrpReport report;
    report.threshold = threshold;
    report.metadata = reference.metadata();
    const int64_t vocab = model.config().vocab_size;
    const int64_t kv_heads = model.config().num_key_value_heads;
    const int64_t layers = model.num_layers();
    DeviceBackend* backend = model.embed_tokens().weight().backend();
    for (int64_t k = 0;; ++k) {
        const std::string suffix = "." + std::to_string(k);
        if (!reference.contains("ids" + suffix)) break;
        const std::vector<int64_t> ids = ReadIds(reference, "ids" + suffix);
        const std::vector<int64_t> target = ReadIds(reference, "target" + suffix);
        const auto L = static_cast<int64_t>(ids.size());
        if (target.size() != 1 || target[0] < 0 || target[0] >= vocab) {
            throw std::invalid_argument("CompareToAttnLrp: target" + suffix + " must be one id in the vocabulary");
        }
        std::vector<float> x(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= vocab) {
                throw std::invalid_argument("CompareToAttnLrp: ids" + suffix + " holds an id outside the vocabulary");
            }
            x[i] = static_cast<float>(ids[i]);
        }
        const std::vector<float> expected = ReadFloats(reference, "relevance" + suffix, backend, ids.size());
        const size_t kv_size = static_cast<size_t>(layers * kv_heads * L);
        const std::vector<float> expected_k = ReadFloats(reference, "key_relevance" + suffix, backend, kv_size);
        const std::vector<float> expected_v = ReadFloats(reference, "value_relevance" + suffix, backend, kv_size);

        const Tensor logits = model.forward(Tensor(Shape({1, L}), backend, x));
        const size_t seed_index = static_cast<size_t>((L - 1) * vocab + target[0]);
        std::vector<float> seed(static_cast<size_t>(L * vocab), 0.0f);
        seed[seed_index] = logits.to_host_vector()[seed_index];
        const std::vector<float> got =
            model.propagate_relevance(Tensor(logits.shape(), backend, seed), config).to_host_vector();

        AttnLrpSequenceResult r;
        r.num_tokens = L;
        r.target = target[0];
        r.token_correlation = static_cast<float>(PearsonCorrelation(got, expected));
        float largest = 0.0f;
        for (size_t i = 0; i < got.size(); ++i) {
            r.relevance_sum += got[i];
            r.reference_sum += expected[i];
            largest = std::max(largest, std::fabs(expected[i]));
            r.max_relative_diff = std::max(r.max_relative_diff, std::fabs(got[i] - expected[i]));
        }
        r.max_relative_diff = largest > 0.0f ? r.max_relative_diff / largest : r.max_relative_diff;
        r.key_correlation = r.value_correlation = 1.0f;
        const auto layer_size = static_cast<ptrdiff_t>(kv_heads * L);
        for (int64_t i = 0; i < layers; ++i) {
            MultiHeadAttentionModule& mha = model.layer(i).mha();
            const auto begin = static_cast<ptrdiff_t>(i) * layer_size;
            const std::vector<float> ek(expected_k.begin() + begin, expected_k.begin() + begin + layer_size);
            const std::vector<float> ev(expected_v.begin() + begin, expected_v.begin() + begin + layer_size);
            r.key_correlation = std::min(r.key_correlation,
                                         static_cast<float>(PearsonCorrelation(PerHead(mha.key_relevance(), L, kv_heads), ek)));
            r.value_correlation = std::min(r.value_correlation,
                                           static_cast<float>(PearsonCorrelation(PerHead(mha.value_relevance(), L, kv_heads), ev)));
        }
        report.min_token_correlation = std::min(report.min_token_correlation, r.token_correlation);
        report.max_relative_diff = std::max(report.max_relative_diff, r.max_relative_diff);
        report.min_kv_correlation = std::min({report.min_kv_correlation, r.key_correlation, r.value_correlation});
        report.sequences.push_back(r);
    }
    if (report.sequences.empty()) throw std::invalid_argument("CompareToAttnLrp: the reference file holds no ids.0");
    report.passed = report.min_token_correlation > threshold && report.min_kv_correlation > threshold;
    return report;
}

}  // namespace pulsatrix
