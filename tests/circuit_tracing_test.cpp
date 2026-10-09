// FEAT-10: attribution graphs with cross-layer transcoders, against a PyTorch rendering of
// circuit-tracer's definition (tools/golden/make_circuit_golden.py) on the tiny Llama.
#include "pulsatrix/circuit_tracing.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Floats(const SafetensorsFile& f, const std::string& name) {
    CPUBackend cpu;
    return f.tensor(name, &cpu).to_host_vector();
}

std::vector<int64_t> Ints(const SafetensorsFile& f, const std::string& name) {
    auto [ptr, size] = f.bytes(name);
    std::vector<int64_t> v(size / sizeof(int64_t));
    std::memcpy(v.data(), ptr, size);
    return v;
}

constexpr int64_t kF = 6;

CrossLayerTranscoder GoldenTranscoder(const SafetensorsFile& g, int64_t d, CPUBackend* cpu) {
    const std::vector<int64_t> spans = Ints(g, "spans");
    const auto L = static_cast<int64_t>(spans.size());
    CrossLayerTranscoder clt(L, d, cpu);
    for (int64_t l = 0; l < L; ++l) {
        std::vector<Tensor> dec;
        for (int64_t k = 0; k < spans[static_cast<size_t>(l)]; ++k) {
            dec.emplace_back(Shape({kF, d}), cpu, Floats(g, "w_dec." + std::to_string(l) + "." + std::to_string(k)));
        }
        clt.set_layer(l, Tensor(Shape({kF, d}), cpu, Floats(g, "w_enc." + std::to_string(l))), Floats(g, "b_enc." + std::to_string(l)),
                      std::vector<float>(kF, 0.05f), std::move(dec));
        clt.set_output_bias(l, Floats(g, "b_dec." + std::to_string(l)));
    }
    clt.set_skip(1, Tensor(Shape({d, d}), cpu, Floats(g, "w_skip.1")));
    return clt;
}

SafetensorsFile Golden(const std::string& model) {
    return SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/circuit/" + model + "_circuit.safetensors");
}

class CircuitGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(CircuitGolden, MatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden(GetParam());
    std::unique_ptr<CausalLM> model = LoadCausalLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + GetParam(), &cpu);
    const int64_t d = model->config().hidden_size;
    const CrossLayerTranscoder clt = GoldenTranscoder(g, d, &cpu);
    const std::vector<int64_t> ids = Ints(g, "ids");
    CircuitTraceOptions o;
    o.batch = 7;  // several batches, mixing logit and feature targets
    const CircuitTrace t = TraceCircuit(*model, clt, ids, o);

    // The PyTorch model is the same model: same logits.
    const std::vector<float> ref_logits = Floats(g, "logits");
    const std::vector<float> logits = model->forward(Tensor(Shape({1, 7}), &cpu, std::vector<float>(ids.begin(), ids.end()))).to_host_vector();
    double dl = 0;
    for (size_t i = 0; i < logits.size(); ++i) dl = std::max(dl, std::abs(static_cast<double>(logits[i]) - ref_logits[i]));
    EXPECT_LT(dl, 1e-4);

    // The same active features and logits.
    const std::vector<int64_t> feats = Ints(g, "features");
    const std::vector<float> act = Floats(g, "activations");
    ASSERT_EQ(t.features.size(), feats.size() / 3);
    for (size_t i = 0; i < t.features.size(); ++i) {
        EXPECT_EQ(t.features[i].layer, feats[3 * i]);
        EXPECT_EQ(t.features[i].position, feats[3 * i + 1]);
        EXPECT_EQ(t.features[i].index, feats[3 * i + 2]);
        EXPECT_NEAR(t.features[i].activation, act[i], 1e-4);
    }
    EXPECT_EQ(t.logit_tokens, Ints(g, "logit_tokens"));
    const auto n = static_cast<int64_t>(std::sqrt(static_cast<double>(Floats(g, "adjacency").size())));
    ASSERT_EQ(t.num_nodes(), n);

    // Every edge and every target's value.
    const std::vector<float> a = Floats(g, "adjacency"), values = Floats(g, "target_values");
    double worst = 0, scale = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(t.adjacency[i] - a[i]));
        scale = std::max(scale, std::abs(static_cast<double>(a[i])));
    }
    std::printf("[CircuitTracing] tiny %s: %zu features, %zu logits; max |edge - PyTorch| %.2e of max |edge| %.2f\n", GetParam().c_str(), t.features.size(),
                t.logit_tokens.size(), worst, scale);
    EXPECT_LT(worst, 2e-4);
    for (size_t i = 0; i < values.size(); ++i) EXPECT_NEAR(t.target_values[i], values[i], 2e-4) << i;

    // Structure: nothing reaches back in layers or forward in positions.
    for (int64_t r = 0; r < static_cast<int64_t>(t.features.size()); ++r) {
        for (int64_t s = 0; s < static_cast<int64_t>(t.features.size()); ++s) {
            const auto &tf = t.features[static_cast<size_t>(r)], &sf = t.features[static_cast<size_t>(s)];
            if (sf.layer >= tf.layer || sf.position > tf.position) EXPECT_EQ(t.adjacency[static_cast<size_t>(r * n + s)], 0.0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Models, CircuitGolden, ::testing::Values("llama", "gemma3"));

TEST(CircuitTracing, PrunesAndScoresAsCircuitTracerDoes) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden("llama");
    std::unique_ptr<CausalLM> model = LoadCausalLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/llama", &cpu);
    const CrossLayerTranscoder clt = GoldenTranscoder(g, model->config().hidden_size, &cpu);
    CircuitTraceOptions o;
    const CircuitTrace t = TraceCircuit(*model, clt, Ints(g, "ids"), o);
    const CircuitScores s = ScoreCircuit(t);
    std::printf("[CircuitTracing] replacement %.3f, completeness %.3f\n", s.replacement, s.completeness);
    EXPECT_GT(s.replacement, 0.0);
    EXPECT_LT(s.replacement, 1.0);
    EXPECT_GT(s.completeness, 0.0);
    EXPECT_LE(s.completeness, 1.0);

    // Thresholds of 1 keep every node with influence, and every nonzero edge among them.
    o.node_threshold = 1.0;
    o.edge_threshold = 1.0;
    const AttributionGraph all = ToAttributionGraph(t, o);
    o.node_threshold = 0.8;
    o.edge_threshold = 0.98;
    const AttributionGraph pruned = ToAttributionGraph(t, o);
    EXPECT_LT(pruned.nodes.size(), all.nodes.size());
    EXPECT_LT(pruned.links.size(), all.links.size());
    // Tokens and logits stay; every kept feature has an edge in and out.
    std::map<std::string, int> in, out;
    for (const auto& l : pruned.links) {
        ++out[l.source];
        ++in[l.target];
    }
    int tokens = 0, logits = 0;
    for (const auto& n : pruned.nodes) {
        tokens += n.feature_type == "embedding" ? 1 : 0;
        logits += n.feature_type == "logit" ? 1 : 0;
        if (n.feature_type == "cross layer transcoder") {
            EXPECT_GT(in[n.node_id], 0) << n.node_id;
            EXPECT_GT(out[n.node_id], 0) << n.node_id;
        }
    }
    EXPECT_EQ(tokens, 7);
    EXPECT_EQ(logits, static_cast<int>(t.logit_tokens.size()));
    // The JSON the viewer reads round-trips.
    const AttributionGraph back = ParseNeuronpediaGraph(ToNeuronpediaJson(pruned));
    EXPECT_EQ(back.nodes.size(), pruned.nodes.size());
    EXPECT_EQ(back.links.size(), pruned.links.size());
}

TEST(CircuitTracing, LoadsCircuitTracersTwoLayouts) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden("llama");
    const int64_t d = 32;
    const CrossLayerTranscoder want = GoldenTranscoder(g, d, &cpu);
    auto write = [](const std::filesystem::path& path, const std::vector<std::pair<std::string, Tensor>>& tensors) {
        std::vector<std::pair<std::string, const Tensor*>> refs;
        for (const auto& [name, t] : tensors) refs.emplace_back(name, &t);
        WriteSafetensors(path.string(), refs);
    };
    const auto dir = std::filesystem::temp_directory_path() / "pulsatrix_transcoders";
    std::filesystem::remove_all(dir);
    // Cross-layer, circuit-tracer's lazy format: W_dec_<i> is (F, L - i, d).
    std::filesystem::create_directories(dir / "clt");
    for (int64_t l = 0; l < 2; ++l) {
        const std::string i = std::to_string(l);
        std::vector<std::pair<std::string, Tensor>> enc = {
            {"W_enc_" + i, Tensor(Shape({kF, d}), &cpu, Floats(g, "w_enc." + i))},
            {"b_enc_" + i, Tensor(Shape({kF}), &cpu, Floats(g, "b_enc." + i))},
            {"b_dec_" + i, Tensor(Shape({d}), &cpu, Floats(g, "b_dec." + i))},
            {"threshold_" + i, Tensor(Shape({kF}), &cpu, std::vector<float>(kF, 0.05f))}};
        if (l == 1) enc.push_back({"W_skip_1", Tensor(Shape({d, d}), &cpu, Floats(g, "w_skip.1"))});
        write(dir / "clt" / ("W_enc_" + i + ".safetensors"), enc);
        const int64_t span = l == 0 ? 2 : 1;
        std::vector<float> wd(static_cast<size_t>(kF * span * d));
        for (int64_t k = 0; k < span; ++k) {
            const std::vector<float> w = Floats(g, "w_dec." + i + "." + std::to_string(k));
            for (int64_t f = 0; f < kF; ++f) std::copy_n(w.begin() + f * d, d, wd.begin() + (f * span + k) * d);
        }
        write(dir / "clt" / ("W_dec_" + i + ".safetensors"), {{"W_dec_" + i, Tensor(Shape({kF, span, d}), &cpu, wd)}});
    }
    const CrossLayerTranscoder clt = LoadTranscoders((dir / "clt").string(), 2, &cpu);
    EXPECT_EQ(clt.span(0), 2);
    EXPECT_EQ(clt.span(1), 1);
    EXPECT_TRUE(clt.has_skip(1));
    EXPECT_EQ(clt.decoder_row(0, 3, 1), want.decoder_row(0, 3, 1));
    EXPECT_EQ(clt.encoder_row(1, 2), want.encoder_row(1, 2));
    // Per layer (Gemma Scope 2 as circuit-tracer stores it): layer_<i> with W_enc, W_dec (F, d).
    std::filesystem::create_directories(dir / "plt");
    for (int64_t l = 0; l < 2; ++l) {
        const std::string i = std::to_string(l);
        write(dir / "plt" / ("layer_" + i + ".safetensors"),
              {{"W_enc", Tensor(Shape({kF, d}), &cpu, Floats(g, "w_enc." + i))},
                          {"W_dec", Tensor(Shape({kF, d}), &cpu, Floats(g, "w_dec." + i + ".0"))},
                          {"b_enc", Tensor(Shape({kF}), &cpu, Floats(g, "b_enc." + i))},
                          {"b_dec", Tensor(Shape({d}), &cpu, Floats(g, "b_dec." + i))},
                          {"activation_function.threshold", Tensor(Shape({kF}), &cpu, std::vector<float>(kF, 0.05f))}});
    }
    const CrossLayerTranscoder plt = LoadTranscoders((dir / "plt").string(), 2, &cpu);
    EXPECT_EQ(plt.span(0), 1);
    EXPECT_FALSE(plt.has_skip(1));
    EXPECT_EQ(plt.decoder_row(0, 4, 0), want.decoder_row(0, 4, 0));
    EXPECT_THROW((void)LoadTranscoders((dir / "missing").string(), 2, &cpu), std::invalid_argument);
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace pulsatrix
