// PLM-2: ESM-2 in pulsatrix against transformers. In CI it runs on tests/fixtures/hf_tiny/esm (a tiny
// random ESM-2 and its esm_golden.safetensors from tools/golden/make_esm_golden.py). With
// PULSATRIX_GOLDEN_DIR set, every model directory holding an esm_golden.safetensors is checked too
// (facebook/esm2_t6_8M_UR50D and esm2_t33_650M_UR50D were).
#include "pulsatrix/encoder_lm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string Fixtures() { return std::string(PULSATRIX_TEST_FIXTURES_DIR); }
std::string TinyEsm() { return Fixtures() + "/hf_tiny/esm"; }

std::vector<int64_t> Ids(const SafetensorsFile& f, const std::string& name) {
    auto [ptr, size] = f.bytes(name);
    std::vector<int64_t> ids(size / sizeof(int64_t));
    std::memcpy(ids.data(), ptr, size);
    return ids;
}

Tensor IdTensor(const std::vector<int64_t>& ids, Shape shape, DeviceBackend* backend) {
    return Tensor(std::move(shape), backend, std::vector<float>(ids.begin(), ids.end()));
}

/** @brief max |a - b| relative to max |b|. */
double RelativeError(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double diff = 0, scale = 1e-12;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        diff = std::max(diff, std::abs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::abs(static_cast<double>(b[i])));
    }
    return diff / scale;
}

/** @brief Every output esm_golden.safetensors records, at float32 precision. */
void ExpectMatchesGolden(const std::string& dir, double tolerance) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/esm_golden.safetensors");
    int checked = 0;
    for (int k = 0; golden.contains("ids." + std::to_string(k)); ++k, ++checked) {
        const std::string s = std::to_string(k);
        const std::vector<int64_t> ids = Ids(golden, "ids." + s);
        const int64_t L = static_cast<int64_t>(ids.size());
        const Tensor logits = model->forward(IdTensor(ids, Shape({1, L}), &cpu));
        EXPECT_LT(RelativeError(logits.to_host_vector(), golden.tensor("logits." + s, &cpu).to_host_vector()), tolerance) << dir << " logits " << k;
        for (int64_t i = 0; i <= model->num_layers(); ++i) {
            EXPECT_LT(RelativeError(model->hidden_states()[static_cast<size_t>(i)].to_host_vector(),
                                    golden.tensor("hidden." + s + "." + std::to_string(i), &cpu).to_host_vector()),
                      tolerance)
                << dir << " hidden state " << i << " of sequence " << k;
        }
        EXPECT_LT(RelativeError(model->last_hidden_state().to_host_vector(), golden.tensor("final." + s, &cpu).to_host_vector()), tolerance)
            << dir << " final hidden state " << k;
        for (int64_t i = 0; i < model->num_layers(); ++i) {
            EXPECT_LT(RelativeError(model->layer(i).mha().last_attention_weights().to_host_vector(),
                                    golden.tensor("attn." + s + "." + std::to_string(i), &cpu).to_host_vector()),
                      tolerance)
                << dir << " attention of layer " << i << ", sequence " << k;
        }
    }
    EXPECT_GE(checked, 4);
    // A padded batch: padding is masked out of attention and of token dropout's ratio.
    const std::vector<int64_t> ids = Ids(golden, "batch.ids"), mask = Ids(golden, "batch.mask");
    const int64_t L = static_cast<int64_t>(ids.size() / 2);
    model->set_padding_mask(IdTensor(mask, Shape({2, L}), &cpu));
    const std::vector<float> got = model->forward(IdTensor(ids, Shape({2, L}), &cpu)).to_host_vector();
    const std::vector<float> want = golden.tensor("batch.logits", &cpu).to_host_vector();
    // Compare real tokens only: padding rows are don't-cares.
    const int64_t V = model->config().vocab_size;
    std::vector<float> g, w;
    for (size_t t = 0; t < mask.size(); ++t) {
        if (mask[t] == 0) continue;
        g.insert(g.end(), got.begin() + static_cast<std::ptrdiff_t>(t * V), got.begin() + static_cast<std::ptrdiff_t>((t + 1) * V));
        w.insert(w.end(), want.begin() + static_cast<std::ptrdiff_t>(t * V), want.begin() + static_cast<std::ptrdiff_t>((t + 1) * V));
    }
    EXPECT_LT(RelativeError(g, w), tolerance) << dir << " padded batch";
}

TEST(EncoderLM, TinyEsmMatchesTransformers) { ExpectMatchesGolden(TinyEsm(), 1e-5); }

TEST(EncoderLM, RealModelsInPulsatrixGoldenDirMatchTransformers) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to check downloaded ESM-2 models";
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "esm_golden.safetensors")) continue;
        ExpectMatchesGolden(entry.path().string(), 5e-5);
        ++checked;
    }
    if (checked == 0) GTEST_SKIP() << "no ESM model with an esm_golden.safetensors in " << dir;
}

TEST(EsmTokenizer, MatchesTransformersEsmTokenizer) {
    const TextTokenizer tok = LoadEsmTokenizer(Fixtures() + "/esm/vocab.txt");
    const SafetensorsFile golden = SafetensorsFile::Map(TinyEsm() + "/esm_golden.safetensors");
    const JsonValue cases = ParseJson(golden.metadata().at("tokenizer_cases"));
    ASSERT_GE(cases.as_array().size(), 10u);
    for (size_t j = 0; j < cases.as_array().size(); ++j) {
        const std::string& text = cases.as_array()[j].as_string();
        EXPECT_EQ(tok.encode(text).ids, Ids(golden, "tok." + std::to_string(j))) << "\"" << text << "\"";
    }
    const JsonValue sequences = ParseJson(golden.metadata().at("sequences"));
    for (size_t k = 0; k < sequences.as_array().size(); ++k) {
        EXPECT_EQ(tok.encode(sequences.as_array()[k].as_string()).ids, Ids(golden, "ids." + std::to_string(k)));
    }
    EXPECT_EQ(tok.decode(tok.encode("MKT<mask>").ids), "<cls> M K T <mask> <eos>");
    EXPECT_EQ(tok.decode(tok.encode("MKT<mask>").ids, /*skip_special_tokens=*/true), "M K T");
}

TEST(EncoderLM, GradientsMatchFiniteDifferences) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const std::vector<int64_t> ids = {0, 20, 15, 32, 5, 19, 2};  // <cls> M K <mask> A Y <eos>
    const Tensor x = IdTensor(ids, Shape({1, 7}), &cpu);
    std::mt19937 gen(3);
    std::normal_distribution<float> d;
    std::vector<float> r(7 * 33);
    for (float& v : r) v = d(gen);
    const auto loss = [&] {
        const std::vector<float> y = model->forward(x).to_host_vector();
        double s = 0;
        for (size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * r[i];
        return s;
    };
    (void)loss();
    (void)model->backward(Tensor(Shape({1, 7, 33}), &cpu, r));
    int checked = 0;
    for (auto& p : model->named_parameters()) {
        for (const char* name : {"embed_tokens.weight", "layers.0.mha.q_proj.weight", "layers.1.mlp.fc1.bias", "norm.weight", "lm_head.dense.weight",
                                 "lm_head.norm.bias", "lm_head.bias"}) {
            if (p.name != name) continue;
            const std::vector<float> grad = p.ref.grad->to_host_vector();
            std::vector<float> w = p.ref.value->to_host_vector();
            // embed_tokens row 20 (M) is used twice: as input and by the tied decoder.
            const size_t k = p.name == std::string("embed_tokens.weight") ? 20 * 32 + 3 : w.size() / 3;
            const float h = 1e-2f, w0 = w[k];
            w[k] = w0 + h;
            *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
            const double up = loss();
            w[k] = w0 - h;
            *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
            const double down = loss();
            w[k] = w0;
            *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
            EXPECT_NEAR(grad[k], (up - down) / (2 * h), 2e-3 * (1 + std::abs(grad[k]))) << name;
            ++checked;
        }
    }
    EXPECT_EQ(checked, 7);
}

TEST(EncoderLM, RelevanceReachesEveryTokenAndSplitsByLayer) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const std::vector<int64_t> ids = {0, 20, 15, 32, 5, 19, 2};
    const Tensor logits = model->forward(IdTensor(ids, Shape({1, 7}), &cpu));
    std::vector<float> seed(7 * 33, 0.0f);
    seed[3 * 33 + 5] = logits.to_host_vector()[3 * 33 + 5];  // explain "A" at the masked position
    const Tensor r(Shape({1, 7, 33}), &cpu, seed);
    const std::vector<float> tokens = model->propagate_relevance(r, LRPRuleConfig{}).to_host_vector();
    ASSERT_EQ(tokens.size(), 7u);
    const std::vector<Tensor> layers = model->propagate_relevance_by_layer(r, LRPRuleConfig{});
    ASSERT_EQ(layers.size(), static_cast<size_t>(model->num_layers() + 1));
    const std::vector<float> first = layers.front().to_host_vector();
    // Relative to the largest element: on this random tiny model single dimensions carry about
    // 5e4 while a token's total can be about 20, so the float sum cancels.
    float largest = 0;
    for (float v : first) largest = std::max(largest, std::abs(v));
    for (size_t t = 0; t < 7; ++t) {
        double s = 0;
        for (size_t k = 0; k < 32; ++k) s += first[t * 32 + k];
        EXPECT_NEAR(s, tokens[t], 1e-6 * largest);
    }
    // The masked token's embedding is zeroed, so the epsilon rule gives it no relevance through
    // its own residual path; it still gets some through attention (other tokens' keys and values).
    double total = 0;
    for (float v : tokens) total += std::abs(v);
    EXPECT_GT(total, 0.0);
}

TEST(EncoderLM, RejectsBadInput) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    EXPECT_THROW((void)model->forward(Tensor(Shape({1, 3}), &cpu, {0, 40, 2})), std::invalid_argument);
    EXPECT_THROW((void)model->forward(Tensor(Shape({3}), &cpu, {0, 4, 2})), std::invalid_argument);
    EXPECT_THROW(model->set_padding_mask(Tensor(Shape({1, 3}), &cpu, {1, 0.5f, 1})), std::invalid_argument);
    model->set_padding_mask(Tensor(Shape({1, 3}), &cpu, {1, 1, 0}));
    EXPECT_THROW((void)model->forward(Tensor(Shape({1, 4}), &cpu, {0, 4, 5, 2})), std::invalid_argument);
    model->clear_padding_mask();
    EXPECT_NO_THROW((void)model->forward(Tensor(Shape({1, 4}), &cpu, {0, 4, 5, 2})));
}

TEST(EsmConfig, RefusesWhatEncoderLMDoesntRun) {
    const std::string base = R"({"model_type": "esm", "architectures": ["EsmForMaskedLM"], "vocab_size": 33, "hidden_size": 32,
        "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4, "position_embedding_type": "rotary",
        "token_dropout": true, "mask_token_id": 32, "pad_token_id": 1, "layer_norm_eps": 1e-05)";
    const EncoderLMConfig c = ParseEsmConfig(base + "}");
    EXPECT_EQ(c.hidden_size, 32);
    EXPECT_TRUE(c.token_dropout);
    EXPECT_FLOAT_EQ(c.layer_norm_eps, 1e-5f);
    EXPECT_THROW((void)ParseEsmConfig(base + R"(, "position_embedding_type": "absolute"})"), std::invalid_argument);
    EXPECT_THROW((void)ParseEsmConfig(base + R"(, "is_folding_model": true})"), std::invalid_argument);
    EXPECT_THROW((void)ParseEsmConfig(base + R"(, "emb_layer_norm_before": true})"), std::invalid_argument);
    EXPECT_THROW((void)ParseEsmConfig(base + R"(, "architectures": ["EsmForTokenClassification"]})"), std::invalid_argument);
    EXPECT_THROW((void)ParseEsmConfig(R"({"model_type": "llama"})"), std::invalid_argument);
}

TEST(Fasta, ParsesMultiLineRecordsCommentsAndCrLf) {
    const std::vector<FastaRecord> r = ParseFasta(
        "; a comment\r\n>sp|P0CG48|UBC_HUMAN Polyubiquitin-C OS=Homo sapiens\r\nMQIFVKTLTG\r\nKTITLEVE PS\r\n\r\n>short\nMK*\n>empty\n");
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].id, "sp|P0CG48|UBC_HUMAN");
    EXPECT_EQ(r[0].description, "Polyubiquitin-C OS=Homo sapiens");
    EXPECT_EQ(r[0].sequence, "MQIFVKTLTGKTITLEVEPS");
    EXPECT_EQ(r[1].sequence, "MK*");
    EXPECT_EQ(r[2].sequence, "");
    EXPECT_EQ(ParseFasta(WriteFasta(r, 7))[0].sequence, r[0].sequence);
    EXPECT_EQ(WriteFasta({{"a", "", "MKTAYIA"}}, 3), ">a\nMKT\nAYI\nA\n");
    EXPECT_THROW((void)ParseFasta("MKT\n>a\n"), std::invalid_argument);
    EXPECT_THROW((void)ParseFasta(">\nMKT\n"), std::invalid_argument);
    EXPECT_TRUE(ParseFasta("").empty());
}

}  // namespace
}  // namespace pulsatrix
