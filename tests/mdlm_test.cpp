// DIFF-2: MDLM masked diffusion language models in pulsatrix against the authors' modeling code.
// In CI it runs on tests/fixtures/hf_tiny/mdlm, a tiny random MDLM whose adaLN modulation is random
// rather than zero, so folding it into the weights is tested, with mdlm_golden.safetensors from
// tools/golden/make_mdlm_golden.py. With PULSATRIX_GOLDEN_DIR set, every model directory holding an
// mdlm_golden.safetensors is checked too (kuleshov-group/mdlm-owt was).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string TinyMdlm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/mdlm"; }

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

void ExpectMatchesGolden(const std::string& dir, double tolerance) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    EXPECT_EQ(model->config().head, EncoderLMHead::Linear);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/mdlm_golden.safetensors");
    int checked = 0;
    for (int k = 0; golden.contains("ids." + std::to_string(k)); ++k, ++checked) {
        const std::string s = std::to_string(k);
        const std::vector<int64_t> ids = Ids(golden, "ids." + s);
        const int64_t L = static_cast<int64_t>(ids.size());
        const Tensor logits = model->forward(IdTensor(ids, Shape({1, L}), &cpu));
        EXPECT_LT(RelativeError(logits.to_host_vector(), golden.tensor("logits." + s, &cpu).to_host_vector()), tolerance) << dir << " logits " << k;
        for (int64_t i = 0; i <= model->num_layers(); ++i) {
            // The folded gates scale each block's branches, so the residual stream itself matches.
            EXPECT_LT(RelativeError(model->hidden_states()[static_cast<size_t>(i)].to_host_vector(),
                                    golden.tensor("hidden." + s + "." + std::to_string(i), &cpu).to_host_vector()),
                      tolerance)
                << dir << " hidden state " << i << " of sequence " << k;
        }
        EXPECT_LT(RelativeError(model->last_hidden_state().to_host_vector(), golden.tensor("final." + s, &cpu).to_host_vector()), tolerance)
            << dir << " final modulated LayerNorm " << k;
    }
    EXPECT_GE(checked, 4);
    const std::vector<int64_t> ids = Ids(golden, "batch.ids");
    const int64_t L = static_cast<int64_t>(ids.size() / 2);
    EXPECT_LT(RelativeError(model->forward(IdTensor(ids, Shape({2, L}), &cpu)).to_host_vector(), golden.tensor("batch.logits", &cpu).to_host_vector()),
              tolerance)
        << dir << " batch";
}

TEST(Mdlm, TinyMdlmMatchesTheAuthorsCode) { ExpectMatchesGolden(TinyMdlm(), 1e-5); }

TEST(Mdlm, RealModelsInPulsatrixGoldenDirMatchTheAuthorsCode) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to check a downloaded MDLM";
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "mdlm_golden.safetensors")) continue;
        ExpectMatchesGolden(entry.path().string(), 5e-5);
        ++checked;
    }
    if (checked == 0) GTEST_SKIP() << "no MDLM with an mdlm_golden.safetensors in " << dir;
}

TEST(Mdlm, LoadsAsAnEncoderWithALinearHead) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyMdlm(), &cpu);
    const EncoderLMConfig& c = model->config();
    EXPECT_EQ(c.model_type, "mdlm");
    EXPECT_EQ(c.mask_token_id, c.vocab_size - 1);
    EXPECT_EQ(c.intermediate_size, 4 * c.hidden_size);
    EXPECT_FALSE(c.attention_bias);
    EXPECT_FALSE(c.token_dropout);
    std::vector<std::string> names;
    for (const NamedParamRef& p : model->named_parameters()) names.push_back(p.name);
    for (const char* name : {"embed_tokens.weight", "layers.0.mha.q_proj.weight", "layers.1.norm2.bias", "norm.bias", "lm_head.weight", "lm_head.bias"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
    }
    EXPECT_EQ(std::find(names.begin(), names.end(), "layers.0.mha.q_proj.bias"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "lm_head.dense.weight"), names.end());
}

TEST(Mdlm, GradientsAndRelevanceRunThroughTheLinearHead) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyMdlm(), &cpu);
    const int64_t V = model->config().vocab_size, L = 5;
    const Tensor ids = IdTensor({3, V - 1, 7, V - 1, 12}, Shape({1, L}), &cpu);
    const Tensor logits = model->forward(ids);
    // Explain one masked position's logit for token 9.
    std::vector<float> seed(static_cast<size_t>(L * V), 0.0f);
    const size_t target = static_cast<size_t>(1 * V + 9);
    seed[target] = logits.to_host_vector()[target];
    const std::vector<float> r = model->propagate_relevance(Tensor(Shape({1, L, V}), &cpu, seed), LRPRuleConfig{}).to_host_vector();
    ASSERT_EQ(r.size(), static_cast<size_t>(L));
    double total = 0;
    for (float v : r) {
        EXPECT_TRUE(std::isfinite(v));
        total += std::abs(v);
    }
    EXPECT_GT(total, 0.0);
    // The embedding lookup's backward returns one entry per token id, as EncoderLM's does for ESM.
    EXPECT_EQ(model->backward(Tensor(Shape({1, L, V}), &cpu, seed)).numel(), L);
}

TEST(MdlmConfig, RefusesWhatCantBeFolded) {
    const std::string base =
        R"({"architectures": ["MDLM"], "model_type": "mdlm", "vocab_size": 40, "hidden_dim": 32, "cond_dim": 16, "n_blocks": 2,
            "n_heads": 4, "model_length": 64)";
    EXPECT_NO_THROW((void)ParseMdlmConfig(base + R"(, "time_conditioning": false})"));
    EXPECT_THROW((void)ParseMdlmConfig(base + R"(, "time_conditioning": true})"), std::invalid_argument);
    EXPECT_THROW((void)ParseMdlmConfig(R"({"model_type": "esm"})"), std::invalid_argument);
    EXPECT_THROW((void)ParseMdlmConfig(R"({"model_type": "mdlm", "vocab_size": 40})"), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
