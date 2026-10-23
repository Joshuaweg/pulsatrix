// DIFF-4: dLLM's A2D ("autoregressive to diffusion") checkpoints, Qwen3 with bidirectional attention,
// in pulsatrix's CausalLM against the authors' modeling code. In CI it runs on
// tests/fixtures/hf_tiny/a2d_qwen3, a tiny random A2D Qwen3, with a2d_golden.safetensors from
// tools/golden/make_a2d_golden.py. With PULSATRIX_GOLDEN_DIR set, every model directory holding an
// a2d_golden.safetensors is checked too (dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 was).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/masked_diffusion_lrp.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string TinyA2d() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/a2d_qwen3"; }

std::vector<int64_t> Ids(const SafetensorsFile& f, const std::string& name) {
    auto [ptr, size] = f.bytes(name);
    std::vector<int64_t> ids(size / sizeof(int64_t));
    std::memcpy(ids.data(), ptr, size);
    return ids;
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
    std::unique_ptr<CausalLM> model = LoadCausalLM(dir, &cpu);
    EXPECT_FALSE(model->config().causal);
    const DenoiserFn denoise = MakeDenoiser(*model, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/a2d_golden.safetensors");
    int checked = 0;
    for (int k = 0; golden.contains("ids." + std::to_string(k)); ++k, ++checked) {
        const std::string s = std::to_string(k);
        EXPECT_LT(RelativeError(denoise(Ids(golden, "ids." + s)), golden.tensor("logits." + s, &cpu).to_host_vector()), tolerance)
            << dir << " logits " << k;
    }
    EXPECT_GE(checked, 3);
}

TEST(A2d, ConfigIsQwen3WithBidirectionalAttention) {
    const HfModelConfig c = ReadHfConfig(TinyA2d() + "/config.json");
    EXPECT_EQ(c.model_type, "a2d-qwen3");
    EXPECT_FALSE(c.causal);
    EXPECT_TRUE(c.qk_norm);
    EXPECT_TRUE(c.unsupported.empty());
    EXPECT_FALSE(ToAttentionConfig(c).causal);
    EXPECT_TRUE(ReadHfConfig(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/qwen3/config.json").causal);
}

TEST(A2d, TinyA2dQwen3MatchesTheAuthorsCode) { ExpectMatchesGolden(TinyA2d(), 1e-5); }

TEST(A2d, EveryPositionSeesTheWholeSequenceAndThereIsNoNextTokenCache) {
    CPUBackend cpu;
    std::unique_ptr<CausalLM> model = LoadCausalLM(TinyA2d(), &cpu);
    const DenoiserFn denoise = MakeDenoiser(*model, &cpu);
    std::vector<int64_t> ids = {5, 9, 63, 63, 2};
    const std::vector<float> before = denoise(ids);
    ids[4] = 7;
    const std::vector<float> after = denoise(ids);
    const int64_t V = model->config().vocab_size;
    double change = 0;
    for (int64_t v = 0; v < V; ++v) change = std::max(change, std::abs(static_cast<double>(before[v]) - after[v]));
    EXPECT_GT(change, 1e-4) << "position 0 didn't see position 4";
    EXPECT_THROW((void)model->next_token_logits(16), std::logic_error);

    std::unique_ptr<CausalLM> causal = LoadCausalLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/qwen3", &cpu);
    EXPECT_THROW((void)MakeDenoiser(*causal, &cpu), std::invalid_argument);
    EXPECT_THROW((void)CausalLMStepModel(*causal, &cpu), std::invalid_argument);
}

TEST(A2d, CommitmentGraphOnTheTinyA2dIsOnePassPerStepAndBalances) {
    CPUBackend cpu;
    std::unique_ptr<CausalLM> model = LoadCausalLM(TinyA2d(), &cpu);
    const int64_t mask = 63;
    std::vector<int64_t> input(9, mask);
    input[0] = 3;
    input[1] = 11;
    MaskedDiffusionConfig config;
    config.steps = 4;
    config.mask_token_id = mask;
    const MaskedDiffusionResult t = SampleMaskedDiffusion(MakeDenoiser(*model, &cpu), input, config);
    CausalLMStepModel step(*model, &cpu);
    const std::vector<CommitmentTarget> targets = {{7, std::nullopt}, {8, std::nullopt}};
    const CommitmentRelevance all = ExplainCommitments(step, t, mask, targets);
    EXPECT_NEAR(all.value, ReplayCommitmentValue(MakeDenoiser(*model, &cpu), t, input, config, targets), 1e-4);
    EXPECT_NEAR(all.prompt_total() + all.mask_total() + all.leak_total(), all.value, 1e-6 * std::abs(all.value) + 1e-6);
    double scale = 0;
    for (float v : all.prompt) scale += std::abs(v);
    ASSERT_GT(scale, 0.0);
    std::vector<double> prompt(9, 0.0);
    for (const CommitmentTarget& target : targets) {
        const CommitmentRelevance one = ExplainCommitments(step, t, mask, {target});
        for (size_t p = 0; p < 9; ++p) prompt[p] += one.prompt[p];
    }
    for (size_t p = 0; p < 9; ++p) EXPECT_NEAR(all.prompt[p], prompt[p], 1e-4 * scale) << p;
}

TEST(A2d, RealModelsInPulsatrixGoldenDirMatchTheAuthorsCode) {
    const char* root = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (root == nullptr) GTEST_SKIP() << "PULSATRIX_GOLDEN_DIR not set";
    int found = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!std::filesystem::exists(entry.path() / "a2d_golden.safetensors")) continue;
        ++found;
        // A 28-layer model with QK-Norm in fp32: rounding differs from PyTorch's in the last bits.
        ExpectMatchesGolden(entry.path().string(), 1e-4);
    }
    if (found == 0) GTEST_SKIP() << "no a2d_golden.safetensors under " << root;
}

}  // namespace
}  // namespace pulsatrix
