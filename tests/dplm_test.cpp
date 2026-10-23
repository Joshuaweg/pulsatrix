// DIFF-4: DPLM's remasking sampler. UnmaskingSchedule::Remasking against DPLM's own decoding code
// (tools/golden/make_dplm_golden.py runs `generate` and `_reparam_decoding` from DPLM's source): in
// CI on tests/fixtures/hf_tiny/esm, a tiny random ESM-2, and with PULSATRIX_GOLDEN_DIR set, on every
// model directory holding a dplm_golden.safetensors (airkingbd/dplm_150m and dplm_650m were). Every
// step's input must match, so the order positions are filled, taken back and refilled in is DPLM's.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/masked_diffusion_lrp.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::vector<int64_t> Ids(const SafetensorsFile& f, const std::string& name) {
    auto [ptr, size] = f.bytes(name);
    std::vector<int64_t> ids(size / sizeof(int64_t));
    std::memcpy(ids.data(), ptr, size);
    return ids;
}

/** @brief DPLM's sampler settings for an ESM vocabulary: it never generates <cls>, <pad>, <eos>
 *         or X (EsmForDPLM sets their logits to -inf, as it does the mask's). */
MaskedDiffusionConfig DplmConfig(const EncoderLMConfig& model, int64_t steps) {
    MaskedDiffusionConfig c;
    c.steps = steps;
    c.mask_token_id = model.mask_token_id;
    c.schedule = UnmaskingSchedule::Remasking;
    c.suppressed_token_ids = {0, 1, 2, 24};  // <cls>, <pad>, <eos>, X in ESM's vocab.txt
    return c;
}

void ExpectMatchesDplm(const std::string& dir) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/dplm_golden.safetensors");
    const int64_t steps = std::stoll(golden.metadata().at("steps"));
    const MaskedDiffusionConfig config = DplmConfig(model->config(), steps);
    int checked = 0;
    for (int k = 0; golden.contains("input." + std::to_string(k)); ++k, ++checked) {
        const std::string s = std::to_string(k);
        const std::vector<int64_t> input = Ids(golden, "input." + s);
        const std::vector<int64_t> seen = Ids(golden, "steps." + s);
        const size_t L = input.size();
        const MaskedDiffusionResult r = SampleMaskedDiffusion(MakeDenoiser(*model, &cpu), input, config);
        ASSERT_EQ(r.steps.size(), static_cast<size_t>(steps)) << dir << " case " << k;
        for (size_t t = 0; t < r.steps.size(); ++t) {
            EXPECT_EQ(r.steps[t].input, std::vector<int64_t>(seen.begin() + t * L, seen.begin() + (t + 1) * L))
                << dir << " case " << k << " step " << t;
        }
        EXPECT_EQ(r.tokens, Ids(golden, "output." + s)) << dir << " case " << k;
    }
    EXPECT_GE(checked, 3);
}

TEST(Dplm, RemaskingFollowsDplmsOwnDecodingStepByStep) {
    ExpectMatchesDplm(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm");
}

TEST(Dplm, ExplainingARemaskingTrajectoryBalancesAndReplays) {
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm";
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/dplm_golden.safetensors");
    const std::vector<int64_t> input = Ids(golden, "input.0");
    const MaskedDiffusionConfig config = DplmConfig(model->config(), 6);
    const DenoiserFn denoise = MakeDenoiser(*model, &cpu);
    const MaskedDiffusionResult t = SampleMaskedDiffusion(denoise, input, config);
    // Replaying the frozen order, re-masks included, gives the trajectory back.
    const MaskedDiffusionResult again = ReplayMaskedDiffusion(denoise, t, input, config);
    EXPECT_EQ(again.tokens, t.tokens);
    for (size_t s = 0; s < t.steps.size(); ++s) {
        EXPECT_EQ(again.steps[s].input, t.steps[s].input) << s;
        EXPECT_EQ(again.steps[s].remasked, t.steps[s].remasked) << s;
    }

    std::vector<CommitmentTarget> targets;
    for (size_t p = 0; p < input.size(); ++p) {
        if (input[p] == config.mask_token_id) targets.push_back({static_cast<int64_t>(p), std::nullopt});
    }
    EncoderLMStepModel step(*model, &cpu);
    const CommitmentRelevance r = ExplainCommitments(step, t, config.mask_token_id, targets);
    EXPECT_NEAR(r.value, ReplayCommitmentValue(denoise, t, input, config, targets), 1e-4 * std::abs(r.value) + 1e-4);
    EXPECT_NEAR(r.prompt_total() + r.mask_total() + r.leak_total(), r.value, 1e-5 * std::abs(r.value) + 1e-5);
    for (float v : r.prompt) EXPECT_TRUE(std::isfinite(v));

    // The positions written last, explained alone, leave the earlier ones to patch.
    int64_t last = 0;
    for (int64_t s : t.last_write_step) last = std::max(last, s);
    std::vector<CommitmentTarget> late;
    for (size_t p = 0; p < input.size(); ++p) {
        if (t.last_write_step[p] == last) late.push_back({static_cast<int64_t>(p), std::nullopt});
    }
    ASSERT_LT(late.size(), targets.size());
    const CommitmentRelevance graph = ExplainCommitments(step, t, config.mask_token_id, late);
    const CommitmentPatchResult patched = CommitmentPatchCheck(denoise, t, config, late, graph);
    EXPECT_EQ(patched.positions.size(), targets.size() - late.size());
    for (float e : patched.effect) EXPECT_TRUE(std::isfinite(e));
}

TEST(Dplm, RealModelsInPulsatrixGoldenDirMatchDplmsOwnDecoding) {
    const char* root = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (root == nullptr) GTEST_SKIP() << "PULSATRIX_GOLDEN_DIR not set";
    int found = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!std::filesystem::exists(entry.path() / "dplm_golden.safetensors")) continue;
        ++found;
        ExpectMatchesDplm(entry.path().string());
    }
    if (found == 0) GTEST_SKIP() << "no dplm_golden.safetensors under " << root;
}

}  // namespace
}  // namespace pulsatrix
