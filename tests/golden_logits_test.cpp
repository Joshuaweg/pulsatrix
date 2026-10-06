// LLM-6: the golden-logit harness. In CI it runs on the tiny models in tests/fixtures/hf_tiny, whose
// golden.safetensors come from tools/golden/make_golden.py. With PULSATRIX_GOLDEN_DIR set to a
// directory of real model directories (each with config.json, weights and golden.safetensors, as
// make_golden.py writes them), it checks every one of those too.

#include "pulsatrix/golden_logits.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

std::string Tiny(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

class GoldenTinyTest : public ::testing::TestWithParam<const char*> {};

TEST_P(GoldenTinyTest, MatchesTransformersWellInsideTheThreshold) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny(GetParam()), &backend);
    GoldenReport r = CompareToGolden(*model, SafetensorsFile::Map(Tiny(GetParam()) + "/golden.safetensors"));
    EXPECT_TRUE(r.passed) << "max |diff| " << r.max_abs_diff;
    ASSERT_EQ(r.sequences.size(), 3u);
    EXPECT_EQ(r.sequences[2].num_tokens, 16);
    for (const auto& s : r.sequences) EXPECT_EQ(s.argmax_disagreements, 0);
    EXPECT_EQ(r.metadata.at("dtype"), "float32");
}

INSTANTIATE_TEST_SUITE_P(TinyModels, GoldenTinyTest, ::testing::Values("llama", "qwen2", "qwen3"));

// The harness's reason to exist: a wrong weight that still loads is caught.
TEST(GoldenHarnessTest, CatchesASlightlyWrongWeight) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny("llama"), &backend);
    for (NamedParamRef& p : model->named_parameters()) {
        if (p.name == "layers.1.mha.v_proj.weight") {
            std::vector<float> w = p.ref.value->to_host_vector();
            w[5] += 0.01f;
            *p.ref.value = Tensor(p.ref.value->shape(), &backend, w);
        }
    }
    GoldenReport r = CompareToGolden(*model, SafetensorsFile::Map(Tiny("llama") + "/golden.safetensors"));
    EXPECT_FALSE(r.passed);
    EXPECT_GT(r.max_scaled_diff, 1e-3f);
}

// The threshold scales with each position's largest logit once that exceeds 1 (CompareLogits).
TEST(GoldenHarnessTest, ScalesTheThresholdByEachPositionsLargestLogit) {
    // Position 0's logits reach 20, position 1's stay under 1.
    const std::vector<float> expected = {20.0f, -3.0f, 1.0f, 0.5f, -0.25f, 0.1f};
    std::vector<float> got = expected;
    got[1] += 0.015f;  // 7.5e-4 of 20: drift, within 1e-3
    GoldenSequenceResult drift = CompareLogits(got, expected, 2, 3);
    EXPECT_NEAR(drift.max_abs_diff, 0.015f, 1e-6f);
    EXPECT_NEAR(drift.max_scaled_diff, 0.015f / 20.0f, 1e-7f);
    EXPECT_EQ(drift.argmax_disagreements, 0);

    got = expected;
    got[4] += 0.0015f;  // under a largest logit of 0.5 the threshold stays absolute
    EXPECT_NEAR(CompareLogits(got, expected, 2, 3).max_scaled_diff, 0.0015f, 1e-7f);

    got = expected;
    got[2] = std::nanf("");
    EXPECT_TRUE(std::isinf(CompareLogits(got, expected, 2, 3).max_scaled_diff));
    EXPECT_THROW((void)CompareLogits(got, expected, 3, 3), std::invalid_argument);
}

TEST(GoldenHarnessTest, RejectsFilesThatArentGoldens) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny("qwen3"), &backend);
    // The model's own weights: a safetensors file with no ids.0.
    EXPECT_THROW((void)CompareToGolden(*model, SafetensorsFile::Map(Tiny("qwen3") + "/model.safetensors")),
                 std::invalid_argument);
    // Another model's golden has the same sizes, so it reads, and the comparison fails as it should.
    GoldenReport other = CompareToGolden(*model, SafetensorsFile::Map(Tiny("llama") + "/golden.safetensors"));
    EXPECT_FALSE(other.passed);
}

// Real models, when a golden directory is given.
TEST(GoldenRealModelsTest, EveryModelInPulsatrixGoldenDirMatches) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to a directory of models made by tools/golden/make_golden.py";
    }
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "golden.safetensors")) continue;
        CPUBackend backend;
        std::unique_ptr<CausalLM> model = LoadCausalLM(entry.path().string(), &backend);
        GoldenReport r = CompareToGolden(*model, SafetensorsFile::Map((entry.path() / "golden.safetensors").string()));
        EXPECT_TRUE(r.passed) << entry.path() << ": max |diff| " << r.max_abs_diff;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no model directory with a golden.safetensors in " << dir;
}

}  // namespace
}  // namespace pulsatrix
