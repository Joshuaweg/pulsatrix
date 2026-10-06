// LLM-7: AttnLRP parity with LXT. In CI it runs on the tiny models in tests/fixtures/hf_tiny, whose
// attnlrp.safetensors (LXT's AttnLRP) and gradient_x_input.safetensors (LXT's patches off: a
// negative control) come from tools/golden/make_attnlrp_reference.py. With PULSATRIX_GOLDEN_DIR
// set to a directory of real model directories, it checks every one holding an
// attnlrp.safetensors too.

#include "pulsatrix/attnlrp_parity.hpp"

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

class AttnLrpTinyTest : public ::testing::TestWithParam<const char*> {};

// Llama (grouped-query heads), Qwen2 (multi-query, QKV biases) and Qwen3 (QK-Norm).
TEST_P(AttnLrpTinyTest, MatchesLxtToFloatPrecision) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny(GetParam()), &backend);
    AttnLrpReport r = CompareToAttnLrp(*model, SafetensorsFile::Map(Tiny(GetParam()) + "/attnlrp.safetensors"));
    EXPECT_TRUE(r.passed);
    ASSERT_EQ(r.sequences.size(), 3u);
    EXPECT_GT(r.min_token_correlation, 0.99999f);
    EXPECT_GT(r.min_kv_correlation, 0.99999f);
    EXPECT_LT(r.max_relative_diff, 1e-4f);  // not just correlated: the same numbers
    for (const auto& s : r.sequences) EXPECT_NEAR(s.relevance_sum, s.reference_sum, 1e-4f * std::abs(s.reference_sum));
    EXPECT_EQ(r.metadata.at("rule"), "AttnLRP (lxt.efficient)");
}

// The comparison can tell AttnLRP from plain gradient * input on the same model.
TEST_P(AttnLrpTinyTest, DoesNotMatchPlainGradientTimesInput) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny(GetParam()), &backend);
    AttnLrpReport r = CompareToAttnLrp(*model, SafetensorsFile::Map(Tiny(GetParam()) + "/gradient_x_input.safetensors"));
    EXPECT_FALSE(r.passed);
    EXPECT_LT(r.min_token_correlation, 0.9f);
    EXPECT_EQ(r.metadata.at("rule"), "gradient * input");
}

INSTANTIATE_TEST_SUITE_P(TinyModels, AttnLrpTinyTest, ::testing::Values("llama", "qwen2", "qwen3"));

TEST(AttnLrpHarnessTest, PearsonCorrelation) {
    EXPECT_NEAR(PearsonCorrelation({1, 2, 3, 4}, {2, 4, 6, 8}), 1.0, 1e-12);
    EXPECT_NEAR(PearsonCorrelation({1, 2, 3, 4}, {-1, -2, -3, -4}), -1.0, 1e-12);
    EXPECT_NEAR(PearsonCorrelation({1, 2, 3, 4}, {10, 10, 10, 10}), 0.0, 1e-12);  // constant: no correlation
    EXPECT_THROW((void)PearsonCorrelation({1, 2}, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW((void)PearsonCorrelation({}, {}), std::invalid_argument);
}

TEST(AttnLrpHarnessTest, ExposesKeyAndValueRelevanceOnlyAfterARelevancePass) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny("llama"), &backend);
    MultiHeadAttentionModule& mha = model->layer(0).mha();
    EXPECT_THROW((void)mha.key_relevance(), std::logic_error);
    EXPECT_THROW((void)mha.value_relevance(), std::logic_error);
    Tensor logits = model->forward(Tensor(Shape({1, 4}), &backend, std::vector<float>{1, 2, 3, 4}));
    std::vector<float> seed(static_cast<size_t>(logits.numel()), 0.0f);
    seed.back() = 1.0f;
    (void)model->propagate_relevance(Tensor(logits.shape(), &backend, seed), LxtAttnLrpConfig());
    const HfModelConfig& c = model->config();
    const Shape kv({4, c.num_key_value_heads * c.head_dim});
    EXPECT_EQ(mha.key_relevance().shape(), kv);
    EXPECT_EQ(mha.value_relevance().shape(), kv);
}

TEST(AttnLrpHarnessTest, RejectsFilesThatArentReferences) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Tiny("qwen3"), &backend);
    // The model's own weights hold no ids.0; a golden-logit file has ids but no target.
    EXPECT_THROW((void)CompareToAttnLrp(*model, SafetensorsFile::Map(Tiny("qwen3") + "/model.safetensors")),
                 std::invalid_argument);
    EXPECT_ANY_THROW((void)CompareToAttnLrp(*model, SafetensorsFile::Map(Tiny("qwen3") + "/golden.safetensors")));
}

// Real models, when a golden directory is given.
TEST(AttnLrpRealModelsTest, EveryModelInPulsatrixGoldenDirMatchesLxt) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to a directory of models made by tools/golden/make_golden.py "
                        "and make_attnlrp_reference.py";
    }
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "attnlrp.safetensors")) continue;
        CPUBackend backend;
        std::unique_ptr<CausalLM> model = LoadCausalLM(entry.path().string(), &backend);
        AttnLrpReport r = CompareToAttnLrp(*model, SafetensorsFile::Map((entry.path() / "attnlrp.safetensors").string()));
        EXPECT_TRUE(r.passed) << entry.path() << ": token r " << r.min_token_correlation << ", K/V r "
                              << r.min_kv_correlation;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no model directory with an attnlrp.safetensors in " << dir;
}

}  // namespace
}  // namespace pulsatrix
