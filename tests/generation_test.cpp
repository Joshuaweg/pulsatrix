// LLM-4: generation. Logit processing is checked against Hugging Face's own processors
// (tools/generate_generation_reference_values.py, transformers 5.18.0); decoding against
// models whose next-token distribution is known.

#include "pulsatrix/generation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/determinism.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// ---- Reference values: tools/generate_generation_reference_values.py output, verbatim ----
const std::vector<float> kRepetitionPenalty = {
    1.25f, -0.5f, 2.0f, 0.576923072f, -1.5f, 0.192307696f, 3.0f, -0.25f, 1.5f, 0.384615391f, -1.0f, 2.5f,
};
const std::vector<float> kTemperatureTopK = {
    -kInf, -kInf, 2.85714293f, -kInf, -kInf, -kInf, 4.28571415f, -kInf, 2.14285707f, -kInf, -kInf, 3.57142854f,
};
const std::vector<float> kTopP = {
    -kInf, -kInf, 2.0f, -kInf, -kInf, -kInf, 3.0f, -kInf, 1.5f, -kInf, -kInf, 2.5f,
};
const std::vector<float> kMinP = {
    1.25f, -kInf, 2.0f, 0.75f, -kInf, -kInf, 3.0f, -kInf, 1.5f, -kInf, -kInf, 2.5f,
};
const std::vector<float> kMinNewTokens = {
    1.25f, -0.5f, 2.0f, 0.75f, -1.5f, 0.25f, -kInf, -0.25f, 1.5f, 0.5f, -1.0f, 2.5f,
};
const std::vector<float> kCombined = {
    -kInf, -kInf, 2.5f, -kInf, -kInf, -kInf, 3.75f, -kInf, 1.875f, -kInf, -kInf, 3.125f,
};

const std::vector<float> kLogits{1.25f, -0.5f, 2.0f, 0.75f, -1.5f, 0.25f, 3.0f, -0.25f, 1.5f, 0.5f, -1.0f, 2.5f};
const std::vector<int64_t> kTokens{3, 5, 5, 9};

void ExpectLogits(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isinf(expected[i])) {
            EXPECT_EQ(actual[i], expected[i]) << "token " << i;
        } else {
            EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "token " << i;
        }
    }
}

std::vector<float> Processed(const GenerationConfig& c, int64_t num_new = 0) {
    std::vector<float> l = kLogits;
    ProcessLogits(l, kTokens, num_new, c);
    return l;
}

TEST(GenerationTest, LogitProcessingMatchesHuggingFace) {
    GenerationConfig c;
    c.repetition_penalty = 1.3f;
    ExpectLogits(Processed(c), kRepetitionPenalty);

    c = {};
    c.do_sample = true;
    c.temperature = 0.7f;
    c.top_k = 4;
    ExpectLogits(Processed(c), kTemperatureTopK);

    c = {};
    c.do_sample = true;
    c.top_p = 0.8f;
    ExpectLogits(Processed(c), kTopP);

    c = {};
    c.do_sample = true;
    c.min_p = 0.1f;
    ExpectLogits(Processed(c), kMinP);

    c = {};
    c.min_new_tokens = 2;
    c.eos_token_ids = {6};
    ExpectLogits(Processed(c, /*num_new=*/1), kMinNewTokens);
    ExpectLogits(Processed(c, /*num_new=*/2), kLogits);

    c = {};
    c.do_sample = true;
    c.repetition_penalty = 1.2f;
    c.temperature = 0.8f;
    c.top_k = 6;
    c.top_p = 0.9f;
    c.min_p = 0.05f;
    ExpectLogits(Processed(c), kCombined);

    // Greedy decoding ignores the sampling warpers.
    c.do_sample = false;
    c.repetition_penalty = 1.0f;
    ExpectLogits(Processed(c), kLogits);
}

// A bigram model: the next token is (last + 1) mod 5 with high probability.
std::vector<float> Bigram(const std::vector<int64_t>& tokens) {
    std::vector<float> l(5, 0.0f);
    l[static_cast<size_t>((tokens.back() + 1) % 5)] = 4.0f;
    return l;
}

TEST(GenerationTest, GreedyFollowsTheModelAndStopsAtEos) {
    GenerationConfig c;
    c.max_new_tokens = 10;
    c.eos_token_ids = {4};
    GenerationResult r = Generate(Bigram, {1}, c);
    EXPECT_EQ(r.new_tokens, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(r.tokens, (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(r.finish_reason, FinishReason::Eos);
    // log p = 4 - log(e^4 + 4).
    ASSERT_EQ(r.log_probs.size(), 3u);
    EXPECT_NEAR(r.log_probs[0], 4.0f - std::log(std::exp(4.0f) + 4.0f), 1e-6f);

    c.eos_token_ids = {};
    c.max_new_tokens = 7;
    r = Generate(Bigram, {1}, c);
    EXPECT_EQ(r.new_tokens.size(), 7u);
    EXPECT_EQ(r.finish_reason, FinishReason::Length);
    EXPECT_EQ(r.new_tokens.back(), 3);  // 2 3 4 0 1 2 3
}

TEST(GenerationTest, MinNewTokensHoldsOffEos) {
    GenerationConfig c;
    c.max_new_tokens = 10;
    c.eos_token_ids = {2};
    EXPECT_EQ(Generate(Bigram, {1}, c).new_tokens, (std::vector<int64_t>{2}));
    c.min_new_tokens = 3;
    GenerationResult r = Generate(Bigram, {1}, c);
    // EOS (2) banned for the first 3 tokens; the model's next favourite is then token 0 (tied, lowest).
    ASSERT_GE(r.new_tokens.size(), 4u);
    EXPECT_NE(r.new_tokens[0], 2);
    EXPECT_EQ(r.finish_reason, FinishReason::Eos);
}

TEST(GenerationTest, SamplingIsSeededAndFollowsTheDistribution) {
    // A model whose next-token distribution never changes: draw many tokens in one call.
    const std::vector<float> fixed{0.0f, 1.0f, 2.0f, -1.0f};
    auto model = [&](const std::vector<int64_t>&) { return fixed; };
    GenerationConfig c;
    c.do_sample = true;
    c.temperature = 0.5f;  // p ~ exp(2 * logit)
    c.max_new_tokens = 40000;
    c.seed = 17;
    GenerationResult r = Generate(model, {0}, c);
    std::map<int64_t, int> counts;
    for (int64_t t : r.new_tokens) ++counts[t];
    double z = 0.0;
    for (float l : fixed) z += std::exp(2.0 * l);
    for (int64_t t = 0; t < 4; ++t) {
        const double p = std::exp(2.0 * fixed[static_cast<size_t>(t)]) / z;
        EXPECT_NEAR(counts[t] / 40000.0, p, 0.01) << "token " << t;
    }
    // Same seed, same tokens; another seed, other tokens.
    c.max_new_tokens = 50;
    EXPECT_EQ(Generate(model, {0}, c).new_tokens, Generate(model, {0}, c).new_tokens);
    GenerationConfig other = c;
    other.seed = 18;
    EXPECT_NE(Generate(model, {0}, c).new_tokens, Generate(model, {0}, other).new_tokens);
    // Without an explicit seed, set_seed() makes it reproducible.
    c.seed.reset();
    set_seed(5);
    const auto first = Generate(model, {0}, c).new_tokens;
    set_seed(5);
    EXPECT_EQ(Generate(model, {0}, c).new_tokens, first);
    // top_k = 1 makes sampling greedy.
    c.top_k = 1;
    for (int64_t t : Generate(model, {0}, c).new_tokens) EXPECT_EQ(t, 2);
}

// A real causal LM stack: embedding, a causal transformer block, the tied head.
TEST(GenerationTest, ModuleStackAdapterReadsTheLastPosition) {
    CPUBackend backend;
    EmbeddingModule embedding(6, 4, &backend);
    std::vector<float> table(24);
    for (size_t i = 0; i < table.size(); ++i) table[i] = 0.2f * static_cast<float>((i * 7 + 3) % 11) - 1.0f;
    embedding.set_weight(table);
    AttentionConfig ac;
    ac.d_model = 4;
    ac.num_heads = 2;
    ac.num_kv_heads = 1;
    ac.rope_layout = RoPELayout::RotateHalf;
    ac.causal = true;
    TransformerBlock block(ac, 8, &backend);
    for (ParamRef& p : block.parameters()) {
        std::vector<float> v(static_cast<size_t>(p.value->numel()));
        for (size_t i = 0; i < v.size(); ++i) v[i] = 0.1f * static_cast<float>((i * 5 + 2) % 9) - 0.4f;
        *p.value = Tensor(p.value->shape(), &backend, v);
    }
    TiedLMHeadModule head(embedding, &backend);
    NextTokenLogitsFn lm = MakeNextTokenLogits({&embedding, &block, &head}, &backend);

    const std::vector<int64_t> prompt{1, 4, 2};
    const std::vector<float> last = lm(prompt);
    ASSERT_EQ(last.size(), 6u);
    Tensor full = head.forward(block.forward(embedding.forward(Tensor(Shape({1, 3}), &backend, {1.0f, 4.0f, 2.0f}))));
    for (size_t v = 0; v < 6; ++v) EXPECT_FLOAT_EQ(last[v], full.data()[2 * 6 + static_cast<int64_t>(v)]);
    // Causal: the logits after a prefix don't depend on what follows it.
    const std::vector<float> prefix = lm({1, 4});
    for (size_t v = 0; v < 6; ++v) EXPECT_NEAR(prefix[v], full.data()[1 * 6 + static_cast<int64_t>(v)], 1e-5f);

    GenerationConfig c;
    c.max_new_tokens = 5;
    GenerationResult r = Generate(lm, prompt, c);
    EXPECT_EQ(r.new_tokens.size(), 5u);
    for (int64_t t : r.new_tokens) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, 6);
    }
}

TEST(GenerationTest, RejectsBadInput) {
    GenerationConfig c;
    EXPECT_THROW((void)Generate(Bigram, {}, c), std::invalid_argument);
    c.temperature = 0.0f;
    EXPECT_THROW((void)Generate(Bigram, {1}, c), std::invalid_argument);
    c = {};
    c.top_p = 0.0f;
    EXPECT_THROW((void)Generate(Bigram, {1}, c), std::invalid_argument);
    c = {};
    c.top_k = -1;
    EXPECT_THROW((void)Generate(Bigram, {1}, c), std::invalid_argument);
    auto nan_model = [](const std::vector<int64_t>&) { return std::vector<float>{0.0f, std::nanf("")}; };
    EXPECT_THROW((void)Generate(nan_model, {1}, GenerationConfig{}), std::invalid_argument);
    auto empty_model = [](const std::vector<int64_t>&) { return std::vector<float>{}; };
    EXPECT_THROW((void)Generate(empty_model, {1}, GenerationConfig{}), std::invalid_argument);
    // Every token banned: EOS is the whole vocabulary and min_new_tokens holds it off.
    GenerationConfig banned;
    banned.min_new_tokens = 1;
    banned.eos_token_ids = {0, 1, 2, 3, 4};
    EXPECT_THROW((void)Generate(Bigram, {1}, banned), std::invalid_argument);
    EXPECT_THROW((void)MakeNextTokenLogits({}, nullptr), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
