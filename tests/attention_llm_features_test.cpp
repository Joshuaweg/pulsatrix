// LLM-1: the attention features real checkpoints need -- grouped-query attention, head_dim
// independent of d_model, optional biases, causal and key padding masks, RoPE layout and
// position offset. tests/attention_reference_test.cpp checks the same features against
// Hugging Face; these tests check the properties that define each one, plus gradients by
// finite differences and the relevance rules.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rope_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Pattern(int64_t n, float scale, float shift) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = scale * static_cast<float>((i * 7 + 3) % 11 - 5) + shift;
    }
    return v;
}

// A Llama-shaped config: 4 query heads sharing 2 K/V heads, head_dim 4 (query width 16 vs.
// d_model 6), rotate-half RoPE, causal.
AttentionConfig GqaConfig() {
    AttentionConfig c;
    c.d_model = 6;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 4;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 50.0f;
    c.qkv_bias = true;
    c.out_bias = false;
    c.causal = true;
    return c;
}

void FillWeights(MultiHeadAttentionModule& m) {
    const int64_t d = m.d_model(), q = m.num_heads() * m.head_dim(), kv = m.num_kv_heads() * m.head_dim();
    m.q_proj().set_weight(Pattern(d * q, 0.11f, 0.02f));
    m.k_proj().set_weight(Pattern(d * kv, 0.13f, -0.01f));
    m.v_proj().set_weight(Pattern(d * kv, 0.09f, 0.03f));
    m.out_proj().set_weight(Pattern(q * d, 0.07f, -0.02f));
    if (m.q_proj().uses_bias()) {
        m.q_proj().set_bias(Pattern(q, 0.05f, 0.0f));
        m.k_proj().set_bias(Pattern(kv, -0.04f, 0.01f));
        m.v_proj().set_bias(Pattern(kv, 0.03f, 0.0f));
    }
    if (m.out_proj().uses_bias()) {
        m.out_proj().set_bias(Pattern(d, 0.02f, 0.0f));
    }
}

float Objective(MultiHeadAttentionModule& m, const Tensor& x, const std::vector<float>& g) {
    Tensor y = m.forward(x);
    double s = 0.0;
    for (int64_t i = 0; i < y.numel(); ++i) {
        s += static_cast<double>(y.data()[i]) * g[static_cast<size_t>(i)];
    }
    return static_cast<float>(s);
}

class AttentionLlmFeaturesTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// ---- Configuration -------------------------------------------------------------------------

TEST_F(AttentionLlmFeaturesTest, ConfigResolvesDefaultsAndSizesProjections) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    EXPECT_EQ(m.num_heads(), 4);
    EXPECT_EQ(m.num_kv_heads(), 2);
    EXPECT_EQ(m.head_dim(), 4);
    EXPECT_EQ(m.q_proj().weight().shape(), Shape({6, 16}));
    EXPECT_EQ(m.k_proj().weight().shape(), Shape({6, 8}));
    EXPECT_EQ(m.v_proj().weight().shape(), Shape({6, 8}));
    EXPECT_EQ(m.out_proj().weight().shape(), Shape({16, 6}));

    AttentionConfig plain;
    plain.d_model = 8;
    plain.num_heads = 2;
    MultiHeadAttentionModule p(plain, &backend);
    EXPECT_EQ(p.config().num_kv_heads, 2);
    EXPECT_EQ(p.config().head_dim, 4);
}

TEST_F(AttentionLlmFeaturesTest, NamedParametersOmitDisabledBiases) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    std::vector<std::string> names;
    for (const NamedParamRef& p : m.named_parameters()) {
        names.push_back(p.name);
    }
    const std::vector<std::string> expected{"q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias",
                                            "v_proj.weight", "v_proj.bias", "out_proj.weight"};
    EXPECT_EQ(names, expected);
}

TEST_F(AttentionLlmFeaturesTest, ConstructorRejectsInvalidGroupedQueryShapes) {
    AttentionConfig c = GqaConfig();
    c.num_kv_heads = 3;  // 4 query heads don't split into groups of 3
    EXPECT_THROW(MultiHeadAttentionModule(c, &backend), std::invalid_argument);
    c.num_kv_heads = -1;
    EXPECT_THROW(MultiHeadAttentionModule(c, &backend), std::invalid_argument);
    c = GqaConfig();
    c.head_dim = 3;  // RoPE needs an even head_dim
    EXPECT_THROW(MultiHeadAttentionModule(c, &backend), std::invalid_argument);
    c.use_rope = false;  // ...but without RoPE an odd, explicit head_dim is fine
    EXPECT_NO_THROW(MultiHeadAttentionModule(c, &backend));
}

// ---- Grouped-query attention ---------------------------------------------------------------

// GQA is MHA whose K/V weights repeat across each group: build both and compare forward,
// input gradient and relevance. The K/V weight columns of head h are those of kv head h / 2.
TEST_F(AttentionLlmFeaturesTest, GroupedQueryAttentionEqualsMhaWithRepeatedKvWeights) {
    AttentionConfig gqa_config = GqaConfig();
    MultiHeadAttentionModule gqa(gqa_config, &backend);
    FillWeights(gqa);
    AttentionConfig mha_config = gqa_config;
    mha_config.num_kv_heads = 4;
    MultiHeadAttentionModule mha(mha_config, &backend);
    mha.q_proj().set_weight(gqa.q_proj().weight().to_host_vector());
    mha.q_proj().set_bias(gqa.q_proj().bias().to_host_vector());
    mha.out_proj().set_weight(gqa.out_proj().weight().to_host_vector());
    auto repeat_columns = [](const std::vector<float>& w, int64_t rows, int64_t kv_heads, int64_t head_dim,
                             int64_t group) {
        std::vector<float> out;
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t h = 0; h < kv_heads * group; ++h) {
                for (int64_t e = 0; e < head_dim; ++e) {
                    out.push_back(w[static_cast<size_t>(r * kv_heads * head_dim + (h / group) * head_dim + e)]);
                }
            }
        }
        return out;
    };
    for (auto [src, dst] : {std::pair{&gqa.k_proj(), &mha.k_proj()}, std::pair{&gqa.v_proj(), &mha.v_proj()}}) {
        dst->set_weight(repeat_columns(src->weight().to_host_vector(), 6, 2, 4, 2));
        dst->set_bias(repeat_columns(src->bias().to_host_vector(), 1, 2, 4, 2));
    }

    Tensor x(Shape({2, 3, 6}), &backend, Pattern(36, 0.3f, 0.1f));
    Tensor y_gqa = gqa.forward(x);
    Tensor y_mha = mha.forward(x);
    for (int64_t i = 0; i < y_gqa.numel(); ++i) {
        EXPECT_NEAR(y_gqa.data()[i], y_mha.data()[i], 1e-5f) << "output " << i;
    }
    Tensor g(Shape({2, 3, 6}), &backend, Pattern(36, 0.2f, -0.1f));
    Tensor dx_gqa = gqa.backward(g);
    Tensor dx_mha = mha.backward(g);
    for (int64_t i = 0; i < dx_gqa.numel(); ++i) {
        EXPECT_NEAR(dx_gqa.data()[i], dx_mha.data()[i], 1e-5f) << "input gradient " << i;
    }
    // The shared K weight's gradient is the sum of its two copies' gradients.
    const std::vector<float> gk_gqa = gqa.k_proj().weight_grad().to_host_vector();
    const std::vector<float> gk_mha = mha.k_proj().weight_grad().to_host_vector();
    for (int64_t r = 0; r < 6; ++r) {
        for (int64_t kvh = 0; kvh < 2; ++kvh) {
            for (int64_t e = 0; e < 4; ++e) {
                const float copies = gk_mha[static_cast<size_t>(r * 16 + (2 * kvh) * 4 + e)] +
                                     gk_mha[static_cast<size_t>(r * 16 + (2 * kvh + 1) * 4 + e)];
                EXPECT_NEAR(gk_gqa[static_cast<size_t>(r * 8 + kvh * 4 + e)], copies, 1e-5f);
            }
        }
    }
    Tensor rel = Tensor(Shape({2, 3, 6}), &backend, Pattern(36, 0.5f, 0.2f));
    Tensor r_gqa = gqa.propagate_relevance(rel, LRPRuleConfig{1e-6f});
    Tensor r_mha = mha.propagate_relevance(rel, LRPRuleConfig{1e-6f});
    for (int64_t i = 0; i < r_gqa.numel(); ++i) {
        EXPECT_NEAR(r_gqa.data()[i], r_mha.data()[i], 1e-4f * std::max(1.0f, std::fabs(r_mha.data()[i])))
            << "relevance " << i;
    }
}

// ---- Masks ---------------------------------------------------------------------------------

TEST_F(AttentionLlmFeaturesTest, CausalOutputIgnoresLaterTokens) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    FillWeights(m);
    std::vector<float> a = Pattern(1 * 4 * 6, 0.3f, 0.0f);
    std::vector<float> b = a;
    for (size_t i = 2 * 6; i < b.size(); ++i) {
        b[i] += 1.5f;  // change tokens 2 and 3
    }
    Tensor ya = m.forward(Tensor(Shape({1, 4, 6}), &backend, a));
    Tensor yb = m.forward(Tensor(Shape({1, 4, 6}), &backend, b));
    for (int64_t i = 0; i < 2 * 6; ++i) {
        EXPECT_FLOAT_EQ(ya.data()[i], yb.data()[i]) << "token " << i / 6 << " saw a later token";
    }
    EXPECT_NE(ya.data()[2 * 6], yb.data()[2 * 6]);

    const Tensor& w = m.last_attention_weights();  // (1, 4, 4, 4)
    for (int64_t h = 0; h < 4; ++h) {
        for (int64_t i = 0; i < 4; ++i) {
            for (int64_t j = i + 1; j < 4; ++j) {
                EXPECT_EQ(w.data()[(h * 4 + i) * 4 + j], 0.0f);
            }
        }
    }
}

TEST_F(AttentionLlmFeaturesTest, PaddingKeysDoNotAffectRealTokens) {
    AttentionConfig c = GqaConfig();
    c.causal = false;  // padding alone, so every real query could otherwise see the pads
    MultiHeadAttentionModule m(c, &backend);
    FillWeights(m);
    m.set_key_padding_mask(Tensor(Shape({1, 4}), &backend, {1, 1, 1, 0}));
    std::vector<float> a = Pattern(24, 0.3f, 0.0f);
    std::vector<float> b = a;
    for (size_t i = 18; i < 24; ++i) {
        b[i] = -3.0f;  // the padding token's contents
    }
    Tensor ya = m.forward(Tensor(Shape({1, 4, 6}), &backend, a));
    Tensor yb = m.forward(Tensor(Shape({1, 4, 6}), &backend, b));
    for (int64_t i = 0; i < 18; ++i) {
        EXPECT_FLOAT_EQ(ya.data()[i], yb.data()[i]) << "element " << i;
    }

    // Masking the pad gives the same output as leaving it out.
    MultiHeadAttentionModule short_m(c, &backend);
    FillWeights(short_m);
    Tensor y_short = short_m.forward(Tensor(Shape({1, 3, 6}), &backend, std::vector<float>(a.begin(), a.begin() + 18)));
    for (int64_t i = 0; i < 18; ++i) {
        EXPECT_NEAR(ya.data()[i], y_short.data()[i], 1e-5f) << "element " << i;
    }
}

// Left padding under a causal mask leaves query 0 with no visible key. It gets uniform weights
// (Hugging Face's behaviour), never NaN, and its masked scores pass no gradient.
TEST_F(AttentionLlmFeaturesTest, FullyMaskedRowIsUniformAndFinite) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    FillWeights(m);
    m.set_key_padding_mask(Tensor(Shape({1, 3}), &backend, {0, 1, 1}));
    Tensor y = m.forward(Tensor(Shape({1, 3, 6}), &backend, Pattern(18, 0.3f, 0.0f)));
    for (int64_t i = 0; i < y.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(y.data()[i]));
    }
    const Tensor& w = m.last_attention_weights();  // (1, 4, 3, 3); row 0 of each head
    for (int64_t h = 0; h < 4; ++h) {
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(w.data()[(h * 3 + 0) * 3 + j], 1.0f / 3.0f, 1e-6f);
        }
        EXPECT_EQ(w.data()[(h * 3 + 1) * 3 + 0], 0.0f);  // query 1 can't see the pad
    }
    Tensor dx = m.backward(Tensor(Shape({1, 3, 6}), &backend, Pattern(18, 0.2f, 0.1f)));
    for (int64_t i = 0; i < dx.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(dx.data()[i]));
    }
    Tensor r = m.propagate_relevance(Tensor(Shape({1, 3, 6}), &backend, Pattern(18, 0.2f, 0.1f)), LRPRuleConfig{});
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(r.data()[i]));
    }
}

// With no relevance on a padding query (the caller's job), the padding token receives none:
// no real query attends to it, and its own query row starts at zero.
TEST_F(AttentionLlmFeaturesTest, PaddingTokenReceivesNoRelevance) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    FillWeights(m);
    m.set_key_padding_mask(Tensor(Shape({2, 3}), &backend, {1, 1, 0, 1, 1, 1}));
    (void)m.forward(Tensor(Shape({2, 3, 6}), &backend, Pattern(36, 0.3f, 0.0f)));
    std::vector<float> rel = Pattern(36, 0.4f, 0.3f);
    for (size_t i = 12; i < 18; ++i) {
        rel[i] = 0.0f;  // sequence 0, token 2 is padding
    }
    Tensor r = m.propagate_relevance(Tensor(Shape({2, 3, 6}), &backend, rel), LRPRuleConfig{1e-6f});
    for (int64_t i = 12; i < 18; ++i) {
        EXPECT_EQ(r.data()[i], 0.0f) << "padding feature " << i - 12;
    }
}

TEST_F(AttentionLlmFeaturesTest, PaddingMaskShapeMustMatchInput) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    EXPECT_THROW(m.set_key_padding_mask(Tensor(Shape({4}), &backend)), std::invalid_argument);
    m.set_key_padding_mask(Tensor(Shape({1, 5}), &backend));
    EXPECT_THROW((void)m.forward(Tensor(Shape({1, 4, 6}), &backend)), std::invalid_argument);
    m.clear_key_padding_mask();
    EXPECT_FALSE(m.has_key_padding_mask());
    EXPECT_NO_THROW((void)m.forward(Tensor(Shape({1, 4, 6}), &backend)));
}

// ---- Gradients -----------------------------------------------------------------------------

// Every LLM-1 feature at once -- GQA, head_dim != d_model / heads, biases on one side only,
// rotate-half RoPE with an offset, a causal mask and padding -- by central differences over
// the input and every parameter.
TEST_F(AttentionLlmFeaturesTest, BackwardMatchesFiniteDifferencesWithEveryFeature) {
    AttentionConfig c = GqaConfig();
    c.use_qk_norm = true;
    MultiHeadAttentionModule m(c, &backend);
    FillWeights(m);
    m.set_position_offset(5);
    m.set_key_padding_mask(Tensor(Shape({2, 3}), &backend, {1, 1, 0, 0, 1, 1}));
    const Shape shape({2, 3, 6});
    std::vector<float> xv = Pattern(36, 0.25f, 0.05f);
    std::vector<float> g = Pattern(36, 0.3f, -0.1f);
    for (size_t i = 12; i < 18; ++i) {
        g[i] = 0.0f;  // padding queries carry no loss
    }
    for (size_t i = 18; i < 24; ++i) {
        g[i] = 0.0f;
    }
    Tensor x(shape, &backend, xv);
    (void)m.forward(x);
    Tensor dx = m.backward(Tensor(shape, &backend, g));

    const float h = 1e-3f;
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> p = xv, q = xv;
        p[i] += h;
        q[i] -= h;
        const float numeric =
            (Objective(m, Tensor(shape, &backend, p), g) - Objective(m, Tensor(shape, &backend, q), g)) / (2 * h);
        EXPECT_NEAR(dx.data()[static_cast<int64_t>(i)], numeric, 3e-3f) << "input " << i;
    }

    std::vector<ParamRef> params = m.parameters();
    std::vector<std::vector<float>> analytic;
    for (const ParamRef& p : params) {
        analytic.push_back(p.grad->to_host_vector());
    }
    for (size_t pi = 0; pi < params.size(); ++pi) {
        for (int64_t i = 0; i < params[pi].value->numel(); ++i) {
            const float original = params[pi].value->data()[i];
            params[pi].value->data()[i] = original + h;
            const float fp = Objective(m, x, g);
            params[pi].value->data()[i] = original - h;
            const float fm = Objective(m, x, g);
            params[pi].value->data()[i] = original;
            EXPECT_NEAR(analytic[pi][static_cast<size_t>(i)], (fp - fm) / (2 * h), 3e-3f)
                << "parameter " << pi << ", element " << i;
        }
    }
}

// A fully masked row's weights are a constant (uniform), so its output depends on V but not
// on Q or K. With loss on that row, the analytic gradient must still match finite differences,
// which it only does if the masked scores pass no gradient back into Q and K.
TEST_F(AttentionLlmFeaturesTest, FullyMaskedRowPassesNoGradientToQueriesOrKeys) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    FillWeights(m);
    m.set_key_padding_mask(Tensor(Shape({1, 3}), &backend, {0, 1, 1}));
    const Shape shape({1, 3, 6});
    std::vector<float> xv = Pattern(18, 0.25f, 0.05f);
    const std::vector<float> g = Pattern(18, 0.3f, -0.1f);  // loss on the padding query too
    (void)m.forward(Tensor(shape, &backend, xv));
    Tensor dx = m.backward(Tensor(shape, &backend, g));
    const float h = 1e-3f;
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> p = xv, q = xv;
        p[i] += h;
        q[i] -= h;
        const float numeric =
            (Objective(m, Tensor(shape, &backend, p), g) - Objective(m, Tensor(shape, &backend, q), g)) / (2 * h);
        EXPECT_NEAR(dx.data()[static_cast<int64_t>(i)], numeric, 3e-3f) << "input " << i;
    }
}

// ---- Position offset -----------------------------------------------------------------------

// Attention scores under RoPE depend only on relative position, so a uniform offset leaves
// the output unchanged -- the offset matters once keys and queries are processed in
// different pieces (the KV cache, LLM-5). What it must do here is reach RoPE.
TEST_F(AttentionLlmFeaturesTest, PositionOffsetReachesRopeButNotOutput) {
    MultiHeadAttentionModule m(GqaConfig(), &backend);
    FillWeights(m);
    Tensor x(Shape({1, 3, 6}), &backend, Pattern(18, 0.3f, 0.0f));
    Tensor y0 = m.forward(x);
    m.set_position_offset(7);
    EXPECT_EQ(m.position_offset(), 7);
    Tensor y7 = m.forward(x);
    for (int64_t i = 0; i < y0.numel(); ++i) {
        EXPECT_NEAR(y0.data()[i], y7.data()[i], 1e-4f);
    }
    EXPECT_THROW(m.set_position_offset(-1), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
