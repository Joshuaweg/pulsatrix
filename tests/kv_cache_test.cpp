// LLM-5: the KV cache. The falsifier from the research notes: cached and uncached logits differ.
// Every test here runs a sequence in pieces through the cache and compares it with one full pass.

#include "pulsatrix/kv_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/generation.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Pattern(int64_t n, float scale, float shift, int64_t mul = 7) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = scale * static_cast<float>((i * mul + 3) % 11 - 5) + shift;
    }
    return v;
}

void Randomize(Module& m, float scale) {
    int64_t k = 0;
    for (ParamRef& p : m.parameters()) {
        *p.value = Tensor(p.value->shape(), p.value->backend(), Pattern(p.value->numel(), scale, 0.01f, 5 + k++ % 4));
    }
}

AttentionConfig QwenLike() {
    AttentionConfig c;
    c.d_model = 8;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 6;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 100.0f;
    c.use_qk_norm = true;
    c.qkv_bias = true;
    c.out_bias = false;
    c.causal = true;
    return c;
}

// Rows [from, from + count) of every batch entry of an (N, L, d) tensor.
std::vector<float> Rows(const Tensor& t, int64_t from, int64_t count) {
    const int64_t N = t.shape().dim(0), L = t.shape().dim(1), d = t.shape().dim(2);
    const std::vector<float> all = t.to_host_vector();
    std::vector<float> out;
    for (int64_t n = 0; n < N; ++n) {
        out.insert(out.end(), all.begin() + (n * L + from) * d, all.begin() + (n * L + from + count) * d);
    }
    return out;
}

Tensor Slice(const Tensor& t, int64_t from, int64_t count) {
    return Tensor(Shape({t.shape().dim(0), count, t.shape().dim(2)}), t.backend(), Rows(t, from, count));
}

void ExpectClose(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol * std::max(1.0f, std::fabs(b[i]))) << "element " << i;
    }
}

class KVCacheTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(KVCacheTest, ChunkedAttentionMatchesOneFullPass) {
    MultiHeadAttentionModule attn(QwenLike(), &backend);
    Randomize(attn, 0.15f);
    attn.set_position_offset(2);
    Tensor x(Shape({2, 6, 8}), &backend, Pattern(96, 0.3f, 0.05f));
    Tensor full = attn.forward(x);

    KVCache cache = attn.MakeKVCache(2, 8);
    int64_t at = 0;
    for (int64_t piece : {3, 1, 2}) {
        Tensor out = attn.forward_cached(Slice(x, at, piece), cache);
        ExpectClose(out.to_host_vector(), Rows(full, at, piece));
        at += piece;
        EXPECT_EQ(cache.length(), at);
    }
}

TEST_F(KVCacheTest, CachedAttentionHonorsThePaddingMaskOverEveryPosition) {
    MultiHeadAttentionModule attn(QwenLike(), &backend);
    Randomize(attn, 0.15f);
    Tensor x(Shape({1, 5, 8}), &backend, Pattern(40, 0.3f, 0.0f));
    attn.set_key_padding_mask(Tensor(Shape({1, 5}), &backend, {0, 1, 1, 1, 1}));  // left padding
    Tensor full = attn.forward(x);
    KVCache cache = attn.MakeKVCache(1, 5);
    attn.set_key_padding_mask(Tensor(Shape({1, 3}), &backend, {0, 1, 1}));
    // Row 0 is the padding query: it sees no key, so its uniform weights spread over whatever keys
    // exist (5 in the full pass, 3 here). Its output means nothing either way; the real rows match.
    const std::vector<float> first = attn.forward_cached(Slice(x, 0, 3), cache).to_host_vector();
    ExpectClose(std::vector<float>(first.begin() + 8, first.end()), Rows(full, 1, 2));
    attn.set_key_padding_mask(Tensor(Shape({1, 5}), &backend, {0, 1, 1, 1, 1}));
    ExpectClose(attn.forward_cached(Slice(x, 3, 2), cache).to_host_vector(), Rows(full, 3, 2));
    // A mask that doesn't cover every cached position is rejected.
    cache.truncate(3);
    attn.set_key_padding_mask(Tensor(Shape({1, 2}), &backend, {1, 1}));
    EXPECT_THROW((void)attn.forward_cached(Slice(x, 3, 2), cache), std::invalid_argument);
}

TEST_F(KVCacheTest, ChunkedTransformerBlockMatchesOneFullPass) {
    TransformerBlock block(QwenLike(), 12, &backend, 1e-5f);
    Randomize(block, 0.12f);
    Tensor x(Shape({1, 7, 8}), &backend, Pattern(56, 0.25f, 0.1f));
    Tensor full = block.forward(x);
    KVCache cache = block.mha().MakeKVCache(1, 7);
    ExpectClose(block.forward_cached(Slice(x, 0, 4), cache).to_host_vector(), Rows(full, 0, 4));
    for (int64_t i = 4; i < 7; ++i) {
        ExpectClose(block.forward_cached(Slice(x, i, 1), cache).to_host_vector(), Rows(full, i, 1));
    }
}

TEST_F(KVCacheTest, CacheBookkeepingAndErrors) {
    MultiHeadAttentionModule attn(QwenLike(), &backend);
    KVCache cache = attn.MakeKVCache(1, 4);
    EXPECT_EQ(cache.keys().shape(), Shape({1, 2, 4, 6}));
    Tensor x(Shape({1, 3, 8}), &backend, Pattern(24, 0.3f, 0.0f));
    (void)attn.forward_cached(x, cache);
    EXPECT_EQ(cache.length(), 3);
    EXPECT_THROW((void)attn.forward_cached(x, cache), std::invalid_argument);  // 6 > 4
    EXPECT_EQ(cache.length(), 3);                                              // unchanged on failure
    cache.truncate(1);
    EXPECT_EQ(cache.length(), 1);
    EXPECT_THROW(cache.truncate(2), std::invalid_argument);
    cache.reset();
    EXPECT_EQ(cache.length(), 0);
    // Inference only: no backward after a cached pass.
    EXPECT_THROW((void)attn.backward(Tensor(Shape({1, 3, 8}), &backend)), std::logic_error);
    EXPECT_THROW((void)attn.propagate_relevance(Tensor(Shape({1, 3, 8}), &backend), LRPRuleConfig{}),
                 std::logic_error);
    KVCache wrong(1, 4, 6, 4, &backend);  // 4 K/V heads, the layer has 2
    EXPECT_THROW((void)attn.forward_cached(x, wrong), std::invalid_argument);
    KVCache batch2 = attn.MakeKVCache(2, 4);
    EXPECT_THROW((void)attn.forward_cached(x, batch2), std::invalid_argument);
    EXPECT_THROW(KVCache(1, 0, 6, 4, &backend), std::invalid_argument);
}

// A two-block language model, generated with and without the cache.
class CachedGenerationTest : public KVCacheTest {
protected:
    EmbeddingModule embedding{11, 8, &backend};
    TransformerBlock b1{QwenLike(), 12, &backend, 1e-5f};
    TransformerBlock b2{QwenLike(), 12, &backend, 1e-5f};
    TiedLMHeadModule head{embedding, &backend};

    void SetUp() override {
        embedding.set_weight(Pattern(88, 0.3f, 0.0f, 3));
        Randomize(b1, 0.12f);
        Randomize(b2, 0.1f);
    }
};

TEST_F(CachedGenerationTest, CachedLogitsEqualUncachedAtEveryStep) {
    NextTokenLogitsFn plain = MakeNextTokenLogits({&embedding, &b1, &b2, &head}, &backend);
    NextTokenLogitsFn cached = MakeCachedNextTokenLogits(embedding, {&b1, &b2}, {&head}, &backend, 32);
    std::vector<int64_t> seq{4, 1, 7};
    for (int step = 0; step < 8; ++step) {
        const std::vector<float> a = plain(seq);
        ExpectClose(cached(seq), a);
        seq.push_back(static_cast<int64_t>(std::max_element(a.begin(), a.end()) - a.begin()));
    }
    // Same tokens through Generate, greedy and sampled.
    GenerationConfig g;
    g.max_new_tokens = 10;
    EXPECT_EQ(Generate(plain, {2, 9}, g).tokens, Generate(cached, {2, 9}, g).tokens);
    g.do_sample = true;
    g.seed = 3;
    g.top_k = 5;
    EXPECT_EQ(Generate(plain, {2, 9}, g).tokens, Generate(cached, {2, 9}, g).tokens);
}

TEST_F(CachedGenerationTest, ANewHistoryReusesOnlyTheSharedPrefix) {
    NextTokenLogitsFn plain = MakeNextTokenLogits({&embedding, &b1, &b2, &head}, &backend);
    NextTokenLogitsFn cached = MakeCachedNextTokenLogits(embedding, {&b1, &b2}, {&head}, &backend, 16);
    for (const std::vector<int64_t>& seq : std::vector<std::vector<int64_t>>{
             {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 5}, {1, 2}, {8, 8, 8, 8, 8}, {8}}) {
        ExpectClose(cached(seq), plain(seq));
    }
    EXPECT_THROW((void)cached(std::vector<int64_t>(17, 1)), std::invalid_argument);
    EXPECT_THROW((void)cached({}), std::invalid_argument);
}

TEST_F(CachedGenerationTest, HeadLayersSeeTheLastPositionOnly) {
    // A final RMSNorm before the head, as in Llama-family models.
    RMSNormModule norm(8, &backend, backend.device(), 1e-5f);
    norm.set_gamma(Pattern(8, 0.1f, 1.0f));
    NextTokenLogitsFn cached = MakeCachedNextTokenLogits(embedding, {&b1, &b2}, {&norm, &head}, &backend, 8);
    Tensor hidden = b2.forward(b1.forward(embedding.forward(Tensor(Shape({1, 3}), &backend, {5.0f, 0.0f, 6.0f}))));
    Tensor last(Shape({1, 8}), &backend, Rows(hidden, 2, 1));
    ExpectClose(cached({5, 0, 6}), head.forward(norm.forward(last)).to_host_vector());
}

}  // namespace
}  // namespace pulsatrix
