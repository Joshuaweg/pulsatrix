// PLM-1: EncoderBlock, FeedForwardModule and ActivationModule. The ESM-2 and BERT layers are checked
// against transformers' own EsmLayer and BertLayer (tools/generate_encoder_reference.py).
#include "pulsatrix/encoder_block.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/activation_module.hpp"
#include "pulsatrix/causal_lm.hpp"  // LoadWeights
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/feed_forward_module.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/encoder/" + name; }

std::vector<float> Random(size_t n, unsigned seed, float scale = 0.5f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = d(gen);
    return v;
}

void Randomize(Module& m, unsigned seed) {
    for (auto& p : m.named_parameters()) {
        *p.ref.value = Tensor(p.ref.value->shape(), p.ref.value->backend(), Random(static_cast<size_t>(p.ref.value->numel()), seed++));
    }
}

void ExpectNear(const std::vector<float>& a, const std::vector<float>& b, float tol, const std::string& what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    EXPECT_LE(worst, tol) << what;
}

AttentionConfig Attention(bool rope) {
    AttentionConfig a;
    a.d_model = 32;
    a.num_heads = 4;
    a.use_rope = rope;
    a.rope_layout = RoPELayout::RotateHalf;
    return a;
}

/** @brief Loads a reference layer, runs it, and compares output and input gradient with transformers'. */
void ExpectMatchesTransformers(EncoderBlock& block, const std::string& dir, const std::vector<WeightMapping>& mapping, CPUBackend& cpu) {
    const HfCheckpoint ck = HfCheckpoint::Open(Fixture(dir));
    WeightLoadOptions o;
    o.ignore = {"ref.input", "ref.output", "ref.output_grad", "ref.input_grad"};
    (void)LoadWeights(block, ck, mapping, o);  // strict: every parameter loaded, every weight used
    const Tensor x = ck.tensor("ref.input", &cpu);
    ExpectNear(block.forward(x).to_host_vector(), ck.tensor("ref.output", &cpu).to_host_vector(), 2e-5f, dir + " output");
    ExpectNear(block.backward(ck.tensor("ref.output_grad", &cpu)).to_host_vector(), ck.tensor("ref.input_grad", &cpu).to_host_vector(), 2e-5f,
               dir + " input gradient");
}

std::vector<WeightMapping> AttentionMapping(const std::string& self, const std::string& out) {
    using T = WeightTransform;
    std::vector<WeightMapping> m;
    for (const auto& [hf, ours] : {std::pair{"query", "q_proj"}, std::pair{"key", "k_proj"}, std::pair{"value", "v_proj"}}) {
        m.push_back({self + hf + ".weight", std::string("mha.") + ours + ".weight", T::Transpose});
        m.push_back({self + hf + ".bias", std::string("mha.") + ours + ".bias"});
    }
    m.push_back({out + ".weight", "mha.out_proj.weight", T::Transpose});
    m.push_back({out + ".bias", "mha.out_proj.bias"});
    m.push_back({"intermediate.dense.weight", "mlp.fc1.weight", T::Transpose});
    m.push_back({"intermediate.dense.bias", "mlp.fc1.bias"});
    m.push_back({"output.dense.weight", "mlp.fc2.weight", T::Transpose});
    m.push_back({"output.dense.bias", "mlp.fc2.bias"});
    return m;
}

void AddNorm(std::vector<WeightMapping>& m, const std::string& hf, const std::string& ours) {
    m.push_back({hf + ".weight", ours + ".weight"});
    m.push_back({hf + ".bias", ours + ".bias"});
}

// ESM-2's layer: pre-LayerNorm, rotate-half RoPE on scaled queries, exact-GELU MLP.
TEST(EncoderBlock, MatchesTransformersEsmLayer) {
    CPUBackend cpu;
    EncoderBlock block(Attention(/*rope=*/true), 48, &cpu);  // the defaults are ESM-2's
    auto m = AttentionMapping("attention.self.", "attention.output.dense");
    AddNorm(m, "attention.LayerNorm", "norm1");
    AddNorm(m, "LayerNorm", "norm2");
    ExpectMatchesTransformers(block, "esm_layer", m, cpu);
}

// BERT's layer: post-LayerNorm (eps 1e-12), no positions inside the layer.
TEST(EncoderBlock, MatchesTransformersBertLayer) {
    CPUBackend cpu;
    EncoderBlockOptions o;
    o.norm_position = NormPosition::Post;
    o.norm_eps = 1e-12f;
    EncoderBlock block(Attention(/*rope=*/false), 48, &cpu, o);
    auto m = AttentionMapping("attention.self.", "attention.output.dense");
    AddNorm(m, "attention.output.LayerNorm", "norm1");
    AddNorm(m, "output.LayerNorm", "norm2");
    ExpectMatchesTransformers(block, "bert_layer", m, cpu);
}

// Pre-RMSNorm with a gated SiLU MLP is TransformerBlock: same output, gradients and relevance.
TEST(EncoderBlock, LlamaLayoutEqualsTransformerBlock) {
    CPUBackend cpu;
    AttentionConfig a = Attention(true);
    EncoderBlockOptions o;
    o.norm = NormType::RMSNorm;
    o.norm_eps = 1e-6f;
    o.mlp = MlpType::Gated;
    EncoderBlock enc(a, 48, &cpu, o);
    TransformerBlock ref(a, 48, &cpu, TransformerBlockOptions{});
    Randomize(enc, 10);
    auto ref_params = ref.named_parameters();
    for (auto& p : enc.named_parameters()) {
        std::string name = p.name;
        if (name.rfind("mlp.", 0) == 0) name = "swiglu." + name.substr(4);
        bool found = false;
        for (auto& q : ref_params) {
            if (q.name == name) {
                *q.ref.value = *p.ref.value;
                found = true;
            }
        }
        ASSERT_TRUE(found) << p.name;
    }
    const Shape shape({2, 5, 32});
    const Tensor x(shape, &cpu, Random(320, 1)), g(shape, &cpu, Random(320, 2));
    ExpectNear(enc.forward(x).to_host_vector(), ref.forward(x).to_host_vector(), 1e-6f, "output");
    ExpectNear(enc.backward(g).to_host_vector(), ref.backward(g).to_host_vector(), 1e-6f, "input gradient");
    for (auto& p : enc.named_parameters()) {
        std::string name = p.name.rfind("mlp.", 0) == 0 ? "swiglu." + p.name.substr(4) : p.name;
        for (auto& q : ref_params) {
            if (q.name == name) ExpectNear(p.ref.grad->to_host_vector(), q.ref.grad->to_host_vector(), 1e-5f, p.name + " gradient");
        }
    }
    ExpectNear(enc.propagate_relevance(g, LRPRuleConfig{}).to_host_vector(), ref.propagate_relevance(g, LRPRuleConfig{}).to_host_vector(), 1e-6f,
               "relevance");
}

// Parameter gradients against finite differences of sum(y * r), in both layouts.
TEST(EncoderBlock, ParameterGradientsMatchFiniteDifferences) {
    for (NormPosition position : {NormPosition::Pre, NormPosition::Post}) {
        CPUBackend cpu;
        AttentionConfig a = Attention(true);
        a.d_model = 8;
        a.num_heads = 2;
        EncoderBlockOptions o;
        o.norm_position = position;
        EncoderBlock block(a, 12, &cpu, o);
        Randomize(block, 20);
        for (auto& p : block.named_parameters()) {
            if (p.name.find("norm") != std::string::npos && p.name.find("weight") != std::string::npos) {
                std::vector<float> w = p.ref.value->to_host_vector();
                for (float& v : w) v += 1.0f;  // LayerNorm gains around 1
                *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
            }
        }
        const Shape shape({1, 4, 8});
        const Tensor x(shape, &cpu, Random(32, 3)), r(shape, &cpu, Random(32, 4));
        const auto loss = [&] {
            const std::vector<float> y = block.forward(x).to_host_vector(), rv = r.to_host_vector();
            double s = 0;
            for (size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * rv[i];
            return s;
        };
        (void)loss();
        (void)block.backward(r);
        for (const char* name : {"mha.q_proj.weight", "mha.v_proj.bias", "norm1.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc2.bias"}) {
            for (auto& p : block.named_parameters()) {
                if (p.name != name) continue;
                const std::vector<float> grad = p.ref.grad->to_host_vector();
                std::vector<float> w = p.ref.value->to_host_vector();
                const size_t k = w.size() / 2;
                const float h = 1e-2f, w0 = w[k];
                w[k] = w0 + h;
                *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
                const double up = loss();
                w[k] = w0 - h;
                *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
                const double down = loss();
                w[k] = w0;
                *p.ref.value = Tensor(p.ref.value->shape(), &cpu, w);
                EXPECT_NEAR(grad[k], (up - down) / (2 * h), 2e-3 * (1 + std::abs(grad[k])))
                    << name << (position == NormPosition::Pre ? " (pre)" : " (post)");
            }
        }
    }
}

// LRP is linear in the incoming relevance (what VIZ-4's per-position graph relies on), in both layouts.
TEST(EncoderBlock, RelevanceIsLinear) {
    for (NormPosition position : {NormPosition::Pre, NormPosition::Post}) {
        CPUBackend cpu;
        EncoderBlockOptions o;
        o.norm_position = position;
        EncoderBlock block(Attention(true), 48, &cpu, o);
        Randomize(block, 30);
        const Shape shape({1, 5, 32});
        (void)block.forward(Tensor(shape, &cpu, Random(160, 5)));
        const std::vector<float> r1 = Random(160, 6), r2 = Random(160, 7);
        std::vector<float> sum(160);
        for (size_t i = 0; i < 160; ++i) sum[i] = 2.0f * r1[i] - 0.5f * r2[i];
        const std::vector<float> a = block.propagate_relevance(Tensor(shape, &cpu, r1), LRPRuleConfig{}).to_host_vector();
        const std::vector<float> b = block.propagate_relevance(Tensor(shape, &cpu, r2), LRPRuleConfig{}).to_host_vector();
        const std::vector<float> c = block.propagate_relevance(Tensor(shape, &cpu, sum), LRPRuleConfig{}).to_host_vector();
        std::vector<float> expect(160);
        for (size_t i = 0; i < 160; ++i) expect[i] = 2.0f * a[i] - 0.5f * b[i];
        // Relative: with random weights a residual split's a + b can be near zero, so relevance
        // reaches about 1e6 here (post-norm) and float rounding is absolute in that scale.
        float scale = 0;
        for (float v : expect) scale = std::max(scale, std::abs(v));
        ExpectNear(c, expect, 1e-5f * scale, position == NormPosition::Pre ? "pre" : "post");
    }
}

TEST(EncoderBlock, NamesParametersByRole) {
    CPUBackend cpu;
    EncoderBlock plain(Attention(true), 48, &cpu);
    std::vector<std::string> names;
    for (auto& p : plain.named_parameters()) names.push_back(p.name);
    for (const char* expected : {"norm1.weight", "norm1.bias", "mha.q_proj.weight", "mha.out_proj.bias", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.bias"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end()) << expected;
    }
    EncoderBlockOptions o;
    o.mlp = MlpType::Gated;
    o.norm = NormType::RMSNorm;
    EncoderBlock gated(Attention(true), 48, &cpu, o);
    bool has_gate = false, has_norm_bias = false;
    for (auto& p : gated.named_parameters()) {
        has_gate = has_gate || p.name == "mlp.gate_proj.weight";
        has_norm_bias = has_norm_bias || p.name == "norm1.bias";
    }
    EXPECT_TRUE(has_gate);
    EXPECT_FALSE(has_norm_bias);  // RMSNorm has no bias
}

TEST(EncoderBlock, RejectsBadInput) {
    CPUBackend cpu;
    EXPECT_THROW(EncoderBlock(Attention(true), 0, &cpu), std::invalid_argument);
    EncoderBlock block(Attention(true), 48, &cpu);
    EXPECT_THROW((void)block.forward(Tensor(Shape({5, 32}), &cpu)), std::invalid_argument);
    EXPECT_THROW((void)block.backward(Tensor(Shape({1, 5, 32}), &cpu)), std::logic_error);
    EXPECT_THROW((void)block.propagate_relevance(Tensor(Shape({1, 5, 32}), &cpu), LRPRuleConfig{}), std::logic_error);
    (void)block.forward(Tensor(Shape({1, 5, 32}), &cpu));
    EXPECT_THROW((void)block.backward(Tensor(Shape({1, 4, 32}), &cpu)), std::invalid_argument);
}

TEST(FeedForwardModule, IsLinearActivationLinear) {
    CPUBackend cpu;
    FeedForwardModule ff(4, 6, &cpu);
    Randomize(ff, 40);
    const Tensor x(Shape({2, 3, 4}), &cpu, Random(24, 8));
    const std::vector<float> y = ff.forward(x).to_host_vector();
    // The same computation, module by module.
    Tensor flat(x);
    flat.reshape(Shape({6, 4}));
    const Tensor hidden = ff.fc1().forward(flat);
    std::vector<float> h = hidden.to_host_vector();
    cpu.elementwise(ElementwiseOp::Gelu, h.data(), h.data(), h.size());
    const std::vector<float> expect = ff.fc2().forward(Tensor(Shape({6, 6}), &cpu, h)).to_host_vector();
    ExpectNear(y, expect, 1e-6f, "output");
    EXPECT_THROW((void)ff.forward(Tensor(Shape({3, 5}), &cpu)), std::invalid_argument);
    EXPECT_THROW(FeedForwardModule(4, 0, &cpu), std::invalid_argument);
}

TEST(ActivationModule, AppliesTheOpAndPassesRelevanceThrough) {
    CPUBackend cpu;
    ActivationModule gelu(ElementwiseOp::Gelu, &cpu);
    const Tensor x(Shape({2, 3}), &cpu, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
    const std::vector<float> y = gelu.forward(x).to_host_vector();
    for (size_t i = 0; i < y.size(); ++i) {
        const float v = x.to_host_vector()[i];
        EXPECT_NEAR(y[i], 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f))), 1e-6f);
    }
    const Tensor r(Shape({2, 3}), &cpu, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(gelu.propagate_relevance(r, LRPRuleConfig{}).to_host_vector(), r.to_host_vector());
    EXPECT_THROW(ActivationModule(ElementwiseOp::Exp, &cpu), std::invalid_argument);
    EXPECT_THROW(ActivationModule(ElementwiseOp::Neg, &cpu), std::invalid_argument);
    ActivationModule fresh(ElementwiseOp::Relu, &cpu);
    EXPECT_THROW((void)fresh.backward(r), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
