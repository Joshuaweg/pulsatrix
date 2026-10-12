// TRN-8: LRP through LoRA, with the relevance split between the base weight and the adapter.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/mse_loss.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

std::vector<float> Random(std::size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = scale * normal(rng);
    return v;
}

float MaxAbs(const std::vector<float>& v) {
    float m = 0;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

double Sum(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return s;
}

class LoraRelevanceTest : public ::testing::Test {
protected:
    CPUBackend backend;
    const int64_t in = 6, out = 5, n = 3;

    LinearModule Adapted(uint32_t seed, float b_scale) {
        LinearModule l(in, out, &backend);
        l.set_weight(Random(static_cast<std::size_t>(in * out), seed, 0.5f));
        l.set_bias(Random(static_cast<std::size_t>(out), seed + 1, 0.2f));
        LoRAConfig c;
        c.rank = 2;
        c.alpha = 4;
        l.attach_lora(c);
        l.lora()->set_b(Random(static_cast<std::size_t>(2 * out), seed + 2, b_scale));
        return l;
    }
    Tensor Input(uint32_t seed) { return Tensor(Shape({n, in}), &backend, Random(static_cast<std::size_t>(n * in), seed)); }
    Tensor Relevance(uint32_t seed) { return Tensor(Shape({n, out}), &backend, Random(static_cast<std::size_t>(n * out), seed)); }
};

// Epsilon: each path gets x_i w_ij R_j / stab(z_j), with z the whole layer's pre-bias output.
TEST_F(LoraRelevanceTest, EpsilonSplitIsEachPathsShareOfTheMergedRelevance) {
    LinearModule l = Adapted(1, 0.5f);
    const Tensor x = Input(4), r = Relevance(5);
    const std::vector<float> z = l.forward(x).to_host_vector();  // with the bias; subtracted below
    const std::vector<float> split = l.propagate_relevance(r, LRPRuleConfig{}).to_host_vector();
    ASSERT_TRUE(l.lora()->has_relevance());
    const std::vector<float> base = l.lora()->last_base_relevance().to_host_vector();
    const std::vector<float> adapter = l.lora()->last_adapter_relevance().to_host_vector();

    const std::vector<float> xv = x.to_host_vector(), rv = r.to_host_vector(), w = l.weight().to_host_vector(),
                             u = l.lora()->update().to_host_vector(), bias = l.bias().to_host_vector();
    const float eps = LRPRuleConfig{}.epsilon;
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t i = 0; i < in; ++i) {
            double want_base = 0, want_adapter = 0;
            for (int64_t j = 0; j < out; ++j) {
                const double zj = z[static_cast<std::size_t>(s * out + j)] - bias[static_cast<std::size_t>(j)];
                const double ratio = rv[static_cast<std::size_t>(s * out + j)] / (zj + (zj >= 0 ? eps : -eps));
                const double xi = xv[static_cast<std::size_t>(s * in + i)];
                want_base += xi * w[static_cast<std::size_t>(i * out + j)] * ratio;
                want_adapter += xi * u[static_cast<std::size_t>(i * out + j)] * ratio;
            }
            const std::size_t k = static_cast<std::size_t>(s * in + i);
            EXPECT_NEAR(base[k], want_base, 1e-4 * (1 + std::fabs(want_base)));
            EXPECT_NEAR(adapter[k], want_adapter, 1e-4 * (1 + std::fabs(want_adapter)));
            EXPECT_NEAR(split[k], base[k] + adapter[k], 1e-6 * (1 + std::fabs(split[k])));
        }
    }

    LRPRuleConfig merged;
    merged.lora_relevance = LoRARelevance::Merged;
    (void)l.forward(x);
    const std::vector<float> whole = l.propagate_relevance(r, merged).to_host_vector();
    EXPECT_FALSE(l.lora()->has_relevance());
    EXPECT_LT(MaxAbsDiff(split, whole), 1e-5f * MaxAbs(whole));
}

TEST_F(LoraRelevanceTest, AnAdapterThatChangesNothingGetsNoRelevance) {
    LinearModule l = Adapted(6, 0.0f);  // B = 0, as right after attach_lora()
    LinearModule plain(in, out, &backend);
    plain.set_weight(l.weight().to_host_vector());
    plain.set_bias(l.bias().to_host_vector());
    const Tensor x = Input(7), r = Relevance(8);
    (void)l.forward(x);
    (void)plain.forward(x);
    LRPRuleConfig gamma;
    gamma.rule = LRPRule::Gamma;
    for (const LRPRuleConfig& config : {LRPRuleConfig{}, gamma}) {
        const std::vector<float> got = l.propagate_relevance(r, config).to_host_vector();
        for (float v : l.lora()->last_adapter_relevance().to_host_vector()) EXPECT_EQ(v, 0.0f);
        EXPECT_LT(MaxAbsDiff(got, plain.propagate_relevance(r, config).to_host_vector()), 1e-6f);
    }
}

// Gamma favours positive weights, and the positive part of W + sAB isn't W+ plus (sAB)+: split and
// merged disagree, though both conserve the relevance.
TEST_F(LoraRelevanceTest, GammaSplitConservesButDiffersFromMerged) {
    LinearModule l(in, out, &backend, /*use_bias=*/false);
    l.set_weight(Random(static_cast<std::size_t>(in * out), 9, 0.5f));
    LoRAConfig c;
    c.rank = 2;
    c.alpha = 4;
    l.attach_lora(c);
    l.lora()->set_b(Random(static_cast<std::size_t>(2 * out), 10, 1.0f));
    const Tensor x(Shape({n, in}), &backend, [&] {
        std::vector<float> v = Random(static_cast<std::size_t>(n * in), 11);
        for (float& e : v) e = std::fabs(e);  // nonnegative inputs, as after a ReLU
        return v;
    }());
    const Tensor r = Relevance(12);
    LRPRuleConfig gamma;
    gamma.rule = LRPRule::Gamma;
    gamma.gamma = 0.5f;
    (void)l.forward(x);
    const std::vector<float> split = l.propagate_relevance(r, gamma).to_host_vector();
    gamma.lora_relevance = LoRARelevance::Merged;
    (void)l.forward(x);
    const std::vector<float> merged = l.propagate_relevance(r, gamma).to_host_vector();
    const double total = Sum(r.to_host_vector());
    EXPECT_NEAR(Sum(split), total, 1e-4 * (1 + std::fabs(total)));
    EXPECT_NEAR(Sum(merged), total, 1e-4 * (1 + std::fabs(total)));
    EXPECT_GT(MaxAbsDiff(split, merged), 1e-2f * MaxAbs(merged));
}

// A base layer that ignores input 3, fine-tuned to use it: the adapter's relevance lands on input 3.
TEST(LoraRelevanceTraining, TheAdapterCarriesWhatTheFineTuneAdded) {
    CPUBackend backend;
    LinearModule l(4, 1, &backend, /*use_bias=*/false);
    l.set_weight({1.0f, 1.0f, 1.0f, 0.0f});
    LoRAConfig c;
    c.rank = 1;
    c.alpha = 1;
    c.seed = 3;
    l.attach_lora(c);
    l.set_requires_grad(false, "weight");
    std::vector<float> xs = Random(64 * 4, 13), ys(64);
    for (int s = 0; s < 64; ++s) {
        const float* x = &xs[static_cast<std::size_t>(4 * s)];
        ys[static_cast<std::size_t>(s)] = x[0] + x[1] + x[2] + 2 * x[3];
    }
    const Tensor x(Shape({64, 4}), &backend, xs), y(Shape({64, 1}), &backend, ys);
    AdamOptimizer opt(5e-2f, &backend);
    MSELoss mse(&backend);
    float loss = 0;
    for (int step = 0; step < 400; ++step) {
        opt.zero_grad(l);
        loss = mse.forward(l.forward(x), y);
        (void)l.backward(mse.backward());
        opt.step(l);
    }
    ASSERT_LT(loss, 1e-3f);

    const Tensor probe(Shape({1, 4}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    const Tensor prediction = l.forward(probe);
    (void)l.propagate_relevance(prediction, LRPRuleConfig{});
    const std::vector<float> base = l.lora()->last_base_relevance().to_host_vector();
    const std::vector<float> adapter = l.lora()->last_adapter_relevance().to_host_vector();
    EXPECT_EQ(base[3], 0.0f);                                   // W never read input 3
    EXPECT_NEAR(adapter[3], 2.0, 0.05);                         // the new term, 2 x_3, is all the adapter's
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(adapter[i], 0.0, 0.05) << i;  // and it left the rest alone
    EXPECT_NEAR(base[0] + base[1] + base[2], 3.0, 0.05);
}

class LoraModelRelevanceTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    const std::vector<float> tokens = {3, 17, 42, 5, 60, 8};

    void SetUp() override { LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config())); }

    // The relevance of the last position's top logit, per token.
    std::vector<float> Explain(const LRPRuleConfig& config) {
        const Tensor logits = model->forward(Tensor(Shape({1, 6}), &backend, tokens));
        std::vector<float> r(static_cast<std::size_t>(logits.numel()), 0.0f), l = logits.to_host_vector();
        const auto last = l.begin() + 5 * 64;
        const auto top = std::max_element(last, last + 64);
        r[static_cast<std::size_t>(top - l.begin())] = *top;
        return model->propagate_relevance(Tensor(logits.shape(), &backend, r), config).to_host_vector();
    }
};

TEST_F(LoraModelRelevanceTest, EveryAdaptedLayerReportsItsSplit) {
    const std::vector<float> split = Explain(LRPRuleConfig{});
    const std::vector<LoraLayerRelevance> layers = LoraRelevanceSplit(*model);
    ASSERT_EQ(layers.size(), 14u);
    EXPECT_EQ(layers.front().name, "layers.0.mha.q_proj");
    double adapter = 0;
    for (const LoraLayerRelevance& layer : layers) {
        EXPECT_EQ(layer.base.shape(), layer.adapter.shape()) << layer.name;
        EXPECT_NEAR(layer.base_total, Sum(layer.base.to_host_vector()), 1e-6 * (1 + std::fabs(layer.base_total)));
        adapter += std::fabs(layer.adapter_total);
    }
    EXPECT_GT(adapter, 0.0);

    // With the epsilon rule, splitting doesn't change the token relevance.
    LRPRuleConfig merged;
    merged.lora_relevance = LoRARelevance::Merged;
    const std::vector<float> whole = Explain(merged);
    EXPECT_LT(MaxAbsDiff(split, whole), 1e-4f * MaxAbs(whole));
    EXPECT_THROW((void)LoraRelevanceSplit(*model), std::logic_error);  // Merged records nothing

    SetLoraEnabled(*model, false);
    (void)Explain(LRPRuleConfig{});
    EXPECT_THROW((void)LoraRelevanceSplit(*model), std::logic_error);
    SetLoraEnabled(*model, true);
    MergeLora(*model);
    (void)Explain(LRPRuleConfig{});
    EXPECT_THROW((void)LoraRelevanceSplit(*model), std::logic_error);
}

TEST_F(LoraModelRelevanceTest, ANewForwardClearsTheSplit) {
    (void)Explain(LRPRuleConfig{});
    LinearModule& q = model->layer(0).mha().q_proj();
    ASSERT_TRUE(q.lora()->has_relevance());
    (void)model->forward(Tensor(Shape({1, 6}), &backend, tokens));
    EXPECT_FALSE(q.lora()->has_relevance());
    EXPECT_THROW((void)q.lora()->last_adapter_relevance(), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
