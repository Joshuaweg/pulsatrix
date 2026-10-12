// TRN-11: relevance of a fine-tune minus its base model's, by input and by token, and intruder
// dimensions (Shuttleworth et al., 2024).
#include "pulsatrix/fine_tune_diff.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/ia3.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/loreft.hpp"
#include "pulsatrix/lora.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }
const std::vector<float> kTokens = {3, 17, 42, 5, 60, 8};

std::vector<float> Random(std::size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = scale * normal(rng);
    return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float m = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// A 4 -> 2 layer whose class 0 ignores input 3, and a LoRA update that adds 2 x_3 to it. Under the epsilon
// rule seeded with the logit itself, each input's relevance is x_i W_i0, so the difference is x_i (delta W)_i0:
// 2 x_3 on input 3 and nothing elsewhere.
TEST(FineTuneDiff, TheDifferenceIsWhatTheAdapterAdded) {
    CPUBackend backend;
    LinearModule l(4, 2, &backend, /*use_bias=*/false);
    l.set_weight({1.0f, 0.5f, 1.0f, -0.5f, 1.0f, 0.2f, 0.0f, 0.3f});  // (in, out); W_30 = 0
    LoRAConfig c;
    c.rank = 1;
    c.alpha = 1;
    l.attach_lora(c);
    l.lora()->set_a({0.0f, 0.0f, 0.0f, 1.0f});  // A = e_3
    l.lora()->set_b({2.0f, 0.0f});              // B = 2 e_0
    const Tensor x(Shape({2, 4}), &backend, {1.0f, 2.0f, 0.5f, 1.5f, -1.0f, 0.5f, 2.0f, 1.0f});
    DiffTarget target;
    target.classes = {0, 0};
    const RelevanceDiff d = DiffAdapterRelevance(l, x, target);
    const std::vector<float> diff = d.difference.to_host_vector(), xv = x.to_host_vector();
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(diff[static_cast<std::size_t>(s * 4 + i)], i == 3 ? 2.0f * xv[static_cast<std::size_t>(s * 4 + 3)] : 0.0f, 1e-5)
                << s << " " << i;
        }
        EXPECT_NEAR(d.tuned_scores[static_cast<std::size_t>(s)] - d.base_scores[static_cast<std::size_t>(s)],
                    2.0 * xv[static_cast<std::size_t>(s * 4 + 3)], 1e-5);
    }
    EXPECT_TRUE(l.lora()->enabled());  // switched back on

    // Two separate models give the same, with the base as its own model.
    LinearModule base(4, 2, &backend, false), tuned(4, 2, &backend, false);
    base.set_weight(l.weight().to_host_vector());
    tuned.set_weight(l.lora()->effective_weight(l.weight()).to_host_vector());
    EXPECT_LT(MaxAbsDiff(DiffRelevance(base, tuned, x, target).difference.to_host_vector(), diff), 1e-5f);

    // Without a target, both explain the fine-tuned model's choice.
    const RelevanceDiff chosen = DiffRelevance(base, tuned, x);
    const std::vector<float> logits = tuned.forward(x).to_host_vector();
    for (int s = 0; s < 2; ++s) {
        const int64_t top = logits[static_cast<std::size_t>(2 * s)] >= logits[static_cast<std::size_t>(2 * s + 1)] ? 0 : 1;
        EXPECT_EQ(chosen.classes[static_cast<std::size_t>(s)], top);
    }

    l.merge_lora();
    EXPECT_THROW((void)DiffAdapterRelevance(l, x, target), std::invalid_argument);
    l.unmerge_lora();
    l.detach_lora();
    EXPECT_THROW((void)DiffAdapterRelevance(l, x, target), std::invalid_argument);  // nothing to switch off
    target.classes = {0, 2};
    EXPECT_THROW((void)DiffRelevance(base, tuned, x, target), std::invalid_argument);
}

class FineTuneDiffModelTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    std::unique_ptr<CausalLM> plain = LoadCausalLM(Fixture("qwen2"), &backend);
    const Tensor ids = Tensor(Shape({1, 6}), &backend, kTokens);

    // The base half must be the plain model's relevance for the same token, and the tuned half the
    // adapted model's; the adapters must be on again afterwards.
    void Check(const std::vector<float>& adapted_logits) {
        const RelevanceDiff d = DiffAdapterRelevance(*model, ids);
        ASSERT_EQ(d.tuned.shape(), Shape({1, 6}));
        DiffTarget same;
        same.classes = d.classes;
        const RelevanceDiff against_plain = DiffRelevance(*plain, *model, ids, same);
        EXPECT_LT(MaxAbsDiff(d.base.to_host_vector(), against_plain.base.to_host_vector()), 1e-5f);
        EXPECT_LT(MaxAbsDiff(d.tuned.to_host_vector(), against_plain.tuned.to_host_vector()), 1e-5f);
        float largest = 0;
        for (float v : d.difference.to_host_vector()) largest = std::max(largest, std::fabs(v));
        EXPECT_GT(largest, 1e-3f);
        EXPECT_EQ(model->forward(ids).to_host_vector(), adapted_logits);
    }
};

TEST_F(FineTuneDiffModelTest, LoraAdapters) {
    LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config()));
    Check(model->forward(ids).to_host_vector());
}

TEST_F(FineTuneDiffModelTest, IA3Vectors) {
    LoadIA3Adapter(*model, Fixture("qwen2_ia3"), HuggingFaceMapping(model->config()));
    Check(model->forward(ids).to_host_vector());
}

TEST_F(FineTuneDiffModelTest, LoReFTInterventions) {
    (void)LoadLoReFT(*model, Fixture("qwen2_loreft") + "/loreft.safetensors");
    Check(model->forward(ids).to_host_vector());
    EXPECT_TRUE(model->interventions_enabled());
}

TEST(TokenRelevanceShiftTest, AveragesEachTokensShareOfTheScore) {
    CPUBackend backend;
    RelevanceDiff d{Tensor(Shape({2, 3}), &backend, {2.0f, 1.0f, 1.0f, 0.0f, 3.0f, 1.0f}),
                    Tensor(Shape({2, 3}), &backend, {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f}),
                    Tensor(Shape({2, 3}), &backend, {1.0f, 0.0f, 1.0f, -1.0f, 2.0f, 1.0f}),
                    {0, 0},
                    {4.0, 4.0},
                    {2.0, 2.0}};
    const Tensor ids(Shape({2, 3}), &backend, {7.0f, 8.0f, 9.0f, 8.0f, 7.0f, 9.0f});
    TokenRelevanceShift shift;
    shift.add(ids, d);
    // Token 7: tuned shares 2/4 and 3/4, base 1/2 and 1/2. Token 8: 1/4 and 0, base 1/2 and 1/2. Token 9: 1/4 twice, base 0.
    const std::vector<TokenShift> all = shift.all();
    ASSERT_EQ(all.size(), 3u);
    EXPECT_EQ(all[0].token, 7);
    EXPECT_EQ(all[0].count, 2);
    EXPECT_DOUBLE_EQ(all[0].tuned, 0.625);
    EXPECT_DOUBLE_EQ(all[0].base, 0.5);
    EXPECT_DOUBLE_EQ(all[1].difference, 0.125 - 0.5);
    EXPECT_DOUBLE_EQ(all[2].difference, 0.25);
    const std::vector<TokenShift> up = shift.top(1), down = shift.top(1, /*decrease=*/true);
    EXPECT_EQ(up[0].token, 9);
    EXPECT_EQ(down[0].token, 8);
    EXPECT_TRUE(shift.top(5, false, 3).empty());
    EXPECT_THROW(shift.add(Tensor(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f}), d), std::invalid_argument);
}

TEST(IntruderDimensionsTest, ANewTopDirectionIsAnIntruderAndASmallNudgeIsNot) {
    CPUBackend backend;
    const int64_t in = 64, out = 32;
    const std::vector<float> w0 = Random(static_cast<std::size_t>(in * out), 1, 0.1f);
    const Tensor base(Shape({in, out}), &backend, w0);

    const IntruderResult same = IntruderDimensions(base, base);
    EXPECT_EQ(same.count, 0);
    for (double s : same.max_similarity) EXPECT_NEAR(s, 1.0, 1e-6);

    std::vector<float> nudged = w0;
    const std::vector<float> noise = Random(nudged.size(), 2, 1e-3f);
    for (std::size_t i = 0; i < nudged.size(); ++i) nudged[i] += noise[i];
    const IntruderResult small = IntruderDimensions(base, Tensor(Shape({in, out}), &backend, nudged), 3);
    EXPECT_EQ(small.count, 0);
    for (double s : small.max_similarity) EXPECT_GT(s, 0.9);

    // A rank-1 update v u^T far larger than W0, with u spread evenly over W0's singular vectors: the top
    // singular vector becomes u, whose largest |cos| with any of them is small.
    std::vector<float> u(static_cast<std::size_t>(out), 1.0f / std::sqrt(static_cast<float>(out)));
    for (std::size_t a = 1; a < u.size(); a += 2) u[a] = -u[a];
    const std::vector<float> v = Random(static_cast<std::size_t>(in), 3);
    std::vector<float> intruded = w0;
    for (int64_t i = 0; i < in; ++i)
        for (int64_t a = 0; a < out; ++a) intruded[static_cast<std::size_t>(i * out + a)] += 2.0f * v[static_cast<std::size_t>(i)] * u[static_cast<std::size_t>(a)];
    const IntruderResult big = IntruderDimensions(base, Tensor(Shape({in, out}), &backend, intruded), 1);
    ASSERT_EQ(big.max_similarity.size(), 1u);
    EXPECT_LT(big.max_similarity[0], 0.6);
    EXPECT_EQ(big.count, 1);
    EXPECT_GT(big.singular_values[0], 10.0);

    EXPECT_THROW((void)IntruderDimensions(base, Tensor(Shape({out, in}), &backend, w0)), std::invalid_argument);
    EXPECT_THROW((void)IntruderDimensions(base, base, 0), std::invalid_argument);
    EXPECT_THROW((void)IntruderDimensions(base, base, 10, 1.5), std::invalid_argument);
}

TEST(IntruderDimensionsTest, PerLayerForLoraAndForTwoModels) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    std::unique_ptr<CausalLM> twin = LoadCausalLM(Fixture("qwen2"), &backend);
    for (const LayerIntruders& l : IntruderDimensions(*twin, *model, 4)) EXPECT_EQ(l.result.count, 0) << l.name;
    LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config()));
    const std::vector<LayerIntruders> layers = LoraIntruderDimensions(*model, 4);
    ASSERT_EQ(layers.size(), 14u);
    EXPECT_EQ(layers[0].name, "layers.0.mha.q_proj");
    EXPECT_EQ(layers[0].result.max_similarity.size(), 4u);
    EXPECT_EQ(IntruderDimensions(*twin, *model, 4).size(), 15u);  // 14 projections and lm_head
    RemoveLora(*model);
    EXPECT_THROW((void)LoraIntruderDimensions(*model), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
