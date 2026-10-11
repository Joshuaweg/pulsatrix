// INT-6: Concept Relevance Propagation. Conditional heatmaps add up to the LRP heatmap, a concept's
// heatmap stays where the concept looks, conditions nest across layers, and relevance and
// activation pick different reference samples.
#include "pulsatrix/concept_relevance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {
namespace {

float Uniform(std::mt19937& rng, float lo, float hi) { return lo + (hi - lo) * static_cast<float>(rng()) / 4294967296.0f; }

std::vector<float> Random(std::size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::vector<float> v(n);
    for (float& x : v) x = Uniform(rng, lo, hi);
    return v;
}

void ExpectNear(const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], tolerance) << "at " << i;
}

std::vector<float> Sum(const std::vector<Attribution>& maps) {
    std::vector<float> total(static_cast<std::size_t>(maps.front().values.numel()), 0.0f);
    for (const Attribution& m : maps) {
        const std::vector<float> v = m.values.to_host_vector();
        for (std::size_t i = 0; i < v.size(); ++i) total[i] += v[i];
    }
    return total;
}

// Conv(2 -> 3, 3x3) -> ReLU -> Flatten -> Linear(27 -> 2), no biases. Filter 0 reads only input
// channel 0, filter 1 only input channel 1, and filter 2 both.
class ConvCrpTest : public ::testing::Test {
protected:
    CPUBackend backend;
    Conv2DModule conv{2, 3, 3, 3, &backend};
    ReluModule relu{&backend};
    FlattenModule flatten{&backend};
    LinearModule fc{27, 2, &backend};
    ExplainerContext ctx{{&conv, &relu, &flatten, &fc}};

    void SetUp() override {
        std::vector<float> kernel = Random(3 * 2 * 3 * 3, 1, -0.5f, 1.0f);  // (out, in, kh, kw)
        for (int k = 0; k < 9; ++k) {
            kernel[static_cast<std::size_t>(0 * 18 + 1 * 9 + k)] = 0.0f;  // filter 0 ignores input channel 1
            kernel[static_cast<std::size_t>(1 * 18 + 0 * 9 + k)] = 0.0f;  // filter 1 ignores input channel 0
        }
        conv.set_kernel(kernel);
        conv.set_bias(std::vector<float>(3, 0.0f));
        fc.set_weight(Random(27 * 2, 2, -0.3f, 1.0f));
        fc.set_bias(std::vector<float>(2, 0.0f));
    }

    Tensor Input(int64_t n, uint32_t seed) { return Tensor(Shape({n, 2, 5, 5}), &backend, Random(static_cast<std::size_t>(n * 50), seed, 0.0f, 1.0f)); }
};

TEST_F(ConvCrpTest, ChannelHeatmapsAddUpToTheLrpHeatmapUnderEveryRule) {
    const Tensor x = Input(2, 3);
    for (const LRP& lrp : {LRP(), LRP::epsilon_plus(), LRP::epsilon_alpha2_beta1()}) {
        const Attribution full = lrp.explain(ctx, x, 0, &backend);
        const std::vector<Attribution> maps = ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, {{{1, {0}}}, {{1, {1}}}, {{1, {2}}}}, &backend);
        ASSERT_EQ(maps.size(), 3u);
        EXPECT_EQ(maps[1].method, "crp");
        EXPECT_EQ(maps[1].metadata.at("condition"), "1:1");
        EXPECT_EQ(maps[0].values.shape(), x.shape());
        ExpectNear(Sum(maps), full.values.to_host_vector(), 1e-5f);
        // All three channels at once is the plain heatmap.
        ExpectNear(ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, {{{1, {0, 1, 2}}}}, &backend)[0].values.to_host_vector(),
                   full.values.to_host_vector(), 1e-5f);
    }
}

TEST_F(ConvCrpTest, ChannelRelevanceSumsToTheExplainedLogit) {
    const Tensor x = Input(2, 4);
    const Tensor logits = ctx.forward_pass(x);
    const std::vector<float> y = logits.to_host_vector();
    const Tensor at_layer = LayerRelevance(LRP(), ctx, x, LRPTarget{{1}}, 2, &backend);  // after the ReLU
    EXPECT_EQ(at_layer.shape(), Shape({2, 3, 3, 3}));
    const std::vector<double> per_channel = ChannelRelevance(at_layer);
    ASSERT_EQ(per_channel.size(), 6u);
    for (int n = 0; n < 2; ++n) {
        EXPECT_NEAR(per_channel[static_cast<std::size_t>(n * 3)] + per_channel[static_cast<std::size_t>(n * 3 + 1)] +
                        per_channel[static_cast<std::size_t>(n * 3 + 2)],
                    y[static_cast<std::size_t>(n * 2 + 1)], 1e-4);
    }
}

// The falsifier: a concept's heatmap is where the concept looks, and nowhere else.
TEST_F(ConvCrpTest, AConceptsHeatmapStaysInTheInputItReads) {
    const Tensor x = Input(1, 5);
    const std::vector<Attribution> maps = ConditionalHeatmaps(LRP(), ctx, x, LRPTarget{{0}}, {{{1, {0}}}, {{1, {1}}}}, &backend);
    const std::vector<float> first = maps[0].values.to_host_vector(), second = maps[1].values.to_host_vector();
    double first_on_0 = 0, first_on_1 = 0, second_on_0 = 0, second_on_1 = 0;
    for (int i = 0; i < 25; ++i) {
        first_on_0 += std::fabs(first[static_cast<std::size_t>(i)]);
        first_on_1 += std::fabs(first[static_cast<std::size_t>(25 + i)]);
        second_on_0 += std::fabs(second[static_cast<std::size_t>(i)]);
        second_on_1 += std::fabs(second[static_cast<std::size_t>(25 + i)]);
    }
    EXPECT_GT(first_on_0, 0.01);
    EXPECT_EQ(first_on_1, 0.0);
    EXPECT_EQ(second_on_0, 0.0);
    EXPECT_GT(second_on_1, 0.01);
}

// Conditions nest: a concept's heatmap is the sum of its heatmaps conditioned further on each
// channel of a lower layer.
TEST(ConceptRelevance, ConditionsOnTwoLayersNest) {
    CPUBackend backend;
    LinearModule l1{4, 5, &backend}, l2{5, 3, &backend}, l3{3, 2, &backend};
    ReluModule r1{&backend}, r2{&backend};
    l1.set_weight(Random(20, 11, -0.5f, 1.0f));
    l2.set_weight(Random(15, 12, -0.5f, 1.0f));
    l3.set_weight(Random(6, 13, -0.5f, 1.0f));
    for (LinearModule* l : {&l1, &l2, &l3}) l->set_bias(std::vector<float>(static_cast<std::size_t>(l == &l1 ? 5 : l == &l2 ? 3 : 2), 0.0f));
    ExplainerContext ctx({&l1, &r1, &l2, &r2, &l3});
    const Tensor x(Shape({3, 4}), &backend, Random(12, 14, 0.0f, 1.0f));
    const LRP lrp;
    // Node 4 is r2's output (3 units); node 2 is r1's (5 units).
    const std::vector<Attribution> top = ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, {{{4, {1}}}}, &backend);
    std::vector<ConceptCondition> nested;
    for (int64_t d = 0; d < 5; ++d) nested.push_back({{4, {1}}, {2, {d}}});
    ExpectNear(Sum(ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, nested, &backend)), top[0].values.to_host_vector(), 1e-5f);
    // The order masks are listed in doesn't matter.
    ExpectNear(ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, {{{2, {3}}, {4, {1}}}}, &backend)[0].values.to_host_vector(),
               ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, {{{4, {1}}, {2, {3}}}}, &backend)[0].values.to_host_vector(), 0.0f);

    // Splitting a relevance pass anywhere changes nothing.
    const Tensor seed = lrp.seed_relevance(ctx, x, LRPTarget{{0}}, &backend);
    const std::vector<LRPRuleConfig> configs = lrp.rule_configs(ctx);
    const Tensor whole = ctx.relevance_pass(seed, configs);
    const Tensor split = ctx.relevance_pass_through(ctx.relevance_pass_through(seed, configs, 3, 5), configs, 0, 3);
    ExpectNear(split.to_host_vector(), whole.to_host_vector(), 0.0f);
    EXPECT_THROW((void)ctx.relevance_pass_through(seed, configs, 3, 2), std::invalid_argument);
    EXPECT_THROW((void)ctx.relevance_pass_through(seed, configs, 0, 6), std::invalid_argument);
}

// Relevance maximization picks the inputs where a channel works for the class; activation
// maximization those where it fires, whether or not that reaches the output.
TEST(ConceptRelevance, RelevanceAndActivationPickDifferentReferenceSamples) {
    CPUBackend backend;
    // h = ReLU(x) (two units), then the output is ReLU(h0 - h1): h0 helps only when it beats h1.
    LinearModule pass{2, 2, &backend}, diff{2, 1, &backend};
    ReluModule r1{&backend}, r2{&backend};
    pass.set_weight({1, 0, 0, 1});
    pass.set_bias({0, 0});
    diff.set_weight({1, -1});
    diff.set_bias({0});
    ExplainerContext ctx({&pass, &r1, &diff, &r2});
    // Row 0 has the most h0, but h1 is larger, so none of it reaches the output.
    const Tensor candidates(Shape({4, 2}), &backend, std::vector<float>{5, 6, 3, 1, 4, 2, 1, 0.5f});
    const LRP lrp;
    EXPECT_EQ(ConceptReferenceSamples(lrp, ctx, candidates, 0, 2, 0, 2, ReferenceMode::Activation, &backend),
              (std::vector<int64_t>{0, 2}));
    EXPECT_EQ(ConceptReferenceSamples(lrp, ctx, candidates, 0, 2, 0, 2, ReferenceMode::Relevance, &backend),
              (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(ConceptReferenceSamples(lrp, ctx, candidates, 0, 2, 0, 10, ReferenceMode::Relevance, &backend).size(), 4u);
    EXPECT_THROW((void)ConceptReferenceSamples(lrp, ctx, candidates, 0, 2, 2, 1, ReferenceMode::Relevance, &backend), std::invalid_argument);
    EXPECT_THROW((void)ConceptReferenceSamples(lrp, ctx, candidates, 0, 2, 0, 0, ReferenceMode::Activation, &backend), std::invalid_argument);
}

TEST_F(ConvCrpTest, RejectsConditionsItCannotApply) {
    const Tensor x = Input(1, 6);
    const LRP lrp;
    auto run = [&](std::vector<ConceptCondition> c) { return ConditionalHeatmaps(lrp, ctx, x, LRPTarget{{0}}, c, &backend); };
    EXPECT_THROW((void)run({{}}), std::invalid_argument);
    EXPECT_THROW((void)run({{{0, {0}}}}), std::invalid_argument);           // the input isn't a layer
    EXPECT_THROW((void)run({{{5, {0}}}}), std::invalid_argument);           // past the output
    EXPECT_THROW((void)run({{{1, {3}}}}), std::invalid_argument);           // no such channel
    EXPECT_THROW((void)run({{{1, {0}}, {1, {1}}}}), std::invalid_argument);  // the same layer twice
    EXPECT_THROW((void)LayerRelevance(lrp, ctx, x, LRPTarget{{0}}, 0, &backend), std::invalid_argument);
    EXPECT_THROW((void)ChannelRelevance(Tensor(Shape({3}), &backend, std::vector<float>{1, 2, 3})), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
