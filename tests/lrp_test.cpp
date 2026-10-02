#include "pulsatrix/lrp.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

// LRP campaign Mission 1: the whole-model LRP explainer.
namespace pulsatrix {
namespace {

class LRPTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Linear(3, 2) -> ReLU -> Linear(2, 2), zero biases: every rule in the stack conserves, so
    // the input relevance must sum to the seeded output relevance (up to epsilon).
    LinearModule l1{3, 2, &backend};
    ReluModule relu{&backend};
    LinearModule l2{2, 2, &backend};
    void SetUp() override {
        l1.set_weight({0.5f, -1.0f, 1.5f, 0.25f, -0.75f, 2.0f});
        l1.set_bias({0.0f, 0.0f});
        l2.set_weight({1.0f, -0.5f, 0.75f, 2.0f});
        l2.set_bias({0.0f, 0.0f});
    }
};

// Single Linear layer, no bias: relevance of input i for target t is x_i * w_it (epsilon -> 0),
// and it sums to the logit y_t.
TEST_F(LRPTest, SingleLinearLayerMatchesHandComputedContributions) {
    LinearModule lin(3, 2, &backend);
    lin.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // (in=3, out=2)
    lin.set_bias({0.0f, 0.0f});
    ExplainerContext ctx({&lin});
    Tensor x(Shape({1, 3}), &backend, {1.0f, -1.0f, 2.0f});
    LRP lrp;
    Attribution a = lrp.explain(ctx, x, /*target_index=*/1, &backend);
    // y_1 = 1*2 - 1*4 + 2*6 = 10; contributions x_i * w_i1 = 2, -4, 12.
    EXPECT_EQ(a.method, "lrp");
    EXPECT_NEAR(a.values.data()[0], 2.0f, 1e-5f);
    EXPECT_NEAR(a.values.data()[1], -4.0f, 1e-5f);
    EXPECT_NEAR(a.values.data()[2], 12.0f, 1e-5f);
    EXPECT_EQ(a.metadata.at("seed"), "output_value");
    EXPECT_EQ(a.metadata.at("rule"), "epsilon");
}

TEST_F(LRPTest, BiasFreeMlpConservesTheExplainedLogit) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    Tensor y = ctx.forward_pass(x);
    Attribution a = LRP().explain(ctx, x, 0, &backend);
    float sum = 0.0f;
    for (int64_t i = 0; i < 3; ++i) {
        sum += a.values.data()[i];
    }
    EXPECT_NEAR(sum, y.data()[0], 1e-4f);
    EXPECT_NEAR(std::stof(a.metadata.at("relevance_in_sum")), y.data()[0], 1e-4f);
    EXPECT_NEAR(std::stof(a.metadata.at("relevance_out_sum")), y.data()[0], 1e-6f);
}

TEST_F(LRPTest, OneHotSeedDistributesUnitRelevance) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    Attribution a = LRP().explain(ctx, x, LRPTarget{{1}, {}, LRPSeed::OneHot}, &backend);
    float sum = 0.0f;
    for (int64_t i = 0; i < 3; ++i) {
        sum += a.values.data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
    EXPECT_EQ(a.metadata.at("seed"), "one_hot");
}

TEST_F(LRPTest, PerRowTargetsExplainEachRowIndependently) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor batch(Shape({2, 3}), &backend, {1.0f, 2.0f, 0.5f, -1.0f, 0.5f, 1.0f});
    Attribution both = LRP().explain(ctx, batch, LRPTarget{{0, 1}}, &backend);

    Tensor row0(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    Tensor row1(Shape({1, 3}), &backend, {-1.0f, 0.5f, 1.0f});
    Attribution a0 = LRP().explain(ctx, row0, 0, &backend);
    Attribution a1 = LRP().explain(ctx, row1, 1, &backend);
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(both.values.data()[i], a0.values.data()[i]);
        EXPECT_FLOAT_EQ(both.values.data()[3 + i], a1.values.data()[i]);
    }
}

// LRP is linear in the output relevance for fixed activations, so a contrastive seed must give
// exactly explanation(target) - explanation(contrast).
TEST_F(LRPTest, ContrastEqualsDifferenceOfExplanations) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    Attribution contrast = LRP().explain(ctx, x, LRPTarget{{0}, {1}}, &backend);
    Attribution t = LRP().explain(ctx, x, 0, &backend);
    Attribution c = LRP().explain(ctx, x, 1, &backend);
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(contrast.values.data()[i], t.values.data()[i] - c.values.data()[i], 1e-5f);
    }
    EXPECT_EQ(contrast.metadata.at("contrasts"), "1");
}

TEST_F(LRPTest, MatchesManualLayerByLayerPropagation) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    LRPRuleConfig config{0.01f};
    Attribution a = LRP(config).explain(ctx, x, 1, &backend);

    Tensor y = ctx.forward_pass(x);
    Tensor seed(y.shape(), &backend, {0.0f, y.data()[1]});
    Tensor r = l1.propagate_relevance(relu.propagate_relevance(l2.propagate_relevance(seed, config), config),
                                      config);
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(a.values.data()[i], r.data()[i]);
    }
    EXPECT_EQ(a.metadata.at("epsilon"), std::to_string(0.01f));
}

TEST_F(LRPTest, RejectsInvalidTargets) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor batch(Shape({2, 3}), &backend, {1.0f, 2.0f, 0.5f, -1.0f, 0.5f, 1.0f});
    EXPECT_THROW((void)LRP().explain(ctx, batch, 2, &backend), std::invalid_argument);          // out of range
    EXPECT_THROW((void)LRP().explain(ctx, batch, -1, &backend), std::invalid_argument);
    EXPECT_THROW((void)LRP().explain(ctx, batch, LRPTarget{{0, 1, 0}}, &backend), std::invalid_argument);  // 3 != N
    EXPECT_THROW((void)LRP().explain(ctx, batch, LRPTarget{{0}, {0}}, &backend), std::invalid_argument);   // c == t
    EXPECT_THROW((void)LRP().explain(ctx, batch, LRPTarget{{}}, &backend), std::invalid_argument);         // empty
}

TEST_F(LRPTest, RejectsNonRank2Output) {
    LinearModule lin(3, 2, &backend);
    ExplainerContext ctx({&relu});
    Tensor v(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW((void)LRP().explain(ctx, v, 0, &backend), std::invalid_argument);
}

TEST_F(LRPTest, RelevancePassRefusesAfterPatchedForward) {
    ExplainerContext ctx({&l1, &relu, &l2});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
    Tensor y = ctx.forward_pass(x);
    Tensor patch(Shape({1, 2}), &backend, {0.0f, 0.0f});
    (void)ctx.forward_pass_with_patch(x, 1, patch);
    Tensor seed(y.shape(), &backend, {1.0f, 0.0f});
    EXPECT_THROW((void)ctx.relevance_pass(seed, LRPRuleConfig{}), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
