// CFS-5 and CFS-6: gradient counterfactuals (Wachter et al. 2017) and growing spheres (Laugel et
// al. 2018). On a linear classifier the L1-optimal
// counterfactual is known in closed form: move the single feature with the largest
// |w_target - w_other| * scale just past the decision boundary. The search must find it.

#include "pulsatrix/counterfactual.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {
namespace {

class CounterfactualTest : public ::testing::Test {
protected:
    CPUBackend backend;
    // Logits z = x W + b for 3 features and 2 classes. Column differences w1 - w0 per feature:
    // f0: 1.0, f1: -0.5, f2: 2.0. So f2 is the cheapest feature to move toward class 1.
    LinearModule model{3, 2, &backend};
    Tensor input{Shape({1, 3}), &backend, {1.0f, 1.0f, -1.0f}};

    void SetUp() override {
        model.set_weight({0.0f, 1.0f,   // f0
                          0.5f, 0.0f,   // f1
                          -1.0f, 1.0f}); // f2
        model.set_bias({0.5f, 0.0f});
    }
    // z1 - z0 at x: (1 - 0) * 1 + (0 - 0.5) * 1 + (1 + 1) * -1 + (0 - 0.5) = 1 - 0.5 - 2 - 0.5 = -2.
    std::vector<float> Logits(const Tensor& x) { return model.forward(x).to_host_vector(); }
};

TEST_F(CounterfactualTest, MovesOnlyTheCheapestFeatureJustPastTheBoundary) {
    ExplainerContext ctx({&model});
    ASSERT_LT(Logits(input)[1], Logits(input)[0]);
    CounterfactualResult r = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1));
    ASSERT_TRUE(r.valid);
    const std::vector<float> cf = r.counterfactual.to_host_vector();
    EXPECT_EQ(r.num_changed, 1);
    EXPECT_FLOAT_EQ(cf[0], 1.0f);
    EXPECT_FLOAT_EQ(cf[1], 1.0f);
    // Closing a gap of 2 through f2 (slope 2) takes a change of 1: the L1 minimum.
    EXPECT_NEAR(cf[2], 0.0f, 0.05f);
    EXPECT_NEAR(r.distance, 1.0f, 0.05f);
    EXPECT_GE(Logits(r.counterfactual)[1], Logits(r.counterfactual)[0]);
    EXPECT_FLOAT_EQ(r.output_before, Logits(input)[1]);
}

// The scale changes which move is cheapest: with f2 four times as costly, f0 wins (gap 2 at
// slope 1 costs 2, f2 now costs 1 * 4).
TEST_F(CounterfactualTest, ScaleDecidesWhichFeatureIsCheapest) {
    ExplainerContext ctx({&model});
    CounterfactualConstraints c;
    c.scale = {1.0f, 1.0f, 0.25f};
    CounterfactualResult r = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), c);
    ASSERT_TRUE(r.valid);
    const std::vector<float> cf = r.counterfactual.to_host_vector();
    EXPECT_NEAR(cf[0], 3.0f, 0.1f);
    EXPECT_FLOAT_EQ(cf[2], -1.0f);
}

TEST_F(CounterfactualTest, RespectsImmutableFeaturesAndLimits) {
    ExplainerContext ctx({&model});
    CounterfactualConstraints c;
    c.immutable = {2};
    CounterfactualResult r = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), c);
    ASSERT_TRUE(r.valid);
    EXPECT_FLOAT_EQ(r.counterfactual.to_host_vector()[2], -1.0f);
    EXPECT_NEAR(r.counterfactual.to_host_vector()[0], 3.0f, 0.1f);  // next cheapest: f0, slope 1

    // Every route blocked: f2 frozen, f0 capped at 1.5, f1 can only move up (the wrong way).
    c.lower = {-10.0f, 1.0f, -10.0f};
    c.upper = {1.5f, 10.0f, 10.0f};
    CounterfactualOptions quick;
    quick.max_rounds = 2;
    quick.steps_per_round = 100;
    CounterfactualResult blocked = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), c, quick);
    EXPECT_FALSE(blocked.valid);
    const std::vector<float> cf = blocked.counterfactual.to_host_vector();
    EXPECT_LE(cf[0], 1.5f);
    EXPECT_GE(cf[1], 1.0f);
    EXPECT_EQ(blocked.rounds, 2);
}

TEST_F(CounterfactualTest, ReachesARegressionRange) {
    LinearModule reg(3, 1, &backend);
    reg.set_weight({1.0f, 2.0f, 0.0f});
    reg.set_bias({0.0f});
    ExplainerContext ctx({&reg});
    CounterfactualResult r = FindCounterfactual(ctx, input, CounterfactualTarget::ToRange(0, 5.0f, 6.0f));
    ASSERT_TRUE(r.valid);
    EXPECT_GE(r.output_after, 5.0f);
    EXPECT_LE(r.output_after, 6.0f);
    EXPECT_FLOAT_EQ(r.output_before, 3.0f);
    EXPECT_EQ(r.num_changed, 1);  // f1 has the larger slope
}

// Integer and one-hot features are fixed up at the end, and validity is judged after that.
TEST_F(CounterfactualTest, RoundsIntegerFeaturesAndProjectsOneHotGroups) {
    LinearModule m(4, 2, &backend);
    // Features 1 and 2 one-hot encode a category; category "2" strongly favors class 1.
    m.set_weight({0.0f, 0.3f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.2f});
    m.set_bias({1.0f, 0.0f});
    ExplainerContext ctx({&m});
    Tensor x(Shape({1, 4}), &backend, {1.0f, 1.0f, 0.0f, 2.0f});
    CounterfactualConstraints c;
    c.integer = {0, 3};
    c.one_hot_groups = {{1, 2}};
    CounterfactualResult r = FindCounterfactual(ctx, x, CounterfactualTarget::ToClass(1, 0.1f), c);
    const std::vector<float> cf = r.counterfactual.to_host_vector();
    EXPECT_EQ(cf[0], std::round(cf[0]));
    EXPECT_EQ(cf[3], std::round(cf[3]));
    EXPECT_EQ(cf[1] + cf[2], 1.0f);
    EXPECT_TRUE(cf[1] == 0.0f || cf[1] == 1.0f);
    // The search moves f2 only partway (f2 has slope 3, so 0.13 closes the gap); snapping back to
    // one-hot would undo that, so the category switches to f2 outright.
    EXPECT_TRUE(r.valid);
    EXPECT_EQ(cf[2], 1.0f);
    const std::vector<float> z = m.forward(r.counterfactual).to_host_vector();
    EXPECT_GE(z[1] - z[0], 0.1f);
}

TEST_F(CounterfactualTest, WorksThroughANonlinearModel) {
    LinearModule l1(3, 4, &backend), l2(4, 2, &backend);
    ReluModule relu(&backend);
    l1.set_weight({0.5f, -0.3f, 0.8f, 0.1f, -0.6f, 0.4f, 0.2f, -0.7f, 0.9f, 0.3f, -0.5f, 0.6f});
    l1.set_bias({0.1f, 0.0f, -0.1f, 0.2f});
    l2.set_weight({1.0f, -1.0f, -0.5f, 0.8f, 0.7f, -0.2f, -0.4f, 0.9f});
    l2.set_bias({0.3f, -0.3f});
    ExplainerContext ctx({&l1, &relu, &l2});
    const std::vector<float> z = ctx.forward_pass(input).to_host_vector();
    const int64_t other = z[0] >= z[1] ? 1 : 0;
    CounterfactualResult r = FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(other, 0.05f));
    ASSERT_TRUE(r.valid);
    const std::vector<float> zc = ctx.forward_pass(r.counterfactual).to_host_vector();
    EXPECT_GE(zc[static_cast<size_t>(other)] - zc[static_cast<size_t>(1 - other)], 0.05f - 1e-5f);
    EXPECT_GT(r.distance, 0.0f);
}

TEST_F(CounterfactualTest, MedianAbsoluteDeviation) {
    std::vector<Tensor> bg;
    for (float v : {1.0f, 2.0f, 4.0f, 7.0f, 100.0f}) {
        bg.emplace_back(Shape({2}), &backend, std::vector<float>{v, 3.0f});
    }
    // Median 4; deviations 3, 2, 0, 3, 96 -> median 3. The constant feature gets 1.
    const std::vector<float> mad = MedianAbsoluteDeviation(bg);
    EXPECT_FLOAT_EQ(mad[0], 3.0f);
    EXPECT_FLOAT_EQ(mad[1], 1.0f);
    EXPECT_THROW((void)MedianAbsoluteDeviation({}), std::invalid_argument);
}

TEST_F(CounterfactualTest, RejectsBadInput) {
    ExplainerContext ctx({&model});
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(2)), std::invalid_argument);
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToRange(0, 2.0f, 1.0f)),
                 std::invalid_argument);
    CounterfactualConstraints bad;
    bad.scale = {1.0f};
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), bad), std::invalid_argument);
    bad = {};
    bad.immutable = {3};
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), bad), std::invalid_argument);
    bad = {};
    bad.scale = {1.0f, 0.0f, 1.0f};
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), bad), std::invalid_argument);
    CounterfactualOptions o;
    o.learning_rate = 0.0f;
    EXPECT_THROW((void)FindCounterfactual(ctx, input, CounterfactualTarget::ToClass(1), {}, o), std::invalid_argument);
}

// ---- CFS-6: growing spheres ----------------------------------------------------------------

// A model with no gradient anywhere: class 1 iff x0 > 2 and x1 < 0 (a two-rule tree).
std::vector<float> RuleLogits(const std::vector<float>& v) {
    const bool yes = v[0] > 2.0f && v[1] < 0.0f;
    return {yes ? 0.0f : 1.0f, yes ? 1.0f : 0.0f};
}

TEST_F(CounterfactualTest, GrowingSpheresHandlesAModelWithNoGradient) {
    auto rule = [this](const Tensor& x) { return Tensor(Shape({1, 2}), &backend, RuleLogits(x.to_host_vector())); };
    Tensor x(Shape({1, 3}), &backend, {1.0f, 1.0f, 5.0f});
    GrowingSpheresOptions o;
    o.seed = 11;
    CounterfactualResult r = GrowingSpheresCounterfactual(rule, x, CounterfactualTarget::ToClass(1), {}, o);
    ASSERT_TRUE(r.valid);
    const std::vector<float> cf = r.counterfactual.to_host_vector();
    EXPECT_GT(cf[0], 2.0f);
    EXPECT_LT(cf[1], 0.0f);
    EXPECT_EQ(cf[2], 5.0f);  // not needed, so put back exactly
    EXPECT_EQ(r.num_changed, 2);
    EXPECT_GT(r.rounds, 1);
    // The same seed gives the same counterfactual.
    CounterfactualResult again = GrowingSpheresCounterfactual(rule, x, CounterfactualTarget::ToClass(1), {}, o);
    EXPECT_EQ(again.counterfactual.to_host_vector(), cf);
}

TEST_F(CounterfactualTest, GrowingSpheresComesCloseToTheGradientSearchOnALinearModel) {
    auto predict = [this](const Tensor& x) { return model.forward(x); };
    CounterfactualResult r = GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1));
    ASSERT_TRUE(r.valid);
    EXPECT_GE(r.distance, 1.0f - 1e-3f);  // the L1 optimum (see the first test)
    EXPECT_LT(r.distance, 2.0f);
    const std::vector<float> z = model.forward(r.counterfactual).to_host_vector();
    EXPECT_GE(z[1], z[0]);
}

TEST_F(CounterfactualTest, GrowingSpheresRespectsConstraints) {
    auto predict = [this](const Tensor& x) { return model.forward(x); };
    CounterfactualConstraints c;
    c.immutable = {2};
    c.lower = {-10.0f, -10.0f, -10.0f};
    c.upper = {10.0f, 1.5f, 10.0f};
    CounterfactualResult r = GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1), c);
    ASSERT_TRUE(r.valid);
    const std::vector<float> cf = r.counterfactual.to_host_vector();
    EXPECT_EQ(cf[2], -1.0f);
    EXPECT_LE(cf[1], 1.5f);

    // Already there: nothing changes. Impossible: reported invalid after max_layers.
    CounterfactualResult same = GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(0));
    EXPECT_TRUE(same.valid);
    EXPECT_EQ(same.num_changed, 0);
    c.immutable = {0, 1, 2};
    EXPECT_FALSE(GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1), c).valid);
    c.immutable = {2};
    c.upper = {1.5f, 10.0f, 10.0f};
    c.lower = {-10.0f, 1.0f, -10.0f};
    GrowingSpheresOptions few;
    few.samples_per_layer = 50;
    few.max_layers = 6;
    CounterfactualResult blocked = GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1), c, few);
    EXPECT_FALSE(blocked.valid);
    EXPECT_EQ(blocked.rounds, 6);
}

TEST_F(CounterfactualTest, GrowingSpheresRejectsBadOptions) {
    auto predict = [this](const Tensor& x) { return model.forward(x); };
    GrowingSpheresOptions o;
    o.initial_radius = 0.0f;
    EXPECT_THROW((void)GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(1), {}, o),
                 std::invalid_argument);
    EXPECT_THROW((void)GrowingSpheresCounterfactual(predict, input, CounterfactualTarget::ToClass(5)),
                 std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
