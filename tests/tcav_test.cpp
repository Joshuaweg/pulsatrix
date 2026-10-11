// INT-6: TCAV. CAVs, conceptual sensitivities, Kim et al.'s significance test against random
// concepts, and the Student t-test it rests on, checked against the t distribution's closed forms.
#include "pulsatrix/tcav.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {
namespace {

// Uniform in [lo, hi) from mt19937's raw output, the same on every platform.
float Uniform(std::mt19937& rng, float lo, float hi) { return lo + (hi - lo) * static_cast<float>(rng()) / 4294967296.0f; }

class TcavTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Linear(6, 4) passing the first four inputs through -> ReLU -> Linear(4, 2). Class 0's logit
    // is 2 h0 - h1: it rises with h0, falls with h1, and ignores h2 and h3.
    LinearModule l1{6, 4, &backend};
    ReluModule relu{&backend};
    LinearModule l2{4, 2, &backend};
    void SetUp() override {
        std::vector<float> w1(6 * 4, 0.0f);
        for (int k = 0; k < 4; ++k) w1[static_cast<std::size_t>(k * 4 + k)] = 1.0f;  // (in, out)
        l1.set_weight(w1);
        l1.set_bias(std::vector<float>(4, 0.0f));
        l2.set_weight({2.0f, 0.5f, -1.0f, 0.5f, 0.0f, 0.5f, 0.0f, 0.5f});
        l2.set_bias({0.0f, 0.0f});
    }

    // n rows of six inputs, each uniform in [0, 2), except input `high`, uniform in [2, 3).
    Tensor Examples(int64_t n, int high, uint32_t seed) {
        std::mt19937 rng(seed);
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int k = 0; k < 6; ++k) v.push_back(k == high ? Uniform(rng, 2.0f, 3.0f) : Uniform(rng, 0.0f, 2.0f));
        }
        return Tensor(Shape({n, 6}), &backend, v);
    }

    std::vector<Tensor> RandomSets(int count) {
        std::vector<Tensor> sets;
        for (int s = 0; s < count; ++s) sets.push_back(Examples(30, -1, 100 + static_cast<uint32_t>(s)));
        return sets;
    }
};

// ---- the t-test ------------------------------------------------------------------------------------

TEST(StudentTTest, MatchesTheClosedFormsOfTheTDistribution) {
    // Two samples of two: 2 degrees of freedom, where the two-sided p is 1 - |t| / sqrt(2 + t^2).
    const TTestResult two = StudentTTest({0, 1}, {2, 4});
    // Means 0.5 and 3, pooled variance (0.5 + 2) / 2 = 1.25, so t = -2.5 / sqrt(1.25 * (1/2 + 1/2)).
    EXPECT_NEAR(two.t, -2.5 / std::sqrt(1.25), 1e-12);
    EXPECT_EQ(two.degrees_of_freedom, 2);
    EXPECT_NEAR(two.p_value, 1 - std::fabs(two.t) / std::sqrt(2 + two.t * two.t), 1e-12);

    // Two samples of three: 4 degrees of freedom, where it is 1 - t (t^2 + 6) / (t^2 + 4)^1.5.
    const TTestResult four = StudentTTest({1, 2, 4}, {0.5, 0, 1.5});
    EXPECT_EQ(four.degrees_of_freedom, 4);
    const double t = std::fabs(four.t);
    EXPECT_NEAR(four.p_value, 1 - t * (t * t + 6) / std::pow(t * t + 4, 1.5), 1e-12);

    // Symmetric in its samples, and p is 1 at t = 0.
    EXPECT_NEAR(StudentTTest({2, 4}, {0, 1}).p_value, two.p_value, 1e-15);
    EXPECT_NEAR(StudentTTest({1, 2, 3}, {3, 2, 1}).p_value, 1.0, 1e-12);
}

TEST(StudentTTest, MatchesTheTablesCriticalValue) {
    // With 18 degrees of freedom, |t| = 2.10092204024104 is the two-sided 5% point. Shift one sample of
    // ten so its mean differs by exactly that many standard errors.
    std::vector<double> a = {0.3, -1.2, 0.8, 1.9, -0.4, 0.0, 1.1, -0.7, 0.5, -2.3};
    std::vector<double> b = {1.4, 0.2, -0.9, 0.6, -1.5, 0.9, 0.1, -0.3, 1.7, -2.2};
    double ma = 0, mb = 0;
    for (double v : a) ma += v / 10;
    for (double v : b) mb += v / 10;
    double ss = 0;
    for (double v : a) ss += (v - ma) * (v - ma);
    for (double v : b) ss += (v - mb) * (v - mb);
    const double se = std::sqrt(ss / 18 * (1.0 / 10 + 1.0 / 10));
    // Shifting a doesn't change se, so this sets t to the critical value.
    const double shift = 2.10092204024104 * se - (ma - mb);
    for (double& v : a) v += shift;
    const TTestResult r = StudentTTest(a, b);
    EXPECT_NEAR(r.t, 2.10092204024104, 1e-9);
    EXPECT_NEAR(r.p_value, 0.05, 1e-8);
}

TEST(StudentTTest, HandlesSamplesWithoutVarianceAndRejectsTooFewValues) {
    EXPECT_EQ(StudentTTest({1, 1}, {1, 1}).p_value, 1.0);
    const TTestResult apart = StudentTTest({1, 1}, {0, 0});
    EXPECT_EQ(apart.p_value, 0.0);
    EXPECT_TRUE(std::isinf(apart.t));
    EXPECT_THROW((void)StudentTTest({1}, {1, 2}), std::invalid_argument);
    EXPECT_THROW((void)StudentTTest({1, NAN}, {1, 2}), std::invalid_argument);
}

// ---- CAVs ----------------------------------------------------------------------------------------

TEST_F(TcavTest, ACavPointsAtTheConceptAndReportsHeldOutAccuracy) {
    ExplainerContext ctx({&l1, &relu, &l2});
    const Tensor concept_acts = LayerActivations(ctx, Examples(30, 0, 1), 1);
    const Tensor random_acts = LayerActivations(ctx, Examples(30, -1, 2), 1);
    EXPECT_EQ(concept_acts.shape().dim(1), 4);
    for (CavMethod method : {CavMethod::LogisticRegression, CavMethod::DifferenceOfMeans}) {
        CavOptions o;
        o.method = method;
        const Cav cav = TrainCav(concept_acts, random_acts, o);
        ASSERT_EQ(cav.direction.size(), 4u);
        double norm = 0;
        for (double d : cav.direction) norm += d * d;
        EXPECT_NEAR(norm, 1.0, 1e-12);
        EXPECT_GT(cav.direction[0], 0.8) << static_cast<int>(method);  // towards high h0
        // Difference of means cuts at the midpoint, where random examples near the top of their
        // range fall on the wrong side: 0.85 here, against logistic regression's 0.95.
        EXPECT_GT(cav.accuracy, 0.8) << static_cast<int>(method);
    }
    // Two samples of the same distribution can't be told apart on held-out examples.
    const Cav chance = TrainCav(LayerActivations(ctx, Examples(60, -1, 3), 1), LayerActivations(ctx, Examples(60, -1, 4), 1));
    EXPECT_LT(chance.accuracy, 0.75);

    EXPECT_THROW((void)TrainCav(concept_acts, Tensor(Shape({30, 3}), &backend, std::vector<float>(90, 1.0f))), std::invalid_argument);
    EXPECT_THROW((void)TrainCav(Tensor(Shape({2, 4}), &backend, std::vector<float>(8, 1.0f)), random_acts), std::invalid_argument);
    CavOptions bad;
    bad.holdout_every = 1;
    EXPECT_THROW((void)TrainCav(concept_acts, random_acts, bad), std::invalid_argument);
}

TEST_F(TcavTest, LayerGradientsAreTheLogitsGradientAtTheLayer) {
    ExplainerContext ctx({&l1, &relu, &l2});
    // The second row has h1 below zero, so the ReLU blocks its gradient there.
    const Tensor x(Shape({2, 6}), &backend, std::vector<float>{1, 1, 1, 1, 0, 0, 1, -1, 1, 1, 0, 0});
    const std::vector<float> g = LayerGradients(ctx, x, 1, 0, &backend).to_host_vector();
    EXPECT_EQ(g, (std::vector<float>{2, -1, 0, 0, 2, 0, 0, 0}));
    EXPECT_THROW((void)LayerGradients(ctx, x, 4, 0, &backend), std::invalid_argument);
    EXPECT_THROW((void)LayerGradients(ctx, x, 1, 2, &backend), std::invalid_argument);
    EXPECT_THROW((void)LayerActivations(ctx, x, 9), std::invalid_argument);

    Cav cav;
    cav.direction = {0.6, -0.8, 0, 0};
    const std::vector<double> s = ConceptualSensitivities(LayerGradients(ctx, x, 1, 0, &backend), cav);
    EXPECT_NEAR(s[0], 2 * 0.6 + 0.8, 1e-12);
    EXPECT_NEAR(s[1], 2 * 0.6, 1e-12);
    EXPECT_EQ(TcavScore({1.0, -1.0, 0.0, 2.0}), 0.5);  // zero isn't positive
}

// ---- the falsifier: significance against random concepts -----------------------------------------

TEST_F(TcavTest, OnlyConceptsTheClassDependsOnAreSignificant) {
    ExplainerContext ctx({&l1, &relu, &l2});
    const Tensor class_inputs = Examples(40, -1, 7);
    const std::vector<Tensor> randoms = RandomSets(10);

    // High h0 raises class 0's logit: TCAV 1 against every random set.
    const TcavResult up = RunTcav(ctx, 1, 0, class_inputs, Examples(300, 0, 11), randoms, &backend);
    EXPECT_EQ(up.mean_score, 1.0);
    EXPECT_TRUE(up.significant) << up.test.p_value;
    // High h1 lowers it: TCAV 0, just as significant.
    const TcavResult down = RunTcav(ctx, 1, 0, class_inputs, Examples(300, 1, 12), randoms, &backend);
    EXPECT_EQ(down.mean_score, 0.0);
    EXPECT_TRUE(down.significant) << down.test.p_value;
    // High h2 is a real, separable concept (its CAVs are accurate), but the logit ignores it: its
    // scores are no different from random directions'.
    const TcavResult ignored = RunTcav(ctx, 1, 0, class_inputs, Examples(300, 2, 13), randoms, &backend);
    for (double a : ignored.cav_accuracies) EXPECT_GT(a, 0.85);
    EXPECT_FALSE(ignored.significant) << ignored.test.p_value;

    // Random directions score all over the place: one score alone would mislead.
    double lo = 1, hi = 0;
    for (double s : up.random_scores) {
        lo = std::min(lo, s);
        hi = std::max(hi, s);
    }
    EXPECT_GT(hi - lo, 0.5);
    EXPECT_EQ(up.scores.size(), 10u);
    EXPECT_EQ(up.random_scores.size(), 10u);

    EXPECT_THROW((void)RunTcav(ctx, 1, 0, class_inputs, Examples(30, 0, 11), {randoms[0], randoms[1]}, &backend), std::invalid_argument);
}

// A trap the test doesn't remove: every CAV of a concept shares the concept's examples, so their
// sampling noise tilts all of them the same way. With few concept examples an ignored concept
// passes the test far more often than alpha says; with more, the tilt shrinks.
TEST_F(TcavTest, FewConceptExamplesMakeAnIgnoredConceptLookSignificant) {
    ExplainerContext ctx({&l1, &relu, &l2});
    const Tensor class_inputs = Examples(40, -1, 7);
    int few = 0, many = 0;
    for (uint32_t trial = 0; trial < 40; ++trial) {
        std::vector<Tensor> randoms;
        for (uint32_t s = 0; s < 10; ++s) randoms.push_back(Examples(30, -1, 5000 + trial * 100 + s));
        few += RunTcav(ctx, 1, 0, class_inputs, Examples(30, 2, 1000 + trial), randoms, &backend).significant ? 1 : 0;
        many += RunTcav(ctx, 1, 0, class_inputs, Examples(300, 2, 1000 + trial), randoms, &backend).significant ? 1 : 0;
    }
    // At alpha = 0.05, about 2 in 40 are expected by chance. Here 8 pass with 30 examples, 0 with 300.
    EXPECT_GE(few, 5) << "with 30 concept examples";
    EXPECT_LE(many, 2) << "with 300 concept examples";
}

}  // namespace
}  // namespace pulsatrix
