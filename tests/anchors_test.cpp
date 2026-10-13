// XAI-4: Anchors on models whose anchors are known, the KL bounds, and validation.

#include "pulsatrix/anchors.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class AnchorsTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::vector<Tensor> background;

    void SetUp() override {
        for (int i = 0; i < 400; ++i) {
            std::vector<float> r(3);
            for (int a = 0; a < 3; ++a) {
                const double di = i, da = a;
                r[static_cast<size_t>(a)] = static_cast<float>(std::sin(0.3 + 0.37 * di + 1.13 * da + 0.05 * di * da + 0.71 * di * di));
            }
            background.emplace_back(Shape({1, 3}), &backend, r);
        }
    }

    Tensor Input(float a, float b, float c) { return Tensor(Shape({1, 3}), &backend, {a, b, c}); }

    // Two outputs; the class is 1 when @p rule holds.
    std::function<Tensor(const Tensor&)> Classifier(std::function<bool(const std::vector<float>&)> rule) {
        return [this, rule](const Tensor& in) {
            const bool one = rule(in.to_host_vector());
            return Tensor(Shape({1, 2}), &backend, {one ? 0.0f : 1.0f, one ? 1.0f : 0.0f});
        };
    }

    double Covered(const Anchor& a) {
        int n = 0;
        for (const Tensor& b : background) n += a.applies_to(b.to_host_vector()) ? 1 : 0;
        return static_cast<double>(n) / static_cast<double>(background.size());
    }
};

// One feature decides: the anchor is that feature's bin and nothing else.
TEST_F(AnchorsTest, OneDecidingFeatureIsTheWholeAnchor) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] > 0.2f; });
    const Anchor a = FindAnchor(model, Input(0.9f, 0.0f, 0.0f), background);
    EXPECT_EQ(a.predicted_class, 1);
    ASSERT_TRUE(a.meets_threshold);
    ASSERT_EQ(a.predicates.size(), 1u);
    EXPECT_EQ(a.predicates[0].feature, 0);
    EXPECT_GT(a.predicates[0].lower, 0.2);  // the top quartile lies wholly above the model's threshold
    EXPECT_TRUE(std::isinf(a.predicates[0].upper));
    EXPECT_DOUBLE_EQ(a.precision, 1.0);
    EXPECT_GE(a.precision_lower, 0.9);
    EXPECT_NEAR(a.coverage, 0.25, 0.01);
    EXPECT_DOUBLE_EQ(a.coverage, Covered(a));
    EXPECT_TRUE(a.applies_to({0.9f, 0.0f, 0.0f}));
    EXPECT_GT(a.model_calls, a.samples);
}

// x0 > median keeps class 1 for about 93% of inputs. With the reference implementation's
// epsilon_stop of 0.05, sampling noise often lifts that predicate plus an irrelevant one over 0.95,
// and the wider, wrong rule wins on coverage. Requiring the lower bound to reach the threshold
// (the default here) keeps the irrelevant features out, whatever the seed.
TEST_F(AnchorsTest, StrictAcceptanceKeepsIrrelevantFeaturesOut) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] > 0.2f; });
    for (uint64_t seed = 0; seed < 20; ++seed) {
        AnchorOptions options;
        options.seed = seed;
        const Anchor a = FindAnchor(model, Input(0.9f, 0.0f, 0.0f), background, options);
        ASSERT_TRUE(a.meets_threshold) << "seed " << seed;
        ASSERT_EQ(a.predicates.size(), 1u) << "seed " << seed << ": " << DescribeAnchor(a);
        EXPECT_EQ(a.predicates[0].feature, 0);
    }
}

// Both features are needed: either alone leaves the other free to flip the class. Predicates are
// half-lines at the percentile cuts, as in the reference implementation, so the anchor can be wider
// than the input's own bins.
TEST_F(AnchorsTest, AConjunctionNeedsBothPredicates) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] > 0.0f && v[1] > 0.0f; });
    const Anchor a = FindAnchor(model, Input(0.9f, 0.9f, 0.0f), background);
    EXPECT_EQ(a.predicted_class, 1);
    ASSERT_TRUE(a.meets_threshold);
    ASSERT_EQ(a.predicates.size(), 2u);
    std::vector<int64_t> features = {a.predicates[0].feature, a.predicates[1].feature};
    std::sort(features.begin(), features.end());
    EXPECT_EQ(features, (std::vector<int64_t>{0, 1}));
    EXPECT_DOUBLE_EQ(a.precision, 1.0);
    EXPECT_DOUBLE_EQ(a.coverage, Covered(a));
    // The half-lines at the medians, close to the model's own rule (x0 > 0 and x1 > 0), not the
    // top quartiles where the input lies: about a quarter of the data, not a sixteenth.
    for (const AnchorPredicate& p : a.predicates) {
        EXPECT_NEAR(p.lower, 0.0, 0.1);
        EXPECT_TRUE(std::isinf(p.upper));
    }
    EXPECT_GT(a.coverage, 0.2);
}

// The other side of the same model: class 0 needs only one feature to stay low.
TEST_F(AnchorsTest, TheAnchorFollowsThePrediction) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] > 0.0f && v[1] > 0.0f; });
    const Anchor a = FindAnchor(model, Input(-0.9f, 0.9f, 0.0f), background);
    EXPECT_EQ(a.predicted_class, 0);
    ASSERT_TRUE(a.meets_threshold);
    ASSERT_EQ(a.predicates.size(), 1u);
    EXPECT_EQ(a.predicates[0].feature, 0);
    EXPECT_TRUE(std::isinf(a.predicates[0].lower));
    EXPECT_LT(a.predicates[0].upper, 0.0);
}

TEST_F(AnchorsTest, AModelThatAlwaysAgreesNeedsNoPredicate) {
    const auto model = Classifier([](const std::vector<float>&) { return true; });
    const Anchor a = FindAnchor(model, Input(0.1f, 0.2f, 0.3f), background);
    EXPECT_TRUE(a.meets_threshold);
    EXPECT_TRUE(a.predicates.empty());
    EXPECT_DOUBLE_EQ(a.coverage, 1.0);
    EXPECT_EQ(DescribeAnchor(a), "ALWAYS THEN class 1 (precision 1, coverage 1)");
}

// A class that changes faster than any bin: no rule of quartile bins keeps it.
TEST_F(AnchorsTest, SaysSoWhenNoAnchorReachesTheThreshold) {
    const auto model = Classifier([](const std::vector<float>& v) { return std::sin(40.0f * (v[0] + v[1] + v[2])) > 0.0f; });
    AnchorOptions options;
    options.max_samples_per_candidate = 2000;
    const Anchor a = FindAnchor(model, Input(0.1f, 0.2f, 0.3f), background, options);
    EXPECT_FALSE(a.meets_threshold);
    EXPECT_LT(a.precision, 0.95);
    EXPECT_NE(DescribeAnchor(a).find("below the precision threshold"), std::string::npos);
}

TEST_F(AnchorsTest, SeededAndValidated) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] + 0.3f * v[2] > 0.1f; });
    AnchorOptions options;
    options.seed = 11;
    const Anchor a = FindAnchor(model, Input(0.5f, 0.0f, 0.5f), background, options);
    const Anchor b = FindAnchor(model, Input(0.5f, 0.0f, 0.5f), background, options);
    EXPECT_EQ(DescribeAnchor(a), DescribeAnchor(b));
    EXPECT_EQ(a.samples, b.samples);
    EXPECT_LE(a.precision_lower, a.precision);
    EXPECT_GE(a.precision_upper, a.precision);

    EXPECT_THROW((void)FindAnchor(model, Input(0, 0, 0), {}), std::invalid_argument);
    EXPECT_THROW((void)FindAnchor(model, Input(0, 0, 0), {Tensor(Shape({1, 2}), &backend)}), std::invalid_argument);
    AnchorOptions bad;
    bad.percentiles = {50.0, 25.0};
    EXPECT_THROW((void)FindAnchor(model, Input(0, 0, 0), background, bad), std::invalid_argument);
    bad = AnchorOptions();
    bad.precision_threshold = 1.5;
    EXPECT_THROW((void)FindAnchor(model, Input(0, 0, 0), background, bad), std::invalid_argument);
    bad = AnchorOptions();
    bad.max_samples_per_candidate = 10;  // below one batch
    EXPECT_THROW((void)FindAnchor(model, Input(0, 0, 0), background, bad), std::invalid_argument);
}

TEST_F(AnchorsTest, MaxAnchorSizeLimitsTheRule) {
    const auto model = Classifier([](const std::vector<float>& v) { return v[0] > 0.0f && v[1] > 0.0f; });
    AnchorOptions options;
    options.max_anchor_size = 1;
    const Anchor a = FindAnchor(model, Input(0.9f, 0.9f, 0.0f), background, options);
    EXPECT_LE(a.predicates.size(), 1u);
    EXPECT_FALSE(a.meets_threshold);
}

TEST_F(AnchorsTest, DescribesAndWritesTheRule) {
    Anchor a;
    a.predicted_class = 1;
    AnchorPredicate income;
    income.feature = 0;
    income.lower = 0.71;
    AnchorPredicate debt;
    debt.feature = 2;
    debt.lower = -0.5;
    debt.upper = -0.12;
    a.predicates = {income, debt};
    a.precision = 0.97;
    a.coverage = 0.21;
    a.meets_threshold = true;
    EXPECT_EQ(DescribeAnchor(a, {"income", "age", "debt"}),
              "IF income > 0.71 AND -0.5 < debt <= -0.12 THEN class 1 (precision 0.97, coverage 0.21)");
    EXPECT_EQ(DescribeAnchor(a), "IF x[0] > 0.71 AND -0.5 < x[2] <= -0.12 THEN class 1 (precision 0.97, coverage 0.21)");
    EXPECT_TRUE(a.applies_to({0.8f, 9.0f, -0.2f}));
    EXPECT_FALSE(a.applies_to({0.8f, 9.0f, -0.1f}));
    EXPECT_THROW((void)a.applies_to({0.8f}), std::invalid_argument);

    const JsonValue doc = ParseJson(WriteJson(ToJson(a, {"income"})));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.anchor.v1");
    const JsonValue& first = doc.find("predicates")->as_array()[0];
    EXPECT_EQ(first.find("name")->as_string(), "income");
    EXPECT_TRUE(first.find("upper")->is_null());
    EXPECT_DOUBLE_EQ(first.find("lower")->as_double(), 0.71);
    EXPECT_EQ(doc.find("predicates")->as_array()[1].find("name")->as_string(), "x[2]");
    EXPECT_TRUE(doc.find("meets_threshold")->as_bool());
}

TEST(AnchorBounds, MatchTheKlDefinition) {
    EXPECT_NEAR(BernoulliKl(0.5, 0.5), 0.0, 1e-12);
    // 0.3 ln(0.3/0.6) + 0.7 ln(0.7/0.4)
    EXPECT_NEAR(BernoulliKl(0.3, 0.6), 0.3 * std::log(0.5) + 0.7 * std::log(1.75), 1e-12);
    for (double p : {0.0, 0.2, 0.5, 0.9, 1.0}) {
        for (double level : {0.01, 0.1, 1.0}) {
            const double up = KlUpperBound(p, level), lo = KlLowerBound(p, level);
            EXPECT_GE(up, p);
            EXPECT_LE(lo, p);
            EXPECT_LE(up, 1.0);
            EXPECT_GE(lo, 0.0);
            // Bisection to 2^-16 of the starting interval: the bound sits just past KL = level.
            if (up < 1.0 && p < 1.0) EXPECT_NEAR(BernoulliKl(p, up), level, 0.02 + 0.05 * level) << p << " " << level;
        }
    }
    // More samples, a smaller level, tighter bounds.
    EXPECT_LT(KlUpperBound(0.9, 0.01) - KlLowerBound(0.9, 0.01), KlUpperBound(0.9, 0.1) - KlLowerBound(0.9, 0.1));
}

}  // namespace
}  // namespace pulsatrix
