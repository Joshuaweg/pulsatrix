// XAI-4: pertinent negatives and positives (CEM) on linear models whose answers can be worked out by
// hand.

#include "pulsatrix/contrastive.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

class ContrastiveTest : public ::testing::Test {
protected:
    CPUBackend backend;
    LinearModule linear{4, 2, &backend};
    ExplainerContext ctx{{&linear}};

    Tensor Input(std::vector<float> v) { return Tensor(Shape({1, 4}), &backend, v); }
};

// z0 = x0 + x1, z1 = x2 + 0.5 x3. The input (1, 0, 0, 0) is class 0, and x1, x2, x3 are absent.
// Adding x1 only strengthens class 0; adding x2 (or, at twice the cost, x3) flips it. The elastic
// net spreads the change: minimizing 0.1 (a + b) + a^2 + b^2 subject to a + b / 2 = 1 gives
// a = 0.81, b = 0.38.
TEST_F(ContrastiveTest, APertinentNegativeAddsOnlyWhatFlipsTheClass) {
    linear.set_weight({1, 0, 1, 0, 0, 1, 0, 0.5f});
    linear.set_bias({0, 0});
    const ContrastiveResult r = FindPertinentNegative(ctx, Input({1, 0, 0, 0}));
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.original_class, 0);
    EXPECT_EQ(r.explanation_class, 1);
    EXPECT_EQ(r.delta[0], 0.0f);  // present: not part of a pertinent negative
    EXPECT_EQ(r.delta[1], 0.0f);  // absent, but adding it would only confirm class 0
    EXPECT_NEAR(r.delta[2], 0.81f, 0.01f);
    EXPECT_NEAR(r.delta[3], 0.38f, 0.01f);
    EXPECT_EQ(r.features, (std::vector<int64_t>{2, 3}));
    EXPECT_FLOAT_EQ(r.explanation[0], 1.0f);
    EXPECT_NEAR(r.l1, r.delta[2] + r.delta[3], 1e-6);
}

TEST_F(ContrastiveTest, UpperLimitsWhatAPertinentNegativeMayAdd) {
    linear.set_weight({1, 0, 1, 0, 0, 1, 0, 0.5f});
    linear.set_bias({0, 0});
    ContrastiveOptions options;
    options.upper = {1, 1, 0, 10};  // x2 may not be raised at all
    ContrastiveResult r = FindPertinentNegative(ctx, Input({1, 0, 0, 0}), options);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.delta[2], 0.0f);
    EXPECT_GT(r.delta[3], 2.0f - 0.05f);  // 0.5 x3 must pass 1
    // Nothing absent can flip it: not found, and the input comes back unchanged.
    options.upper = {1, 1, 0, 0};
    r = FindPertinentNegative(ctx, Input({1, 0, 0, 0}), options);
    EXPECT_FALSE(r.found);
    EXPECT_EQ(r.explanation_class, 0);
    EXPECT_TRUE(r.features.empty());
}

// z0 = x0 + 0.5 x1, z1 = x2 + x3 + 0.5. The input (1, 1, 0.2, 0) is class 0; nothing (the baseline)
// is class 1. The pertinent positive keeps just enough of x0 and x1 to pass z1, and none of x2,
// which only helps class 1: minimizing 0.1 (a + b) + a^2 + b^2 subject to a + b / 2 = 0.5 gives
// a = 0.41, b = 0.18.
TEST_F(ContrastiveTest, APertinentPositiveKeepsTheLeastThatStillDecides) {
    linear.set_weight({1, 0, 0.5f, 0, 0, 1, 0, 1});
    linear.set_bias({0, 0.5f});
    const ContrastiveResult r = FindPertinentPositive(ctx, Input({1, 1, 0.2f, 0}));
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.original_class, 0);
    EXPECT_EQ(r.explanation_class, 0);
    EXPECT_EQ(r.delta[2], 0.0f);
    EXPECT_EQ(r.delta[3], 0.0f);
    EXPECT_NEAR(r.delta[0], 0.41f, 0.01f);
    EXPECT_NEAR(r.delta[1], 0.18f, 0.01f);
    EXPECT_LT(r.l1, 2.2);  // less than the whole input
    for (size_t j = 0; j < 4; ++j) EXPECT_FLOAT_EQ(r.explanation[j], r.delta[j]);  // baseline 0

    const JsonValue doc = ParseJson(WriteJson(ToJson(r, "pertinent_positive")));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.contrastive.v1");
    EXPECT_EQ(doc.find("kind")->as_string(), "pertinent_positive");
    EXPECT_TRUE(doc.find("found")->as_bool());
    EXPECT_EQ(doc.find("features")->as_array().size(), 2u);
}

// With a baseline of 1, "absent" means equal to 1, and a pertinent positive is measured from it.
TEST_F(ContrastiveTest, TheBaselineSaysWhatAbsentMeans) {
    linear.set_weight({1, 0, 1, 0, 0, 1, 0, 0.5f});
    linear.set_bias({0, 0});
    ContrastiveOptions options;
    options.baseline = {1, 1, 1, 1};
    // (3, 1, 1, 1): z0 = 4, z1 = 1.5. Only x1, x2, x3 equal the baseline, so only they may grow.
    const ContrastiveResult r = FindPertinentNegative(ctx, Input({3, 1, 1, 1}), options);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.delta[0], 0.0f);
    EXPECT_EQ(r.delta[1], 0.0f);
    EXPECT_GT(r.delta[2], 0.0f);
}

TEST_F(ContrastiveTest, Validates) {
    linear.set_weight({1, 0, 1, 0, 0, 1, 0, 0.5f});
    ContrastiveOptions options;
    options.baseline = {0, 0};
    EXPECT_THROW((void)FindPertinentNegative(ctx, Input({1, 0, 0, 0}), options), std::invalid_argument);
    options = ContrastiveOptions();
    options.upper = {1};
    EXPECT_THROW((void)FindPertinentPositive(ctx, Input({1, 0, 0, 0}), options), std::invalid_argument);
    options = ContrastiveOptions();
    options.steps = 0;
    EXPECT_THROW((void)FindPertinentNegative(ctx, Input({1, 0, 0, 0}), options), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
