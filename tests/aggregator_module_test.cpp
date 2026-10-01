#include "pulsatrix/aggregator_module.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class AggregatorModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using AggregatorModuleDeathTest = AggregatorModuleTest;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_F(AggregatorModuleTest, ConstructorRejectsPEqualsZero) {
    EXPECT_THROW({ AggregatorModule agg(&backend, 0.0f); }, std::invalid_argument);
}

TEST_F(AggregatorModuleTest, DefaultPIsTwo) {
    AggregatorModule agg(&backend);
    EXPECT_FLOAT_EQ(agg.p(), 2.0f);
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

TEST_F(AggregatorModuleTest, PEqualsOneComputesArithmeticMean) {
    AggregatorModule agg(&backend, 1.0f);
    Tensor x(Shape({4}), &backend, {0.2f, 0.4f, 0.6f, 0.8f});
    Tensor y = agg.forward(x);

    ASSERT_EQ(y.rank(), 0);
    EXPECT_NEAR(y.data()[0], 0.5f, 1e-5f);
}

TEST_F(AggregatorModuleTest, PEqualsTwoComputesRootMeanSquare) {
    AggregatorModule agg(&backend, 2.0f);
    Tensor x(Shape({3}), &backend, {0.3f, 0.4f, 0.5f});
    Tensor y = agg.forward(x);

    const float expected = std::sqrt((0.3f * 0.3f + 0.4f * 0.4f + 0.5f * 0.5f) / 3.0f);
    EXPECT_NEAR(y.data()[0], expected, 1e-5f);
}

TEST_F(AggregatorModuleTest, ReducesLeadingAxisOnlyKeepingTrailingDims) {
    AggregatorModule agg(&backend, 1.0f);
    // shape (2, 3): 2 groundings, 3 independent "columns" (e.g. 3 independent formulas).
    Tensor x(Shape({2, 3}), &backend, {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f});
    Tensor y = agg.forward(x);

    ASSERT_EQ(y.shape(), Shape({3}));
    EXPECT_NEAR(y.data()[0], 0.5f, 1e-5f);   // mean(0.2, 0.8)
    EXPECT_NEAR(y.data()[1], 0.7f, 1e-5f);   // mean(0.4, 1.0)
    EXPECT_NEAR(y.data()[2], 0.9f, 1e-5f);   // mean(0.6, 1.2)
}

TEST_F(AggregatorModuleTest, SingletonBatchReturnsTheSingleValueUnchanged) {
    AggregatorModule agg(&backend, 2.0f);
    Tensor x(Shape({1}), &backend, {0.42f});
    Tensor y = agg.forward(x);

    EXPECT_NEAR(y.data()[0], 0.42f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Backward -- central finite differences (p=2), exact formula (p=1)
// ---------------------------------------------------------------------------

TEST_F(AggregatorModuleTest, BackwardMatchesCentralFiniteDifferencesForPEqualsTwo) {
    AggregatorModule agg(&backend, 2.0f);
    std::vector<float> x_values{0.3f, 0.6f, 0.9f, 0.2f};
    Tensor x(Shape({4}), &backend, x_values);
    (void)agg.forward(x);
    Tensor grad_out(Shape({}), &backend, {1.0f});
    Tensor grad_in = agg.backward(grad_out);

    auto agg_of = [&](const std::vector<float>& xv) {
        AggregatorModule probe(&backend, 2.0f);
        Tensor xp(Shape({4}), &backend, xv);
        Tensor y = probe.forward(xp);
        return y.data()[0];
    };

    const float h = 1e-3f;
    for (size_t i = 0; i < x_values.size(); ++i) {
        std::vector<float> xp = x_values;
        std::vector<float> xm = x_values;
        xp[i] += h;
        xm[i] -= h;
        const float numeric = (agg_of(xp) - agg_of(xm)) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 5e-3f) << "x[" << i << "]";
    }
}

TEST_F(AggregatorModuleTest, BackwardIsUniformOneOverNForPEqualsOne) {
    AggregatorModule agg(&backend, 1.0f);
    Tensor x(Shape({4}), &backend, {0.1f, 0.9f, 0.5f, 0.5f});  // arbitrary -- mean's gradient
                                                                // doesn't depend on the values
    (void)agg.forward(x);
    Tensor grad_out(Shape({}), &backend, {2.0f});
    Tensor grad_in = agg.backward(grad_out);

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(grad_in.data()[i], 2.0f * 0.25f, 1e-5f) << "x[" << i << "]";
    }
}

TEST_F(AggregatorModuleTest, BackwardBeforeForwardThrows) {
    AggregatorModule agg(&backend);
    Tensor grad_out(Shape({}), &backend, {1.0f});
    EXPECT_THROW({ (void)agg.backward(grad_out); }, std::logic_error);
}

TEST_F(AggregatorModuleTest, BackwardShapeMismatchThrows) {
    AggregatorModule agg(&backend);
    Tensor x(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    (void)agg.forward(x);
    Tensor grad_out(Shape({2}), &backend, {1.0f, 1.0f});  // wrong -- forward's output is rank 0
    EXPECT_THROW({ (void)agg.backward(grad_out); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP -- hand-derivable conservation
// ---------------------------------------------------------------------------

TEST_F(AggregatorModuleTest, PropagateRelevanceConservesExactlyForPEqualsOneWithZeroEpsilon) {
    AggregatorModule agg(&backend, 1.0f);
    Tensor x(Shape({4}), &backend, {0.2f, 0.4f, 0.6f, 0.8f});
    Tensor y = agg.forward(x);
    Tensor relevance_out(Shape({}), &backend, {y.data()[0]});
    LRPRuleConfig config;
    config.epsilon = 0.0f;
    Tensor relevance_in = agg.propagate_relevance(relevance_out, config);

    float sum = 0.0f;
    for (int64_t i = 0; i < 4; ++i) sum += relevance_in.data()[i];
    EXPECT_NEAR(sum, y.data()[0], 1e-5f);
}

TEST_F(AggregatorModuleTest, PropagateRelevanceConservesNearExactlyForPEqualsTwo) {
    AggregatorModule agg(&backend, 2.0f);
    Tensor x(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
    Tensor y = agg.forward(x);
    Tensor relevance_out(Shape({}), &backend, {y.data()[0]});
    Tensor relevance_in = agg.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum = 0.0f;
    for (int64_t i = 0; i < 4; ++i) sum += relevance_in.data()[i];
    EXPECT_NEAR(sum, y.data()[0], 1e-4f);
}

TEST_F(AggregatorModuleTest, PropagateRelevanceBeforeForwardThrows) {
    AggregatorModule agg(&backend);
    Tensor relevance_out(Shape({}), &backend, {1.0f});
    EXPECT_THROW({ (void)agg.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(AggregatorModuleTest, PropagateRelevanceShapeMismatchThrows) {
    AggregatorModule agg(&backend);
    Tensor x(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    (void)agg.forward(x);
    Tensor relevance_out(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)agg.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

// Rank-0 (bare-scalar, no batch axis) input is rejected -- there is no leading axis to
// reduce, unlike a rank-0 *output*, which is the valid degenerate result of reducing a
// rank-1 input.
TEST_F(AggregatorModuleTest, ForwardRejectsRankZeroInput) {
    AggregatorModule agg(&backend);
    Tensor scalar(Shape({}), &backend, {0.5f});
    EXPECT_THROW({ (void)agg.forward(scalar); }, std::invalid_argument);
}

// Empty batch -- inherited from Module::forward()'s own NVI precondition, not
// re-implemented here.
TEST_F(AggregatorModuleTest, ForwardRejectsEmptyBatch) {
    AggregatorModule agg(&backend);
    Tensor x(Shape({0}), &backend);
    EXPECT_THROW({ (void)agg.forward(x); }, std::invalid_argument);
}

// x outside [0,1] -- deliberately not rejected (see header's note): well-defined
// arithmetic for a positive-p, positive-x case even outside [0,1].
TEST_F(AggregatorModuleTest, ForwardHandlesOutOfRangeInputsWithoutError) {
    AggregatorModule agg(&backend, 2.0f);
    Tensor x(Shape({2}), &backend, {2.0f, 3.0f});
    Tensor y = agg.forward(x);
    const float expected = std::sqrt((4.0f + 9.0f) / 2.0f);
    EXPECT_NEAR(y.data()[0], expected, 1e-5f);
}

// Negative x with non-integer p -- deliberately not rejected: std::pow(negative,
// non-integer) is NaN by IEEE-754 definition, and this module propagates that honestly
// rather than silently clamping or throwing.
TEST_F(AggregatorModuleTest, ForwardHandlesNegativeInputWithNonIntegerPWithoutThrowing) {
    AggregatorModule agg(&backend, 2.5f);
    Tensor x(Shape({2}), &backend, {-0.5f, 0.5f});
    Tensor y = agg.forward(x);
    EXPECT_TRUE(std::isnan(y.data()[0]));
}

}  // namespace
}  // namespace pulsatrix
