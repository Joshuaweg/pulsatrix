#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/softmax_module.hpp"

namespace exai {
namespace {

class SoftmaxModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    SoftmaxModule softmax{&backend};
};

// Hand-computed: softmax([1, 2]) = [1/(1+e), e/(1+e)] = [sigmoid(-1), sigmoid(1)]
// = [0.26894142137, 0.73105857863] to 11 decimal places.
TEST_F(SoftmaxModuleTest, ForwardMatchesHandComputedTwoElementRow) {
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor y = softmax.forward(x);

    EXPECT_NEAR(y.data()[0], 0.26894142137f, 1e-6f);
    EXPECT_NEAR(y.data()[1], 0.73105857863f, 1e-6f);
}

// Hand-computed: softmax([0, 1, 2]); with e = 2.718281828, e^2 = 7.389056099,
// sum = 1 + 2.718281828 + 7.389056099 = 11.107337927,
// -> [0.09003057317, 0.24472847105, 0.66524095578].
TEST_F(SoftmaxModuleTest, ForwardMatchesHandComputedThreeElementRow) {
    Tensor x(Shape({3}), &backend, {0.0f, 1.0f, 2.0f});
    Tensor y = softmax.forward(x);

    EXPECT_NEAR(y.data()[0], 0.09003057317f, 1e-6f);
    EXPECT_NEAR(y.data()[1], 0.24472847105f, 1e-6f);
    EXPECT_NEAR(y.data()[2], 0.66524095578f, 1e-6f);
}

// Softmax is shift-invariant: adding a constant to a whole row leaves the output unchanged.
// This also exercises the numerically-stable (subtract-row-max) path with large magnitudes,
// where a naive exp() would overflow to inf and produce NaN.
TEST_F(SoftmaxModuleTest, ForwardIsNumericallyStableForLargeLogits) {
    Tensor x(Shape({3}), &backend, {1000.0f, 1001.0f, 1002.0f});
    Tensor y = softmax.forward(x);

    EXPECT_TRUE(std::isfinite(y.data()[0]));
    EXPECT_NEAR(y.data()[0], 0.09003057317f, 1e-6f);
    EXPECT_NEAR(y.data()[1], 0.24472847105f, 1e-6f);
    EXPECT_NEAR(y.data()[2], 0.66524095578f, 1e-6f);
}

// The last-axis, per-row semantics (not a single global reduction): a rank-3 (2, 3, 4)
// tensor has 2*3 = 6 independent softmax rows of 4 elements, each of which sums to 1.
// A global softmax would instead make the grand total 1.
TEST_F(SoftmaxModuleTest, ForwardRowsEachSumToOneOnRank3Tensor) {
    std::vector<float> values(24);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.3f * static_cast<float>(i) - 2.0f;  // spread across positive and negative
    }
    Tensor x(Shape({2, 3, 4}), &backend, values);
    Tensor y = softmax.forward(x);

    for (int64_t row = 0; row < 6; ++row) {
        float row_sum = 0.0f;
        for (int64_t j = 0; j < 4; ++j) {
            row_sum += y.data()[row * 4 + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "row " << row;
    }
    // Grand total is 6, not 1 -- proof this is per-row, not a global softmax.
    float total = 0.0f;
    for (int64_t i = 0; i < y.numel(); ++i) {
        total += y.data()[i];
    }
    EXPECT_NEAR(total, 6.0f, 1e-4f);
}

// Rows are independent: perturbing one row must not change any other row's output.
TEST_F(SoftmaxModuleTest, ForwardRowsAreIndependent) {
    Tensor a(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 0.5f, -0.5f, 0.0f});
    Tensor ya = softmax.forward(a);
    std::vector<float> row0{ya.data()[0], ya.data()[1], ya.data()[2]};

    Tensor b(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 10.0f, -7.0f, 4.0f});
    Tensor yb = softmax.forward(b);

    EXPECT_FLOAT_EQ(yb.data()[0], row0[0]);
    EXPECT_FLOAT_EQ(yb.data()[1], row0[1]);
    EXPECT_FLOAT_EQ(yb.data()[2], row0[2]);
}

// Central finite differences against the analytic Jacobian-vector product, on a multi-row
// (2, 3) input. f(x) = sum_k(grad_out[k] * softmax(x)[k]); d f / d x[i] must equal
// backward(grad_out)[i].
TEST_F(SoftmaxModuleTest, BackwardMatchesCentralFiniteDifferencesOnMultiRowInput) {
    const std::vector<float> base{0.5f, -1.2f, 2.0f, 0.1f, 0.7f, -0.3f};
    const std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f};

    Tensor x(Shape({2, 3}), &backend, base);
    (void)softmax.forward(x);
    Tensor grad_out(Shape({2, 3}), &backend, grad_out_values);
    Tensor grad_in = softmax.backward(grad_out);

    const float h = 1e-3f;
    for (size_t i = 0; i < base.size(); ++i) {
        std::vector<float> plus = base;
        std::vector<float> minus = base;
        plus[i] += h;
        minus[i] -= h;

        SoftmaxModule probe(&backend);
        Tensor xp(Shape({2, 3}), &backend, plus);
        Tensor yp = probe.forward(xp);
        Tensor xm(Shape({2, 3}), &backend, minus);
        Tensor ym = probe.forward(xm);

        float fp = 0.0f;
        float fm = 0.0f;
        for (size_t k = 0; k < base.size(); ++k) {
            fp += grad_out_values[k] * yp.data()[static_cast<int64_t>(k)];
            fm += grad_out_values[k] * ym.data()[static_cast<int64_t>(k)];
        }
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 1e-3f) << "element " << i;
    }
}

// Gradient of sum(softmax(x)) is identically zero (each row's output sums to a constant 1),
// which is exactly what the Jacobian formula's subtracted term enforces.
TEST_F(SoftmaxModuleTest, BackwardOfUniformGradientIsZero) {
    Tensor x(Shape({2, 3}), &backend, {0.5f, -1.2f, 2.0f, 0.1f, 0.7f, -0.3f});
    (void)softmax.forward(x);
    Tensor grad_out(Shape({2, 3}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    Tensor grad_in = softmax.backward(grad_out);

    for (int64_t i = 0; i < grad_in.numel(); ++i) {
        EXPECT_NEAR(grad_in.data()[i], 0.0f, 1e-6f) << "element " << i;
    }
}

// AttnLRP Eq. 13, worked out by hand for a 2-element row and written here as literal
// expected numbers (an independent transcription of the formula, not a re-run of the
// implementation):
//   x = [1, 2]  ->  s = [0.26894142137, 0.73105857863]
//   R_out = [0.6, 0.4],  sum_j R_out[j] = 1.0
//   R_in[0] = x[0] * (R_out[0] - s[0]*1.0) = 1 * (0.6 - 0.26894142137) =  0.33105857863
//   R_in[1] = x[1] * (R_out[1] - s[1]*1.0) = 2 * (0.4 - 0.73105857863) = -0.66211715726
TEST_F(SoftmaxModuleTest, PropagateRelevanceMatchesHandWorkedEq13TwoElementRow) {
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)softmax.forward(x);

    Tensor relevance_out(Shape({2}), &backend, {0.6f, 0.4f});
    Tensor relevance_in = softmax.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_NEAR(relevance_in.data()[0], 0.33105857863f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[1], -0.66211715726f, 1e-6f);
}

// Same rule, hand-worked for a 3-element row:
//   x = [0, 1, 2]  ->  s = [0.09003057317, 0.24472847105, 0.66524095578]
//   R_out = [1.0, 0.0, 0.0],  sum_j R_out[j] = 1.0
//   R_in[0] = 0 * (1.0 - 0.09003057317) =  0.0            (the x[i] factor zeroes this out)
//   R_in[1] = 1 * (0.0 - 0.24472847105) = -0.24472847105
//   R_in[2] = 2 * (0.0 - 0.66524095578) = -1.33048191156
TEST_F(SoftmaxModuleTest, PropagateRelevanceMatchesHandWorkedEq13ThreeElementRow) {
    Tensor x(Shape({3}), &backend, {0.0f, 1.0f, 2.0f});
    (void)softmax.forward(x);

    Tensor relevance_out(Shape({3}), &backend, {1.0f, 0.0f, 0.0f});
    Tensor relevance_in = softmax.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_NEAR(relevance_in.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[1], -0.24472847105f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[2], -1.33048191156f, 1e-6f);
}

// Eq. 13 is applied per row, with each row's own sum_j(R_out[j]) -- not one global sum.
// Row 0 is the hand-worked [1, 2] case above (R_out sums to 1.0); row 1 uses a different
// R_out total, so a global-sum implementation would get row 0 wrong.
TEST_F(SoftmaxModuleTest, PropagateRelevanceUsesPerRowRelevanceSums) {
    Tensor x(Shape({2, 2}), &backend, {1.0f, 2.0f, 1.0f, 2.0f});
    (void)softmax.forward(x);

    Tensor relevance_out(Shape({2, 2}), &backend, {0.6f, 0.4f, 2.0f, 3.0f});
    Tensor relevance_in = softmax.propagate_relevance(relevance_out, LRPRuleConfig{});

    // Row 0: identical to the hand-worked two-element case.
    EXPECT_NEAR(relevance_in.data()[0], 0.33105857863f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[1], -0.66211715726f, 1e-6f);
    // Row 1: sum_j R_out[j] = 5.0
    //   R_in[0] = 1 * (2.0 - 0.26894142137*5) = 2.0 - 1.34470710685 =  0.65529289315
    //   R_in[1] = 2 * (3.0 - 0.73105857863*5) = 2 * (3.0 - 3.65529289315) = -1.3105857863
    EXPECT_NEAR(relevance_in.data()[2], 0.65529289315f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[3], -1.3105857863f, 1e-6f);
}

// AttnLRP Eq. 13 is a first-order Deep-Taylor-Decomposition approximation around a nonzero
// reference point and carries a residual "hidden bias term": relevance is NOT conserved.
// This test therefore measures and reports the gap rather than asserting it away. Do not
// "fix" a nonzero result here by adding a stabilizer or rescaling -- see
// mission_softmax_module.md's Requirements section.
TEST_F(SoftmaxModuleTest, PropagateRelevanceConservationGapIsMeasuredNotAssumedZero) {
    Tensor x(Shape({2, 3}), &backend, {0.5f, -1.2f, 2.0f, 0.1f, 0.7f, -0.3f});
    (void)softmax.forward(x);

    Tensor relevance_out(Shape({2, 3}), &backend, {1.0f, 0.5f, -0.25f, 2.0f, 0.75f, 0.0f});
    Tensor relevance_in = softmax.propagate_relevance(relevance_out, LRPRuleConfig{});

    double sum_out = 0.0;
    double sum_in = 0.0;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) {
        sum_out += relevance_out.data()[i];
        sum_in += relevance_in.data()[i];
    }
    const double gap = sum_out - sum_in;

    std::cout << "[ SOFTMAX  ] AttnLRP Eq. 13 conservation gap on shape (2,3): "
              << "sum(R_out)=" << sum_out << ", sum(R_in)=" << sum_in << ", gap=" << gap << std::endl;
    RecordProperty("lrp_conservation_gap", std::to_string(gap));

    // Everything must stay finite -- the gap is an approximation error, not a blow-up.
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(relevance_in.data()[i]));
    }
    // Positive assertion that this rule genuinely does NOT conserve (guards against someone
    // later "fixing" it into a conserving rule without revisiting the mission's rationale).
    EXPECT_GT(std::abs(gap), 1e-3) << "Eq. 13 is not expected to conserve; a ~0 gap here means "
                                      "the rule was changed away from the cited formula";
    // Characterization of the measured value at the time of writing (see mission close-out).
    // Measured: sum(R_out) = 4.0, sum(R_in) = -2.83971, gap = 6.83971 (double-precision
    // reference computation of the same formula agrees to 5 decimals).
    EXPECT_NEAR(gap, 6.83971, 1e-4);
}

// Degenerate but legal: a single-element row. softmax([c]) == [1] for any c, and Eq. 13
// collapses to R_in = x * (R_out - 1 * R_out) = 0.
TEST_F(SoftmaxModuleTest, SingleElementRowIsUnityForwardAndZeroRelevance) {
    Tensor x(Shape({3, 1}), &backend, {5.0f, -5.0f, 0.0f});
    Tensor y = softmax.forward(x);
    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[2], 1.0f);

    Tensor relevance_out(Shape({3, 1}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor relevance_in = softmax.propagate_relevance(relevance_out, LRPRuleConfig{});
    EXPECT_NEAR(relevance_in.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[1], 0.0f, 1e-6f);
    EXPECT_NEAR(relevance_in.data()[2], 0.0f, 1e-6f);
}

TEST_F(SoftmaxModuleTest, HasNoLearnableParameters) {
    EXPECT_TRUE(softmax.parameters().empty());
}

TEST_F(SoftmaxModuleTest, OpTypeIsActivation) {
    EXPECT_EQ(softmax.op_type(), OpType::Activation);
}

TEST_F(SoftmaxModuleTest, ForwardRejectsEmptyInput) {
    Tensor empty(Shape({0}), &backend);
    EXPECT_THROW({ (void)softmax.forward(empty); }, std::invalid_argument);
}

using SoftmaxModuleDeathTest = SoftmaxModuleTest;

// forward_impl/backward/propagate_relevance all dereference Tensor::data() in raw host
// loops -- undefined behavior on a CUDA-backed Tensor. See RNNModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses (no real GPU needed: DeviceType::Cuda over
// real CPUBackend memory trips the guard identically). Written from the start of this
// mission, per this campaign's standing adversarial discipline.
TEST_F(SoftmaxModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)softmax.forward(x); }, "EXAI_ASSERT failed");
}

TEST_F(SoftmaxModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)softmax.forward(x);

    Tensor grad_output(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)softmax.backward(grad_output); }, "EXAI_ASSERT failed");
}

TEST_F(SoftmaxModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)softmax.forward(x);

    Tensor relevance_out(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)softmax.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
