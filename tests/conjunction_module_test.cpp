#include "pulsatrix/conjunction_module.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class ConjunctionModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using ConjunctionModuleDeathTest = ConjunctionModuleTest;

// ---------------------------------------------------------------------------
// Forward correctness -- product t-norm (default/primary)
// ---------------------------------------------------------------------------

TEST_F(ConjunctionModuleTest, ProductForwardComputesElementwiseProduct) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({3}), &backend, {0.2f, 0.5f, 1.0f});
    Tensor b(Shape({3}), &backend, {0.5f, 0.5f, 0.0f});
    Tensor y = conj.forward(a, b);

    EXPECT_FLOAT_EQ(y.data()[0], 0.1f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.25f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.0f);
}

TEST_F(ConjunctionModuleTest, ProductForwardMatchesShapeOfOperands) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({2, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});
    Tensor b(Shape({2, 2}), &backend, {0.9f, 0.8f, 0.7f, 0.6f});
    Tensor y = conj.forward(a, b);
    EXPECT_EQ(y.shape(), a.shape());
}

// ---------------------------------------------------------------------------
// Forward correctness -- Lukasiewicz / Godel variants
// ---------------------------------------------------------------------------

TEST_F(ConjunctionModuleTest, LukasiewiczForwardComputesClippedSum) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Lukasiewicz);
    Tensor a(Shape({3}), &backend, {0.8f, 0.3f, 0.9f});
    Tensor b(Shape({3}), &backend, {0.9f, 0.3f, 0.1f});
    Tensor y = conj.forward(a, b);

    EXPECT_NEAR(y.data()[0], 0.7f, 1e-6f);  // max(0, 0.8+0.9-1) = 0.7
    EXPECT_NEAR(y.data()[1], 0.0f, 1e-6f);  // max(0, 0.3+0.3-1) = max(0,-0.4) = 0
    EXPECT_NEAR(y.data()[2], 0.0f, 1e-6f);  // max(0, 0.9+0.1-1) = 0
}

TEST_F(ConjunctionModuleTest, GodelForwardComputesMin) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Godel);
    Tensor a(Shape({3}), &backend, {0.2f, 0.8f, 0.5f});
    Tensor b(Shape({3}), &backend, {0.6f, 0.3f, 0.5f});
    Tensor y = conj.forward(a, b);

    EXPECT_FLOAT_EQ(y.data()[0], 0.2f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.3f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.5f);  // tie -- min is well-defined regardless of tie-break
}

// ---------------------------------------------------------------------------
// Backward -- central finite differences (product), exact formulas (others)
// ---------------------------------------------------------------------------

TEST_F(ConjunctionModuleTest, ProductBackwardMatchesCentralFiniteDifferences) {
    ConjunctionModule conj(&backend);
    std::vector<float> a_values{0.3f, 0.7f, 0.9f};
    std::vector<float> b_values{0.6f, 0.2f, 0.4f};
    std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f};

    Tensor a(Shape({3}), &backend, a_values);
    Tensor b(Shape({3}), &backend, b_values);
    (void)conj.forward(a, b);
    Tensor grad_out(Shape({3}), &backend, grad_out_values);
    Tensor grad_stacked = conj.backward(grad_out);

    auto loss_of = [&](const std::vector<float>& av, const std::vector<float>& bv) {
        ConjunctionModule probe(&backend);
        Tensor ap(Shape({3}), &backend, av);
        Tensor bp(Shape({3}), &backend, bv);
        Tensor y = probe.forward(ap, bp);
        float loss = 0.0f;
        for (size_t k = 0; k < grad_out_values.size(); ++k) {
            loss += grad_out_values[k] * y.data()[static_cast<int64_t>(k)];
        }
        return loss;
    };

    const float h = 1e-3f;
    for (size_t i = 0; i < a_values.size(); ++i) {
        std::vector<float> ap = a_values;
        std::vector<float> am = a_values;
        ap[i] += h;
        am[i] -= h;
        const float numeric = (loss_of(ap, b_values) - loss_of(am, b_values)) / (2.0f * h);
        // grad_stacked is (2, 3): first 3 elements are grad_a.
        EXPECT_NEAR(grad_stacked.data()[static_cast<int64_t>(i)], numeric, 5e-3f) << "a[" << i << "]";
    }
    for (size_t i = 0; i < b_values.size(); ++i) {
        std::vector<float> bp = b_values;
        std::vector<float> bm = b_values;
        bp[i] += h;
        bm[i] -= h;
        const float numeric = (loss_of(a_values, bp) - loss_of(a_values, bm)) / (2.0f * h);
        EXPECT_NEAR(grad_stacked.data()[3 + static_cast<int64_t>(i)], numeric, 5e-3f) << "b[" << i << "]";
    }
}

TEST_F(ConjunctionModuleTest, GodelBackwardRoutesGradientToWinningOperand) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Godel);
    Tensor a(Shape({2}), &backend, {0.2f, 0.9f});  // a wins at index 0, b wins at index 1
    Tensor b(Shape({2}), &backend, {0.6f, 0.3f});
    (void)conj.forward(a, b);
    Tensor grad_out(Shape({2}), &backend, {10.0f, 10.0f});
    Tensor grad_stacked = conj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 10.0f);  // grad_a[0]: a wins
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);   // grad_a[1]: b wins
    EXPECT_FLOAT_EQ(grad_stacked.data()[2], 0.0f);   // grad_b[0]: a wins
    EXPECT_FLOAT_EQ(grad_stacked.data()[3], 10.0f);  // grad_b[1]: b wins
}

TEST_F(ConjunctionModuleTest, GodelBackwardTieRoutesGradientToFirstOperand) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Godel);
    Tensor a(Shape({1}), &backend, {0.5f});
    Tensor b(Shape({1}), &backend, {0.5f});
    (void)conj.forward(a, b);
    Tensor grad_out(Shape({1}), &backend, {7.0f});
    Tensor grad_stacked = conj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 7.0f);  // tie -> a
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);
}

TEST_F(ConjunctionModuleTest, LukasiewiczBackwardZeroInInactiveRegion) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Lukasiewicz);
    Tensor a(Shape({1}), &backend, {0.2f});
    Tensor b(Shape({1}), &backend, {0.3f});  // a+b-1 = -0.5 < 0 -- inactive
    (void)conj.forward(a, b);
    Tensor grad_out(Shape({1}), &backend, {5.0f});
    Tensor grad_stacked = conj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);
}

TEST_F(ConjunctionModuleTest, BackwardBeforeForwardThrows) {
    ConjunctionModule conj(&backend);
    Tensor grad_out(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)conj.backward(grad_out); }, std::logic_error);
}

TEST_F(ConjunctionModuleTest, BackwardShapeMismatchThrows) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    Tensor b(Shape({3}), &backend, {0.4f, 0.5f, 0.6f});
    (void)conj.forward(a, b);
    Tensor grad_out(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)conj.backward(grad_out); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP -- hand-derivable conservation
// ---------------------------------------------------------------------------

TEST_F(ConjunctionModuleTest, ProductPropagateRelevanceConservesNearExactly) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
    Tensor b(Shape({4}), &backend, {0.6f, 0.2f, 0.5f, 0.1f});
    Tensor y = conj.forward(a, b);
    Tensor relevance_in = conj.propagate_relevance(y, LRPRuleConfig{});

    // Bilinear split: r_a[i] + r_b[i] == 2 * (a*b)/(2*y+eps) * y == y * (2y)/(2y+eps) ~= y,
    // when relevance_out == y (canonical LRP setup: total relevance == the value itself).
    for (int64_t i = 0; i < 4; ++i) {
        const float sum = relevance_in.data()[i] + relevance_in.data()[4 + i];
        EXPECT_NEAR(sum, y.data()[i], 1e-4f) << "index " << i;
    }
}

TEST_F(ConjunctionModuleTest, GodelPropagateRelevanceConservesExactly) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Godel);
    Tensor a(Shape({3}), &backend, {0.2f, 0.9f, 0.4f});
    Tensor b(Shape({3}), &backend, {0.6f, 0.3f, 0.4f});
    Tensor y = conj.forward(a, b);
    Tensor relevance_out(Shape({3}), &backend, {1.0f, -0.5f, 2.0f});
    Tensor relevance_in = conj.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 3; ++i) {
        const float sum = relevance_in.data()[i] + relevance_in.data()[3 + i];
        EXPECT_FLOAT_EQ(sum, relevance_out.data()[i]);
    }
}

TEST_F(ConjunctionModuleTest, LukasiewiczPropagateRelevanceZeroInInactiveRegion) {
    ConjunctionModule conj(&backend, ConjunctionModule::TNorm::Lukasiewicz);
    Tensor a(Shape({1}), &backend, {0.2f});
    Tensor b(Shape({1}), &backend, {0.3f});
    Tensor y = conj.forward(a, b);
    Tensor relevance_out(Shape({1}), &backend, {3.0f});
    Tensor relevance_in = conj.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 0.0f);
}

TEST_F(ConjunctionModuleTest, PropagateRelevanceBeforeForwardThrows) {
    ConjunctionModule conj(&backend);
    Tensor relevance_out(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)conj.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

// Inputs outside [0,1] -- deliberately not rejected (see header's note): the formula is
// well-defined arithmetic for any real input, and enforcing the range would reject
// legitimate slight floating-point overshoot from upstream computations. Verified here to
// behave per the plain formula, not to throw/assert.
TEST_F(ConjunctionModuleTest, ProductForwardHandlesOutOfRangeInputsWithoutError) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({2}), &backend, {-0.5f, 1.5f});
    Tensor b(Shape({2}), &backend, {2.0f, -1.0f});
    Tensor y = conj.forward(a, b);
    EXPECT_FLOAT_EQ(y.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], -1.5f);
}

TEST_F(ConjunctionModuleTest, ForwardHandlesDegenerateZeroAndOneOperands) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({2}), &backend, {0.0f, 1.0f});
    Tensor b(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor y = conj.forward(a, b);
    EXPECT_FLOAT_EQ(y.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 1.0f);
}

// Malformed operand shapes -- external boundary, delegated to (and already implemented by)
// Tensor::Stack.
TEST_F(ConjunctionModuleTest, ForwardRejectsMismatchedOperandShapes) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    Tensor b(Shape({2}), &backend, {0.4f, 0.5f});
    EXPECT_THROW({ (void)conj.forward(a, b); }, std::invalid_argument);
}

// Empty-tensor construction -- both operands empty; the stacked tensor is also empty, so
// Module::forward()'s own NVI precondition rejects it (inherited, not re-implemented here).
TEST_F(ConjunctionModuleTest, ForwardRejectsEmptyOperands) {
    ConjunctionModule conj(&backend);
    Tensor a(Shape({0}), &backend);
    Tensor b(Shape({0}), &backend);
    EXPECT_THROW({ (void)conj.forward(a, b); }, std::invalid_argument);
}

// forward_impl reachable directly through the inherited Module::forward(const Tensor&),
// bypassing forward(a, b)/stack_operands() entirely -- a non-stacked (leading dim != 2)
// tensor must be rejected explicitly, not silently misinterpreted.
TEST_F(ConjunctionModuleTest, ForwardRejectsNonStackedInputWithWrongLeadingDimension) {
    ConjunctionModule conj(&backend);
    Tensor not_stacked(Shape({3, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    EXPECT_THROW({ (void)conj.forward(not_stacked); }, std::invalid_argument);
}

// Active region (a + b > 1): the bias-excluded epsilon rule must hand out the output relevance,
// not (a + b) / (a + b - 1) times it. Before the fix the denominator included the -1 bias:
// 2.25x at a = b = 0.9 and ~2000x at a + b - 1 = 5e-4. The 0.5005 + 0.5 case sits just above the
// threshold, where that amplification was largest.
TEST_F(ConjunctionModuleTest, LukasiewiczPropagateRelevanceConservesInActiveRegion) {
    ConjunctionModule m(&backend, ConjunctionModule::TNorm::Lukasiewicz);
    Tensor a(Shape({4}), &backend, {0.9f, 0.5005f, 0.7f, 0.95f});
    Tensor b(Shape({4}), &backend, {0.9f, 0.5f, 0.6f, 0.3f});
    (void)m.forward(a, b);
    Tensor relevance_out(Shape({4}), &backend, {1.0f, -0.5f, 2.0f, 0.25f});
    Tensor relevance_in = m.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 4; ++i) {
        const float sum = relevance_in.data()[i] + relevance_in.data()[4 + i];
        EXPECT_NEAR(sum, relevance_out.data()[i], 1e-4f) << "index " << i;
    }
}

}  // namespace
}  // namespace pulsatrix
