#include "pulsatrix/disjunction_module.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class DisjunctionModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using DisjunctionModuleDeathTest = DisjunctionModuleTest;

// ---------------------------------------------------------------------------
// Forward correctness -- product t-conorm (default/primary)
// ---------------------------------------------------------------------------

TEST_F(DisjunctionModuleTest, ProductForwardComputesProbabilisticSum) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({3}), &backend, {0.2f, 0.5f, 1.0f});
    Tensor b(Shape({3}), &backend, {0.5f, 0.5f, 0.0f});
    Tensor y = disj.forward(a, b);

    EXPECT_NEAR(y.data()[0], 0.6f, 1e-6f);   // 0.2+0.5-0.1
    EXPECT_NEAR(y.data()[1], 0.75f, 1e-6f);  // 0.5+0.5-0.25
    EXPECT_NEAR(y.data()[2], 1.0f, 1e-6f);   // 1.0+0.0-0.0
}

TEST_F(DisjunctionModuleTest, ProductForwardMatchesShapeOfOperands) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({2, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});
    Tensor b(Shape({2, 2}), &backend, {0.9f, 0.8f, 0.7f, 0.6f});
    Tensor y = disj.forward(a, b);
    EXPECT_EQ(y.shape(), a.shape());
}

// ---------------------------------------------------------------------------
// Forward correctness -- Lukasiewicz / Godel variants
// ---------------------------------------------------------------------------

TEST_F(DisjunctionModuleTest, LukasiewiczForwardComputesClippedSum) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Lukasiewicz);
    Tensor a(Shape({3}), &backend, {0.8f, 0.3f, 0.9f});
    Tensor b(Shape({3}), &backend, {0.9f, 0.3f, 0.1f});
    Tensor y = disj.forward(a, b);

    EXPECT_NEAR(y.data()[0], 1.0f, 1e-6f);  // min(1, 1.7) = 1
    EXPECT_NEAR(y.data()[1], 0.6f, 1e-6f);  // min(1, 0.6) = 0.6
    EXPECT_NEAR(y.data()[2], 1.0f, 1e-6f);  // min(1, 1.0) = 1
}

TEST_F(DisjunctionModuleTest, GodelForwardComputesMax) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Godel);
    Tensor a(Shape({3}), &backend, {0.2f, 0.8f, 0.5f});
    Tensor b(Shape({3}), &backend, {0.6f, 0.3f, 0.5f});
    Tensor y = disj.forward(a, b);

    EXPECT_FLOAT_EQ(y.data()[0], 0.6f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.8f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.5f);
}

// ---------------------------------------------------------------------------
// Backward -- central finite differences (product), exact formulas (others)
// ---------------------------------------------------------------------------

TEST_F(DisjunctionModuleTest, ProductBackwardMatchesCentralFiniteDifferences) {
    DisjunctionModule disj(&backend);
    std::vector<float> a_values{0.3f, 0.7f, 0.9f};
    std::vector<float> b_values{0.6f, 0.2f, 0.4f};
    std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f};

    Tensor a(Shape({3}), &backend, a_values);
    Tensor b(Shape({3}), &backend, b_values);
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({3}), &backend, grad_out_values);
    Tensor grad_stacked = disj.backward(grad_out);

    auto loss_of = [&](const std::vector<float>& av, const std::vector<float>& bv) {
        DisjunctionModule probe(&backend);
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

TEST_F(DisjunctionModuleTest, GodelBackwardRoutesGradientToWinningOperand) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Godel);
    Tensor a(Shape({2}), &backend, {0.6f, 0.3f});  // a wins at index 0, b wins at index 1
    Tensor b(Shape({2}), &backend, {0.2f, 0.9f});
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({2}), &backend, {10.0f, 10.0f});
    Tensor grad_stacked = disj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 10.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[3], 10.0f);
}

TEST_F(DisjunctionModuleTest, GodelBackwardTieRoutesGradientToFirstOperand) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Godel);
    Tensor a(Shape({1}), &backend, {0.5f});
    Tensor b(Shape({1}), &backend, {0.5f});
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({1}), &backend, {7.0f});
    Tensor grad_stacked = disj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 7.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);
}

TEST_F(DisjunctionModuleTest, LukasiewiczBackwardZeroInSaturatedRegion) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Lukasiewicz);
    Tensor a(Shape({1}), &backend, {0.8f});
    Tensor b(Shape({1}), &backend, {0.7f});  // a+b = 1.5 >= 1 -- saturated
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({1}), &backend, {5.0f});
    Tensor grad_stacked = disj.backward(grad_out);

    EXPECT_FLOAT_EQ(grad_stacked.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(grad_stacked.data()[1], 0.0f);
}

TEST_F(DisjunctionModuleTest, BackwardBeforeForwardThrows) {
    DisjunctionModule disj(&backend);
    Tensor grad_out(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)disj.backward(grad_out); }, std::logic_error);
}

TEST_F(DisjunctionModuleTest, BackwardShapeMismatchThrows) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    Tensor b(Shape({3}), &backend, {0.4f, 0.5f, 0.6f});
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)disj.backward(grad_out); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP -- hand-derivable conservation
// ---------------------------------------------------------------------------

// The averaged dual-decomposition rule conserves exactly (z_a + z_b == y for every a, b, by
// construction -- see the header's derivation), before the epsilon stabilizer's tiny
// near-exactness gap. This test checks the algebraic identity directly, not just numerically.
TEST_F(DisjunctionModuleTest, ProductPropagateRelevanceConservesNearExactly) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
    Tensor b(Shape({4}), &backend, {0.6f, 0.2f, 0.5f, 0.1f});
    Tensor y = disj.forward(a, b);
    Tensor relevance_in = disj.propagate_relevance(y, LRPRuleConfig{});

    for (int64_t i = 0; i < 4; ++i) {
        const float sum = relevance_in.data()[i] + relevance_in.data()[4 + i];
        EXPECT_NEAR(sum, y.data()[i], 1e-4f) << "index " << i;
    }
}

TEST_F(DisjunctionModuleTest, GodelPropagateRelevanceConservesExactly) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Godel);
    Tensor a(Shape({3}), &backend, {0.6f, 0.3f, 0.4f});
    Tensor b(Shape({3}), &backend, {0.2f, 0.9f, 0.4f});
    Tensor y = disj.forward(a, b);
    Tensor relevance_out(Shape({3}), &backend, {1.0f, -0.5f, 2.0f});
    Tensor relevance_in = disj.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 3; ++i) {
        const float sum = relevance_in.data()[i] + relevance_in.data()[3 + i];
        EXPECT_FLOAT_EQ(sum, relevance_out.data()[i]);
    }
}

TEST_F(DisjunctionModuleTest, LukasiewiczPropagateRelevanceZeroInSaturatedRegion) {
    DisjunctionModule disj(&backend, DisjunctionModule::TConorm::Lukasiewicz);
    Tensor a(Shape({1}), &backend, {0.8f});
    Tensor b(Shape({1}), &backend, {0.7f});
    Tensor y = disj.forward(a, b);
    Tensor relevance_out(Shape({1}), &backend, {3.0f});
    Tensor relevance_in = disj.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 0.0f);
}

TEST_F(DisjunctionModuleTest, PropagateRelevanceBeforeForwardThrows) {
    DisjunctionModule disj(&backend);
    Tensor relevance_out(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)disj.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

TEST_F(DisjunctionModuleTest, ProductForwardHandlesOutOfRangeInputsWithoutError) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({2}), &backend, {-0.5f, 1.5f});
    Tensor b(Shape({2}), &backend, {2.0f, -1.0f});
    Tensor y = disj.forward(a, b);
    // a+b-ab: (-0.5+2.0-(-1.0)) = 2.5 ; (1.5-1.0-(-1.5)) = 2.0
    EXPECT_FLOAT_EQ(y.data()[0], 2.5f);
    EXPECT_FLOAT_EQ(y.data()[1], 2.0f);
}

TEST_F(DisjunctionModuleTest, ForwardHandlesDegenerateZeroAndOneOperands) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({2}), &backend, {0.0f, 1.0f});
    Tensor b(Shape({2}), &backend, {0.0f, 0.0f});
    Tensor y = disj.forward(a, b);
    EXPECT_FLOAT_EQ(y.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 1.0f);
}

TEST_F(DisjunctionModuleTest, ForwardRejectsMismatchedOperandShapes) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    Tensor b(Shape({2}), &backend, {0.4f, 0.5f});
    EXPECT_THROW({ (void)disj.forward(a, b); }, std::invalid_argument);
}

TEST_F(DisjunctionModuleTest, ForwardRejectsEmptyOperands) {
    DisjunctionModule disj(&backend);
    Tensor a(Shape({0}), &backend);
    Tensor b(Shape({0}), &backend);
    EXPECT_THROW({ (void)disj.forward(a, b); }, std::invalid_argument);
}

TEST_F(DisjunctionModuleTest, ForwardRejectsNonStackedInputWithWrongLeadingDimension) {
    DisjunctionModule disj(&backend);
    Tensor not_stacked(Shape({3, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    EXPECT_THROW({ (void)disj.forward(not_stacked); }, std::invalid_argument);
}

TEST_F(DisjunctionModuleDeathTest, ForwardAbortsOnNonCpuStackedInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    DisjunctionModule disj(&backend);
    Tensor a(Shape({2}), &backend, {0.1f, 0.2f}, DeviceType::Cuda);
    Tensor b(Shape({2}), &backend, {0.3f, 0.4f}, DeviceType::Cuda);
    Tensor stacked = DisjunctionModule::stack_operands(a, b, &backend);
    EXPECT_DEATH({ (void)disj.forward(stacked); }, "PULSATRIX_ASSERT failed");
}

TEST_F(DisjunctionModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    DisjunctionModule disj(&backend);
    Tensor a(Shape({2}), &backend, {0.1f, 0.2f});
    Tensor b(Shape({2}), &backend, {0.3f, 0.4f});
    (void)disj.forward(a, b);
    Tensor grad_out(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)disj.backward(grad_out); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
