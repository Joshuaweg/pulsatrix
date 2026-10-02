#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

// GPU-native-kernels Mission 1 primitives, CPUBackend reference semantics against
// hand-computed values. The GPU backends are held to these results by
// backend_primitive_equivalence_cases.hpp.
namespace pulsatrix {
namespace {

class BackendPrimitivesTest : public ::testing::Test {
protected:
    CPUBackend cpu;
};

// A = [[1,2,3],[4,5,6]] (2x3), B = [[1,0],[0,1],[2,-1]] (3x2). A*B = [[7,-1],[16,-1]].
TEST_F(BackendPrimitivesTest, GemmExPlainMatchesHandComputedProduct) {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {1, 0, 0, 1, 2, -1};
    std::vector<float> out(4, -99.0f);  // beta == 0 must ignore this garbage
    cpu.gemm_ex(a.data(), false, b.data(), false, out.data(), 2, 3, 2, 0.0f);
    EXPECT_EQ(out, (std::vector<float>{7, -1, 16, -1}));
}

TEST_F(BackendPrimitivesTest, GemmExTransposeAReadsStoredTranspose) {
    // A stored as its transpose (3x2): [[1,4],[2,5],[3,6]]; op(A) = original A.
    std::vector<float> a_t = {1, 4, 2, 5, 3, 6};
    std::vector<float> b = {1, 0, 0, 1, 2, -1};
    std::vector<float> out(4, 0.0f);
    cpu.gemm_ex(a_t.data(), true, b.data(), false, out.data(), 2, 3, 2, 0.0f);
    EXPECT_EQ(out, (std::vector<float>{7, -1, 16, -1}));
}

TEST_F(BackendPrimitivesTest, GemmExTransposeBReadsStoredTranspose) {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_t = {1, 0, 2, 0, 1, -1};  // B^T (2x3)
    std::vector<float> out(4, 0.0f);
    cpu.gemm_ex(a.data(), false, b_t.data(), true, out.data(), 2, 3, 2, 0.0f);
    EXPECT_EQ(out, (std::vector<float>{7, -1, 16, -1}));
}

TEST_F(BackendPrimitivesTest, GemmExBetaOneAccumulates) {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {1, 0, 0, 1, 2, -1};
    std::vector<float> out = {1, 1, 1, 1};
    cpu.gemm_ex(a.data(), false, b.data(), false, out.data(), 2, 3, 2, 1.0f);
    EXPECT_EQ(out, (std::vector<float>{8, 0, 17, 0}));
}

TEST_F(BackendPrimitivesTest, ColumnSumsWithAndWithoutAccumulate) {
    std::vector<float> in = {1, 2, 3, 4, 5, 6};  // (2x3)
    std::vector<float> out(3, -99.0f);
    cpu.column_sums(in.data(), out.data(), 2, 3, 0.0f);
    EXPECT_EQ(out, (std::vector<float>{5, 7, 9}));
    cpu.column_sums(in.data(), out.data(), 2, 3, 1.0f);
    EXPECT_EQ(out, (std::vector<float>{10, 14, 18}));
}

TEST_F(BackendPrimitivesTest, AddRowVectorBroadcastsAcrossRowsInPlace) {
    std::vector<float> m = {1, 2, 3, 4, 5, 6};
    std::vector<float> row = {10, 20, 30};
    cpu.add_row_vector(m.data(), row.data(), m.data(), 2, 3);
    EXPECT_EQ(m, (std::vector<float>{11, 22, 33, 14, 25, 36}));
}

TEST_F(BackendPrimitivesTest, ElementwiseBackwardReluIsZeroAtZero) {
    std::vector<float> x = {-1.0f, 0.0f, 2.0f};
    std::vector<float> g = {5.0f, 5.0f, 5.0f};
    std::vector<float> out(3);
    cpu.elementwise_backward(ElementwiseOp::Relu, x.data(), g.data(), out.data(), 3);
    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.0f, 5.0f}));  // ReluModule's x > 0 convention
}

TEST_F(BackendPrimitivesTest, ElementwiseBackwardReluMasksNonFiniteGradients) {
    std::vector<float> x = {-1.0f, 0.0f};
    std::vector<float> g = {std::nanf(""), INFINITY};
    std::vector<float> out(2, 1.0f);
    cpu.elementwise_backward(ElementwiseOp::Relu, x.data(), g.data(), out.data(), 2);
    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.0f}));
}

TEST_F(BackendPrimitivesTest, ElementwiseBackwardSmoothActivationsMatchClosedForms) {
    const float x = 0.5f;
    const float g = 2.0f;
    const float s = 1.0f / (1.0f + std::exp(-x));
    const float t = std::tanh(x);
    float out = 0.0f;
    cpu.elementwise_backward(ElementwiseOp::Neg, &x, &g, &out, 1);
    EXPECT_FLOAT_EQ(out, -2.0f);
    cpu.elementwise_backward(ElementwiseOp::Tanh, &x, &g, &out, 1);
    EXPECT_FLOAT_EQ(out, g * (1.0f - t * t));
    cpu.elementwise_backward(ElementwiseOp::Sigmoid, &x, &g, &out, 1);
    EXPECT_FLOAT_EQ(out, g * s * (1.0f - s));
    cpu.elementwise_backward(ElementwiseOp::Silu, &x, &g, &out, 1);
    EXPECT_FLOAT_EQ(out, g * (s + x * s * (1.0f - s)));
}

TEST_F(BackendPrimitivesTest, AxpbyAliasingY) {
    std::vector<float> p = {1, 2, 3};
    std::vector<float> grad = {10, 20, 30};
    cpu.axpby(-0.1f, grad.data(), 1.0f, p.data(), p.data(), 3);  // SGD: p -= 0.1 * g
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
}

TEST_F(BackendPrimitivesTest, DotSumsProducts) {
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, -5, 6};
    EXPECT_FLOAT_EQ(cpu.dot(a.data(), b.data(), 3), 12.0f);
    EXPECT_FLOAT_EQ(cpu.dot(nullptr, nullptr, 0), 0.0f);
}

TEST_F(BackendPrimitivesTest, SoftmaxRowsIsStableForLargeLogits) {
    std::vector<float> in = {1000.0f, 1000.0f, 0.0f, std::log(3.0f)};  // 2 rows x 2
    std::vector<float> out(4);
    cpu.softmax_rows(in.data(), out.data(), 2, 2);
    EXPECT_FLOAT_EQ(out[0], 0.5f);
    EXPECT_FLOAT_EQ(out[1], 0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.25f);
    EXPECT_FLOAT_EQ(out[3], 0.75f);
}

TEST_F(BackendPrimitivesTest, SoftmaxRowsBackwardMatchesJacobianProduct) {
    std::vector<float> y = {0.25f, 0.75f};
    std::vector<float> dy = {1.0f, 0.0f};
    std::vector<float> dx(2);
    cpu.softmax_rows_backward(y.data(), dy.data(), dx.data(), 1, 2);
    // dx = y * (dy - y.dy) = [0.25*(1-0.25), 0.75*(0-0.25)]
    EXPECT_FLOAT_EQ(dx[0], 0.1875f);
    EXPECT_FLOAT_EQ(dx[1], -0.1875f);
}

TEST_F(BackendPrimitivesTest, LogsumexpRowsIsStableForLargeLogits) {
    std::vector<float> in = {1000.0f, 1000.0f};
    float out = 0.0f;
    cpu.logsumexp_rows(in.data(), &out, 1, 2);
    EXPECT_FLOAT_EQ(out, 1000.0f + std::log(2.0f));
}

TEST_F(BackendPrimitivesTest, AdamStepFirstIterationMovesByLearningRate) {
    // At t=1 with bias correction, m_hat = g and v_hat = g^2, so the step is lr * g/(|g|+eps).
    std::vector<float> p = {1.0f, 1.0f};
    std::vector<float> g = {0.5f, -2.0f};
    std::vector<float> m(2, 0.0f);
    std::vector<float> v(2, 0.0f);
    cpu.adam_step(p.data(), g.data(), m.data(), v.data(), 2, 0.1f, 0.9f, 0.999f, 1e-8f, 1.0f - 0.9f, 1.0f - 0.999f);
    EXPECT_NEAR(p[0], 0.9f, 1e-6f);
    EXPECT_NEAR(p[1], 1.1f, 1e-6f);
    EXPECT_FLOAT_EQ(m[0], 0.05f);
    EXPECT_NEAR(v[1], 0.004f, 1e-7f);
}

// ---- Mission 1b ----------------------------------------------------------------------------

TEST_F(BackendPrimitivesTest, AxpbyWithZeroBetaDoesNotReadY) {
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> y = {INFINITY, std::nanf("")};
    std::vector<float> out(2);
    cpu.axpby(3.0f, x.data(), 0.0f, y.data(), out.data(), 2);
    EXPECT_EQ(out, (std::vector<float>{3.0f, 6.0f}));  // not NaN from 0 * inf / 0 * NaN
    cpu.axpby(3.0f, x.data(), 0.0f, nullptr, out.data(), 2);  // y may even be null
    EXPECT_EQ(out, (std::vector<float>{3.0f, 6.0f}));
}

TEST_F(BackendPrimitivesTest, ExpForwardAndBackward) {
    const float x = 0.75f;
    const float g = 2.0f;
    float out = 0.0f;
    cpu.elementwise(ElementwiseOp::Exp, &x, &out, 1);
    EXPECT_FLOAT_EQ(out, std::exp(0.75f));
    cpu.elementwise_backward(ElementwiseOp::Exp, &x, &g, &out, 1);
    EXPECT_FLOAT_EQ(out, 2.0f * std::exp(0.75f));
}

TEST_F(BackendPrimitivesTest, SumAddsSequentially) {
    std::vector<float> v = {1.5f, -2.0f, 4.0f};
    EXPECT_FLOAT_EQ(cpu.sum(v.data(), 3), 3.5f);
    EXPECT_FLOAT_EQ(cpu.sum(nullptr, 0), 0.0f);
}

TEST_F(BackendPrimitivesTest, DropoutIsAPureFunctionOfSeedAndOffset) {
    const size_t n = 4096;
    std::vector<float> in(n, 3.0f), out_a(n), out_b(n), mask_a(n), mask_b(n);
    cpu.dropout_forward(in.data(), out_a.data(), mask_a.data(), n, 0.3f, 1.0f / 0.7f, 99, 0);
    cpu.dropout_forward(in.data(), out_b.data(), mask_b.data(), n, 0.3f, 1.0f / 0.7f, 99, 0);
    EXPECT_EQ(mask_a, mask_b);

    // Offset k shifts the stream: element i at offset 1000 is element 1000 + i at offset 0.
    std::vector<float> shifted_out(n - 1000), shifted_mask(n - 1000);
    cpu.dropout_forward(in.data(), shifted_out.data(), shifted_mask.data(), n - 1000, 0.3f, 1.0f / 0.7f, 99, 1000);
    for (size_t i = 0; i < n - 1000; ++i) {
        ASSERT_EQ(shifted_mask[i], mask_a[1000 + i]) << "at " << i;
    }

    size_t dropped = 0;
    for (size_t i = 0; i < n; ++i) {
        if (mask_a[i] == 0.0f) {
            ++dropped;
            EXPECT_EQ(out_a[i], 0.0f);
        } else {
            EXPECT_FLOAT_EQ(out_a[i], 3.0f / 0.7f);
        }
    }
    const float fraction = static_cast<float>(dropped) / static_cast<float>(n);
    EXPECT_GT(fraction, 0.27f);  // p = 0.3, n = 4096: sigma ~ 0.007
    EXPECT_LT(fraction, 0.33f);
}

TEST_F(BackendPrimitivesTest, DropoutZeroesDroppedNonFiniteInputs) {
    std::vector<float> in(256, INFINITY), out(256), mask(256);
    cpu.dropout_forward(in.data(), out.data(), mask.data(), 256, 0.5f, 2.0f, 7, 0);
    for (size_t i = 0; i < 256; ++i) {
        if (mask[i] == 0.0f) {
            EXPECT_EQ(out[i], 0.0f);  // a select, not inf * 0
        }
    }
}

TEST_F(BackendPrimitivesTest, BceWithLogitsMatchesClosedFormAndIsStable) {
    // x = 0, y = 1: loss = log 2; grad = (0.5 - 1) * scale.
    std::vector<float> x = {0.0f, 100.0f, -100.0f};
    std::vector<float> y = {1.0f, 1.0f, 0.0f};
    std::vector<float> terms(3), grad(3);
    cpu.bce_with_logits(x.data(), y.data(), terms.data(), 3);
    EXPECT_FLOAT_EQ(terms[0], std::log(2.0f));
    // Confident and right: no overflow, and the loss is the denormal log1p(exp(-100)) ~ 4e-44.
    EXPECT_NEAR(terms[1], 0.0f, 1e-30f);
    EXPECT_NEAR(terms[2], 0.0f, 1e-30f);
    cpu.bce_with_logits_grad(x.data(), y.data(), grad.data(), 3, 0.5f);
    EXPECT_FLOAT_EQ(grad[0], -0.25f);
    EXPECT_NEAR(grad[1], 0.0f, 1e-30f);
    EXPECT_NEAR(grad[2], 0.0f, 1e-30f);
}

// ---- Mission 2 -----------------------------------------------------------------------------

TEST_F(BackendPrimitivesTest, Permute0213SwapsTheMiddleAxes) {
    // in (1, 2, 3, 1): [[a0 a1 a2], [b0 b1 b2]] -> out (1, 3, 2, 1): [[a0 b0], [a1 b1], [a2 b2]]
    std::vector<float> in = {0, 1, 2, 10, 11, 12};
    std::vector<float> out(6);
    cpu.permute_0213(in.data(), out.data(), 1, 2, 3, 1);
    EXPECT_EQ(out, (std::vector<float>{0, 10, 1, 11, 2, 12}));
    std::vector<float> back(6);
    cpu.permute_0213(out.data(), back.data(), 1, 3, 2, 1);
    EXPECT_EQ(back, in);
}

TEST_F(BackendPrimitivesTest, GatherAndScatterAddRowsHandleRepeatedIndices) {
    std::vector<float> table = {1, 2, 10, 20, 100, 200};  // 3 rows x 2
    std::vector<float> idx = {2, 0, 2};
    std::vector<float> out(6);
    cpu.gather_rows(table.data(), idx.data(), out.data(), 3, 2);
    EXPECT_EQ(out, (std::vector<float>{100, 200, 1, 2, 100, 200}));
    std::vector<float> acc(6, 0.0f);
    std::vector<float> src = {1, 1, 5, 5, 2, 3};
    cpu.scatter_add_rows(src.data(), idx.data(), acc.data(), 3, 2);
    EXPECT_EQ(acc, (std::vector<float>{5, 5, 0, 0, 3, 4}));
}

TEST_F(BackendPrimitivesTest, RopeRotateInverseUndoesForward) {
    std::vector<float> x = {1.0f, 2.0f, -3.0f, 0.5f};  // 1 slice, 2 positions, head_dim 2
    std::vector<float> c = {1.0f, 0.6f}, s = {0.0f, 0.8f};  // per-position cos/sin
    std::vector<float> y(4), back(4);
    cpu.rope_rotate(x.data(), c.data(), s.data(), y.data(), 1, 2, 2, false);
    EXPECT_FLOAT_EQ(y[0], 1.0f);  // position 0: identity
    EXPECT_FLOAT_EQ(y[1], 2.0f);
    cpu.rope_rotate(y.data(), c.data(), s.data(), back.data(), 1, 2, 2, true);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6f);
    }
}

// ---- Mission 5 -----------------------------------------------------------------------------

TEST_F(BackendPrimitivesTest, Copy2dExtractsAndWritesOneTimestep) {
    // (N=2, L=3, D=2): extract t=1, then write it to t=2 of another sequence.
    std::vector<float> seq = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15};
    std::vector<float> step(4);
    cpu.copy_2d(step.data(), 2, seq.data() + 1 * 2, 3 * 2, 2, 2);
    EXPECT_EQ(step, (std::vector<float>{2, 3, 12, 13}));
    std::vector<float> dst(12, 0.0f);
    cpu.copy_2d(dst.data() + 2 * 2, 3 * 2, step.data(), 2, 2, 2);
    EXPECT_EQ(dst, (std::vector<float>{0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 12, 13}));
}

TEST_F(BackendPrimitivesTest, AccumulateRowsAddsIntoTheRunningValueRowByRow) {
    std::vector<float> in = {1, 2, 3, 4};  // 2 rows x 2
    std::vector<float> out = {10, 20};
    cpu.accumulate_rows(in.data(), out.data(), 2, 2);
    EXPECT_EQ(out, (std::vector<float>{14, 26}));
}

}  // namespace
}  // namespace pulsatrix
