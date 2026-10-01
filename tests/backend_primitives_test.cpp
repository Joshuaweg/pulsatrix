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
    EXPECT_FLOAT_EQ(out, g * s * (1.0f + x * (1.0f - s)));
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

}  // namespace
}  // namespace pulsatrix
