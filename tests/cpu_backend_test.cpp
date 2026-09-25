#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class CPUBackendTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(CPUBackendTest, AllocateReturnsNonNullForPositiveSize) {
    void* ptr = backend.allocate(64);
    ASSERT_NE(ptr, nullptr);
    backend.free(ptr);
}

TEST_F(CPUBackendTest, AllocateZeroBytesReturnsNullByConvention) {
    void* ptr = backend.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CPUBackendTest, FreeOnNullptrIsSafe) {
    EXPECT_NO_THROW(backend.free(nullptr));
}

TEST_F(CPUBackendTest, CopyRoundTripPreservesData) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    void* dst = backend.allocate(src.size() * sizeof(float));
    ASSERT_NE(dst, nullptr);

    backend.copy(dst, src.data(), src.size() * sizeof(float), CopyDirection::HostToHost);

    auto* dst_floats = static_cast<float*>(dst);
    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_FLOAT_EQ(dst_floats[i], src[i]);
    }
    backend.free(dst);
}

TEST_F(CPUBackendTest, CopyZeroBytesIsSafe) {
    void* dst = backend.allocate(4);
    ASSERT_NE(dst, nullptr);
    EXPECT_NO_THROW(backend.copy(dst, nullptr, 0, CopyDirection::HostToHost));
    backend.free(dst);
}

TEST_F(CPUBackendTest, FillSetsEveryElementToValue) {
    constexpr size_t n = 5;
    void* ptr = backend.allocate(n * sizeof(float));
    ASSERT_NE(ptr, nullptr);

    backend.fill(ptr, 3.5f, n);

    auto* floats = static_cast<float*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(floats[i], 3.5f);
    }
    backend.free(ptr);
}

TEST_F(CPUBackendTest, GemmComputesHandVerified2x2Product) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> A*B = [[19, 22], [43, 50]]
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out(4, 0.0f);

    backend.gemm(a.data(), b.data(), out.data(), 2, 2, 2);

    EXPECT_FLOAT_EQ(out[0], 19.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 43.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);
}

TEST_F(CPUBackendTest, GemmHandlesNonSquareDimensions) {
    // A (2x3) * B (3x2): A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]] -> [[58,64],[139,154]]
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {7, 8, 9, 10, 11, 12};
    std::vector<float> out(4, 0.0f);

    backend.gemm(a.data(), b.data(), out.data(), 2, 3, 2);

    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);
}

TEST_F(CPUBackendTest, ElementwiseReluClampsNegativeValuesToZero) {
    std::vector<float> in = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    std::vector<float> out(in.size(), 0.0f);

    backend.elementwise(ElementwiseOp::Relu, in.data(), out.data(), in.size());

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 0.5f);
    EXPECT_FLOAT_EQ(out[4], 2.0f);
}

TEST_F(CPUBackendTest, ElementwiseNegNegatesEveryElement) {
    std::vector<float> in = {1.0f, -1.0f, 0.0f, 3.5f};
    std::vector<float> out(in.size(), 0.0f);

    backend.elementwise(ElementwiseOp::Neg, in.data(), out.data(), in.size());

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], -3.5f);
}

TEST_F(CPUBackendTest, ElementwiseTanhMatchesReferenceValues) {
    std::vector<float> in = {-1.0f, 0.0f, 0.5f, 2.0f};
    std::vector<float> out(in.size(), 0.0f);

    backend.elementwise(ElementwiseOp::Tanh, in.data(), out.data(), in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], std::tanh(in[i]));
    }
    // Odd symmetry and the saturating range are the properties recurrent modules rely on.
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[0], -std::tanh(1.0f));
    EXPECT_LT(out[3], 1.0f);
}

TEST_F(CPUBackendTest, ElementwiseSigmoidMatchesReferenceValues) {
    std::vector<float> in = {-2.0f, 0.0f, 1.0f, 3.0f};
    std::vector<float> out(in.size(), 0.0f);

    backend.elementwise(ElementwiseOp::Sigmoid, in.data(), out.data(), in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 1.0f / (1.0f + std::exp(-in[i])));
    }
    EXPECT_FLOAT_EQ(out[1], 0.5f);  // sigmoid(0) is exactly 1/2
    EXPECT_GT(out[0], 0.0f);
    EXPECT_LT(out[3], 1.0f);
}

TEST_F(CPUBackendTest, ElementwiseSiluIsInputTimesSigmoid) {
    std::vector<float> in = {-3.0f, -1.0f, 0.0f, 1.0f, 4.0f};
    std::vector<float> sigmoid_out(in.size(), 0.0f);
    std::vector<float> out(in.size(), 0.0f);

    backend.elementwise(ElementwiseOp::Sigmoid, in.data(), sigmoid_out.data(), in.size());
    backend.elementwise(ElementwiseOp::Silu, in.data(), out.data(), in.size());

    // silu(x) == x * sigmoid(x) exactly (same shared sigmoid expression), not merely close.
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], in[i] * sigmoid_out[i]);
    }
    EXPECT_FLOAT_EQ(out[2], 0.0f);  // silu(0) == 0
    EXPECT_LT(out[1], 0.0f);        // non-monotonic: stays negative for small negative x
}

TEST_F(CPUBackendTest, ElementwiseHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.elementwise(ElementwiseOp::Relu, nullptr, nullptr, 0));
}

TEST_F(CPUBackendTest, ElementwiseSupportsInPlaceAliasing) {
    std::vector<float> buf = {-1.0f, 2.0f, -3.0f};
    backend.elementwise(ElementwiseOp::Relu, buf.data(), buf.data(), buf.size());

    EXPECT_FLOAT_EQ(buf[0], 0.0f);
    EXPECT_FLOAT_EQ(buf[1], 2.0f);
    EXPECT_FLOAT_EQ(buf[2], 0.0f);
}

TEST_F(CPUBackendTest, AddComputesElementwiseSum) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(3, 0.0f);

    backend.add(a.data(), b.data(), out.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);
}

TEST_F(CPUBackendTest, AddSupportsInPlaceAccumulation) {
    std::vector<float> acc = {1.0f, 2.0f, 3.0f};
    std::vector<float> delta = {0.5f, 0.5f, 0.5f};

    backend.add(acc.data(), delta.data(), acc.data(), 3);  // out aliases a

    EXPECT_FLOAT_EQ(acc[0], 1.5f);
    EXPECT_FLOAT_EQ(acc[1], 2.5f);
    EXPECT_FLOAT_EQ(acc[2], 3.5f);
}

TEST_F(CPUBackendTest, AddHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.add(nullptr, nullptr, nullptr, 0));
}

TEST_F(CPUBackendTest, MulComputesElementwiseProduct) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(3, 0.0f);

    backend.mul(a.data(), b.data(), out.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 40.0f);
    EXPECT_FLOAT_EQ(out[2], 90.0f);
}

TEST_F(CPUBackendTest, MulSupportsInPlaceApplication) {
    std::vector<float> acc = {1.0f, 2.0f, 3.0f};
    std::vector<float> factor = {2.0f, 2.0f, 2.0f};

    backend.mul(acc.data(), factor.data(), acc.data(), 3);  // out aliases a

    EXPECT_FLOAT_EQ(acc[0], 2.0f);
    EXPECT_FLOAT_EQ(acc[1], 4.0f);
    EXPECT_FLOAT_EQ(acc[2], 6.0f);
}

TEST_F(CPUBackendTest, MulHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.mul(nullptr, nullptr, nullptr, 0));
}

}  // namespace
}  // namespace pulsatrix
