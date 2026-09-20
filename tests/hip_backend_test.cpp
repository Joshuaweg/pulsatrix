#include <gtest/gtest.h>

#include <vector>

#include "exai/hip_backend.hpp"
#include "exai/tensor.hpp"

// Mirrors CUDABackendTest's exact test shape (allocate/free round-trip, zero-byte
// convention, copy round-trip, fill correctness), which in turn mirrors CPUBackendTest's --
// deliberate, so all three backends' test suites are structurally comparable, supporting
// Mission 2's numerical-equivalence suite. Run against real hardware (Radeon 8060S,
// gfx1151), not mocked.

namespace exai {
namespace {

class HIPBackendTest : public ::testing::Test {
protected:
    HIPBackend backend;
};

TEST_F(HIPBackendTest, AllocateReturnsNonNullForPositiveSize) {
    void* ptr = backend.allocate(64);
    ASSERT_NE(ptr, nullptr);
    backend.free(ptr);
}

TEST_F(HIPBackendTest, AllocateZeroBytesReturnsNullByConvention) {
    void* ptr = backend.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(HIPBackendTest, FreeOnNullptrIsSafe) {
    EXPECT_NO_THROW(backend.free(nullptr));
}

TEST_F(HIPBackendTest, CopyRoundTripPreservesData) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    void* device_ptr = backend.allocate(src.size() * sizeof(float));
    ASSERT_NE(device_ptr, nullptr);

    backend.copy(device_ptr, src.data(), src.size() * sizeof(float), CopyDirection::HostToDevice);

    std::vector<float> dst(src.size(), 0.0f);
    backend.copy(dst.data(), device_ptr, src.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
    backend.free(device_ptr);
}

TEST_F(HIPBackendTest, FillSetsEveryElementToValue) {
    constexpr size_t n = 5;
    void* device_ptr = backend.allocate(n * sizeof(float));
    ASSERT_NE(device_ptr, nullptr);

    backend.fill(device_ptr, 3.5f, n);

    std::vector<float> host(n, 0.0f);
    backend.copy(host.data(), device_ptr, n * sizeof(float), CopyDirection::DeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host[i], 3.5f);
    }
    backend.free(device_ptr);
}

TEST_F(HIPBackendTest, FillOnZeroElementsIsSafe) {
    EXPECT_NO_THROW(backend.fill(nullptr, 1.0f, 0));
}

TEST_F(HIPBackendTest, GemmComputesHandVerified2x2Product) {
    // Mirrors CUDABackendTest/CPUBackendTest::GemmComputesHandVerified2x2Product exactly.
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> A*B = [[19, 22], [43, 50]]
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};

    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(4 * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), 2,
                 2, 2);

    std::vector<float> out(4, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 19.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 43.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

TEST_F(HIPBackendTest, GemmHandlesNonSquareDimensions) {
    // Mirrors CUDABackendTest/CPUBackendTest::GemmHandlesNonSquareDimensions -- catches a
    // transpose-only (wrong) result, which a shape-only test would miss for a square case.
    // This matters more for HIP than it did for CUDA: the swap-and-transpose argument order
    // is the one construct the charter predicted would not translate automatically, so a
    // square-only check would be exactly the wrong coverage here.
    // A (2x3) * B (3x2): A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]] -> [[58,64],[139,154]]
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {7, 8, 9, 10, 11, 12};

    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(4 * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), 2,
                 3, 2);

    std::vector<float> out(4, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

}  // namespace
}  // namespace exai
