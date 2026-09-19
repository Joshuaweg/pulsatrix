#include <gtest/gtest.h>

#include <vector>

#include "exai/cuda_backend.hpp"

// Mirrors CPUBackendTest's exact test shape (allocate/free round-trip, zero-byte
// convention, copy round-trip, fill correctness) -- deliberate, so the two backends'
// test suites are structurally comparable, supporting Mission 1's numerical-equivalence
// suite. Run against real hardware (RTX 3060), not mocked.

namespace exai {
namespace {

class CUDABackendTest : public ::testing::Test {
protected:
    CUDABackend backend;
};

TEST_F(CUDABackendTest, AllocateReturnsNonNullForPositiveSize) {
    void* ptr = backend.allocate(64);
    ASSERT_NE(ptr, nullptr);
    backend.free(ptr);
}

TEST_F(CUDABackendTest, AllocateZeroBytesReturnsNullByConvention) {
    void* ptr = backend.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CUDABackendTest, FreeOnNullptrIsSafe) {
    EXPECT_NO_THROW(backend.free(nullptr));
}

TEST_F(CUDABackendTest, CopyRoundTripPreservesData) {
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

TEST_F(CUDABackendTest, FillSetsEveryElementToValue) {
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

TEST_F(CUDABackendTest, FillOnZeroElementsIsSafe) {
    EXPECT_NO_THROW(backend.fill(nullptr, 1.0f, 0));
}

}  // namespace
}  // namespace exai
