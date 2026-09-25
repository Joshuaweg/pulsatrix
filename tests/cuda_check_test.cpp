#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <stdexcept>

#include "pulsatrix/cuda_check.hpp"

namespace pulsatrix {
namespace {

TEST(CudaCheckTest, SuccessfulCallDoesNotThrow) {
    void* ptr = nullptr;
    EXPECT_NO_THROW(PULSATRIX_CUDA_CHECK(cudaMalloc(&ptr, 16)));
    cudaFree(ptr);
}

TEST(CudaCheckTest, FailedCallThrowsRuntimeErrorWithMessage) {
    // Request an allocation far beyond any real GPU's memory -- deterministically fails
    // with cudaErrorMemoryAllocation regardless of what else is using this device.
    void* ptr = nullptr;
    constexpr size_t absurd_size = static_cast<size_t>(100) * 1024 * 1024 * 1024 * 1024;  // 100 TB

    bool threw = false;
    try {
        PULSATRIX_CUDA_CHECK(cudaMalloc(&ptr, absurd_size));
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        EXPECT_NE(msg.find("CUDA error"), std::string::npos);
    }
    EXPECT_TRUE(threw);

    // Clear the sticky error state so it doesn't leak into the next test.
    cudaGetLastError();
}

}  // namespace
}  // namespace pulsatrix
