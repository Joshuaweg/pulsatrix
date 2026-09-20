#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#include "pulsatrix/hip_check.hpp"

// Mirrors cuda_check_test.cpp's exact test shape -- deliberate, so the two backends'
// error-handling suites are structurally comparable. Run against real hardware
// (Radeon 8060S, gfx1151), not mocked.

namespace pulsatrix {
namespace {

TEST(HipCheckTest, SuccessfulCallDoesNotThrow) {
    void* ptr = nullptr;
    EXPECT_NO_THROW(PULSATRIX_HIP_CHECK(hipMalloc(&ptr, 16)));
    static_cast<void>(hipFree(ptr));  // HIP marks this [[nodiscard]]; cudaFree is not
}

TEST(HipCheckTest, FailedCallThrowsRuntimeErrorWithMessage) {
    // Request an allocation far beyond any real GPU's memory -- deterministically fails
    // with hipErrorOutOfMemory regardless of what else is using this device. Note this
    // device is an APU whose "VRAM" is a 96 GiB carve-out of system RAM, so the absurd
    // size has to clear that, not a discrete card's 12 GB; 100 TB does.
    void* ptr = nullptr;
    constexpr size_t absurd_size = static_cast<size_t>(100) * 1024 * 1024 * 1024 * 1024;  // 100 TB

    bool threw = false;
    try {
        PULSATRIX_HIP_CHECK(hipMalloc(&ptr, absurd_size));
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        EXPECT_NE(msg.find("HIP error"), std::string::npos);
    }
    EXPECT_TRUE(threw);

    // Clear the sticky error state so it doesn't leak into the next test.
    static_cast<void>(hipGetLastError());
}

}  // namespace
}  // namespace pulsatrix
