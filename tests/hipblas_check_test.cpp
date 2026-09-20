#include <gtest/gtest.h>

#include <hipblas/hipblas.h>
#include <stdexcept>
#include <string>

#include "pulsatrix/hipblas_check.hpp"

// Mirrors cublas_check_test.cpp's exact test shape -- deliberate, so the two backends'
// error-handling suites are structurally comparable.

namespace pulsatrix {
namespace {

TEST(HipblasCheckTest, SuccessfulCallDoesNotThrow) {
    hipblasHandle_t handle = nullptr;
    EXPECT_NO_THROW(PULSATRIX_HIPBLAS_CHECK(hipblasCreate(&handle)));
    static_cast<void>(hipblasDestroy(handle));
}

TEST(HipblasCheckTest, FailedCallThrowsRuntimeErrorWithMessage) {
    // An uninitialized (null) handle deterministically fails with
    // HIPBLAS_STATUS_NOT_INITIALIZED -- hipBLAS inherits cuBLAS's documented behavior that
    // handle validity is checked before any of the data pointers, so the 1x1 dummy args
    // below are never touched.
    hipblasHandle_t uninitialized_handle = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    float dummy = 0.0f;

    bool threw = false;
    try {
        PULSATRIX_HIPBLAS_CHECK(hipblasSgemm(uninitialized_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, 1, 1, 1, &alpha,
                                         &dummy, 1, &dummy, 1, &beta, &dummy, 1));
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        EXPECT_NE(msg.find("hipBLAS error"), std::string::npos);
    }
    EXPECT_TRUE(threw);
}

}  // namespace
}  // namespace pulsatrix
