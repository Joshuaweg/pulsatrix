#include <gtest/gtest.h>

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#include "pulsatrix/cublas_check.hpp"

namespace pulsatrix {
namespace {

TEST(CublasCheckTest, SuccessfulCallDoesNotThrow) {
    cublasHandle_t handle = nullptr;
    EXPECT_NO_THROW(PULSATRIX_CUBLAS_CHECK(cublasCreate(&handle)));
    cublasDestroy(handle);
}

TEST(CublasCheckTest, FailedCallThrowsRuntimeErrorWithMessage) {
    // An uninitialized (null) handle deterministically fails with
    // CUBLAS_STATUS_NOT_INITIALIZED -- per cuBLAS docs, handle validity is checked before
    // any of the data pointers, so the 1x1 dummy args below are never touched.
    cublasHandle_t uninitialized_handle = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    float dummy = 0.0f;

    bool threw = false;
    try {
        PULSATRIX_CUBLAS_CHECK(cublasSgemm(uninitialized_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha,
                                       &dummy, 1, &dummy, 1, &beta, &dummy, 1));
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        EXPECT_NE(msg.find("cuBLAS error"), std::string::npos);
    }
    EXPECT_TRUE(threw);
}

}  // namespace
}  // namespace pulsatrix
