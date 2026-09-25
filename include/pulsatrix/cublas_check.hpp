/** @file cublas_check.hpp
 *  @brief PULSATRIX_CUBLAS_CHECK -- converts a cuBLAS call failure into a thrown C++ exception,
 *         parallel to PULSATRIX_CUDA_CHECK (cuda_check.hpp). Only compiled when PULSATRIX_ENABLE_CUDA
 *         is set.
 */
#pragma once

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

/**
 * @brief Checks a cuBLAS call's cublasStatus_t result; throws std::runtime_error with the
 *        cuBLAS status string and source location if it did not succeed.
 */
#define PULSATRIX_CUBLAS_CHECK(call)                                                                   \
    do {                                                                                            \
        cublasStatus_t pulsatrix_cublas_check_result = (call);                                            \
        if (pulsatrix_cublas_check_result != CUBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("cuBLAS error: ") +                                   \
                                      cublasGetStatusString(pulsatrix_cublas_check_result) + " at " __FILE__ \
                                      + ":" + std::to_string(__LINE__));                                  \
        }                                                                                                   \
    } while (0)
