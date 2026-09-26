/** @file hipblas_check.hpp
 *  @brief PULSATRIX_HIPBLAS_CHECK -- converts a hipBLAS call failure into a thrown C++ exception,
 *         parallel to PULSATRIX_HIP_CHECK (hip_check.hpp). Only compiled when PULSATRIX_ENABLE_HIP
 *         is set.
 *  @ingroup dl_modules
 */
#pragma once

#include <hipblas/hipblas.h>

#include <stdexcept>
#include <string>

/**
 * @brief Checks a hipBLAS call's hipblasStatus_t result; throws std::runtime_error with the
 *        hipBLAS status string and source location if it did not succeed.
 * @note The one construct in this whole backend that HIPIFY could not translate. cuBLAS's
 *       `cublasGetStatusString` has no same-named HIP counterpart -- `hipify-perl --examine`
 *       reported it as the single "unsupported HIP identifier" across all four Phase 1.5
 *       CUDA sources. hipBLAS provides the equivalent under a different name,
 *       `hipblasStatusToString` (hipblas.h: "Returns string representing hipblasStatus_t
 *       value"), so no error-reporting fidelity is lost relative to PULSATRIX_CUBLAS_CHECK.
 */
#define PULSATRIX_HIPBLAS_CHECK(call)                                                                  \
    do {                                                                                            \
        hipblasStatus_t pulsatrix_hipblas_check_result = (call);                                          \
        if (pulsatrix_hipblas_check_result != HIPBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(std::string("hipBLAS error: ") +                                  \
                                      hipblasStatusToString(pulsatrix_hipblas_check_result) + " at " __FILE__ \
                                      + ":" + std::to_string(__LINE__));                                  \
        }                                                                                                   \
    } while (0)
