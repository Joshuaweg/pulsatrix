/** @file cuda_check.hpp
 *  @brief PULSATRIX_CUDA_CHECK -- converts a CUDA runtime failure into a thrown C++ exception
 *         at the DeviceBackend boundary, per
 *         gpu_backend_programming/context_gpu_cuda_kernel_mechanics.md's Error Checking
 *         pattern. Only compiled when PULSATRIX_ENABLE_CUDA is set.
 */
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

/**
 * @brief Checks a CUDA runtime call's cudaError_t result; throws std::runtime_error with
 *        the CUDA error string and source location if it did not succeed.
 * @note Kernel launches are asynchronous -- this macro catches launch-configuration errors
 *       immediately via cudaGetLastError(), but in-kernel runtime errors only surface at the
 *       next synchronizing call. Route every allocate/copy/kernel-launch/sync call in
 *       CUDABackend through this macro, including a cudaGetLastError() check right after
 *       every kernel launch.
 */
#define PULSATRIX_CUDA_CHECK(call)                                                                   \
    do {                                                                                          \
        cudaError_t pulsatrix_cuda_check_result = (call);                                               \
        if (pulsatrix_cuda_check_result != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ") +                                  \
                                      cudaGetErrorString(pulsatrix_cuda_check_result) + " at " __FILE__ + \
                                      ":" + std::to_string(__LINE__));                                 \
        }                                                                                                \
    } while (0)
