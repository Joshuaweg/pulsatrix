/** @file hip_check.hpp
 *  @brief PULSATRIX_HIP_CHECK -- converts a HIP runtime failure into a thrown C++ exception
 *         at the DeviceBackend boundary, per
 *         gpu_backend_programming/context_gpu_cuda_kernel_mechanics.md's Error Checking
 *         pattern. Only compiled when PULSATRIX_ENABLE_HIP is set.
 */
#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

/**
 * @brief Checks a HIP runtime call's hipError_t result; throws std::runtime_error with
 *        the HIP error string and source location if it did not succeed.
 * @note Kernel launches are asynchronous -- this macro catches launch-configuration errors
 *       immediately via hipGetLastError(), but in-kernel runtime errors only surface at the
 *       next synchronizing call. Route every allocate/copy/kernel-launch/sync call in
 *       HIPBackend through this macro, including a hipGetLastError() check right after
 *       every kernel launch.
 * @note Deliberately identical in shape to PULSATRIX_CUDA_CHECK (cuda_check.hpp) -- HIPIFY maps
 *       cudaError_t/cudaSuccess/cudaGetErrorString one-to-one onto their hip* counterparts,
 *       so there is nothing to redesign here and a divergent shape would only make the two
 *       backends harder to compare.
 */
#define PULSATRIX_HIP_CHECK(call)                                                                    \
    do {                                                                                          \
        hipError_t pulsatrix_hip_check_result = (call);                                                 \
        if (pulsatrix_hip_check_result != hipSuccess) {                                                  \
            throw std::runtime_error(std::string("HIP error: ") +                                    \
                                      hipGetErrorString(pulsatrix_hip_check_result) + " at " __FILE__ +    \
                                      ":" + std::to_string(__LINE__));                                 \
        }                                                                                                \
    } while (0)
