/** @file cpu_backend.hpp
 *  @brief CPU implementation of DeviceBackend -- the first, reference DeviceBackend implementation.
 */
#pragma once

#include "exai/device_backend.hpp"

namespace exai {

/**
 * @brief CPU-resident DeviceBackend implementation. Reference implementation every other
 *        backend (CUDABackend, HIPBackend) is checked for numerical equivalence against.
 */
class CPUBackend : public DeviceBackend {
public:
    [[nodiscard]] void* allocate(size_t bytes) override;
    void free(void* ptr) noexcept override;
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override;
    void fill(void* ptr, float value, size_t n) override;
    void gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) override;
    void elementwise(ElementwiseOp op, const float* in, float* out, size_t n) override;
    void add(const float* a, const float* b, float* out, size_t n) override;
};

}  // namespace exai
