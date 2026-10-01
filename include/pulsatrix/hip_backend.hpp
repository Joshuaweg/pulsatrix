/** @file hip_backend.hpp
 *  @brief HIP/ROCm implementation of DeviceBackend. Only compiled when PULSATRIX_ENABLE_HIP is set.
 *  @ingroup dl_modules
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief HIP-resident DeviceBackend implementation, targeting AMD GPUs via ROCm.
 * @note `gemm` uses hipBLAS (`hipblasSgemm`) with the column-major swap-and-transpose trick,
 *       since hipBLAS inherits cuBLAS's column-major assumption and this project's `Tensor`
 *       is row-major -- see `gpu_backend_programming/context_gpu_cublas_cudnn_integration.md`.
 *       `elementwise`/`add` are hand-written kernels, one thread per element.
 * @note Structurally a mirror of CUDABackend, deliberately. Phase 1.6's HIPIFY triage found
 *       39 of 40 references auto-convertible with no warnings in the kernel bodies at all
 *       (`<<<>>>`, `__global__`, `threadIdx`/`blockIdx` are identical in HIP), so a divergent
 *       design would add risk without adding value and would make Mission 2's three-way
 *       equivalence suite harder to read.
 * @note No `warpSize` handling is needed anywhere in this class: every kernel here is
 *       one-thread-per-element with no cross-lane operation, so the CDNA-wavefront-64 hazard
 *       that `context_accel_rocm_hip.md` flags for CUDA ports does not arise. (This project's
 *       dev device, gfx1151, reports warpSize 32 in any case.)
 * @note The same Phase 1.5 scope limit applies here: only Tensor operations that route
 *       entirely through DeviceBackend's own primitives are safe against a HIP-backed Tensor.
 *       Phase 1's Module backward/LRP/optimizer code is host-loop-only and is guarded by
 *       `PULSATRIX_REQUIRE_HOST`, which covers DeviceType::Hip identically
 *       to DeviceType::Cuda -- those guards need no change for this backend.
 */
class HIPBackend : public DeviceBackend {
public:
    HIPBackend();
    ~HIPBackend() override;

    HIPBackend(const HIPBackend&) = delete;
    HIPBackend& operator=(const HIPBackend&) = delete;

    [[nodiscard]] DeviceType device() const noexcept override { return DeviceType::Hip; }

    [[nodiscard]] void* allocate(size_t bytes) override;
    void free(void* ptr) noexcept override;
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override;
    void fill(void* ptr, float value, size_t n) override;
    void gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) override;
    void elementwise(ElementwiseOp op, const float* in, float* out, size_t n) override;
    void add(const float* a, const float* b, float* out, size_t n) override;
    void mul(const float* a, const float* b, float* out, size_t n) override;

private:
    hipStream_t stream_;
    hipblasHandle_t hipblas_handle_;
};

}  // namespace pulsatrix
