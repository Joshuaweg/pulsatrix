/** @file cuda_backend.hpp
 *  @brief CUDA implementation of DeviceBackend. Only compiled when EXAI_ENABLE_CUDA is set.
 */
#pragma once

#include <cuda_runtime.h>

#include "exai/device_backend.hpp"

namespace exai {

/**
 * @brief CUDA-resident DeviceBackend implementation.
 * @note `allocate`/`free`/`copy`/`fill` are implemented (Phase 1.5 Mission 0). `gemm`/
 *       `elementwise`/`add` throw for now -- Mission 1 of this campaign replaces them with
 *       real cuBLAS/kernel implementations, test-first, before this mission closes. This
 *       class exists in this intermediate state only within Mission 0's own commits.
 * @note This is the first of two currently-supported concrete DeviceBackend
 *       implementations. Per the campaign's own scope decision
 *       (campaign_exai_dl_library_phase1_5_cuda_backend.md), only Tensor operations that
 *       route entirely through DeviceBackend's own primitives are safe to run against a
 *       CUDA-backed Tensor today -- most of Phase 1's Module backward/LRP/optimizer code
 *       is not yet backend-generic and will be guarded (Mission 2 of this campaign) rather
 *       than silently producing wrong results if called on a non-CPU Tensor.
 */
class CUDABackend : public DeviceBackend {
public:
    CUDABackend();
    ~CUDABackend() override;

    CUDABackend(const CUDABackend&) = delete;
    CUDABackend& operator=(const CUDABackend&) = delete;

    [[nodiscard]] void* allocate(size_t bytes) override;
    void free(void* ptr) noexcept override;
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override;
    void fill(void* ptr, float value, size_t n) override;
    void gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) override;
    void elementwise(ElementwiseOp op, const float* in, float* out, size_t n) override;
    void add(const float* a, const float* b, float* out, size_t n) override;

private:
    cudaStream_t stream_;
};

}  // namespace exai
