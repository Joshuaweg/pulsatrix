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
    void gemm_ex(const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m, size_t k,
                 size_t n, float beta) override;
    void column_sums(const float* in, float* out, size_t rows, size_t cols, float beta) override;
    void add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols) override;
    void elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in,
                              size_t n) override;
    void axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) override;
    [[nodiscard]] float dot(const float* a, const float* b, size_t n) override;
    void softmax_rows(const float* in, float* out, size_t rows, size_t cols) override;
    void softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) override;
    void logsumexp_rows(const float* in, float* out, size_t rows, size_t cols) override;
    void adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1, float beta2,
                   float eps, float bias_correction1, float bias_correction2) override;
    [[nodiscard]] float sum(const float* in, size_t n) override;
    void dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale, uint64_t seed,
                         uint64_t offset) override;
    void bce_with_logits(const float* logits, const float* target, float* out, size_t n) override;
    void bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n, float scale) override;
    void layer_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                            float* row_std, size_t rows, size_t cols, float eps) override;
    void layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float* row_std,
                             float* grad_in, size_t rows, size_t cols) override;
    void rms_norm_forward(const float* in, const float* gamma, float* out, float* row_rms, size_t rows, size_t cols,
                          float eps) override;
    void rms_norm_backward(const float* grad_out, const float* gamma, const float* in, const float* row_rms,
                           float* grad_in, float* gamma_terms, size_t rows, size_t cols) override;
    void rope_rotate(const float* in, const float* cos_table, const float* sin_table, float* out, size_t num_slices,
                     size_t seq_len, size_t head_dim, bool inverse) override;
    void permute_0213(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) override;
    void gather_rows(const float* table, const float* indices, float* out, size_t count, size_t dim) override;
    void scatter_add_rows(const float* src, const float* indices, float* table, size_t count, size_t dim) override;
    void tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps, float* action,
                               float* std_cache, float* log_prob, size_t rows, size_t cols, float stabilizer,
                               double half_log_two_pi) override;
    void tanh_gaussian_backward(const float* action, const float* std_cache, const float* eps,
                                const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                float* grad_log_std, size_t n, float stabilizer) override;

private:
    hipStream_t stream_;
    hipblasHandle_t hipblas_handle_;
    // One device float that dot() reduces into before copying it to the host; allocated once
    // so dot() costs no per-call device allocation.
    float* dot_result_ = nullptr;
};

}  // namespace pulsatrix
