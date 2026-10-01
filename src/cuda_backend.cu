#include "pulsatrix/cuda_backend.hpp"

#include <stdexcept>

#include "gpu_kernels.cuh"

#include "pulsatrix/cublas_check.hpp"
#include "pulsatrix/cuda_check.hpp"

namespace pulsatrix {

CUDABackend::CUDABackend() {
    PULSATRIX_CUDA_CHECK(cudaStreamCreate(&stream_));
    PULSATRIX_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    PULSATRIX_CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    PULSATRIX_CUDA_CHECK(cudaMalloc(&dot_result_, sizeof(float)));
}

CUDABackend::~CUDABackend() {
    static_cast<void>(cudaFree(dot_result_));
    cublasDestroy(cublas_handle_);
    cudaStreamDestroy(stream_);
}

void* CUDABackend::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    PULSATRIX_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CUDABackend::free(void* ptr) noexcept {
    cudaFree(ptr);
}

void CUDABackend::copy(void* dst, const void* src, size_t bytes, CopyDirection dir) {
    if (bytes == 0) {
        return;
    }
    cudaMemcpyKind kind = cudaMemcpyDefault;
    switch (dir) {
        case CopyDirection::HostToDevice:
            kind = cudaMemcpyHostToDevice;
            break;
        case CopyDirection::DeviceToHost:
            kind = cudaMemcpyDeviceToHost;
            break;
        case CopyDirection::DeviceToDevice:
            kind = cudaMemcpyDeviceToDevice;
            break;
        case CopyDirection::HostToHost:
            kind = cudaMemcpyHostToHost;
            break;
    }
    PULSATRIX_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream_));
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::fill(void* ptr, float value, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_fill(static_cast<float*>(ptr), value, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) {
    // Row-major A(m,k)*B(k,n)=C(m,n) via cuBLAS (column-major): compute C^T = B^T*A^T
    // instead, which cuBLAS computes correctly as a column-major op, and a row-major
    // C(m,n) buffer is the same bytes as a column-major C^T(n,m) buffer. See
    // gpu_backend_programming/context_gpu_cublas_cudnn_integration.md's Column-Major Trap.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    PULSATRIX_CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
                                   static_cast<int>(m), static_cast<int>(k), &alpha, b,
                                   static_cast<int>(n), a, static_cast<int>(k), &beta, out,
                                   static_cast<int>(n)));
}

void CUDABackend::elementwise(ElementwiseOp op, const float* in, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_elementwise(op, in, out, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::add(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_add(a, b, out, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::mul(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_mul(a, b, out, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- GPU-native-kernels Mission 1 primitives (kernels in gpu_kernels.cuh) -----------------

void CUDABackend::gemm_ex(const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m,
                         size_t k, size_t n, float beta) {
    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        // op(A)*op(B) is all zeros; BLAS would reject the zero leading dimensions.
        axpby(beta, out, 0.0f, out, out, m * n);
        return;
    }
    // Same column-major trick as gemm(): compute C^T(n x m) = op(B)^T * op(A)^T. A row-major
    // (r x c) buffer is a column-major (c x r) matrix with leading dimension c, so a stored-
    // transposed operand becomes a plain one in BLAS's view and vice versa -- hence op(B) maps
    // to OP_T exactly when transpose_b, with the leading dimension of the *stored* layout.
    const float alpha = 1.0f;
    PULSATRIX_CUBLAS_CHECK(cublasSgemm(cublas_handle_, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha, b,
                                 static_cast<int>(transpose_b ? k : n), a, static_cast<int>(transpose_a ? m : k),
                                 &beta, out, static_cast<int>(n)));
}

void CUDABackend::column_sums(const float* in, float* out, size_t rows, size_t cols, float beta) {
    if (cols == 0) {
        return;
    }
    gpu::launch_column_sums(in, out, rows, cols, beta, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::launch_add_row_vector(in, row, out, rows, cols, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in,
                                      size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_elementwise_backward(op, x, grad_out, grad_in, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_axpby(alpha, x, beta, y, out, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

float CUDABackend::dot(const float* a, const float* b, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    gpu::launch_dot(a, b, n, dot_result_, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    float result = 0.0f;
    copy(&result, dot_result_, sizeof(float), CopyDirection::DeviceToHost);  // synchronizes
    return result;
}

void CUDABackend::softmax_rows(const float* in, float* out, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::launch_softmax_rows(in, out, rows, cols, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::launch_softmax_rows_backward(y, dy, dx, rows, cols, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::logsumexp_rows(const float* in, float* out, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::launch_logsumexp_rows(in, out, rows, cols, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1,
                           float beta2, float eps, float bias_correction1, float bias_correction2) {
    if (n == 0) {
        return;
    }
    gpu::launch_adam_step(param, grad, m, v, n, lr, beta1, beta2, eps, bias_correction1, bias_correction2, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}


// ---- GPU-native-kernels Mission 1b -------------------------------------------------------

float CUDABackend::sum(const float* in, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    gpu::launch_sum(in, n, dot_result_, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    float result = 0.0f;
    copy(&result, dot_result_, sizeof(float), CopyDirection::DeviceToHost);  // synchronizes
    return result;
}

void CUDABackend::dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale,
                                 uint64_t seed, uint64_t offset) {
    if (n == 0) {
        return;
    }
    gpu::launch_dropout_forward(in, out, mask, n, p, scale, seed, offset, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::bce_with_logits(const float* logits, const float* target, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::launch_bce_with_logits(logits, target, out, n, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n,
                                      float scale) {
    if (n == 0) {
        return;
    }
    gpu::launch_bce_with_logits_grad(logits, target, grad, n, scale, stream_);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace pulsatrix
