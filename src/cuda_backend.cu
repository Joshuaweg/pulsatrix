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
}

CUDABackend::~CUDABackend() {
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

}  // namespace pulsatrix
