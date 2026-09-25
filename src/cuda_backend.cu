#include "pulsatrix/cuda_backend.hpp"

#include <stdexcept>

#include "pulsatrix/cublas_check.hpp"
#include "pulsatrix/cuda_check.hpp"

namespace pulsatrix {

namespace {

__global__ void fill_kernel(float* ptr, float value, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[i] = value;
    }
}

__global__ void relu_kernel(const float* in, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

__global__ void neg_kernel(const float* in, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -in[i];
    }
}

// Device-side logistic sigmoid -- one definition shared by sigmoid_kernel and silu_kernel,
// mirroring CPUBackend::elementwise's host-side sigmoid() helper so both backends compute
// Sigmoid and Silu from the identical expression.
__device__ inline float sigmoid_device(float z) { return 1.0f / (1.0f + expf(-z)); }

__global__ void tanh_kernel(const float* in, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = tanhf(in[i]);
    }
}

__global__ void sigmoid_kernel(const float* in, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sigmoid_device(in[i]);
    }
}

__global__ void silu_kernel(const float* in, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * sigmoid_device(in[i]);
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

}  // namespace

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
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<grid_size, block_size, 0, stream_>>>(static_cast<float*>(ptr), value, n);
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
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    switch (op) {
        case ElementwiseOp::Relu:
            relu_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, n);
            break;
        case ElementwiseOp::Neg:
            neg_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, n);
            break;
        case ElementwiseOp::Tanh:
            tanh_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, n);
            break;
        case ElementwiseOp::Sigmoid:
            sigmoid_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, n);
            break;
        case ElementwiseOp::Silu:
            silu_kernel<<<grid_size, block_size, 0, stream_>>>(in, out, n);
            break;
    }
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::add(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    add_kernel<<<grid_size, block_size, 0, stream_>>>(a, b, out, n);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::mul(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    mul_kernel<<<grid_size, block_size, 0, stream_>>>(a, b, out, n);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace pulsatrix
