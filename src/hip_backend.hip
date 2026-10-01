#include "pulsatrix/hip_backend.hpp"

#include <stdexcept>

#include "pulsatrix/hip_check.hpp"
#include "pulsatrix/hipblas_check.hpp"

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
// mirroring CUDABackend's sigmoid_device() byte-for-byte so both GPU backends compute
// Sigmoid and Silu from the identical expression CPUBackend uses.
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

HIPBackend::HIPBackend() : stream_(nullptr), hipblas_handle_(nullptr) {
    PULSATRIX_HIP_CHECK(hipStreamCreate(&stream_));
    PULSATRIX_HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
    PULSATRIX_HIPBLAS_CHECK(hipblasSetStream(hipblas_handle_, stream_));
}

HIPBackend::~HIPBackend() {
    static_cast<void>(hipblasDestroy(hipblas_handle_));
    // Discarded deliberately: a destructor cannot throw, and there is no recovery from a
    // failed teardown. HIP marks these [[nodiscard]] where the CUDA runtime does not, so
    // unlike CUDABackend the discard has to be explicit to keep the build warning-clean.
    static_cast<void>(hipStreamDestroy(stream_));
}

void* HIPBackend::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    PULSATRIX_HIP_CHECK(hipMalloc(&ptr, bytes));
    return ptr;
}

void HIPBackend::free(void* ptr) noexcept {
    // noexcept by DeviceBackend's contract, so the status is discarded deliberately -- see
    // the destructor's note on HIP's [[nodiscard]] divergence from CUDA.
    static_cast<void>(hipFree(ptr));
}

void HIPBackend::copy(void* dst, const void* src, size_t bytes, CopyDirection dir) {
    if (bytes == 0) {
        return;
    }
    hipMemcpyKind kind = hipMemcpyDefault;
    switch (dir) {
        case CopyDirection::HostToDevice:
            kind = hipMemcpyHostToDevice;
            break;
        case CopyDirection::DeviceToHost:
            kind = hipMemcpyDeviceToHost;
            break;
        case CopyDirection::DeviceToDevice:
            kind = hipMemcpyDeviceToDevice;
            break;
        case CopyDirection::HostToHost:
            kind = hipMemcpyHostToHost;
            break;
    }
    PULSATRIX_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream_));
    PULSATRIX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void HIPBackend::fill(void* ptr, float value, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<grid_size, block_size, 0, stream_>>>(static_cast<float*>(ptr), value, n);
    PULSATRIX_HIP_CHECK(hipGetLastError());
    PULSATRIX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void HIPBackend::gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) {
    // Row-major A(m,k)*B(k,n)=C(m,n) via hipBLAS (column-major): compute C^T = B^T*A^T
    // instead, which hipBLAS computes correctly as a column-major op, and a row-major
    // C(m,n) buffer is the same bytes as a column-major C^T(n,m) buffer. See
    // gpu_backend_programming/context_gpu_cublas_cudnn_integration.md's Column-Major Trap.
    // Argument order is identical to CUDABackend::gemm's cublasSgemm call -- hipBLAS
    // inherits cuBLAS's column-major convention and its parameter list, so this is the same
    // transformation, not a re-derivation. Verified on hardware against a hand-computed
    // non-square product, not just a square one, since a square case cannot distinguish a
    // correct result from a transposed one.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    PULSATRIX_HIPBLAS_CHECK(hipblasSgemm(hipblas_handle_, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(n),
                                     static_cast<int>(m), static_cast<int>(k), &alpha, b,
                                     static_cast<int>(n), a, static_cast<int>(k), &beta, out,
                                     static_cast<int>(n)));
}

void HIPBackend::elementwise(ElementwiseOp op, const float* in, float* out, size_t n) {
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
    PULSATRIX_HIP_CHECK(hipGetLastError());
    PULSATRIX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void HIPBackend::add(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    add_kernel<<<grid_size, block_size, 0, stream_>>>(a, b, out, n);
    PULSATRIX_HIP_CHECK(hipGetLastError());
    PULSATRIX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void HIPBackend::mul(const float* a, const float* b, float* out, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    mul_kernel<<<grid_size, block_size, 0, stream_>>>(a, b, out, n);
    PULSATRIX_HIP_CHECK(hipGetLastError());
    PULSATRIX_HIP_CHECK(hipStreamSynchronize(stream_));
}

}  // namespace pulsatrix
