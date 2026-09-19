#include "exai/cuda_backend.hpp"

#include <stdexcept>

#include "exai/cuda_check.hpp"

namespace exai {

namespace {

__global__ void fill_kernel(float* ptr, float value, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[i] = value;
    }
}

}  // namespace

CUDABackend::CUDABackend() {
    EXAI_CUDA_CHECK(cudaStreamCreate(&stream_));
}

CUDABackend::~CUDABackend() {
    cudaStreamDestroy(stream_);
}

void* CUDABackend::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    EXAI_CUDA_CHECK(cudaMalloc(&ptr, bytes));
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
    EXAI_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream_));
    EXAI_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::fill(void* ptr, float value, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<grid_size, block_size, 0, stream_>>>(static_cast<float*>(ptr), value, n);
    EXAI_CUDA_CHECK(cudaGetLastError());
    EXAI_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::gemm(const float*, const float*, float*, size_t, size_t, size_t) {
    throw std::logic_error("CUDABackend::gemm: not yet implemented (Phase 1.5 Mission 1)");
}

void CUDABackend::elementwise(ElementwiseOp, const float*, float*, size_t) {
    throw std::logic_error("CUDABackend::elementwise: not yet implemented (Phase 1.5 Mission 1)");
}

void CUDABackend::add(const float*, const float*, float*, size_t) {
    throw std::logic_error("CUDABackend::add: not yet implemented (Phase 1.5 Mission 1)");
}

}  // namespace exai
