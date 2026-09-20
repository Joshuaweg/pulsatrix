#include "exai/hip_backend.hpp"

#include <stdexcept>

#include "exai/hip_check.hpp"

namespace exai {

namespace {

__global__ void fill_kernel(float* ptr, float value, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[i] = value;
    }
}

}  // namespace

HIPBackend::HIPBackend() : stream_(nullptr), hipblas_handle_(nullptr) {
    EXAI_HIP_CHECK(hipStreamCreate(&stream_));
}

HIPBackend::~HIPBackend() {
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
    EXAI_HIP_CHECK(hipMalloc(&ptr, bytes));
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
    EXAI_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kind, stream_));
    EXAI_HIP_CHECK(hipStreamSynchronize(stream_));
}

void HIPBackend::fill(void* ptr, float value, size_t n) {
    if (n == 0) {
        return;
    }
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    fill_kernel<<<grid_size, block_size, 0, stream_>>>(static_cast<float*>(ptr), value, n);
    EXAI_HIP_CHECK(hipGetLastError());
    EXAI_HIP_CHECK(hipStreamSynchronize(stream_));
}

// gemm/elementwise/add are Objective 1.2 and 1.3's scope -- staged as throwing stubs rather
// than silently returning, mirroring how Phase 1.5 Mission 0 staged CUDABackend's. A stub
// that throws is the honest state: the method exists because DeviceBackend requires it, and
// calling it before it is implemented is a loud failure, not wrong numbers.
void HIPBackend::gemm(const float*, const float*, float*, size_t, size_t, size_t) {
    throw std::runtime_error("HIPBackend::gemm: not implemented until Phase 1.6 Mission 1 Objective 2");
}

void HIPBackend::elementwise(ElementwiseOp, const float*, float*, size_t) {
    throw std::runtime_error(
        "HIPBackend::elementwise: not implemented until Phase 1.6 Mission 1 Objective 3");
}

void HIPBackend::add(const float*, const float*, float*, size_t) {
    throw std::runtime_error("HIPBackend::add: not implemented until Phase 1.6 Mission 1 Objective 3");
}

}  // namespace exai
