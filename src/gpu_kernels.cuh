// Device kernels shared by CUDABackend (src/cuda_backend.cu, nvcc) and HIPBackend
// (src/hip_backend.hip, hipcc). Private to src/ -- never included by a public header.
//
// One source for both vendors is deliberate: this project has real AMD hardware (gfx1151)
// to run tests on but no NVIDIA node, so the HIP suite executing *this exact source* plus the
// CUDA compile-only CI job (scripts/ci-gpu-compile.sh) is the strongest CUDA verification
// available. It also removes the duplicated per-op dispatch that let HIPBackend silently skip
// Tanh/Sigmoid/Silu when ElementwiseOp grew (fixed in 87104e1).
//
// Only the CUDA/HIP common subset is used here: __global__/__device__, blockIdx/blockDim/
// threadIdx, <<<>>> launches, and single-precision libm (expf, tanhf). Vendor runtime calls
// (error checks, stream sync) stay in each backend. Everything sits in an anonymous namespace
// so a build with both backends enabled links two independent copies, not a duplicate symbol.
#pragma once

#include <cstddef>
#include <stdexcept>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {
namespace gpu {

constexpr int kBlockSize = 256;

inline int grid_size_for(size_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

__device__ inline size_t global_index() { return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; }

// Logistic sigmoid -- one definition shared by Sigmoid and Silu, the same expression
// CPUBackend uses, so all three backends compute both from 1 / (1 + exp(-z)).
__device__ inline float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

__global__ void fill_kernel(float* ptr, float value, size_t n) {
    size_t i = global_index();
    if (i < n) {
        ptr[i] = value;
    }
}

// One kernel per ElementwiseOp rather than a runtime switch inside the kernel: the op is
// uniform across the launch, so selecting the kernel on the host keeps every thread's body
// branch-free. Each kernel reads in[i] once before writing out[i], so in == out is safe.
__global__ void relu_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

__global__ void neg_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = -in[i];
    }
}

__global__ void tanh_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = tanhf(in[i]);
    }
}

__global__ void sigmoid_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = sigmoid(in[i]);
    }
}

__global__ void silu_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        float x = in[i];
        out[i] = x * sigmoid(x);
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

// Launchers. Stream is cudaStream_t or hipStream_t; both work with the <<<>>> syntax. The
// caller checks the launch error and synchronizes with its own vendor runtime. n == 0 is
// handled by the caller (a zero-sized grid is an invalid launch on both runtimes).

template <typename Stream>
void launch_fill(float* ptr, float value, size_t n, Stream stream) {
    fill_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(ptr, value, n);
}

template <typename Stream>
void launch_elementwise(ElementwiseOp op, const float* in, float* out, size_t n, Stream stream) {
    const int grid = grid_size_for(n);
    switch (op) {
        case ElementwiseOp::Relu:
            relu_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
            return;
        case ElementwiseOp::Neg:
            neg_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
            return;
        case ElementwiseOp::Tanh:
            tanh_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
            return;
        case ElementwiseOp::Sigmoid:
            sigmoid_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
            return;
        case ElementwiseOp::Silu:
            silu_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
            return;
    }
    // Reached only if ElementwiseOp gains a value this switch doesn't handle -- fail loudly
    // rather than leaving out untouched, the exact failure mode this header exists to prevent.
    throw std::invalid_argument("gpu::launch_elementwise: unhandled ElementwiseOp");
}

template <typename Stream>
void launch_add(const float* a, const float* b, float* out, size_t n, Stream stream) {
    add_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(a, b, out, n);
}

template <typename Stream>
void launch_mul(const float* a, const float* b, float* out, size_t n, Stream stream) {
    mul_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(a, b, out, n);
}

}  // namespace gpu
}  // namespace
}  // namespace pulsatrix
