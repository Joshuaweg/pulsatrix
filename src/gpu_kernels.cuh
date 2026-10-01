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

#include "pointwise_math.hpp"
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

__global__ void exp_kernel(const float* in, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = expf(in[i]);
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
        case ElementwiseOp::Exp:
            exp_kernel<<<grid, kBlockSize, 0, stream>>>(in, out, n);
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

// ---- GPU-native-kernels Mission 1 ---------------------------------------------------------
// Reductions are deliberately one thread per output (row or column) looping in CPUBackend's
// order, or a single fixed-shape block for dot: deterministic run to run, no atomics. Faster
// block-cooperative reductions can replace these later behind the same interface.

__global__ void column_sums_kernel(const float* in, float* out, size_t rows, size_t cols, float beta) {
    size_t j = global_index();
    if (j < cols) {
        float acc = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            acc += in[i * cols + j];
        }
        out[j] = (beta == 0.0f) ? acc : beta * out[j] + acc;
    }
}

__global__ void add_row_vector_kernel(const float* in, const float* row, float* out, size_t rows, size_t cols) {
    size_t idx = global_index();
    if (idx < rows * cols) {
        out[idx] = in[idx] + row[idx % cols];
    }
}

// op is uniform across the launch, so the switch costs no divergence.
__global__ void elementwise_backward_kernel(int op, const float* x, const float* grad_out, float* grad_in, size_t n) {
    size_t i = global_index();
    if (i < n) {
        const float xi = x[i];
        float d = 0.0f;
        switch (static_cast<ElementwiseOp>(op)) {
            case ElementwiseOp::Relu:
                // Select, not multiply -- see CPUBackend::elementwise_backward.
                grad_in[i] = xi > 0.0f ? grad_out[i] : 0.0f;
                return;
            case ElementwiseOp::Neg:
                d = -1.0f;
                break;
            case ElementwiseOp::Tanh: {
                const float t = tanhf(xi);
                d = 1.0f - t * t;
                break;
            }
            case ElementwiseOp::Sigmoid: {
                const float sg = sigmoid(xi);
                d = sg * (1.0f - sg);
                break;
            }
            case ElementwiseOp::Silu: {
                const float sg = sigmoid(xi);
                d = sg + xi * sg * (1.0f - sg);
                break;
            }
            case ElementwiseOp::Exp:
                d = expf(xi);
                break;
        }
        grad_in[i] = grad_out[i] * d;
    }
}

__global__ void axpby_kernel(float alpha, const float* x, float beta, const float* y, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        // beta == 0 does not read y (uniform branch) -- see DeviceBackend::axpby.
        out[i] = (beta == 0.0f) ? alpha * x[i] : alpha * x[i] + beta * y[i];
    }
}

// Single block of kBlockSize threads: each thread sums a fixed stride, then a fixed tree.
__global__ void dot_kernel(const float* a, const float* b, size_t n, float* result) {
    __shared__ float partial[kBlockSize];
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += kBlockSize) {
        acc += a[i] * b[i];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *result = partial[0];
    }
}

__global__ void softmax_rows_kernel(const float* in, float* out, size_t rows, size_t cols) {
    size_t r = global_index();
    if (r < rows) {
        const float* x = in + r * cols;
        float* y = out + r * cols;
        float row_max = x[0];
        for (size_t j = 1; j < cols; ++j) {
            row_max = fmaxf(row_max, x[j]);
        }
        float exp_sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            const float e = expf(x[j] - row_max);
            y[j] = e;
            exp_sum += e;
        }
        for (size_t j = 0; j < cols; ++j) {
            y[j] /= exp_sum;
        }
    }
}

__global__ void softmax_rows_backward_kernel(const float* y, const float* dy, float* dx, size_t rows, size_t cols) {
    size_t r = global_index();
    if (r < rows) {
        const size_t base = r * cols;
        float dot_yd = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            dot_yd += y[base + j] * dy[base + j];
        }
        for (size_t j = 0; j < cols; ++j) {
            dx[base + j] = y[base + j] * (dy[base + j] - dot_yd);
        }
    }
}

__global__ void logsumexp_rows_kernel(const float* in, float* out, size_t rows, size_t cols) {
    size_t r = global_index();
    if (r < rows) {
        const float* x = in + r * cols;
        float row_max = x[0];
        for (size_t j = 1; j < cols; ++j) {
            row_max = fmaxf(row_max, x[j]);
        }
        float exp_sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            exp_sum += expf(x[j] - row_max);
        }
        out[r] = row_max + logf(exp_sum);
    }
}

__global__ void adam_step_kernel(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1,
                                 float beta2, float eps, float bias_correction1, float bias_correction2) {
    size_t i = global_index();
    if (i < n) {
        const float g = grad[i];
        const float mi = beta1 * m[i] + (1.0f - beta1) * g;
        const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;
        const float m_hat = mi / bias_correction1;
        const float v_hat = vi / bias_correction2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

template <typename Stream>
void launch_column_sums(const float* in, float* out, size_t rows, size_t cols, float beta, Stream stream) {
    column_sums_kernel<<<grid_size_for(cols), kBlockSize, 0, stream>>>(in, out, rows, cols, beta);
}

template <typename Stream>
void launch_add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols, Stream stream) {
    add_row_vector_kernel<<<grid_size_for(rows * cols), kBlockSize, 0, stream>>>(in, row, out, rows, cols);
}

template <typename Stream>
void launch_elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in, size_t n,
                                 Stream stream) {
    elementwise_backward_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(static_cast<int>(op), x, grad_out,
                                                                            grad_in, n);
}

template <typename Stream>
void launch_axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n, Stream stream) {
    axpby_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(alpha, x, beta, y, out, n);
}

// result is a one-float device buffer owned by the backend.
template <typename Stream>
void launch_dot(const float* a, const float* b, size_t n, float* result, Stream stream) {
    dot_kernel<<<1, kBlockSize, 0, stream>>>(a, b, n, result);
}

template <typename Stream>
void launch_softmax_rows(const float* in, float* out, size_t rows, size_t cols, Stream stream) {
    softmax_rows_kernel<<<grid_size_for(rows), kBlockSize, 0, stream>>>(in, out, rows, cols);
}

template <typename Stream>
void launch_softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols,
                                  Stream stream) {
    softmax_rows_backward_kernel<<<grid_size_for(rows), kBlockSize, 0, stream>>>(y, dy, dx, rows, cols);
}

template <typename Stream>
void launch_logsumexp_rows(const float* in, float* out, size_t rows, size_t cols, Stream stream) {
    logsumexp_rows_kernel<<<grid_size_for(rows), kBlockSize, 0, stream>>>(in, out, rows, cols);
}

template <typename Stream>
void launch_adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1,
                      float beta2, float eps, float bias_correction1, float bias_correction2, Stream stream) {
    adam_step_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(param, grad, m, v, n, lr, beta1, beta2, eps,
                                                                 bias_correction1, bias_correction2);
}

// ---- GPU-native-kernels Mission 1b ---------------------------------------------------------

// Same single-block fixed-order tree as dot_kernel.
__global__ void sum_kernel(const float* in, size_t n, float* result) {
    __shared__ float partial[kBlockSize];
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += kBlockSize) {
        acc += in[i];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *result = partial[0];
    }
}

__global__ void dropout_forward_kernel(const float* in, float* out, float* mask, size_t n, float p, float scale,
                                       uint64_t seed, uint64_t offset) {
    size_t i = global_index();
    if (i < n) {
        const bool keep = !(pointwise::counter_uniform(seed, offset + i) < p);
        mask[i] = keep ? 1.0f : 0.0f;
        out[i] = keep ? in[i] * scale : 0.0f;
    }
}

__global__ void bce_with_logits_kernel(const float* logits, const float* target, float* out, size_t n) {
    size_t i = global_index();
    if (i < n) {
        out[i] = pointwise::bce_with_logits_term(logits[i], target[i]);
    }
}

__global__ void bce_with_logits_grad_kernel(const float* logits, const float* target, float* grad, size_t n,
                                            float scale) {
    size_t i = global_index();
    if (i < n) {
        grad[i] = (pointwise::stable_sigmoid(logits[i]) - target[i]) * scale;
    }
}

template <typename Stream>
void launch_sum(const float* in, size_t n, float* result, Stream stream) {
    sum_kernel<<<1, kBlockSize, 0, stream>>>(in, n, result);
}

template <typename Stream>
void launch_dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale, uint64_t seed,
                            uint64_t offset, Stream stream) {
    dropout_forward_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(in, out, mask, n, p, scale, seed, offset);
}

template <typename Stream>
void launch_bce_with_logits(const float* logits, const float* target, float* out, size_t n, Stream stream) {
    bce_with_logits_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(logits, target, out, n);
}

template <typename Stream>
void launch_bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n, float scale,
                                 Stream stream) {
    bce_with_logits_grad_kernel<<<grid_size_for(n), kBlockSize, 0, stream>>>(logits, target, grad, n, scale);
}

}  // namespace gpu
}  // namespace
}  // namespace pulsatrix
