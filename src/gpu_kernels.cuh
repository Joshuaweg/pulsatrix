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

#include "cnn_math.hpp"
#include "lrp_math.hpp"
#include "pointwise_math.hpp"
#include "recurrent_math.hpp"
#include "rl_math.hpp"
#include "row_math.hpp"
#include "ssm_math.hpp"
#include "top_k_math.hpp"
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
// HIP-5: dot and sum as two deterministic passes. Pass 1 runs reduce_blocks_for(n) blocks, each
// grid-striding over the input and tree-reducing its share to one partial; pass 2 is one block
// tree-reducing the partials in index order. The partition depends only on n, never on the
// hardware, and there are no atomics, so the result is bit-identical run to run.
constexpr size_t kMaxReduceBlocks = 1024;
// Scratch a backend keeps for dot/sum: the result at [0], the partials after it.
constexpr size_t kReduceScratchFloats = 1 + kMaxReduceBlocks;

inline size_t reduce_blocks_for(size_t n) {
    // About four grid-stride iterations per thread before the cap.
    const size_t blocks = (n + 4 * kBlockSize - 1) / (4 * kBlockSize);
    return blocks < 1 ? 1 : (blocks > kMaxReduceBlocks ? kMaxReduceBlocks : blocks);
}

// The block's threads' values, summed by a fixed tree into thread 0's return value.
__device__ inline float block_tree_sum(float acc) {
    __shared__ float partial[kBlockSize];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return partial[0];
}

__global__ void sum_kernel(const float* in, size_t n, float* result);  // defined below

__global__ void dot_partials_kernel(const float* a, const float* b, size_t n, float* partials) {
    float acc = 0.0f;
    for (size_t i = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * kBlockSize) {
        acc += a[i] * b[i];
    }
    const float total = block_tree_sum(acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
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

// scratch is the backend's kReduceScratchFloats-float device buffer; the result lands in scratch[0].
template <typename Stream>
void launch_dot(const float* a, const float* b, size_t n, float* scratch, Stream stream) {
    const size_t blocks = reduce_blocks_for(n);
    dot_partials_kernel<<<static_cast<unsigned>(blocks), kBlockSize, 0, stream>>>(a, b, n, scratch + 1);
    sum_kernel<<<1, kBlockSize, 0, stream>>>(scratch + 1, blocks, scratch);
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

// One block: thread t sums elements t, t + 256, ...; then the fixed tree. Pass 2 of dot and sum,
// and pass 1's per-block work for sum via sum_partials_kernel.
__global__ void sum_kernel(const float* in, size_t n, float* result) {
    float acc = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += kBlockSize) {
        acc += in[i];
    }
    const float total = block_tree_sum(acc);
    if (threadIdx.x == 0) {
        *result = total;
    }
}

__global__ void sum_partials_kernel(const float* in, size_t n, float* partials) {
    float acc = 0.0f;
    for (size_t i = static_cast<size_t>(blockIdx.x) * kBlockSize + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * kBlockSize) {
        acc += in[i];
    }
    const float total = block_tree_sum(acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
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
void launch_sum(const float* in, size_t n, float* scratch, Stream stream) {
    const size_t blocks = reduce_blocks_for(n);
    sum_partials_kernel<<<static_cast<unsigned>(blocks), kBlockSize, 0, stream>>>(in, n, scratch + 1);
    sum_kernel<<<1, kBlockSize, 0, stream>>>(scratch + 1, blocks, scratch);
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

// ---- GPU-native-kernels Mission 2 ----------------------------------------------------------
// One thread per row running the shared rows:: routine (row_math.hpp), except the pure data
// movement kernels (permute, gather), which are one thread per element, and scatter_add,
// one thread per column so repeated indices accumulate in CPU order without atomics.

__global__ void layer_norm_forward_kernel(const float* in, const float* gamma, const float* beta, float* xhat,
                                          float* out, float* row_std, size_t rows, size_t cols, float eps) {
    size_t r = global_index();
    if (r < rows) {
        rows::layer_norm_forward(in + r * cols, gamma, beta, xhat + r * cols, out + r * cols, row_std + r,
                                 static_cast<int64_t>(cols), eps);
    }
}

__global__ void layer_norm_backward_kernel(const float* grad_out, const float* gamma, const float* xhat,
                                           const float* row_std, float* grad_in, size_t rows, size_t cols) {
    size_t r = global_index();
    if (r < rows) {
        rows::layer_norm_backward(grad_out + r * cols, gamma, xhat + r * cols, row_std[r], grad_in + r * cols,
                                  static_cast<int64_t>(cols));
    }
}

__global__ void rms_norm_forward_kernel(const float* in, const float* gamma, float* out, float* row_rms, size_t rows,
                                        size_t cols, float eps) {
    size_t r = global_index();
    if (r < rows) {
        rows::rms_norm_forward(in + r * cols, gamma, out + r * cols, row_rms + r, static_cast<int64_t>(cols), eps);
    }
}

__global__ void rms_norm_backward_kernel(const float* grad_out, const float* gamma, const float* in,
                                         const float* row_rms, float* grad_in, float* gamma_terms, size_t rows,
                                         size_t cols) {
    size_t r = global_index();
    if (r < rows) {
        rows::rms_norm_backward(grad_out + r * cols, gamma, in + r * cols, row_rms[r], grad_in + r * cols,
                                gamma_terms + r * cols, static_cast<int64_t>(cols));
    }
}

__global__ void rope_rotate_kernel(const float* in, const float* cos_table, const float* sin_table, float* out,
                                   size_t total_rows, size_t seq_len, size_t head_dim, bool inverse) {
    size_t row = global_index();
    if (row < total_rows) {
        const size_t half = head_dim / 2;
        const size_t pos = row % seq_len;
        rows::rope_rotate(in + row * head_dim, cos_table + pos * half, sin_table + pos * half, out + row * head_dim,
                          static_cast<int64_t>(half), inverse);
    }
}

__global__ void permute_0213_kernel(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) {
    size_t idx = global_index();  // flat index into out (d0, d2, d1, d3)
    if (idx < d0 * d1 * d2 * d3) {
        const size_t e = idx % d3;
        size_t rest = idx / d3;
        const size_t b = rest % d1;
        rest /= d1;
        const size_t c = rest % d2;
        const size_t a = rest / d2;
        out[idx] = in[((a * d1 + b) * d2 + c) * d3 + e];
    }
}

__global__ void gather_rows_kernel(const float* table, const float* indices, float* out, size_t count, size_t dim) {
    size_t idx = global_index();
    if (idx < count * dim) {
        const size_t i = idx / dim;
        const size_t d = idx % dim;
        out[idx] = table[static_cast<size_t>(indices[i]) * dim + d];
    }
}

__global__ void scatter_add_rows_kernel(const float* src, const float* indices, float* table, size_t count,
                                        size_t dim) {
    size_t d = global_index();
    if (d < dim) {
        for (size_t i = 0; i < count; ++i) {
            table[static_cast<size_t>(indices[i]) * dim + d] += src[i * dim + d];
        }
    }
}

__global__ void tanh_gaussian_forward_kernel(const float* mean, const float* log_std, const float* eps, float* action,
                                             float* std_cache, float* log_prob, size_t rows, size_t cols,
                                             float stabilizer, double half_log_two_pi) {
    size_t r = global_index();
    if (r < rows) {
        const size_t off = r * cols;
        rows::tanh_gaussian_forward(mean + off, log_std + off, eps + off, action + off, std_cache + off, log_prob + r,
                                    static_cast<int64_t>(cols), stabilizer, half_log_two_pi);
    }
}

__global__ void tanh_gaussian_backward_kernel(const float* action, const float* std_cache, const float* eps,
                                              const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                              float* grad_log_std, size_t n, float stabilizer) {
    size_t i = global_index();
    if (i < n) {
        rows::tanh_gaussian_backward_element(action[i], std_cache[i], eps[i], grad_action[i], grad_log_prob[i],
                                             stabilizer, grad_mean + i, grad_log_std + i);
    }
}

// ---- GPU-native-kernels Mission 3 ----------------------------------------------------------
// One thread per output element (or row / column), each running the shared lrp:: routine.

__global__ void lrp_linear_kernel(const float* x, const float* w, const float* z, const float* r, float* r_in,
                                  size_t rows, size_t in_features, size_t out_features, float eps) {
    size_t idx = global_index();
    if (idx < rows * in_features) {
        const size_t n = idx / in_features;
        const size_t i = idx % in_features;
        r_in[idx] = lrp::linear_epsilon(x + n * in_features, w, z + n * out_features, r + n * out_features,
                                        static_cast<int64_t>(i), static_cast<int64_t>(out_features), eps);
    }
}

__global__ void lrp_residual_split_kernel(const float* a, const float* b, const float* r, float* r_a, float* r_b,
                                          size_t n, float eps) {
    size_t i = global_index();
    if (i < n) {
        lrp::residual_split(a[i], b[i], r[i], eps, r_a + i, r_b + i);
    }
}

__global__ void lrp_bilinear_elementwise_kernel(const float* a, const float* b, const float* r, float* r_out,
                                                size_t n, float eps) {
    size_t i = global_index();
    if (i < n) {
        r_out[i] = lrp::bilinear_elementwise(a[i], b[i], r[i], eps);
    }
}

__global__ void lrp_bilinear_r_a_kernel(const float* a, const float* b, const float* o, const float* r_o, float* r_a,
                                        size_t slices, size_t m, size_t p, size_t q, float eps, bool b_transposed) {
    size_t idx = global_index();
    if (idx < slices * m * p) {
        const size_t s = idx / (m * p);
        const size_t rem = idx % (m * p);
        r_a[idx] = lrp::bilinear_matmul_r_a(a + s * m * p, b + s * p * q, o + s * m * q, r_o + s * m * q,
                                            static_cast<int64_t>(rem / p), static_cast<int64_t>(rem % p),
                                            static_cast<int64_t>(p), static_cast<int64_t>(q), eps, b_transposed);
    }
}

__global__ void lrp_bilinear_r_b_kernel(const float* a, const float* b, const float* o, const float* r_o, float* r_b,
                                        size_t slices, size_t m, size_t p, size_t q, float eps, bool b_transposed) {
    size_t idx = global_index();  // index into r_b in its stored layout
    if (idx < slices * p * q) {
        const size_t s = idx / (p * q);
        const size_t rem = idx % (p * q);
        const size_t j = b_transposed ? rem % p : rem / q;
        const size_t k = b_transposed ? rem / p : rem % q;
        r_b[idx] = lrp::bilinear_matmul_r_b(a + s * m * p, b + s * p * q, o + s * m * q, r_o + s * m * q,
                                            static_cast<int64_t>(j), static_cast<int64_t>(k), static_cast<int64_t>(m),
                                            static_cast<int64_t>(p), static_cast<int64_t>(q), eps, b_transposed);
    }
}

__global__ void lrp_softmax_rows_kernel(const float* x, const float* y, const float* r, float* r_in, size_t rows,
                                        size_t cols) {
    size_t row = global_index();
    if (row < rows) {
        const size_t off = row * cols;
        lrp::softmax_row(x + off, y + off, r + off, r_in + off, static_cast<int64_t>(cols));
    }
}

__global__ void lrp_rope_kernel(const float* x, const float* y, const float* r, const float* cos_table,
                                const float* sin_table, float* r_in, size_t total_rows, size_t seq_len,
                                size_t head_dim, float eps) {
    size_t row = global_index();
    if (row < total_rows) {
        const size_t half = head_dim / 2;
        const size_t pos = row % seq_len;
        const size_t off = row * head_dim;
        lrp::rope_position(x + off, y + off, r + off, cos_table + pos * half, sin_table + pos * half, r_in + off,
                           static_cast<int64_t>(half), eps);
    }
}

// op is uniform across the launch, so the switch costs no divergence.
__global__ void logic_pointwise_kernel(int op, int norm, const float* a, const float* b, const float* g_or_r,
                                       const float* y, float* out_a, float* out_b, size_t n, float eps) {
    size_t i = global_index();
    if (i < n) {
        switch (static_cast<LogicOp>(op)) {
            case LogicOp::ConjunctionForward:
                out_a[i] = lrp::conjunction_forward(norm, a[i], b[i]);
                break;
            case LogicOp::ConjunctionBackward:
                lrp::conjunction_backward(norm, a[i], b[i], g_or_r[i], out_a + i, out_b + i);
                break;
            case LogicOp::ConjunctionLrp:
                lrp::conjunction_lrp(norm, a[i], b[i], y[i], g_or_r[i], eps, out_a + i, out_b + i);
                break;
            case LogicOp::DisjunctionForward:
                out_a[i] = lrp::disjunction_forward(norm, a[i], b[i]);
                break;
            case LogicOp::DisjunctionBackward:
                lrp::disjunction_backward(norm, a[i], b[i], g_or_r[i], out_a + i, out_b + i);
                break;
            case LogicOp::DisjunctionLrp:
                lrp::disjunction_lrp(norm, a[i], b[i], y[i], g_or_r[i], eps, out_a + i, out_b + i);
                break;
        }
    }
}

__global__ void aggregator_forward_kernel(const float* x, float* mean_pow, float* out, size_t n, size_t cols,
                                          float p) {
    size_t j = global_index();
    if (j < cols) {
        lrp::aggregator_forward_column(x, mean_pow, out, static_cast<int64_t>(n), static_cast<int64_t>(cols),
                                       static_cast<int64_t>(j), p);
    }
}

__global__ void aggregator_backward_kernel(const float* x, const float* mean_pow, const float* grad_out,
                                           float* grad_in, size_t n, size_t cols, float p) {
    size_t j = global_index();
    if (j < cols) {
        lrp::aggregator_backward_column(x, mean_pow, grad_out, grad_in, static_cast<int64_t>(n),
                                        static_cast<int64_t>(cols), static_cast<int64_t>(j), p);
    }
}

__global__ void aggregator_lrp_kernel(const float* x, const float* mean_pow, const float* r_out, float* r_in,
                                      size_t n, size_t cols, float p, float eps) {
    size_t j = global_index();
    if (j < cols) {
        lrp::aggregator_lrp_column(x, mean_pow, r_out, r_in, static_cast<int64_t>(n), static_cast<int64_t>(cols),
                                   static_cast<int64_t>(j), p, eps);
    }
}

// ---- GPU-native-kernels Mission 4 ----------------------------------------------------------

__global__ void im2col_kernel(const float* in, float* col, size_t n, size_t c, size_t h, size_t w, ConvGeometry g,
                              size_t out_h, size_t out_w) {
    const size_t P = c * g.kh * g.kw, Q = out_h * out_w;
    size_t idx = global_index();
    if (idx < n * P * Q) {
        const size_t e = idx / (P * Q);
        const size_t p = (idx / Q) % P;
        const size_t q = idx % Q;
        col[idx] = cnn::im2col_element(in + e * c * h * w, static_cast<int64_t>(h), static_cast<int64_t>(w), g,
                                       static_cast<int64_t>(out_w), static_cast<int64_t>(p), static_cast<int64_t>(q));
    }
}

__global__ void col2im_add_kernel(const float* col, float* out, size_t n, size_t c, size_t h, size_t w, ConvGeometry g,
                                  size_t out_h, size_t out_w) {
    const size_t P = c * g.kh * g.kw, Q = out_h * out_w;
    size_t idx = global_index();  // flat (n, c, h, w) pixel
    if (idx < n * c * h * w) {
        const size_t iw = idx % w;
        const size_t ih = (idx / w) % h;
        const size_t ch = (idx / (w * h)) % c;
        const size_t e = idx / (w * h * c);
        out[idx] = cnn::col2im_pixel(col + e * P * Q, out[idx], static_cast<int64_t>(h), static_cast<int64_t>(w), g,
                                     static_cast<int64_t>(ch), static_cast<int64_t>(ih), static_cast<int64_t>(iw));
    }
}

__global__ void add_channel_vector_kernel(const float* in, const float* vec, float* out, size_t n, size_t c,
                                          size_t inner) {
    size_t idx = global_index();
    if (idx < n * c * inner) {
        out[idx] = in[idx] + vec[(idx / inner) % c];
    }
}

__global__ void lrp_conv_kernel(const float* col, const float* kernel, const float* pre_bias, const float* r,
                                float* r_col, size_t n, size_t out_channels, size_t p, size_t q, float eps) {
    size_t idx = global_index();
    if (idx < n * p * q) {
        const size_t e = idx / (p * q);
        const size_t pi = (idx / q) % p;
        const size_t qi = idx % q;
        r_col[idx] = cnn::conv_lrp_col(col + e * p * q, kernel, pre_bias + e * out_channels * q,
                                       r + e * out_channels * q,
                                       static_cast<int64_t>(pi), static_cast<int64_t>(qi), static_cast<int64_t>(p),
                                       static_cast<int64_t>(q), static_cast<int64_t>(out_channels), eps);
    }
}

__global__ void lrp_stabilized_divide_kernel(const float* r, const float* denom, const float* gate, float* out,
                                             size_t n, float eps, int gate_mode) {
    size_t i = global_index();
    if (i < n) {
        out[i] = lrp::stabilized_divide(r[i], denom[i], gate_mode == 0 ? 0.0f : gate[i], gate_mode, eps);
    }
}

__global__ void max_pool_forward_kernel(const float* in, float* out, float* argmax, size_t planes, size_t h, size_t w,
                                        size_t kh, size_t kw) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    size_t o = global_index();
    if (o < planes * out_h * out_w) {
        const size_t pl = o / (out_h * out_w);
        const size_t oh = (o / out_w) % out_h;
        const size_t ow = o % out_w;
        cnn::max_pool_window(in + pl * h * w, static_cast<int64_t>(w), static_cast<int64_t>(kh),
                             static_cast<int64_t>(kw), static_cast<int64_t>(oh), static_cast<int64_t>(ow), out + o,
                             argmax + o);
    }
}

// Windows never overlap (stride == kernel), so each destination is written by at most one thread.
__global__ void max_unpool_kernel(const float* src, const float* argmax, float* dst, size_t planes, size_t h,
                                  size_t w, size_t out_plane) {
    size_t o = global_index();
    if (o < planes * out_plane) {
        const size_t pl = o / out_plane;
        dst[pl * h * w + static_cast<size_t>(argmax[o])] = src[o];
    }
}

__global__ void avg_pool_forward_kernel(const float* in, float* out, size_t planes, size_t h, size_t w, size_t kh,
                                        size_t kw) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    size_t o = global_index();
    if (o < planes * out_h * out_w) {
        const size_t pl = o / (out_h * out_w);
        out[o] = cnn::avg_pool_window(in + pl * h * w, static_cast<int64_t>(w), static_cast<int64_t>(kh),
                                      static_cast<int64_t>(kw), static_cast<int64_t>((o / out_w) % out_h),
                                      static_cast<int64_t>(o % out_w));
    }
}

__global__ void avg_pool_backward_kernel(const float* grad_out, float* grad_in, size_t planes, size_t h, size_t w,
                                         size_t kh, size_t kw) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    size_t o = global_index();
    if (o < planes * out_h * out_w) {
        const size_t pl = o / (out_h * out_w);
        const size_t oh = (o / out_w) % out_h;
        const size_t ow = o % out_w;
        const float g = grad_out[o] / static_cast<float>(kh * kw);
        for (size_t i = 0; i < kh; ++i) {
            for (size_t j = 0; j < kw; ++j) {
                grad_in[pl * h * w + (oh * kh + i) * w + (ow * kw + j)] = g;
            }
        }
    }
}

__global__ void lrp_avg_pool_kernel(const float* x, const float* r, float* r_in, size_t planes, size_t h, size_t w,
                                    size_t kh, size_t kw, float eps) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    size_t o = global_index();
    if (o < planes * out_h * out_w) {
        const size_t pl = o / (out_h * out_w);
        cnn::avg_pool_lrp_window(x + pl * h * w, r_in + pl * h * w, static_cast<int64_t>(w), static_cast<int64_t>(kh),
                                 static_cast<int64_t>(kw), static_cast<int64_t>((o / out_w) % out_h),
                                 static_cast<int64_t>(o % out_w), r[o], eps);
    }
}

// ---- HIP-2: BatchNorm as parallel per-channel reductions ------------------------------------
// The old kernels ran one thread per channel over all N * spatial elements of it: 3 or 16
// threads for millions of elements. Now every per-channel sum is a deterministic two-stage
// reduction: groups blocks per channel each reduce a strided share to a partial, then one thread
// per channel adds its partials in order. Normalization and the input gradient run one thread per
// element. No atomics, and the partition depends only on the shape, so results are bit-identical
// run to run. They differ from CPUBackend's sequential sums only at rounding level.

// Blocks per channel: about four grid-stride iterations per thread, at most kMaxReduceBlocks in all.
inline size_t bn_groups_for(size_t per_channel, size_t c) {
    const size_t want = (per_channel + 4 * kBlockSize - 1) / (4 * kBlockSize);
    const size_t cap = c >= kMaxReduceBlocks ? 1 : kMaxReduceBlocks / c;
    return want < 1 ? 1 : (want > cap ? cap : want);
}

// Scratch a backend provides: partials, then two per-channel vectors.
inline size_t bn_scratch_floats(size_t n, size_t c, size_t spatial) {
    return c * bn_groups_for(n * spatial, c) + 2 * c;
}

enum BnTerm : int { kBnX = 0, kBnSquaredDeviation = 1, kBnProduct = 2 };

// partials[ch * groups + g] = block g's share of the sum over channel ch of: a (kBnX),
// (a - mean[ch])^2 (kBnSquaredDeviation) or a * b (kBnProduct).
template <int Term>
__global__ void bn_channel_partials_kernel(const float* a, const float* b, const float* mean, size_t n, size_t c,
                                           size_t spatial, size_t groups, float* partials) {
    const size_t ch = blockIdx.x / groups, g = blockIdx.x % groups;
    const size_t per_channel = n * spatial;
    float acc = 0.0f;
    for (size_t j = g * kBlockSize + threadIdx.x; j < per_channel; j += groups * kBlockSize) {
        const size_t idx = (j / spatial) * c * spatial + ch * spatial + j % spatial;
        if (Term == kBnX) {
            acc += a[idx];
        } else if (Term == kBnSquaredDeviation) {
            const float d = a[idx] - mean[ch];
            acc += d * d;
        } else {
            acc += a[idx] * b[idx];
        }
    }
    const float total = block_tree_sum(acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total;
    }
}

// out[ch] = (partials of ch, added in order) / divisor.
__global__ void bn_combine_kernel(const float* partials, size_t c, size_t groups, float divisor, float* out) {
    const size_t ch = global_index();
    if (ch < c) {
        float s = 0.0f;
        for (size_t g = 0; g < groups; ++g) {
            s += partials[ch * groups + g];
        }
        out[ch] = s / divisor;
    }
}

__global__ void bn_std_kernel(const float* var, size_t c, float eps, float* std_out) {
    const size_t ch = global_index();
    if (ch < c) {
        std_out[ch] = sqrtf(var[ch] + eps);
    }
}

__global__ void bn_normalize_kernel(const float* in, const float* mean, const float* std_dev, const float* gamma,
                                    const float* beta, float* xhat, float* out, size_t total, size_t c,
                                    size_t spatial) {
    const size_t i = global_index();
    if (i < total) {
        const size_t ch = (i / spatial) % c;
        const float xh = (in[i] - mean[ch]) / std_dev[ch];
        xhat[i] = xh;
        out[i] = gamma[ch] * xh + beta[ch];
    }
}

// Training-mode input gradient from the per-channel sums gamma_grad = sum(g * xhat) and
// beta_grad = sum(g): CPUBackend's formula with sum(g * gamma) = gamma * sum(g).
__global__ void bn_input_grad_kernel(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* std_dev, const float* gamma_grad, const float* beta_grad,
                                     float* grad_in, size_t total, size_t c, size_t spatial, float m) {
    const size_t i = global_index();
    if (i < total) {
        const size_t ch = (i / spatial) % c;
        const float gxh = grad_out[i] * gamma[ch];
        const float sum_gxh = gamma[ch] * beta_grad[ch];
        const float sum_gxh_xh = gamma[ch] * gamma_grad[ch];
        grad_in[i] = (m * gxh - sum_gxh - xhat[i] * sum_gxh_xh) / (m * std_dev[ch]);
    }
}

// PyTorch's running-statistics rule (FND-5) from the batch mean and the summed squared deviation.
__global__ void bn_update_running_kernel(const float* mean, const float* squared_deviation, float* running_mean,
                                         float* running_var, size_t c, size_t count, float momentum) {
    const size_t ch = global_index();
    if (ch < c) {
        running_mean[ch] = (1.0f - momentum) * running_mean[ch] + momentum * mean[ch];
        if (count >= 2) {
            running_var[ch] = (1.0f - momentum) * running_var[ch] +
                              momentum * (squared_deviation[ch] / (static_cast<float>(count) - 1.0f));
        }
    }
}

__global__ void bn_eval_input_grad_kernel(const float* grad_out, const float* gamma, const float* std_dev,
                                          float* grad_in, size_t total, size_t c, size_t spatial) {
    const size_t i = global_index();
    if (i < total) {
        const size_t ch = (i / spatial) % c;
        grad_in[i] = grad_out[i] * (gamma[ch] / std_dev[ch]);
    }
}

// out[ch] = sum over channel ch of the Term values, divided by divisor; partials is scratch.
template <int Term, typename Stream>
void bn_reduce(const float* a, const float* b, const float* mean, size_t n, size_t c, size_t spatial,
               float* partials, float divisor, float* out, Stream stream) {
    const size_t groups = bn_groups_for(n * spatial, c);
    bn_channel_partials_kernel<Term><<<static_cast<unsigned>(c * groups), kBlockSize, 0, stream>>>(
        a, b, mean, n, c, spatial, groups, partials);
    bn_combine_kernel<<<grid_size_for(c), kBlockSize, 0, stream>>>(partials, c, groups, divisor, out);
}

// The five DeviceBackend batch_norm_* operations. scratch holds bn_scratch_floats(n, c, spatial).
template <typename Stream>
void launch_batch_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                               float* channel_std, size_t n, size_t c, size_t spatial, float eps, float* scratch,
                               Stream stream) {
    const size_t total = n * c * spatial;
    const auto m = static_cast<float>(n * spatial);
    float* partials = scratch;
    float* mean = scratch + c * bn_groups_for(n * spatial, c);
    float* var = mean + c;
    bn_reduce<kBnX>(in, nullptr, nullptr, n, c, spatial, partials, m, mean, stream);
    bn_reduce<kBnSquaredDeviation>(in, nullptr, mean, n, c, spatial, partials, m, var, stream);
    bn_std_kernel<<<grid_size_for(c), kBlockSize, 0, stream>>>(var, c, eps, channel_std);
    bn_normalize_kernel<<<grid_size_for(total), kBlockSize, 0, stream>>>(in, mean, channel_std, gamma, beta, xhat,
                                                                          out, total, c, spatial);
}

template <typename Stream>
void launch_batch_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                const float* channel_std, float* grad_in, float* gamma_grad, float* beta_grad,
                                size_t n, size_t c, size_t spatial, float* scratch, Stream stream) {
    const size_t total = n * c * spatial;
    bn_reduce<kBnProduct>(grad_out, xhat, nullptr, n, c, spatial, scratch, 1.0f, gamma_grad, stream);
    bn_reduce<kBnX>(grad_out, nullptr, nullptr, n, c, spatial, scratch, 1.0f, beta_grad, stream);
    bn_input_grad_kernel<<<grid_size_for(total), kBlockSize, 0, stream>>>(
        grad_out, gamma, xhat, channel_std, gamma_grad, beta_grad, grad_in, total, c, spatial,
        static_cast<float>(n * spatial));
}

template <typename Stream>
void launch_batch_norm_update_running(const float* in, float* running_mean, float* running_var, size_t n, size_t c,
                                      size_t spatial, float momentum, float* scratch, Stream stream) {
    float* partials = scratch;
    float* mean = scratch + c * bn_groups_for(n * spatial, c);
    float* squared_deviation = mean + c;
    bn_reduce<kBnX>(in, nullptr, nullptr, n, c, spatial, partials, static_cast<float>(n * spatial), mean, stream);
    bn_reduce<kBnSquaredDeviation>(in, nullptr, mean, n, c, spatial, partials, 1.0f, squared_deviation, stream);
    bn_update_running_kernel<<<grid_size_for(c), kBlockSize, 0, stream>>>(mean, squared_deviation, running_mean,
                                                                          running_var, c, n * spatial, momentum);
}

template <typename Stream>
void launch_batch_norm_eval_forward(const float* in, const float* gamma, const float* beta, const float* running_mean,
                                    const float* running_var, float* xhat, float* out, float* channel_std, size_t n,
                                    size_t c, size_t spatial, float eps, Stream stream) {
    const size_t total = n * c * spatial;
    bn_std_kernel<<<grid_size_for(c), kBlockSize, 0, stream>>>(running_var, c, eps, channel_std);
    bn_normalize_kernel<<<grid_size_for(total), kBlockSize, 0, stream>>>(in, running_mean, channel_std, gamma, beta,
                                                                          xhat, out, total, c, spatial);
}

template <typename Stream>
void launch_batch_norm_eval_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* channel_std, float* grad_in, float* gamma_grad, float* beta_grad,
                                     size_t n, size_t c, size_t spatial, float* scratch, Stream stream) {
    const size_t total = n * c * spatial;
    bn_reduce<kBnProduct>(grad_out, xhat, nullptr, n, c, spatial, scratch, 1.0f, gamma_grad, stream);
    bn_reduce<kBnX>(grad_out, nullptr, nullptr, n, c, spatial, scratch, 1.0f, beta_grad, stream);
    bn_eval_input_grad_kernel<<<grid_size_for(total), kBlockSize, 0, stream>>>(grad_out, gamma, channel_std, grad_in,
                                                                                total, c, spatial);
}

__global__ void group_norm_forward_kernel(const float* in, const float* gamma, const float* beta, float* xhat,
                                          float* out, float* group_std, size_t n, size_t c, size_t spatial,
                                          size_t num_groups, float eps) {
    size_t idx = global_index();  // (example, group)
    if (idx < n * num_groups) {
        cnn::group_norm_forward_group(in, gamma, beta, xhat, out, group_std, static_cast<int64_t>(c),
                                      static_cast<int64_t>(spatial), static_cast<int64_t>(num_groups),
                                      static_cast<int64_t>(c / num_groups), static_cast<int64_t>(idx / num_groups),
                                      static_cast<int64_t>(idx % num_groups), eps);
    }
}

__global__ void group_norm_backward_kernel(const float* grad_out, const float* gamma, const float* xhat,
                                           const float* group_std, float* grad_in, size_t n, size_t c,
                                           size_t spatial, size_t num_groups) {
    size_t idx = global_index();
    if (idx < n * num_groups) {
        cnn::group_norm_backward_group(grad_out, gamma, xhat, group_std, grad_in, static_cast<int64_t>(c),
                                       static_cast<int64_t>(spatial), static_cast<int64_t>(num_groups),
                                       static_cast<int64_t>(c / num_groups), static_cast<int64_t>(idx / num_groups),
                                       static_cast<int64_t>(idx % num_groups));
    }
}

__global__ void group_norm_param_grads_kernel(const float* grad_out, const float* xhat, float* gamma_grad,
                                              float* beta_grad, size_t n, size_t c, size_t spatial) {
    size_t ch = global_index();
    if (ch < c) {
        cnn::group_norm_param_grads_channel(grad_out, xhat, gamma_grad, beta_grad, static_cast<int64_t>(n),
                                            static_cast<int64_t>(c), static_cast<int64_t>(spatial),
                                            static_cast<int64_t>(ch));
    }
}

// ---- GPU-native-kernels Mission 5 ----------------------------------------------------------

__global__ void copy_2d_kernel(float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t rows,
                               size_t cols) {
    size_t idx = global_index();
    if (idx < rows * cols) {
        const size_t r = idx / cols;
        const size_t c = idx % cols;
        dst[r * dst_stride + c] = src[r * src_stride + c];
    }
}

// One thread per column, rows in order -- the CPU's association.
__global__ void accumulate_rows_kernel(const float* in, float* out, size_t rows, size_t cols) {
    size_t j = global_index();
    if (j < cols) {
        float acc = out[j];
        for (size_t r = 0; r < rows; ++r) {
            acc += in[r * cols + j];
        }
        out[j] = acc;
    }
}

__global__ void recurrent_cell_kernel(int op, RecurrentCellArgs args, size_t n) {
    size_t i = global_index();
    if (i < n) {
        recurrent::cell(static_cast<RecurrentCellOp>(op), args, static_cast<int64_t>(i));
    }
}

__global__ void gru_lrp_hprev_kernel(const float* h_prev, const float* w_hn, const float* hn, const float* r_term_b,
                                     const float* direct, float* r_hprev, size_t rows, size_t hidden, float eps) {
    size_t idx = global_index();
    if (idx < rows * hidden) {
        const size_t n = idx / hidden;
        r_hprev[idx] = recurrent::gru_lrp_hprev(h_prev + n * hidden, w_hn, hn + n * hidden, r_term_b + n * hidden,
                                                direct + n * hidden, static_cast<int64_t>(idx % hidden),
                                                static_cast<int64_t>(hidden), eps);
    }
}

// ---- GPU-native-kernels Mission 6 ----------------------------------------------------------

// One thread per lane (see SsmPassOp): a recurrence lane walks time in order inside the thread.
__global__ void ssm_pass_kernel(int op, SsmPassArgs args, int64_t lanes) {
    const auto lane = static_cast<int64_t>(global_index());
    if (lane < lanes) {
        ssm::pass(static_cast<SsmPassOp>(op), args, lane);
    }
}

// ---- GPU-native-kernels Mission 7 ----------------------------------------------------------

// One thread per row (per element for PolyakBlend): the row's reductions and writes are its own.
__global__ void rl_rows_kernel(int op, RlRowArgs args) {
    const auto b = static_cast<int64_t>(global_index());
    if (b < args.rows) {
        rl::row(static_cast<RlRowOp>(op), args, b);
    }
}

// ---- FND-3 ----------------------------------------------------------------------------------

// One thread per row: the row's selection buffer is its own slice of the output.
__global__ void top_k_rows_kernel(const float* in, float* values, float* indices, int64_t rows, int64_t cols,
                                  int64_t k, bool largest) {
    const auto r = static_cast<int64_t>(global_index());
    if (r < rows) {
        topk::row(in + r * cols, values + r * k, indices + r * k, cols, k, largest);
    }
}

}  // namespace gpu
}  // namespace
}  // namespace pulsatrix
