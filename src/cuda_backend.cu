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

// ---- GPU-native-kernels Mission 2 (kernels in gpu_kernels.cuh) ---------------------------

void CUDABackend::layer_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* row_std, size_t rows, size_t cols, float eps) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::layer_norm_forward_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        in, gamma, beta, xhat, out, row_std, rows, cols, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* row_std, float* grad_in, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::layer_norm_backward_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        grad_out, gamma, xhat, row_std, grad_in, rows, cols);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::rms_norm_forward(const float* in, const float* gamma, float* out, float* row_rms, size_t rows,
                                  size_t cols, float eps) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::rms_norm_forward_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        in, gamma, out, row_rms, rows, cols, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::rms_norm_backward(const float* grad_out, const float* gamma, const float* in, const float* row_rms,
                                   float* grad_in, float* gamma_terms, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::rms_norm_backward_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        grad_out, gamma, in, row_rms, grad_in, gamma_terms, rows, cols);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::rope_rotate(const float* in, const float* cos_table, const float* sin_table, float* out,
                             size_t num_slices, size_t seq_len, size_t head_dim, bool inverse) {
    const size_t total_rows = num_slices * seq_len;
    if (total_rows == 0 || head_dim == 0) {
        return;
    }
    gpu::rope_rotate_kernel<<<gpu::grid_size_for(total_rows), gpu::kBlockSize, 0, stream_>>>(
        in, cos_table, sin_table, out, total_rows, seq_len, head_dim, inverse);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::permute_0213(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) {
    const size_t n = d0 * d1 * d2 * d3;
    if (n == 0) {
        return;
    }
    gpu::permute_0213_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(in, out, d0, d1, d2, d3);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::gather_rows(const float* table, const float* indices, float* out, size_t count, size_t dim) {
    if (count == 0 || dim == 0) {
        return;
    }
    gpu::gather_rows_kernel<<<gpu::grid_size_for(count * dim), gpu::kBlockSize, 0, stream_>>>(
        table, indices, out, count, dim);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::scatter_add_rows(const float* src, const float* indices, float* table, size_t count, size_t dim) {
    if (count == 0 || dim == 0) {
        return;
    }
    gpu::scatter_add_rows_kernel<<<gpu::grid_size_for(dim), gpu::kBlockSize, 0, stream_>>>(
        src, indices, table, count, dim);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps, float* action,
                                       float* std_cache, float* log_prob, size_t rows, size_t cols, float stabilizer,
                                       double half_log_two_pi) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::tanh_gaussian_forward_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        mean, log_std, eps, action, std_cache, log_prob, rows, cols, stabilizer, half_log_two_pi);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::tanh_gaussian_backward(const float* action, const float* std_cache, const float* eps,
                                        const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                        float* grad_log_std, size_t n, float stabilizer) {
    if (n == 0) {
        return;
    }
    gpu::tanh_gaussian_backward_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(
        action, std_cache, eps, grad_action, grad_log_prob, grad_mean, grad_log_std, n, stabilizer);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- GPU-native-kernels Mission 3 (kernels in gpu_kernels.cuh) ---------------------------

void CUDABackend::lrp_linear(const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                            size_t in_features, size_t out_features, float eps) {
    if (rows == 0 || in_features == 0) {
        return;
    }
    gpu::lrp_linear_kernel<<<gpu::grid_size_for(rows * in_features), gpu::kBlockSize, 0, stream_>>>(
        x, w, z, r, r_in, rows, in_features, out_features, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_residual_split(const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n,
                                    float eps) {
    if (n == 0) {
        return;
    }
    gpu::lrp_residual_split_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(
        a, b, r, r_a, r_b, n, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_bilinear_elementwise(const float* a, const float* b, const float* r, float* r_out, size_t n,
                                          float eps) {
    if (n == 0) {
        return;
    }
    gpu::lrp_bilinear_elementwise_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(
        a, b, r, r_out, n, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_bilinear_matmul(const float* a, const float* b, const float* o, const float* r_o, float* r_a,
                                     float* r_b, size_t slices, size_t m, size_t p, size_t q, float eps,
                                     bool b_transposed) {
    if (slices == 0 || m == 0 || p == 0 || q == 0) {
        return;
    }
    gpu::lrp_bilinear_r_a_kernel<<<gpu::grid_size_for(slices * m * p), gpu::kBlockSize, 0, stream_>>>(
        a, b, o, r_o, r_a, slices, m, p, q, eps, b_transposed);
    gpu::lrp_bilinear_r_b_kernel<<<gpu::grid_size_for(slices * p * q), gpu::kBlockSize, 0, stream_>>>(
        a, b, o, r_o, r_b, slices, m, p, q, eps, b_transposed);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_softmax_rows(const float* x, const float* y, const float* r, float* r_in, size_t rows,
                                  size_t cols) {
    if (rows == 0 || cols == 0) {
        return;
    }
    gpu::lrp_softmax_rows_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        x, y, r, r_in, rows, cols);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_rope(const float* x, const float* y, const float* r, const float* cos_table,
                          const float* sin_table, float* r_in, size_t slices, size_t seq_len, size_t head_dim,
                          float eps) {
    if (slices * seq_len == 0 || head_dim == 0) {
        return;
    }
    gpu::lrp_rope_kernel<<<gpu::grid_size_for(slices * seq_len), gpu::kBlockSize, 0, stream_>>>(
        x, y, r, cos_table, sin_table, r_in, slices * seq_len, seq_len, head_dim, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::logic_pointwise(LogicOp op, int norm, const float* a, const float* b, const float* g_or_r,
                                 const float* y, float* out_a, float* out_b, size_t n, float eps) {
    if (n == 0) {
        return;
    }
    gpu::logic_pointwise_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(
        static_cast<int>(op), norm, a, b, g_or_r, y, out_a, out_b, n, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::aggregator_forward(const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p) {
    if (cols == 0) {
        return;
    }
    gpu::aggregator_forward_kernel<<<gpu::grid_size_for(cols), gpu::kBlockSize, 0, stream_>>>(
        x, mean_pow, out, n, cols, p);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::aggregator_backward(const float* x, const float* mean_pow, const float* grad_out, float* grad_in,
                                     size_t n, size_t cols, float p) {
    if (cols == 0) {
        return;
    }
    gpu::aggregator_backward_kernel<<<gpu::grid_size_for(cols), gpu::kBlockSize, 0, stream_>>>(
        x, mean_pow, grad_out, grad_in, n, cols, p);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::aggregator_lrp(const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n,
                                size_t cols, float p, float eps) {
    if (cols == 0) {
        return;
    }
    gpu::aggregator_lrp_kernel<<<gpu::grid_size_for(cols), gpu::kBlockSize, 0, stream_>>>(
        x, mean_pow, r_out, r_in, n, cols, p, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- GPU-native-kernels Mission 4 (kernels in gpu_kernels.cuh) ---------------------------

void CUDABackend::im2col(const float* in, float* col, size_t n, size_t c, size_t h, size_t w, size_t kh, size_t kw) {
    if (n == 0 || c == 0) {
        return;
    }
    const size_t total = n * c * kh * kw * (h - kh + 1) * (w - kw + 1);
    gpu::im2col_kernel<<<gpu::grid_size_for(total), gpu::kBlockSize, 0, stream_>>>(
        in, col, n, c, h, w, kh, kw);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::col2im_add(const float* col, float* out, size_t n, size_t c, size_t h, size_t w, size_t kh, size_t kw)
                             {
    if (n == 0 || c == 0) {
        return;
    }
    gpu::col2im_add_kernel<<<gpu::grid_size_for(n * c * h * w), gpu::kBlockSize, 0, stream_>>>(
        col, out, n, c, h, w, kh, kw);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::add_channel_vector(const float* in, const float* vec, float* out, size_t n, size_t c, size_t inner) {
    if (n * c * inner == 0) {
        return;
    }
    gpu::add_channel_vector_kernel<<<gpu::grid_size_for(n * c * inner), gpu::kBlockSize, 0, stream_>>>(
        in, vec, out, n, c, inner);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_conv(const float* col, const float* kernel, const float* pre_bias, const float* r, float* r_col,
                           size_t n, size_t out_channels, size_t p, size_t q, float eps) {
    if (n * p * q == 0) {
        return;
    }
    gpu::lrp_conv_kernel<<<gpu::grid_size_for(n * p * q), gpu::kBlockSize, 0, stream_>>>(
        col, kernel, pre_bias, r, r_col, n, out_channels, p, q, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_stabilized_divide(const float* r, const float* denom, const float* gate, float* out, size_t n,
                                        float eps, LrpGate gate_mode) {
    if (n == 0) {
        return;
    }
    gpu::lrp_stabilized_divide_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(
        r, denom, gate, out, n, eps, static_cast<int>(gate_mode));
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::max_pool_forward(const float* in, float* out, float* argmax, size_t planes, size_t h, size_t w, size_t
                                   kh, size_t kw) {
    if (planes == 0) {
        return;
    }
    const size_t total = planes * ((h - kh) / kh + 1) * ((w - kw) / kw + 1);
    gpu::max_pool_forward_kernel<<<gpu::grid_size_for(total), gpu::kBlockSize, 0, stream_>>>(
        in, out, argmax, planes, h, w, kh, kw);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::max_unpool(const float* src, const float* argmax, float* dst, size_t planes, size_t h, size_t w,
                             size_t kh, size_t kw) {
    if (planes == 0) {
        return;
    }
    const size_t out_plane = ((h - kh) / kh + 1) * ((w - kw) / kw + 1);
    gpu::max_unpool_kernel<<<gpu::grid_size_for(planes * out_plane), gpu::kBlockSize, 0, stream_>>>(
        src, argmax, dst, planes, h, w, out_plane);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::avg_pool_forward(const float* in, float* out, size_t planes, size_t h, size_t w, size_t kh, size_t kw)
                                   {
    if (planes == 0) {
        return;
    }
    const size_t total = planes * ((h - kh) / kh + 1) * ((w - kw) / kw + 1);
    gpu::avg_pool_forward_kernel<<<gpu::grid_size_for(total), gpu::kBlockSize, 0, stream_>>>(
        in, out, planes, h, w, kh, kw);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::avg_pool_backward(const float* grad_out, float* grad_in, size_t planes, size_t h, size_t w, size_t kh,
                                    size_t kw) {
    if (planes == 0) {
        return;
    }
    const size_t total = planes * ((h - kh) / kh + 1) * ((w - kw) / kw + 1);
    gpu::avg_pool_backward_kernel<<<gpu::grid_size_for(total), gpu::kBlockSize, 0, stream_>>>(
        grad_out, grad_in, planes, h, w, kh, kw);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::lrp_avg_pool(const float* x, const float* r, float* r_in, size_t planes, size_t h, size_t w, size_t
                               kh, size_t kw, float eps) {
    if (planes == 0) {
        return;
    }
    const size_t total = planes * ((h - kh) / kh + 1) * ((w - kw) / kw + 1);
    gpu::lrp_avg_pool_kernel<<<gpu::grid_size_for(total), gpu::kBlockSize, 0, stream_>>>(
        x, r, r_in, planes, h, w, kh, kw, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::batch_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                     float* channel_std, size_t n, size_t c, size_t spatial, float eps) {
    if (c == 0) {
        return;
    }
    gpu::batch_norm_forward_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        in, gamma, beta, xhat, out, channel_std, n, c, spatial, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::batch_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float*
                                      channel_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t
                                      c, size_t spatial) {
    if (c == 0) {
        return;
    }
    gpu::batch_norm_backward_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        grad_out, gamma, xhat, channel_std, grad_in, gamma_grad, beta_grad, n, c, spatial);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::batch_norm_update_running(const float* in, float* running_mean, float* running_var, size_t n, size_t c,
                                           size_t spatial, float momentum) {
    if (c == 0) {
        return;
    }
    gpu::batch_norm_update_running_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        in, running_mean, running_var, n, c, spatial, momentum);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::batch_norm_eval_forward(const float* in, const float* gamma, const float* beta,
                                         const float* running_mean, const float* running_var, float* xhat, float* out,
                                         float* channel_std, size_t n, size_t c, size_t spatial, float eps) {
    if (c == 0) {
        return;
    }
    gpu::batch_norm_eval_forward_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        in, gamma, beta, running_mean, running_var, xhat, out, channel_std, n, c, spatial, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::batch_norm_eval_backward(const float* grad_out, const float* gamma, const float* xhat,
                                          const float* channel_std, float* grad_in, float* gamma_grad,
                                          float* beta_grad, size_t n, size_t c, size_t spatial) {
    if (c == 0) {
        return;
    }
    gpu::batch_norm_eval_backward_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        grad_out, gamma, xhat, channel_std, grad_in, gamma_grad, beta_grad, n, c, spatial);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::group_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                     float* group_std, size_t n, size_t c, size_t spatial, size_t num_groups, float eps)
                                     {
    if (n * num_groups == 0) {
        return;
    }
    gpu::group_norm_forward_kernel<<<gpu::grid_size_for(n * num_groups), gpu::kBlockSize, 0, stream_>>>(
        in, gamma, beta, xhat, out, group_std, n, c, spatial, num_groups, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::group_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float*
                                      group_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t
                                      c, size_t spatial, size_t num_groups) {
    if (n * num_groups == 0) {
        return;
    }
    gpu::group_norm_param_grads_kernel<<<gpu::grid_size_for(c), gpu::kBlockSize, 0, stream_>>>(
        grad_out, xhat, gamma_grad, beta_grad, n, c, spatial);
    gpu::group_norm_backward_kernel<<<gpu::grid_size_for(n * num_groups), gpu::kBlockSize, 0, stream_>>>(
        grad_out, gamma, xhat, group_std, grad_in, n, c, spatial, num_groups);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- GPU-native-kernels Mission 5 (kernels in gpu_kernels.cuh) ---------------------------

void CUDABackend::copy_2d(float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t rows,
                         size_t cols) {
    if (rows * cols == 0) {
        return;
    }
    gpu::copy_2d_kernel<<<gpu::grid_size_for(rows * cols), gpu::kBlockSize, 0, stream_>>>(dst, dst_stride, src,
                                                                                          src_stride, rows, cols);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::accumulate_rows(const float* in, float* out, size_t rows, size_t cols) {
    if (cols == 0) {
        return;
    }
    gpu::accumulate_rows_kernel<<<gpu::grid_size_for(cols), gpu::kBlockSize, 0, stream_>>>(in, out, rows, cols);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::recurrent_cell(RecurrentCellOp op, const RecurrentCellArgs& args, size_t n) {
    if (n == 0) {
        return;
    }
    gpu::recurrent_cell_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(static_cast<int>(op), args, n);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::gru_lrp_hprev(const float* h_prev, const float* w_hn, const float* hn, const float* r_term_b,
                               const float* direct, float* r_hprev, size_t rows, size_t hidden, float eps) {
    if (rows * hidden == 0) {
        return;
    }
    gpu::gru_lrp_hprev_kernel<<<gpu::grid_size_for(rows * hidden), gpu::kBlockSize, 0, stream_>>>(
        h_prev, w_hn, hn, r_term_b, direct, r_hprev, rows, hidden, eps);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::ssm_pass(SsmPassOp op, const SsmPassArgs& args) {
    const int64_t lanes = ssm::lanes(op, args);
    if (lanes <= 0) {
        return;
    }
    const auto n = static_cast<size_t>(lanes);
    gpu::ssm_pass_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(static_cast<int>(op), args, lanes);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CUDABackend::rl_rows(RlRowOp op, const RlRowArgs& args) {
    if (args.rows <= 0) {
        return;
    }
    const auto n = static_cast<size_t>(args.rows);
    gpu::rl_rows_kernel<<<gpu::grid_size_for(n), gpu::kBlockSize, 0, stream_>>>(static_cast<int>(op), args);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ---- FND-3: selection ----------------------------------------------------------------------

void CUDABackend::top_k_rows(const float* in, float* values, float* indices, size_t rows, size_t cols, size_t k,
                            bool largest) {
    if (rows == 0) {
        return;
    }
    gpu::top_k_rows_kernel<<<gpu::grid_size_for(rows), gpu::kBlockSize, 0, stream_>>>(
        in, values, indices, static_cast<int64_t>(rows), static_cast<int64_t>(cols), static_cast<int64_t>(k), largest);
    PULSATRIX_CUDA_CHECK(cudaGetLastError());
    PULSATRIX_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace pulsatrix
