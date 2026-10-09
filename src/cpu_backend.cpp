#include "pulsatrix/cpu_backend.hpp"

#include "pointwise_math.hpp"
#include "cnn_math.hpp"
#include "lrp_math.hpp"
#include "recurrent_math.hpp"
#include "rl_math.hpp"
#include "row_math.hpp"
#include "ssm_math.hpp"
#include "top_k_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pulsatrix {

namespace {
// Single-precision logistic sigmoid. Kept as one definition so ElementwiseOp::Sigmoid and
// ElementwiseOp::Silu are guaranteed bit-identical on their shared subexpression -- this is
// also the exact expression LSTMModule/GRUModule used in their own host loops before those
// were refactored onto this backend primitive, which is what makes that refactor
// behavior-preserving.
float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

/**
 * @brief Runs `body(j0, j1)` over column ranges that together cover `[0, n)`, on several threads
 *        when the multiply (`m * k * n` multiply-adds) is big enough to pay for starting them.
 * @note Splitting columns, not rows, keeps every output element's arithmetic on one thread and
 *       in one order, so results don't depend on the thread count, and a single-row multiply
 *       (one generated token) still spreads out.
 */
template <typename Body>
void ForColumnRanges(size_t m, size_t k, size_t n, const Body& body) {
    constexpr size_t kMinWorkPerThread = size_t{1} << 21;
    constexpr size_t kMinColumnsPerThread = 64;
    const size_t work = m * k * n;
    size_t threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    threads = std::min({threads, work / kMinWorkPerThread, n / kMinColumnsPerThread});
    if (threads <= 1) {
        body(size_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    const size_t chunk = (n + threads - 1) / threads;
    for (size_t t = 1; t < threads; ++t) {
        const size_t j0 = std::min(n, t * chunk);
        const size_t j1 = std::min(n, j0 + chunk);
        pool.emplace_back([&body, j0, j1] { body(j0, j1); });
    }
    body(size_t{0}, std::min(n, chunk));
    for (std::thread& t : pool) t.join();
}

/**
 * @brief Row i of op(A) * B over columns [j0, j1) into acc, for B stored (k x n): the i-p-j
 *        order, so the inner loop reads B and acc contiguously and vectorizes.
 * @note Each acc[j] still starts at 0 and adds its products in increasing p, as the plain i-j-p
 *       dot product does, so results are bit-identical to it.
 */
void GemmRowRange(const float* a, bool transpose_a, const float* b, float* acc, size_t i, size_t m, size_t k,
                  size_t n, size_t j0, size_t j1) {
    std::fill(acc, acc + (j1 - j0), 0.0f);
    for (size_t p = 0; p < k; ++p) {
        const float a_ip = transpose_a ? a[p * m + i] : a[i * k + p];
        const float* b_row = b + p * n + j0;
        for (size_t j = 0; j < j1 - j0; ++j) {
            acc[j] += a_ip * b_row[j];
        }
    }
}
}  // namespace

void* CPUBackend::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;  // by convention -- see device_backend.hpp's allocate() doc
    }
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        throw std::runtime_error("CPUBackend::allocate: failed to allocate " + std::to_string(bytes) + " bytes");
    }
    return ptr;
}

void CPUBackend::free(void* ptr) noexcept {
    std::free(ptr);
}

void CPUBackend::copy(void* dst, const void* src, size_t bytes, CopyDirection /*dir*/) {
    // CopyDirection is ignored on CPUBackend -- every buffer lives in the same (host) memory
    // space, so there is no device-specific transfer path to select between. CUDABackend and
    // HIPBackend (Phase 1.5/1.6) use it to pick host<->device vs. device<->device transfer APIs.
    if (bytes == 0) {
        return;
    }
    std::memcpy(dst, src, bytes);
}

void CPUBackend::fill(void* ptr, float value, size_t n) {
    auto* floats = static_cast<float*>(ptr);
    std::fill(floats, floats + n, value);
}

void CPUBackend::gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) {
    ForColumnRanges(m, k, n, [&](size_t j0, size_t j1) {
        std::vector<float> acc(j1 - j0);
        for (size_t i = 0; i < m; ++i) {
            GemmRowRange(a, false, b, acc.data(), i, m, k, n, j0, j1);
            std::copy(acc.begin(), acc.end(), out + i * n + j0);
        }
    });
}

void CPUBackend::elementwise(ElementwiseOp op, const float* in, float* out, size_t n) {
    switch (op) {
        case ElementwiseOp::Relu:
            for (size_t i = 0; i < n; ++i) {
                out[i] = std::max(in[i], 0.0f);
            }
            break;
        case ElementwiseOp::Neg:
            for (size_t i = 0; i < n; ++i) {
                out[i] = -in[i];
            }
            break;
        case ElementwiseOp::Tanh:
            for (size_t i = 0; i < n; ++i) {
                out[i] = std::tanh(in[i]);
            }
            break;
        case ElementwiseOp::Sigmoid:
            for (size_t i = 0; i < n; ++i) {
                out[i] = sigmoid(in[i]);
            }
            break;
        case ElementwiseOp::Silu:
            for (size_t i = 0; i < n; ++i) {
                out[i] = in[i] * sigmoid(in[i]);
            }
            break;
        case ElementwiseOp::Exp:
            for (size_t i = 0; i < n; ++i) {
                out[i] = std::exp(in[i]);
            }
            break;
        case ElementwiseOp::GeluTanh:
            for (size_t i = 0; i < n; ++i) {
                out[i] = pointwise::gelu_tanh(in[i]);
            }
            break;
        case ElementwiseOp::Gelu:
            for (size_t i = 0; i < n; ++i) {
                out[i] = pointwise::gelu(in[i]);
            }
            break;
    }
}

void CPUBackend::add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void CPUBackend::mul(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// ---- GPU-native-kernels Mission 1 primitives -------------------------------------------
// Summation orders here are the reference the GPU kernels in gpu_kernels.cuh mirror, and
// match the module host loops they replace, so CPU results are unchanged by the migration.

void CPUBackend::gemm_ex(const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m,
                         size_t k, size_t n, float beta) {
    // beta == 0 must overwrite without reading out (it may be uninitialized).
    auto store = [&](float* dst, float acc) { *dst = (beta == 0.0f) ? acc : beta * *dst + acc; };
    ForColumnRanges(m, k, n, [&](size_t j0, size_t j1) {
        if (transpose_b) {
            // B^T: row j of B is contiguous in p, so the plain dot product already reads it in order.
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = j0; j < j1; ++j) {
                    float acc = 0.0f;
                    for (size_t p = 0; p < k; ++p) {
                        const float a_ip = transpose_a ? a[p * m + i] : a[i * k + p];
                        acc += a_ip * b[j * k + p];
                    }
                    store(&out[i * n + j], acc);
                }
            }
            return;
        }
        std::vector<float> acc(j1 - j0);
        for (size_t i = 0; i < m; ++i) {
            GemmRowRange(a, transpose_a, b, acc.data(), i, m, k, n, j0, j1);
            for (size_t j = j0; j < j1; ++j) store(&out[i * n + j], acc[j - j0]);
        }
    });
}

void CPUBackend::column_sums(const float* in, float* out, size_t rows, size_t cols, float beta) {
    for (size_t j = 0; j < cols; ++j) {
        float acc = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            acc += in[i * cols + j];
        }
        out[j] = (beta == 0.0f) ? acc : beta * out[j] + acc;
    }
}

void CPUBackend::add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out[i * cols + j] = in[i * cols + j] + row[j];
        }
    }
}

void CPUBackend::elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in,
                                      size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        const float g = grad_out[i];
        float d = 0.0f;
        switch (op) {
            case ElementwiseOp::Relu:
                // A select, not g * {0,1}: a NaN/inf gradient where x <= 0 must still yield
                // 0, exactly as ReluModule's original masked loop did.
                grad_in[i] = xi > 0.0f ? g : 0.0f;
                continue;
            case ElementwiseOp::Neg:
                d = -1.0f;
                break;
            case ElementwiseOp::Tanh: {
                const float t = std::tanh(xi);
                d = 1.0f - t * t;
                break;
            }
            case ElementwiseOp::Sigmoid: {
                const float s = sigmoid(xi);
                d = s * (1.0f - s);
                break;
            }
            case ElementwiseOp::Silu: {
                // s + x*s*(1-s): SwiGLUModule's original expression, kept verbatim so its CPU
                // results are unchanged by the migration.
                const float s = sigmoid(xi);
                d = s + xi * s * (1.0f - s);
                break;
            }
            case ElementwiseOp::Exp:
                d = std::exp(xi);
                break;
            case ElementwiseOp::GeluTanh:
                d = pointwise::gelu_tanh_grad(xi);
                break;
            case ElementwiseOp::Gelu:
                d = pointwise::gelu_grad(xi);
                break;
        }
        grad_in[i] = g * d;
    }
}

void CPUBackend::axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) {
    if (beta == 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = alpha * x[i];  // y not read -- see the interface note
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = alpha * x[i] + beta * y[i];
    }
}

float CPUBackend::dot(const float* a, const float* b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

void CPUBackend::softmax_rows(const float* in, float* out, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* x = in + r * cols;
        float* y = out + r * cols;
        float row_max = x[0];
        for (size_t j = 1; j < cols; ++j) {
            row_max = std::max(row_max, x[j]);
        }
        float exp_sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            const float e = std::exp(x[j] - row_max);
            y[j] = e;
            exp_sum += e;
        }
        for (size_t j = 0; j < cols; ++j) {
            y[j] /= exp_sum;
        }
    }
}

void CPUBackend::softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
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

void CPUBackend::logsumexp_rows(const float* in, float* out, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* x = in + r * cols;
        float row_max = x[0];
        for (size_t j = 1; j < cols; ++j) {
            row_max = std::max(row_max, x[j]);
        }
        float exp_sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            exp_sum += std::exp(x[j] - row_max);
        }
        out[r] = row_max + std::log(exp_sum);
    }
}

void CPUBackend::adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1,
                           float beta2, float eps, float bias_correction1, float bias_correction2) {
    for (size_t i = 0; i < n; ++i) {
        const float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        const float m_hat = m[i] / bias_correction1;
        const float v_hat = v[i] / bias_correction2;
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// ---- GPU-native-kernels Mission 1b ---------------------------------------------------------

float CPUBackend::sum(const float* in, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        acc += in[i];
    }
    return acc;
}

void CPUBackend::dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale,
                                 uint64_t seed, uint64_t offset) {
    for (size_t i = 0; i < n; ++i) {
        const bool keep = !(pointwise::counter_uniform(seed, offset + i) < p);
        mask[i] = keep ? 1.0f : 0.0f;
        out[i] = keep ? in[i] * scale : 0.0f;
    }
}

void CPUBackend::bce_with_logits(const float* logits, const float* target, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = pointwise::bce_with_logits_term(logits[i], target[i]);
    }
}

void CPUBackend::bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n,
                                      float scale) {
    for (size_t i = 0; i < n; ++i) {
        grad[i] = (pointwise::stable_sigmoid(logits[i]) - target[i]) * scale;
    }
}

// ---- GPU-native-kernels Mission 2 ----------------------------------------------------------

void CPUBackend::layer_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* row_std, size_t rows, size_t cols, float eps) {
    const auto D = static_cast<int64_t>(cols);
    for (size_t r = 0; r < rows; ++r) {
        rows::layer_norm_forward(in + r * cols, gamma, beta, xhat + r * cols, out + r * cols, row_std + r, D, eps);
    }
}

void CPUBackend::layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* row_std, float* grad_in, size_t rows, size_t cols) {
    const auto D = static_cast<int64_t>(cols);
    for (size_t r = 0; r < rows; ++r) {
        rows::layer_norm_backward(grad_out + r * cols, gamma, xhat + r * cols, row_std[r], grad_in + r * cols, D);
    }
}

void CPUBackend::rms_norm_forward(const float* in, const float* gamma, float* out, float* row_rms, size_t rows,
                                  size_t cols, float eps) {
    const auto D = static_cast<int64_t>(cols);
    for (size_t r = 0; r < rows; ++r) {
        rows::rms_norm_forward(in + r * cols, gamma, out + r * cols, row_rms + r, D, eps);
    }
}

void CPUBackend::rms_norm_backward(const float* grad_out, const float* gamma, const float* in, const float* row_rms,
                                   float* grad_in, float* gamma_terms, size_t rows, size_t cols) {
    const auto D = static_cast<int64_t>(cols);
    for (size_t r = 0; r < rows; ++r) {
        rows::rms_norm_backward(grad_out + r * cols, gamma, in + r * cols, row_rms[r], grad_in + r * cols,
                                gamma_terms + r * cols, D);
    }
}

void CPUBackend::rope_rotate(const float* in, const float* cos_table, const float* sin_table, float* out,
                             size_t num_slices, size_t seq_len, size_t head_dim, bool inverse, bool rotate_half) {
    const size_t half = head_dim / 2;
    for (size_t row = 0; row < num_slices * seq_len; ++row) {
        const size_t pos = row % seq_len;
        rows::rope_rotate(in + row * head_dim, cos_table + pos * half, sin_table + pos * half, out + row * head_dim,
                          static_cast<int64_t>(half), inverse, rotate_half);
    }
}

void CPUBackend::attention_mask_fill(float* scores, const float* key_keep, size_t batch, size_t heads, size_t q_len,
                                     size_t k_len, bool causal, size_t q_offset, size_t window, float value) {
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t i = 0; i < q_len; ++i) {
                float* row = scores + ((b * heads + h) * q_len + i) * k_len;
                for (size_t j = 0; j < k_len; ++j) {
                    if (rows::attention_masked(key_keep, b, i, j, k_len, causal, q_offset, window)) {
                        row[j] = value;
                    }
                }
            }
        }
    }
}

void CPUBackend::permute_0213(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) {
    for (size_t a = 0; a < d0; ++a) {
        for (size_t b = 0; b < d1; ++b) {
            for (size_t c = 0; c < d2; ++c) {
                const float* src = in + ((a * d1 + b) * d2 + c) * d3;
                float* dst = out + ((a * d2 + c) * d1 + b) * d3;
                for (size_t e = 0; e < d3; ++e) {
                    dst[e] = src[e];
                }
            }
        }
    }
}

void CPUBackend::gather_rows(const float* table, const float* indices, float* out, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        const float* row = table + static_cast<size_t>(indices[i]) * dim;
        for (size_t d = 0; d < dim; ++d) {
            out[i * dim + d] = row[d];
        }
    }
}

void CPUBackend::scatter_add_rows(const float* src, const float* indices, float* table, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        float* row = table + static_cast<size_t>(indices[i]) * dim;
        for (size_t d = 0; d < dim; ++d) {
            row[d] += src[i * dim + d];
        }
    }
}

void CPUBackend::tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps, float* action,
                                       float* std_cache, float* log_prob, size_t rows, size_t cols, float stabilizer,
                                       double half_log_two_pi) {
    const auto D = static_cast<int64_t>(cols);
    for (size_t r = 0; r < rows; ++r) {
        const size_t off = r * cols;
        rows::tanh_gaussian_forward(mean + off, log_std + off, eps + off, action + off, std_cache + off, log_prob + r,
                                    D, stabilizer, half_log_two_pi);
    }
}

void CPUBackend::tanh_gaussian_backward(const float* action, const float* std_cache, const float* eps,
                                        const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                        float* grad_log_std, size_t n, float stabilizer) {
    for (size_t i = 0; i < n; ++i) {
        rows::tanh_gaussian_backward_element(action[i], std_cache[i], eps[i], grad_action[i], grad_log_prob[i],
                                             stabilizer, grad_mean + i, grad_log_std + i);
    }
}

// ---- GPU-native-kernels Mission 3 ----------------------------------------------------------

void CPUBackend::lrp_linear(const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                            size_t in_features, size_t out_features, float eps) {
    const auto out = static_cast<int64_t>(out_features);
    for (size_t n = 0; n < rows; ++n) {
        for (size_t i = 0; i < in_features; ++i) {
            r_in[n * in_features + i] = lrp::linear_epsilon(x + n * in_features, w, z + n * out_features,
                                                            r + n * out_features, static_cast<int64_t>(i), out, eps);
        }
    }
}

void CPUBackend::lrp_residual_split(const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n,
                                    float eps) {
    for (size_t i = 0; i < n; ++i) {
        lrp::residual_split(a[i], b[i], r[i], eps, r_a + i, r_b + i);
    }
}

void CPUBackend::lrp_bilinear_elementwise(const float* a, const float* b, const float* r, float* r_out, size_t n,
                                          float eps) {
    for (size_t i = 0; i < n; ++i) {
        r_out[i] = lrp::bilinear_elementwise(a[i], b[i], r[i], eps);
    }
}

void CPUBackend::lrp_bilinear_matmul(const float* a, const float* b, const float* o, const float* r_o, float* r_a,
                                     float* r_b, size_t slices, size_t m, size_t p, size_t q, float eps,
                                     bool b_transposed) {
    const auto M = static_cast<int64_t>(m), P = static_cast<int64_t>(p), Q = static_cast<int64_t>(q);
    for (size_t s = 0; s < slices; ++s) {
        const float* as = a + s * m * p;
        const float* bs = b + s * p * q;
        const float* os = o + s * m * q;
        const float* rs = r_o + s * m * q;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < P; ++j) {
                r_a[s * m * p + static_cast<size_t>(i * P + j)] =
                    lrp::bilinear_matmul_r_a(as, bs, os, rs, i, j, P, Q, eps, b_transposed);
            }
        }
        for (int64_t j = 0; j < P; ++j) {
            for (int64_t k = 0; k < Q; ++k) {
                const size_t out_idx = b_transposed ? static_cast<size_t>(k * P + j) : static_cast<size_t>(j * Q + k);
                r_b[s * p * q + out_idx] = lrp::bilinear_matmul_r_b(as, bs, os, rs, j, k, M, P, Q, eps, b_transposed);
            }
        }
    }
}

void CPUBackend::lrp_softmax_rows(const float* x, const float* y, const float* r, float* r_in, size_t rows,
                                  size_t cols) {
    for (size_t row = 0; row < rows; ++row) {
        const size_t off = row * cols;
        lrp::softmax_row(x + off, y + off, r + off, r_in + off, static_cast<int64_t>(cols));
    }
}

void CPUBackend::lrp_rope(const float* x, const float* y, const float* r, const float* cos_table,
                          const float* sin_table, float* r_in, size_t slices, size_t seq_len, size_t head_dim,
                          float eps, bool rotate_half) {
    const size_t half = head_dim / 2;
    for (size_t row = 0; row < slices * seq_len; ++row) {
        const size_t pos = row % seq_len;
        const size_t off = row * head_dim;
        lrp::rope_position(x + off, y + off, r + off, cos_table + pos * half, sin_table + pos * half, r_in + off,
                           static_cast<int64_t>(half), eps, rotate_half);
    }
}

void CPUBackend::logic_pointwise(LogicOp op, int norm, const float* a, const float* b, const float* g_or_r,
                                 const float* y, float* out_a, float* out_b, size_t n, float eps) {
    for (size_t i = 0; i < n; ++i) {
        switch (op) {
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

void CPUBackend::aggregator_forward(const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p) {
    for (size_t j = 0; j < cols; ++j) {
        lrp::aggregator_forward_column(x, mean_pow, out, static_cast<int64_t>(n), static_cast<int64_t>(cols),
                                       static_cast<int64_t>(j), p);
    }
}

void CPUBackend::aggregator_backward(const float* x, const float* mean_pow, const float* grad_out, float* grad_in,
                                     size_t n, size_t cols, float p) {
    for (size_t j = 0; j < cols; ++j) {
        lrp::aggregator_backward_column(x, mean_pow, grad_out, grad_in, static_cast<int64_t>(n),
                                        static_cast<int64_t>(cols), static_cast<int64_t>(j), p);
    }
}

void CPUBackend::aggregator_lrp(const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n,
                                size_t cols, float p, float eps) {
    for (size_t j = 0; j < cols; ++j) {
        lrp::aggregator_lrp_column(x, mean_pow, r_out, r_in, static_cast<int64_t>(n), static_cast<int64_t>(cols),
                                   static_cast<int64_t>(j), p, eps);
    }
}

// ---- GPU-native-kernels Mission 4 ----------------------------------------------------------

void CPUBackend::im2col(const float* in, float* col, size_t n, size_t c, size_t h, size_t w, const ConvGeometry& g) {
    const auto out_w = static_cast<size_t>(cnn::conv_out_size(static_cast<int64_t>(w), static_cast<int64_t>(g.kw),
                                                               static_cast<int64_t>(g.stride_w),
                                                               static_cast<int64_t>(g.pad_w)));
    const auto out_h = static_cast<size_t>(cnn::conv_out_size(static_cast<int64_t>(h), static_cast<int64_t>(g.kh),
                                                               static_cast<int64_t>(g.stride_h),
                                                               static_cast<int64_t>(g.pad_h)));
    const size_t P = c * g.kh * g.kw, Q = out_h * out_w;
    for (size_t e = 0; e < n; ++e) {
        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < Q; ++q) {
                col[(e * P + p) * Q + q] =
                    cnn::im2col_element(in + e * c * h * w, static_cast<int64_t>(h), static_cast<int64_t>(w), g,
                                        static_cast<int64_t>(out_w), static_cast<int64_t>(p), static_cast<int64_t>(q));
            }
        }
    }
}

void CPUBackend::col2im_add(const float* col, float* out, size_t n, size_t c, size_t h, size_t w,
                            const ConvGeometry& g) {
    const auto out_h = static_cast<size_t>(cnn::conv_out_size(static_cast<int64_t>(h), static_cast<int64_t>(g.kh),
                                                               static_cast<int64_t>(g.stride_h),
                                                               static_cast<int64_t>(g.pad_h)));
    const auto out_w = static_cast<size_t>(cnn::conv_out_size(static_cast<int64_t>(w), static_cast<int64_t>(g.kw),
                                                               static_cast<int64_t>(g.stride_w),
                                                               static_cast<int64_t>(g.pad_w)));
    const size_t P = c * g.kh * g.kw, Q = out_h * out_w;
    for (size_t e = 0; e < n; ++e) {
        for (size_t ch = 0; ch < c; ++ch) {
            for (size_t ih = 0; ih < h; ++ih) {
                for (size_t iw = 0; iw < w; ++iw) {
                    float* px = out + ((e * c + ch) * h + ih) * w + iw;
                    *px = cnn::col2im_pixel(col + e * P * Q, *px, static_cast<int64_t>(h), static_cast<int64_t>(w), g,
                                            static_cast<int64_t>(ch), static_cast<int64_t>(ih),
                                            static_cast<int64_t>(iw));
                }
            }
        }
    }
}

void CPUBackend::add_channel_vector(const float* in, const float* vec, float* out, size_t n, size_t c, size_t inner) {
    for (size_t idx = 0; idx < n * c * inner; ++idx) {
        out[idx] = in[idx] + vec[(idx / inner) % c];
    }
}

void CPUBackend::lrp_conv(const float* col, const float* kernel, const float* pre_bias, const float* r, float* r_col,
                          size_t n, size_t out_channels, size_t p, size_t q, float eps) {
    for (size_t e = 0; e < n; ++e) {
        for (size_t pi = 0; pi < p; ++pi) {
            for (size_t qi = 0; qi < q; ++qi) {
                r_col[(e * p + pi) * q + qi] = cnn::conv_lrp_col(
                    col + e * p * q, kernel, pre_bias + e * out_channels * q, r + e * out_channels * q,
                    static_cast<int64_t>(pi), static_cast<int64_t>(qi), static_cast<int64_t>(p),
                    static_cast<int64_t>(q), static_cast<int64_t>(out_channels), eps);
            }
        }
    }
}

void CPUBackend::lrp_stabilized_divide(const float* r, const float* denom, const float* gate, float* out, size_t n,
                                       float eps, LrpGate gate_mode) {
    const int mode = static_cast<int>(gate_mode);
    for (size_t i = 0; i < n; ++i) {
        out[i] = lrp::stabilized_divide(r[i], denom[i], mode == 0 ? 0.0f : gate[i], mode, eps);
    }
}

void CPUBackend::max_pool_forward(const float* in, float* out, float* argmax, size_t planes, size_t h, size_t w,
                                  size_t kh, size_t kw, size_t sh, size_t sw, size_t ph, size_t pw) {
    const auto H = static_cast<int64_t>(h), W = static_cast<int64_t>(w);
    const int64_t out_h = cnn::pool_out_size(H, static_cast<int64_t>(kh), static_cast<int64_t>(sh), static_cast<int64_t>(ph));
    const int64_t out_w = cnn::pool_out_size(W, static_cast<int64_t>(kw), static_cast<int64_t>(sw), static_cast<int64_t>(pw));
    for (size_t pl = 0; pl < planes; ++pl) {
        for (int64_t oh = 0; oh < out_h; ++oh) {
            for (int64_t ow = 0; ow < out_w; ++ow) {
                const size_t o = (pl * static_cast<size_t>(out_h) + static_cast<size_t>(oh)) * static_cast<size_t>(out_w) + static_cast<size_t>(ow);
                cnn::max_pool_window(in + pl * h * w, H, W, static_cast<int64_t>(kh), static_cast<int64_t>(kw),
                                     static_cast<int64_t>(sh), static_cast<int64_t>(sw), static_cast<int64_t>(ph),
                                     static_cast<int64_t>(pw), oh, ow, out + o, argmax + o);
            }
        }
    }
}

void CPUBackend::max_unpool(const float* src, const float* argmax, float* dst, size_t planes, size_t h, size_t w,
                            size_t kh, size_t kw, size_t sh, size_t sw, size_t ph, size_t pw) {
    const auto H = static_cast<int64_t>(h), W = static_cast<int64_t>(w);
    const int64_t out_h = cnn::pool_out_size(H, static_cast<int64_t>(kh), static_cast<int64_t>(sh), static_cast<int64_t>(ph));
    const int64_t out_w = cnn::pool_out_size(W, static_cast<int64_t>(kw), static_cast<int64_t>(sw), static_cast<int64_t>(pw));
    const size_t out_plane = static_cast<size_t>(out_h * out_w);
    for (size_t pl = 0; pl < planes; ++pl) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                dst[pl * h * w + static_cast<size_t>(y * W + x)] = cnn::max_unpool_gather(
                    src + pl * out_plane, argmax + pl * out_plane, W, out_h, out_w, static_cast<int64_t>(kh),
                    static_cast<int64_t>(kw), static_cast<int64_t>(sh), static_cast<int64_t>(sw), static_cast<int64_t>(ph),
                    static_cast<int64_t>(pw), y, x);
            }
        }
    }
}

void CPUBackend::avg_pool_forward(const float* in, float* out, size_t planes, size_t h, size_t w, size_t kh,
                                  size_t kw) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    for (size_t pl = 0; pl < planes; ++pl) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                out[(pl * out_h + oh) * out_w + ow] =
                    cnn::avg_pool_window(in + pl * h * w, static_cast<int64_t>(w), static_cast<int64_t>(kh),
                                         static_cast<int64_t>(kw), static_cast<int64_t>(oh), static_cast<int64_t>(ow));
            }
        }
    }
}

void CPUBackend::avg_pool_backward(const float* grad_out, float* grad_in, size_t planes, size_t h, size_t w,
                                   size_t kh, size_t kw) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    for (size_t pl = 0; pl < planes; ++pl) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const float g = grad_out[(pl * out_h + oh) * out_w + ow] / static_cast<float>(kh * kw);
                for (size_t i = 0; i < kh; ++i) {
                    for (size_t j = 0; j < kw; ++j) {
                        grad_in[pl * h * w + (oh * kh + i) * w + (ow * kw + j)] = g;
                    }
                }
            }
        }
    }
}

void CPUBackend::lrp_avg_pool(const float* x, const float* r, float* r_in, size_t planes, size_t h, size_t w,
                              size_t kh, size_t kw, float eps) {
    const size_t out_h = (h - kh) / kh + 1, out_w = (w - kw) / kw + 1;
    for (size_t pl = 0; pl < planes; ++pl) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                cnn::avg_pool_lrp_window(x + pl * h * w, r_in + pl * h * w, static_cast<int64_t>(w),
                                         static_cast<int64_t>(kh), static_cast<int64_t>(kw), static_cast<int64_t>(oh),
                                         static_cast<int64_t>(ow), r[(pl * out_h + oh) * out_w + ow], eps);
            }
        }
    }
}

void CPUBackend::batch_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* channel_std, size_t n, size_t c, size_t spatial, float eps) {
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::batch_norm_forward_channel(in, gamma, beta, xhat, out, channel_std, static_cast<int64_t>(n),
                                        static_cast<int64_t>(c), static_cast<int64_t>(spatial),
                                        static_cast<int64_t>(ch), eps);
    }
}

void CPUBackend::batch_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* channel_std, float* grad_in, float* gamma_grad, float* beta_grad,
                                     size_t n, size_t c, size_t spatial) {
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::batch_norm_backward_channel(grad_out, gamma, xhat, channel_std, grad_in, gamma_grad, beta_grad,
                                         static_cast<int64_t>(n), static_cast<int64_t>(c),
                                         static_cast<int64_t>(spatial), static_cast<int64_t>(ch));
    }
}

void CPUBackend::batch_norm_update_running(const float* in, float* running_mean, float* running_var, size_t n,
                                           size_t c, size_t spatial, float momentum) {
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::batch_norm_update_running_channel(in, running_mean, running_var, static_cast<int64_t>(n),
                                               static_cast<int64_t>(c), static_cast<int64_t>(spatial),
                                               static_cast<int64_t>(ch), momentum);
    }
}

void CPUBackend::batch_norm_eval_forward(const float* in, const float* gamma, const float* beta,
                                         const float* running_mean, const float* running_var, float* xhat, float* out,
                                         float* channel_std, size_t n, size_t c, size_t spatial, float eps) {
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::batch_norm_eval_forward_channel(in, gamma, beta, running_mean, running_var, xhat, out, channel_std,
                                             static_cast<int64_t>(n), static_cast<int64_t>(c),
                                             static_cast<int64_t>(spatial), static_cast<int64_t>(ch), eps);
    }
}

void CPUBackend::batch_norm_eval_backward(const float* grad_out, const float* gamma, const float* xhat,
                                          const float* channel_std, float* grad_in, float* gamma_grad,
                                          float* beta_grad, size_t n, size_t c, size_t spatial) {
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::batch_norm_eval_backward_channel(grad_out, gamma, xhat, channel_std, grad_in, gamma_grad, beta_grad,
                                              static_cast<int64_t>(n), static_cast<int64_t>(c),
                                              static_cast<int64_t>(spatial), static_cast<int64_t>(ch));
    }
}

void CPUBackend::group_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* group_std, size_t n, size_t c, size_t spatial, size_t num_groups,
                                    float eps) {
    const size_t group_size = c / num_groups;
    for (size_t e = 0; e < n; ++e) {
        for (size_t g = 0; g < num_groups; ++g) {
            cnn::group_norm_forward_group(in, gamma, beta, xhat, out, group_std, static_cast<int64_t>(c),
                                          static_cast<int64_t>(spatial), static_cast<int64_t>(num_groups),
                                          static_cast<int64_t>(group_size), static_cast<int64_t>(e),
                                          static_cast<int64_t>(g), eps);
        }
    }
}

void CPUBackend::group_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* group_std, float* grad_in, float* gamma_grad, float* beta_grad,
                                     size_t n, size_t c, size_t spatial, size_t num_groups) {
    const size_t group_size = c / num_groups;
    for (size_t ch = 0; ch < c; ++ch) {
        cnn::group_norm_param_grads_channel(grad_out, xhat, gamma_grad, beta_grad, static_cast<int64_t>(n),
                                            static_cast<int64_t>(c), static_cast<int64_t>(spatial),
                                            static_cast<int64_t>(ch));
    }
    for (size_t e = 0; e < n; ++e) {
        for (size_t g = 0; g < num_groups; ++g) {
            cnn::group_norm_backward_group(grad_out, gamma, xhat, group_std, grad_in, static_cast<int64_t>(c),
                                           static_cast<int64_t>(spatial), static_cast<int64_t>(num_groups),
                                           static_cast<int64_t>(group_size), static_cast<int64_t>(e),
                                           static_cast<int64_t>(g));
        }
    }
}

// ---- GPU-native-kernels Mission 5 ----------------------------------------------------------

void CPUBackend::copy_2d(float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t rows,
                         size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * dst_stride, src + r * src_stride, cols * sizeof(float));
    }
}

void CPUBackend::accumulate_rows(const float* in, float* out, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < cols; ++j) {
            out[j] += in[r * cols + j];
        }
    }
}

void CPUBackend::recurrent_cell(RecurrentCellOp op, const RecurrentCellArgs& args, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        recurrent::cell(op, args, static_cast<int64_t>(i));
    }
}

void CPUBackend::gru_lrp_hprev(const float* h_prev, const float* w_hn, const float* hn, const float* r_term_b,
                               const float* direct, float* r_hprev, size_t rows, size_t hidden, float eps) {
    for (size_t n = 0; n < rows; ++n) {
        for (size_t kk = 0; kk < hidden; ++kk) {
            r_hprev[n * hidden + kk] =
                recurrent::gru_lrp_hprev(h_prev + n * hidden, w_hn, hn + n * hidden, r_term_b + n * hidden,
                                         direct + n * hidden, static_cast<int64_t>(kk), static_cast<int64_t>(hidden),
                                         eps);
        }
    }
}

// ---- GPU-native-kernels Mission 6 ----------------------------------------------------------

void CPUBackend::ssm_pass(SsmPassOp op, const SsmPassArgs& args) {
    const int64_t lanes = ssm::lanes(op, args);
    for (int64_t lane = 0; lane < lanes; ++lane) {
        ssm::pass(op, args, lane);
    }
}

// ---- GPU-native-kernels Mission 7 ----------------------------------------------------------

void CPUBackend::rl_rows(RlRowOp op, const RlRowArgs& args) {
    for (int64_t b = 0; b < args.rows; ++b) {
        rl::row(op, args, b);
    }
}

// ---- FND-3: selection ----------------------------------------------------------------------

void CPUBackend::top_k_rows(const float* in, float* values, float* indices, size_t rows, size_t cols, size_t k,
                            bool largest) {
    for (size_t r = 0; r < rows; ++r) {
        topk::row(in + r * cols, values + r * k, indices + r * k, static_cast<int64_t>(cols),
                  static_cast<int64_t>(k), largest);
    }
}

}  // namespace pulsatrix
