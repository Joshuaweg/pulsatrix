#include "pulsatrix/cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pulsatrix {

namespace {
// Single-precision logistic sigmoid. Kept as one definition so ElementwiseOp::Sigmoid and
// ElementwiseOp::Silu are guaranteed bit-identical on their shared subexpression -- this is
// also the exact expression LSTMModule/GRUModule used in their own host loops before those
// were refactored onto this backend primitive, which is what makes that refactor
// behavior-preserving.
float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
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
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
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
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                const float a_ip = transpose_a ? a[p * m + i] : a[i * k + p];
                const float b_pj = transpose_b ? b[j * k + p] : b[p * n + j];
                acc += a_ip * b_pj;
            }
            // beta == 0 must overwrite without reading out (it may be uninitialized).
            out[i * n + j] = (beta == 0.0f) ? acc : beta * out[i * n + j] + acc;
        }
    }
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
        }
        grad_in[i] = g * d;
    }
}

void CPUBackend::axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) {
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

}  // namespace pulsatrix
