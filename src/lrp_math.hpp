// LRP rules and logic-module pointwise math shared verbatim by CPUBackend (loops) and the GPU
// kernels (one thread per output element / row / column). Every body is the pre-campaign
// module host loop for one output, unchanged in expression and summation order, so CPU
// results are bit-identical and the GPU evaluates the same source. Private to src/.
// GPU-native-kernels Mission 3.
#pragma once

#include <cstdint>

#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE

namespace pulsatrix {
namespace lrp {

// z + eps * sign(z), with sign(0) == +1 -- the stabilizer every epsilon rule in this codebase
// uses.
PULSATRIX_HOST_DEVICE inline float stabilize(float z, float eps) { return z + eps * ((z >= 0.0f) ? 1.0f : -1.0f); }

// ---- LinearModule epsilon rule ----------------------------------------------------------------
// relevance_in[n][i] = sum_j (x[n][i] * w[i][j] / stab(z[n][j])) * r[n][j], j ascending.
PULSATRIX_HOST_DEVICE inline float linear_epsilon(const float* x_row, const float* w, const float* z_row,
                                                  const float* r_row, int64_t i, int64_t out_features, float eps) {
    float acc = 0.0f;
    const float x_i = x_row[i];
    for (int64_t j = 0; j < out_features; ++j) {
        const float denom = stabilize(z_row[j], eps);
        acc += (x_i * w[i * out_features + j] / denom) * r_row[j];
    }
    return acc;
}

// ---- Residual / TransformerBlock epsilon split -------------------------------------------------
PULSATRIX_HOST_DEVICE inline void residual_split(float a, float b, float r, float eps, float* r_a, float* r_b) {
    const float y = a + b;
    const float denom = stabilize(y, eps);
    *r_a = (a / denom) * r;
    *r_b = (b / denom) * r;
}

// ---- Elementwise bilinear product (SwiGLU gate * up), AttnLRP Eq. 15 ---------------------------
PULSATRIX_HOST_DEVICE inline float bilinear_elementwise(float a, float b, float r, float eps) {
    const float c = a * b;
    const float denom = 2.0f * c + eps * ((c >= 0.0f) ? 1.0f : -1.0f);
    return (a * b / denom) * r;
}

// ---- Bilinear matmul O = A @ B, AttnLRP Eq. 15 (MultiHeadAttention) -----------------------------
// The 2* in the denominator is the whole point of Eq. 15 (Achtibat et al. 2024) and is NOT this
// codebase's usual additive epsilon rule: it splits each output's relevance evenly between the
// two operands (each operand's shares sum to R_O/2), which is what makes a product of two
// activations -- rather than an activation and a fixed weight -- attributable at all. The 1*
// denominator of LinearModule/RNNModule/RoPEModule would double the relevance handed out.
// A (M x P); B (P x Q), or stored as B^T (Q x P) when b_transposed; O, R_O (M x Q).
// r_a[i][j] = sum_k (A[i][j] * B[j][k] * R_O[i][k] / (2 O[i][k] + eps sign)), k ascending;
// r_b[j][k] = the same terms summed over i ascending -- each output element summed by its own
// thread in the same order the original nested loop accumulated it.
PULSATRIX_HOST_DEVICE inline float bilinear_scaled_r(const float* o, const float* r_o, int64_t idx, float eps) {
    const float o_ik = o[idx];
    const float denom = 2.0f * o_ik + eps * ((o_ik >= 0.0f) ? 1.0f : -1.0f);
    return r_o[idx] / denom;
}

PULSATRIX_HOST_DEVICE inline float bilinear_b_at(const float* b, int64_t j, int64_t k, int64_t P, int64_t Q,
                                                 bool b_transposed) {
    return b_transposed ? b[k * P + j] : b[j * Q + k];
}

PULSATRIX_HOST_DEVICE inline float bilinear_matmul_r_a(const float* a, const float* b, const float* o,
                                                       const float* r_o, int64_t i, int64_t j, int64_t P, int64_t Q,
                                                       float eps, bool b_transposed) {
    float acc = 0.0f;
    for (int64_t k = 0; k < Q; ++k) {
        acc += a[i * P + j] * bilinear_b_at(b, j, k, P, Q, b_transposed) * bilinear_scaled_r(o, r_o, i * Q + k, eps);
    }
    return acc;
}

PULSATRIX_HOST_DEVICE inline float bilinear_matmul_r_b(const float* a, const float* b, const float* o,
                                                       const float* r_o, int64_t j, int64_t k, int64_t M, int64_t P,
                                                       int64_t Q, float eps, bool b_transposed) {
    float acc = 0.0f;
    for (int64_t i = 0; i < M; ++i) {
        acc += a[i * P + j] * bilinear_b_at(b, j, k, P, Q, b_transposed) * bilinear_scaled_r(o, r_o, i * Q + k, eps);
    }
    return acc;
}

// ---- SoftmaxModule rule (one row) --------------------------------------------------------------
PULSATRIX_HOST_DEVICE inline void softmax_row(const float* x, const float* y, const float* r, float* r_in,
                                              int64_t cols) {
    float relevance_sum = 0.0f;
    for (int64_t j = 0; j < cols; ++j) {
        relevance_sum += r[j];
    }
    for (int64_t i = 0; i < cols; ++i) {
        r_in[i] = x[i] * (r[i] - y[i] * relevance_sum);
    }
}

// ---- RoPEModule epsilon rule (one position's head_dim features) ---------------------------------
// Each input of a pair receives relevance from both outputs of the pair, accumulated in the
// original order (source 1, then source 2).
PULSATRIX_HOST_DEVICE inline void rope_position(const float* x, const float* y, const float* r, const float* cos_row,
                                                const float* sin_row, float* r_in, int64_t half, float eps) {
    for (int64_t i = 0; i < half; ++i) {
        const float c = cos_row[i];
        const float s = sin_row[i];
        const int64_t lo = 2 * i;
        const int64_t hi = 2 * i + 1;
        const float denom0 = stabilize(y[lo], eps);
        const float denom1 = stabilize(y[hi], eps);
        float r_lo = 0.0f;
        float r_hi = 0.0f;
        r_lo += (x[lo] * c / denom0) * r[lo];
        r_hi += (x[hi] * -s / denom0) * r[lo];
        r_lo += (x[lo] * s / denom1) * r[hi];
        r_hi += (x[hi] * c / denom1) * r[hi];
        r_in[lo] = r_lo;
        r_in[hi] = r_hi;
    }
}

// ---- Conjunction / Disjunction (t-norms / t-conorms) -------------------------------------------
// norm: 0 Product, 1 Lukasiewicz, 2 Godel -- the declaration order of both modules' enums.
// One element; out_b is unused by the forward ops.

PULSATRIX_HOST_DEVICE inline float conjunction_forward(int norm, float a, float b) {
    switch (norm) {
        case 0:
            return a * b;
        case 1: {
            const float z = a + b - 1.0f;
            return (z > 0.0f) ? z : 0.0f;
        }
        default:
            return (a <= b) ? a : b;
    }
}

PULSATRIX_HOST_DEVICE inline void conjunction_backward(int norm, float a, float b, float g, float* ga, float* gb) {
    switch (norm) {
        case 0:
            *ga = b * g;
            *gb = a * g;
            return;
        case 1: {
            const float z = a + b - 1.0f;
            const float gz = (z > 0.0f) ? g : 0.0f;
            *ga = gz;
            *gb = gz;
            return;
        }
        default: {
            const bool a_wins = a <= b;  // tie -> a
            *ga = a_wins ? g : 0.0f;
            *gb = a_wins ? 0.0f : g;
            return;
        }
    }
}

PULSATRIX_HOST_DEVICE inline void conjunction_lrp(int norm, float a, float b, float y, float r, float eps, float* ra,
                                                  float* rb) {
    switch (norm) {
        case 0: {
            // AttnLRP Eq. 15-shaped bilinear split, elementwise case: both operands get half.
            const float denom = 2.0f * y + eps * ((y >= 0.0f) ? 1.0f : -1.0f);
            const float contribution = (a * b / denom) * r;
            *ra = contribution;
            *rb = contribution;
            return;
        }
        case 1: {
            // Active region: LinearModule-style bias-excluded epsilon rule on z = a + b - 1.
            // Inactive region: both local derivatives are 0 (matches backward), so both get 0.
            const float z = a + b - 1.0f;
            if (z > 0.0f) {
                const float denom = z + eps;  // z > 0 here, sign(z) == +1
                *ra = (a / denom) * r;
                *rb = (b / denom) * r;
            } else {
                *ra = 0.0f;
                *rb = 0.0f;
            }
            return;
        }
        default: {
            // Exact conservation: the winning (smaller, tie -> a) operand is a pure identity map
            // (y == that operand), so it receives all of r; the other receives 0.
            const bool a_wins = a <= b;
            *ra = a_wins ? r : 0.0f;
            *rb = a_wins ? 0.0f : r;
            return;
        }
    }
}

PULSATRIX_HOST_DEVICE inline float disjunction_forward(int norm, float a, float b) {
    switch (norm) {
        case 0:
            return a + b - a * b;
        case 1: {
            const float z = a + b;
            return (z < 1.0f) ? z : 1.0f;
        }
        default:
            return (a >= b) ? a : b;
    }
}

PULSATRIX_HOST_DEVICE inline void disjunction_backward(int norm, float a, float b, float g, float* ga, float* gb) {
    switch (norm) {
        case 0:
            *ga = (1.0f - b) * g;
            *gb = (1.0f - a) * g;
            return;
        case 1: {
            const float z = a + b;
            const float gz = (z < 1.0f) ? g : 0.0f;
            *ga = gz;
            *gb = gz;
            return;
        }
        default: {
            const bool a_wins = a >= b;  // tie -> a
            *ga = a_wins ? g : 0.0f;
            *gb = a_wins ? 0.0f : g;
            return;
        }
    }
}

PULSATRIX_HOST_DEVICE inline void disjunction_lrp(int norm, float a, float b, float y, float r, float eps, float* ra,
                                                  float* rb) {
    switch (norm) {
        case 0: {
            // Averaged dual-decomposition epsilon rule: z_a + z_b == y exactly for every a, b
            // (derivation in DisjunctionModule's header).
            const float denom = stabilize(y, eps);
            const float z_a = a * (2.0f - b) * 0.5f;
            const float z_b = b * (2.0f - a) * 0.5f;
            *ra = (z_a / denom) * r;
            *rb = (z_b / denom) * r;
            return;
        }
        case 1: {
            const float z = a + b;
            if (z < 1.0f) {
                const float denom = stabilize(z, eps);
                *ra = (a / denom) * r;
                *rb = (b / denom) * r;
            } else {
                *ra = 0.0f;
                *rb = 0.0f;
            }
            return;
        }
        default: {
            const bool a_wins = a >= b;
            *ra = a_wins ? r : 0.0f;
            *rb = a_wins ? 0.0f : r;
            return;
        }
    }
}

// ---- AggregatorModule (power mean over the leading axis; one column j) --------------------------
// x is (n x cols) row-major; every routine walks one column top to bottom, as the original did.

PULSATRIX_HOST_DEVICE inline void aggregator_forward_column(const float* x, float* mean_pow, float* out, int64_t n,
                                                            int64_t cols, int64_t j, float p) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += powf(x[i * cols + j], p);
    }
    const float m = sum / static_cast<float>(n);
    mean_pow[j] = m;
    out[j] = powf(m, 1.0f / p);
}

PULSATRIX_HOST_DEVICE inline void aggregator_backward_column(const float* x, const float* mean_pow,
                                                             const float* grad_out, float* grad_in, int64_t n,
                                                             int64_t cols, int64_t j, float p) {
    const float exponent_m = 1.0f / p - 1.0f;
    const float m_pow = powf(mean_pow[j], exponent_m);
    for (int64_t i = 0; i < n; ++i) {
        const float x_pow = powf(x[i * cols + j], p - 1.0f);
        grad_in[i * cols + j] = grad_out[j] * (m_pow * x_pow) / static_cast<float>(n);
    }
}

PULSATRIX_HOST_DEVICE inline void aggregator_lrp_column(const float* x, const float* mean_pow, const float* r_out,
                                                        float* r_in, int64_t n, int64_t cols, int64_t j, float p,
                                                        float eps) {
    const float denom_raw = mean_pow[j] * static_cast<float>(n);
    const float denom = stabilize(denom_raw, eps);
    for (int64_t i = 0; i < n; ++i) {
        const float contribution = powf(x[i * cols + j], p) / denom;
        r_in[i * cols + j] = contribution * r_out[j];
    }
}

}  // namespace lrp
}  // namespace pulsatrix
