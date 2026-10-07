// Per-row routines shared verbatim by CPUBackend (one call per row in a loop) and the GPU
// kernels (one thread per row), so the CPU reference and the device evaluate literally the
// same source. Each body is the pre-campaign module host loop for one row, unchanged in
// expression and order. Private to src/. GPU-native-kernels Mission 2.
#pragma once

#include <cstdint>

#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE

namespace pulsatrix {
namespace rows {

// ---- LayerNorm (LayerNormModule) ------------------------------------------------------------

PULSATRIX_HOST_DEVICE inline void layer_norm_forward(const float* x, const float* gamma, const float* beta,
                                                     float* xhat, float* out, float* std_out, int64_t D, float eps) {
    const float Df = static_cast<float>(D);
    float sum = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        sum += x[i];
    }
    const float mu = sum / Df;
    float sum_sq_diff = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        const float d = x[i] - mu;
        sum_sq_diff += d * d;
    }
    const float var = sum_sq_diff / Df;
    const float std_dev = sqrtf(var + eps);
    *std_out = std_dev;
    for (int64_t i = 0; i < D; ++i) {
        const float xh = (x[i] - mu) / std_dev;
        xhat[i] = xh;
        out[i] = gamma[i] * xh + beta[i];
    }
}

// grad_xhat = grad_out * gamma is recomputed in the second pass rather than stored: the same
// float product twice, so no per-row scratch is needed and the result is unchanged.
PULSATRIX_HOST_DEVICE inline void layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                                      float std_dev, float* grad_in, int64_t D) {
    const float Df = static_cast<float>(D);
    float sum_grad_xhat = 0.0f;
    float sum_grad_xhat_xhat = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        const float g = grad_out[i] * gamma[i];
        sum_grad_xhat += g;
        sum_grad_xhat_xhat += g * xhat[i];
    }
    for (int64_t i = 0; i < D; ++i) {
        const float g = grad_out[i] * gamma[i];
        grad_in[i] = (Df * g - sum_grad_xhat - xhat[i] * sum_grad_xhat_xhat) / (Df * std_dev);
    }
}

// ---- RMSNorm (RMSNormModule) ----------------------------------------------------------------

PULSATRIX_HOST_DEVICE inline void rms_norm_forward(const float* x, const float* gamma, float* out, float* rms_out,
                                                   int64_t D, float eps) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        sum_sq += x[i] * x[i];
    }
    const float ms = sum_sq / static_cast<float>(D);
    const float rms = sqrtf(ms + eps);
    *rms_out = rms;
    for (int64_t i = 0; i < D; ++i) {
        out[i] = gamma[i] * x[i] / rms;
    }
}

// gamma_terms[i] = grad_out[i] * x[i] / rms: this row's contribution to gamma's gradient,
// summed over rows afterwards by column_sums in the original row order.
PULSATRIX_HOST_DEVICE inline void rms_norm_backward(const float* grad_out, const float* gamma, const float* x,
                                                    float rms, float* grad_in, float* gamma_terms, int64_t D) {
    const float Df = static_cast<float>(D);
    float dot = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        dot += grad_out[i] * gamma[i] * x[i];
    }
    for (int64_t i = 0; i < D; ++i) {
        gamma_terms[i] = grad_out[i] * x[i] / rms;
        grad_in[i] = grad_out[i] * gamma[i] / rms - (x[i] / (Df * rms * rms * rms)) * dot;
    }
}

// ---- RoPE (RoPEModule) ----------------------------------------------------------------------

// One position's head_dim features, rotated pairwise by precomputed cos/sin (half entries).
// inverse == false is forward; inverse == true is the transpose rotation backward() applies.
// Pair i is (2i, 2i+1), or (i, i + half) when rotate_half (the Hugging Face Llama layout).
PULSATRIX_HOST_DEVICE inline void rope_rotate(const float* in, const float* cos_row, const float* sin_row, float* out,
                                              int64_t half, bool inverse, bool rotate_half) {
    for (int64_t i = 0; i < half; ++i) {
        const float c = cos_row[i];
        const float s = sin_row[i];
        const int64_t lo = rotate_half ? i : 2 * i;
        const int64_t hi = rotate_half ? i + half : 2 * i + 1;
        const float x0 = in[lo];
        const float x1 = in[hi];
        if (inverse) {
            out[lo] = x0 * c + x1 * s;
            out[hi] = -x0 * s + x1 * c;
        } else {
            out[lo] = x0 * c - x1 * s;
            out[hi] = x0 * s + x1 * c;
        }
    }
}

// ---- Attention masks (MultiHeadAttentionModule, LLM-1) ------------------------------------------

// Whether score (b, ., i, j) is masked: a causal future key, a key further back than a sliding
// window (window > 0: query i + q_offset sees keys j + window > i + q_offset), or a key key_keep
// marks as padding.
PULSATRIX_HOST_DEVICE inline bool attention_masked(const float* key_keep, size_t b, size_t i, size_t j, size_t k_len,
                                                   bool causal, size_t q_offset, size_t window) {
    if (causal && j > i + q_offset) {
        return true;
    }
    if (causal && window > 0 && j + window <= i + q_offset) {
        return true;
    }
    return key_keep != nullptr && key_keep[b * k_len + j] == 0.0f;
}

// ---- TanhGaussianPolicy -----------------------------------------------------------------------

// One (action_dim) row: squashed sample, cached std, and the row log-probability accumulated
// in double exactly as the original host loop did.
PULSATRIX_HOST_DEVICE inline void tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps,
                                                        float* action, float* std_cache, float* log_prob,
                                                        int64_t D, float stabilizer, double half_log_two_pi) {
    double row_log_prob = 0.0;
    for (int64_t d = 0; d < D; ++d) {
        const float std_value = expf(log_std[d]);
        const float u = mean[d] + std_value * eps[d];
        const float a = tanhf(u);
        std_cache[d] = std_value;
        action[d] = a;
        const double eps_value = static_cast<double>(eps[d]);
        const double squash_correction =
            log(1.0 - static_cast<double>(a) * static_cast<double>(a) + static_cast<double>(stabilizer));
        row_log_prob +=
            -0.5 * eps_value * eps_value - static_cast<double>(log_std[d]) - half_log_two_pi - squash_correction;
    }
    *log_prob = static_cast<float>(row_log_prob);
}

PULSATRIX_HOST_DEVICE inline void tanh_gaussian_backward_element(float a, float std_value, float eps,
                                                                 float grad_action, float grad_log_prob,
                                                                 float stabilizer, float* grad_mean,
                                                                 float* grad_log_std) {
    const float one_minus_a_sq = 1.0f - a * a;
    const float dlogprob_du = 2.0f * a * one_minus_a_sq / (one_minus_a_sq + stabilizer);
    const float grad_u = grad_action * one_minus_a_sq + grad_log_prob * dlogprob_du;
    *grad_mean = grad_u;
    *grad_log_std = grad_u * std_value * eps - grad_log_prob;
}

}  // namespace rows
}  // namespace pulsatrix
