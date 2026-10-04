// Convolution, pooling and spatial-normalization routines shared verbatim by CPUBackend
// (loops) and the GPU kernels (one thread per output element / channel / group). Each body is
// one output's worth of the pre-campaign module host loop, unchanged in expression and
// summation order. Private to src/. GPU-native-kernels Mission 4.
#pragma once

#include <cstdint>

#include "lrp_math.hpp"        // lrp::stabilize
#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE

namespace pulsatrix {
namespace cnn {

// ---- Conv2D (stride 1, no padding; Conv2DModule) ----------------------------------------------
// One example's input (C, H, W); col is (P = C*kh*kw, Q = out_h*out_w), p = (c*kh + i)*kw + j.

PULSATRIX_HOST_DEVICE inline float im2col_element(const float* in, int64_t H, int64_t W, int64_t kh, int64_t kw,
                                                  int64_t out_w, int64_t p, int64_t q) {
    const int64_t j = p % kw;
    const int64_t i = (p / kw) % kh;
    const int64_t c = p / (kw * kh);
    const int64_t oh = q / out_w;
    const int64_t ow = q % out_w;
    return in[(c * H + oh + i) * W + ow + j];
}

// The value col2im accumulates into input pixel (c, ih, iw): its contributions in the order the
// original scatter loop added them -- q ascending, i.e. kernel row i and then column j descending
// -- starting from the pixel's existing value.
PULSATRIX_HOST_DEVICE inline float col2im_pixel(const float* col, float existing, int64_t H, int64_t W, int64_t kh,
                                                int64_t kw, int64_t c, int64_t ih, int64_t iw) {
    const int64_t out_h = H - kh + 1;
    const int64_t out_w = W - kw + 1;
    const int64_t Q = out_h * out_w;
    float acc = existing;
    for (int64_t i = kh - 1; i >= 0; --i) {
        const int64_t oh = ih - i;
        if (oh < 0 || oh >= out_h) {
            continue;
        }
        for (int64_t j = kw - 1; j >= 0; --j) {
            const int64_t ow = iw - j;
            if (ow < 0 || ow >= out_w) {
                continue;
            }
            const int64_t p = (c * kh + i) * kw + j;
            acc += col[p * Q + oh * out_w + ow];
        }
    }
    return acc;
}

// Conv2D epsilon rule for one im2col entry (p, q) of one example: sum over output channels in
// order of (a * w / stab(z)) * r.
PULSATRIX_HOST_DEVICE inline float conv_lrp_col(const float* col, const float* kernel, const float* pre_bias,
                                                const float* r, int64_t p, int64_t q, int64_t P, int64_t Q,
                                                int64_t out_channels, float eps) {
    float acc = 0.0f;
    const float a = col[p * Q + q];
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        const float denom = lrp::stabilize(pre_bias[oc * Q + q], eps);
        acc += (a * kernel[oc * P + p] / denom) * r[oc * Q + q];
    }
    return acc;
}

// ---- Pooling (stride == kernel; Max/AvgPool2DModule) -------------------------------------------
// One (H, W) plane; output element (oh, ow) covers rows oh*kh .. +kh, columns ow*kw .. +kw.

// Strict > keeps the first maximum in row-major window order. argmax is the flat in-plane index.
PULSATRIX_HOST_DEVICE inline void max_pool_window(const float* plane, int64_t W, int64_t kh, int64_t kw, int64_t oh,
                                                  int64_t ow, float* out, float* argmax) {
    float best = -INFINITY;
    int64_t best_flat = 0;
    for (int64_t i = 0; i < kh; ++i) {
        for (int64_t j = 0; j < kw; ++j) {
            const int64_t flat = (oh * kh + i) * W + (ow * kw + j);
            const float v = plane[flat];
            if (v > best) {
                best = v;
                best_flat = flat;
            }
        }
    }
    *out = best;
    *argmax = static_cast<float>(best_flat);
}

PULSATRIX_HOST_DEVICE inline float avg_pool_window(const float* plane, int64_t W, int64_t kh, int64_t kw, int64_t oh,
                                                   int64_t ow) {
    float sum = 0.0f;
    for (int64_t i = 0; i < kh; ++i) {
        for (int64_t j = 0; j < kw; ++j) {
            sum += plane[(oh * kh + i) * W + (ow * kw + j)];
        }
    }
    return sum / static_cast<float>(kh * kw);
}

// Epsilon rule for one window: every input of the window gets (a / K / stab(z)) * r.
PULSATRIX_HOST_DEVICE inline void avg_pool_lrp_window(const float* plane, float* r_plane, int64_t W, int64_t kh,
                                                      int64_t kw, int64_t oh, int64_t ow, float r, float eps) {
    const float inv_k = 1.0f / static_cast<float>(kh * kw);
    float z = 0.0f;
    for (int64_t i = 0; i < kh; ++i) {
        for (int64_t j = 0; j < kw; ++j) {
            z += plane[(oh * kh + i) * W + (ow * kw + j)] * inv_k;
        }
    }
    const float denom = lrp::stabilize(z, eps);
    for (int64_t i = 0; i < kh; ++i) {
        for (int64_t j = 0; j < kw; ++j) {
            const int64_t idx = (oh * kh + i) * W + (ow * kw + j);
            r_plane[idx] = (plane[idx] * inv_k / denom) * r;
        }
    }
}

// ---- BatchNorm (per channel over (N, spatial)) ---------------------------------------------------
// x is (N, C, spatial); element (n, c, s) at n*C*spatial + c*spatial + s.

PULSATRIX_HOST_DEVICE inline void batch_norm_forward_channel(const float* x, const float* gamma, const float* beta,
                                                             float* xhat, float* out, float* std_out, int64_t N,
                                                             int64_t C, int64_t spatial, int64_t c, float eps) {
    const int64_t per_example = C * spatial;
    const auto M = static_cast<float>(N * spatial);
    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            sum += x[n * per_example + c * spatial + s];
        }
    }
    const float mu = sum / M;
    float sum_sq_diff = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const float d = x[n * per_example + c * spatial + s] - mu;
            sum_sq_diff += d * d;
        }
    }
    const float var = sum_sq_diff / M;
    const float std_dev = sqrtf(var + eps);
    std_out[c] = std_dev;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            const float xh = (x[idx] - mu) / std_dev;
            xhat[idx] = xh;
            out[idx] = gamma[c] * xh + beta[c];
        }
    }
}

PULSATRIX_HOST_DEVICE inline void batch_norm_backward_channel(const float* grad_out, const float* gamma,
                                                              const float* xhat, const float* std_in, float* grad_in,
                                                              float* gamma_grad, float* beta_grad, int64_t N,
                                                              int64_t C, int64_t spatial, int64_t c) {
    const int64_t per_example = C * spatial;
    const auto M = static_cast<float>(N * spatial);
    const float std_c = std_in[c];
    float gsum = 0.0f;
    float bsum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            gsum += grad_out[idx] * xhat[idx];
            bsum += grad_out[idx];
        }
    }
    gamma_grad[c] = gsum;
    beta_grad[c] = bsum;
    float sum_grad_xhat = 0.0f;
    float sum_grad_xhat_xhat = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            const float gxh = grad_out[idx] * gamma[c];
            sum_grad_xhat += gxh;
            sum_grad_xhat_xhat += gxh * xhat[idx];
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            const float gxh = grad_out[idx] * gamma[c];
            grad_in[idx] = (M * gxh - sum_grad_xhat - xhat[idx] * sum_grad_xhat_xhat) / (M * std_c);
        }
    }
}

// Folds this batch's channel statistics into the running ones (FND-5), PyTorch's rule:
// running = (1 - momentum) * running + momentum * batch, with the unbiased batch variance. The
// mean is summed in batch_norm_forward_channel's order, so it is the mean that forward used.
// With one value per channel the variance is undefined and the running variance is kept.
PULSATRIX_HOST_DEVICE inline void batch_norm_update_running_channel(const float* x, float* running_mean,
                                                                    float* running_var, int64_t N, int64_t C,
                                                                    int64_t spatial, int64_t c, float momentum) {
    const int64_t per_example = C * spatial;
    const int64_t count = N * spatial;
    const auto M = static_cast<float>(count);
    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            sum += x[n * per_example + c * spatial + s];
        }
    }
    const float mu = sum / M;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    if (count < 2) {
        return;
    }
    float sum_sq_diff = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const float d = x[n * per_example + c * spatial + s] - mu;
            sum_sq_diff += d * d;
        }
    }
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * (sum_sq_diff / (M - 1.0f));
}

// Eval-mode BatchNorm: a fixed per-channel affine map from the running statistics, so every
// element depends only on itself -- never on the rest of the batch.
PULSATRIX_HOST_DEVICE inline void batch_norm_eval_forward_channel(const float* x, const float* gamma,
                                                                  const float* beta, const float* running_mean,
                                                                  const float* running_var, float* xhat, float* out,
                                                                  float* std_out, int64_t N, int64_t C,
                                                                  int64_t spatial, int64_t c, float eps) {
    const int64_t per_example = C * spatial;
    const float mu = running_mean[c];
    const float std_dev = sqrtf(running_var[c] + eps);
    std_out[c] = std_dev;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            const float xh = (x[idx] - mu) / std_dev;
            xhat[idx] = xh;
            out[idx] = gamma[c] * xh + beta[c];
        }
    }
}

// Gradient of the eval-mode affine map: the statistics are constants, so the input gradient is
// grad_out * gamma / std, with no batch coupling. gamma/beta gradients overwritten, per channel.
PULSATRIX_HOST_DEVICE inline void batch_norm_eval_backward_channel(const float* grad_out, const float* gamma,
                                                                   const float* xhat, const float* std_in,
                                                                   float* grad_in, float* gamma_grad,
                                                                   float* beta_grad, int64_t N, int64_t C,
                                                                   int64_t spatial, int64_t c) {
    const int64_t per_example = C * spatial;
    const float scale = gamma[c] / std_in[c];
    float gsum = 0.0f;
    float bsum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * per_example + c * spatial + s;
            gsum += grad_out[idx] * xhat[idx];
            bsum += grad_out[idx];
            grad_in[idx] = grad_out[idx] * scale;
        }
    }
    gamma_grad[c] = gsum;
    beta_grad[c] = bsum;
}

// ---- GroupNorm (per (example, group) over (group_size channels, spatial)) ------------------------

PULSATRIX_HOST_DEVICE inline void group_norm_forward_group(const float* x, const float* gamma, const float* beta,
                                                           float* xhat, float* out, float* std_out, int64_t C,
                                                           int64_t spatial, int64_t num_groups, int64_t group_size,
                                                           int64_t n, int64_t g, float eps) {
    const int64_t base = n * C * spatial;
    const auto N_g = static_cast<float>(group_size * spatial);
    const int64_t c_start = g * group_size;
    const int64_t c_end = c_start + group_size;
    float sum = 0.0f;
    for (int64_t c = c_start; c < c_end; ++c) {
        for (int64_t s = 0; s < spatial; ++s) {
            sum += x[base + c * spatial + s];
        }
    }
    const float mu = sum / N_g;
    float sum_sq_diff = 0.0f;
    for (int64_t c = c_start; c < c_end; ++c) {
        for (int64_t s = 0; s < spatial; ++s) {
            const float d = x[base + c * spatial + s] - mu;
            sum_sq_diff += d * d;
        }
    }
    const float var = sum_sq_diff / N_g;
    const float std_dev = sqrtf(var + eps);
    std_out[n * num_groups + g] = std_dev;
    for (int64_t c = c_start; c < c_end; ++c) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = base + c * spatial + s;
            const float xh = (x[idx] - mu) / std_dev;
            xhat[idx] = xh;
            out[idx] = gamma[c] * xh + beta[c];
        }
    }
}

PULSATRIX_HOST_DEVICE inline void group_norm_backward_group(const float* grad_out, const float* gamma,
                                                            const float* xhat, const float* std_in, float* grad_in,
                                                            int64_t C, int64_t spatial, int64_t num_groups,
                                                            int64_t group_size, int64_t n, int64_t g) {
    const int64_t base = n * C * spatial;
    const auto N_g = static_cast<float>(group_size * spatial);
    const int64_t c_start = g * group_size;
    const int64_t c_end = c_start + group_size;
    const float std_g = std_in[n * num_groups + g];
    float sum_grad_xhat = 0.0f;
    float sum_grad_xhat_xhat = 0.0f;
    for (int64_t c = c_start; c < c_end; ++c) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = base + c * spatial + s;
            const float gxh = grad_out[idx] * gamma[c];
            sum_grad_xhat += gxh;
            sum_grad_xhat_xhat += gxh * xhat[idx];
        }
    }
    for (int64_t c = c_start; c < c_end; ++c) {
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = base + c * spatial + s;
            const float gxh = grad_out[idx] * gamma[c];
            grad_in[idx] = (N_g * gxh - sum_grad_xhat - xhat[idx] * sum_grad_xhat_xhat) / (N_g * std_g);
        }
    }
}

// gamma/beta gradient for channel c: per-example sums over spatial, accumulated over examples in
// order from 0 -- the original "local[c] += per-example sum" association.
PULSATRIX_HOST_DEVICE inline void group_norm_param_grads_channel(const float* grad_out, const float* xhat,
                                                                 float* gamma_grad, float* beta_grad, int64_t N,
                                                                 int64_t C, int64_t spatial, int64_t c) {
    float gamma_acc = 0.0f;
    float beta_acc = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        float gsum = 0.0f;
        float bsum = 0.0f;
        for (int64_t s = 0; s < spatial; ++s) {
            const int64_t idx = n * C * spatial + c * spatial + s;
            gsum += grad_out[idx] * xhat[idx];
            bsum += grad_out[idx];
        }
        gamma_acc += gsum;
        beta_acc += bsum;
    }
    gamma_grad[c] = gamma_acc;
    beta_grad[c] = beta_acc;
}

}  // namespace cnn
}  // namespace pulsatrix
