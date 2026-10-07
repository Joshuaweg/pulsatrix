// Scalar math shared verbatim by CPUBackend (g++) and the GPU kernels (nvcc/hipcc), so the
// CPU reference and the device kernels evaluate literally the same expressions. Private to
// src/. GPU-native-kernels Mission 1b.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define PULSATRIX_HOST_DEVICE __host__ __device__
#else
#define PULSATRIX_HOST_DEVICE
#endif

namespace pulsatrix {
namespace pointwise {

// splitmix64 finalizer over seed + (k + 1) * golden gamma -- a stateless, counter-based
// generator: element k's draw depends only on (seed, k), so any backend, any thread
// schedule, produces the same mask. The top 24 bits fill a float mantissa exactly, giving a
// uniform in [0, 1) with no rounding up to 1.
PULSATRIX_HOST_DEVICE inline float counter_uniform(uint64_t seed, uint64_t k) {
    uint64_t z = seed + (k + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

// GELU, tanh approximation (Hugging Face's gelu_pytorch_tanh, PyTorch's approximate="tanh"):
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the gate of Gemma's GeGLU MLP (LLM-9).
PULSATRIX_HOST_DEVICE inline float gelu_tanh(float x) {
    const float kBeta = 0.7978845608028654f;  // sqrt(2 / pi)
    const float kKappa = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kBeta * (x + kKappa * x * x * x)));
}

// d/dx gelu_tanh(x) = 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3 * 0.044715 x^2).
PULSATRIX_HOST_DEVICE inline float gelu_tanh_grad(float x) {
    const float kBeta = 0.7978845608028654f;
    const float kKappa = 0.044715f;
    const float x2 = x * x;
    const float t = tanhf(kBeta * (x + kKappa * x2 * x));
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * kBeta * (1.0f + 3.0f * kKappa * x2);
}

// Exact GELU (PyTorch's default, Hugging Face's "gelu"): x * Phi(x) = 0.5 x (1 + erf(x / sqrt 2)),
// the MLP activation of BERT and ESM (PLM-1).
PULSATRIX_HOST_DEVICE inline float gelu(float x) {
    const float kInvSqrt2 = 0.7071067811865476f;
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

// d/dx gelu(x) = Phi(x) + x phi(x), with phi the standard normal density.
PULSATRIX_HOST_DEVICE inline float gelu_grad(float x) {
    const float kInvSqrt2 = 0.7071067811865476f;
    const float kInvSqrt2Pi = 0.3989422804014327f;  // 1 / sqrt(2 pi)
    return 0.5f * (1.0f + erff(x * kInvSqrt2)) + x * kInvSqrt2Pi * expf(-0.5f * x * x);
}

// Overflow-free logistic sigmoid: the exponent argument is never positive. Moved here from
// bce_with_logits_loss.cpp unchanged.
PULSATRIX_HOST_DEVICE inline float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

// max(x, 0) - x*y + log1p(exp(-|x|)), in BCEWithLogitsLoss's original evaluation order.
// (x < 0 ? 0 : x) reproduces std::max(x, 0.0f) exactly, including for -0 and NaN, where
// fmaxf would not.
PULSATRIX_HOST_DEVICE inline float bce_with_logits_term(float x, float y) {
    const float relu_x = (x < 0.0f) ? 0.0f : x;
    return relu_x - x * y + log1pf(expf(-fabsf(x)));
}

}  // namespace pointwise
}  // namespace pulsatrix
