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
