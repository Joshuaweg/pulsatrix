#include "exai/cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace exai {

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

}  // namespace exai
