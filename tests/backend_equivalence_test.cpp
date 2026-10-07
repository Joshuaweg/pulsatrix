#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "backend_primitive_equivalence_cases.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cuda_backend.hpp"
#include "pulsatrix/device_backend.hpp"

// The literal Phase 1.5 exit gate: CPUBackend and CUDABackend must produce numerically
// equivalent results on DeviceBackend's own primitives. Deterministic seeded pseudo-random
// inputs (not hand-picked small values -- CUDABackendTest already covers those) so this
// suite exercises inputs too large to hand-derive, on real hardware.
//
// kBackendEquivalenceTolerance is looser than exact equality: GPU reduction order (cuBLAS's
// internal tiling/accumulation strategy vs. CPUBackend's sequential accumulation) legitimately
// produces different floating-point rounding for gemm, not a bug -- see
// gpu_backend_programming/context_gpu_devicebackend_implementation.md's Numerical-Equivalence
// Test Pattern section. elementwise/add have no reduction (one FLOP per output element), so
// they're expected to match far more tightly than this bound requires; the single tolerance
// is kept intentionally uniform across all four cases for simplicity, not tuned tighter
// per-op, since none of these cases are anywhere near the bound in practice.
namespace pulsatrix {
namespace {

constexpr float kBackendEquivalenceTolerance = 1e-4f;

std::vector<float> RandomVector(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

// Tanh/Sigmoid/Silu share one body: each is a single transcendental per element, so the
// only legitimate CPU/GPU divergence is libm-vs-device-math ulp rounding, far inside the bound.
void ExpectElementwiseMatchesCPU(CPUBackend& cpu, CUDABackend& cuda, ElementwiseOp op, unsigned seed) {
    std::vector<float> in = RandomVector(1000, seed);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(op, in.data(), cpu_out.data(), in.size());

    void* device_in = cuda.allocate(in.size() * sizeof(float));
    void* device_out = cuda.allocate(in.size() * sizeof(float));
    cuda.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.elementwise(op, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> cuda_out(in.size(), 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_in);
    cuda.free(device_out);
}

class BackendEquivalenceTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    CUDABackend cuda;
};

TEST_F(BackendEquivalenceTest, GemmMatchesCPUBackendOnRandomInput) {
    constexpr size_t m = 17, k = 23, n = 11;  // deliberately non-square, non-power-of-two
    std::vector<float> a = RandomVector(m * k, /*seed=*/1);
    std::vector<float> b = RandomVector(k * n, /*seed=*/2);

    std::vector<float> cpu_out(m * n, 0.0f);
    cpu.gemm(a.data(), b.data(), cpu_out.data(), m, k, n);

    void* device_a = cuda.allocate(a.size() * sizeof(float));
    void* device_b = cuda.allocate(b.size() * sizeof(float));
    void* device_out = cuda.allocate(m * n * sizeof(float));
    cuda.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), m, k,
              n);
    std::vector<float> cuda_out(m * n, 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_a);
    cuda.free(device_b);
    cuda.free(device_out);
}

TEST_F(BackendEquivalenceTest, ElementwiseReluMatchesCPUBackendOnRandomInput) {
    std::vector<float> in = RandomVector(1000, /*seed=*/3);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(ElementwiseOp::Relu, in.data(), cpu_out.data(), in.size());

    void* device_in = cuda.allocate(in.size() * sizeof(float));
    void* device_out = cuda.allocate(in.size() * sizeof(float));
    cuda.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.elementwise(ElementwiseOp::Relu, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> cuda_out(in.size(), 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_in);
    cuda.free(device_out);
}

TEST_F(BackendEquivalenceTest, ElementwiseNegMatchesCPUBackendOnRandomInput) {
    std::vector<float> in = RandomVector(1000, /*seed=*/4);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(ElementwiseOp::Neg, in.data(), cpu_out.data(), in.size());

    void* device_in = cuda.allocate(in.size() * sizeof(float));
    void* device_out = cuda.allocate(in.size() * sizeof(float));
    cuda.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.elementwise(ElementwiseOp::Neg, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> cuda_out(in.size(), 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_in);
    cuda.free(device_out);
}

TEST_F(BackendEquivalenceTest, AddMatchesCPUBackendOnRandomInput) {
    std::vector<float> a = RandomVector(1000, /*seed=*/5);
    std::vector<float> b = RandomVector(1000, /*seed=*/6);

    std::vector<float> cpu_out(a.size(), 0.0f);
    cpu.add(a.data(), b.data(), cpu_out.data(), a.size());

    void* device_a = cuda.allocate(a.size() * sizeof(float));
    void* device_b = cuda.allocate(b.size() * sizeof(float));
    void* device_out = cuda.allocate(a.size() * sizeof(float));
    cuda.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.add(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), a.size());
    std::vector<float> cuda_out(a.size(), 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_a);
    cuda.free(device_b);
    cuda.free(device_out);
}

TEST_F(BackendEquivalenceTest, MulMatchesCPUBackendOnRandomInput) {
    std::vector<float> a = RandomVector(1000, /*seed=*/7);
    std::vector<float> b = RandomVector(1000, /*seed=*/8);

    std::vector<float> cpu_out(a.size(), 0.0f);
    cpu.mul(a.data(), b.data(), cpu_out.data(), a.size());

    void* device_a = cuda.allocate(a.size() * sizeof(float));
    void* device_b = cuda.allocate(b.size() * sizeof(float));
    void* device_out = cuda.allocate(a.size() * sizeof(float));
    cuda.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    cuda.mul(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), a.size());
    std::vector<float> cuda_out(a.size(), 0.0f);
    cuda.copy(cuda_out.data(), device_out, cuda_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], cuda_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    cuda.free(device_a);
    cuda.free(device_b);
    cuda.free(device_out);
}

TEST_F(BackendEquivalenceTest, ElementwiseTanhMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, cuda, ElementwiseOp::Tanh, /*seed=*/9);
}

TEST_F(BackendEquivalenceTest, ElementwiseSigmoidMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, cuda, ElementwiseOp::Sigmoid, /*seed=*/10);
}

TEST_F(BackendEquivalenceTest, ElementwiseSiluMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, cuda, ElementwiseOp::Silu, /*seed=*/11);
}

TEST_F(BackendEquivalenceTest, ElementwiseGeluTanhMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, cuda, ElementwiseOp::GeluTanh, /*seed=*/12);
}

// GPU-native-kernels Mission 1 primitives -- cases shared with the other GPU backend.
PULSATRIX_PRIMITIVE_EQUIVALENCE_TESTS(BackendEquivalenceTest, cuda)

}  // namespace
}  // namespace pulsatrix
