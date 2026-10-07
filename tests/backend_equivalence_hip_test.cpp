#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "backend_primitive_equivalence_cases.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/hip_backend.hpp"

// Phase 1.6's exit gate, HIP half: CPUBackend and HIPBackend must produce numerically
// equivalent results on DeviceBackend's own primitives, on real gfx1151 hardware.
// Deterministic seeded pseudo-random inputs, structurally identical to
// backend_equivalence_test.cpp's CUDA suite -- same helper, same seeds, same shapes, same
// tolerance. The identity is the point: it makes the CUDA and HIP results directly
// comparable rather than merely both green.
//
// On the three-way gate: the charter asks for CPU/CUDA/HIP agreement, but no available node
// carries both an NVIDIA and an AMD GPU. CPUBackend is this codebase's declared reference
// implementation (cpu_backend.hpp), CPU == CUDA is already committed and RTX-3060-verified
// from Phase 1.5, and CPU == HIP is measured here -- so the three-way relation holds
// transitively through the reference. See campaign Decision Point 3; the direct CUDA/HIP arm
// lives in three_way_equivalence_test.cpp, written but unrunnable here.
//
// kBackendEquivalenceTolerance is deliberately the same 1e-4 constant the CUDA suite uses,
// for the same reason: GPU reduction order (hipBLAS's internal tiling/accumulation strategy
// vs. CPUBackend's sequential accumulation) legitimately produces different floating-point
// rounding for gemm, and that is not a bug. elementwise/add/fill have no reduction (one FLOP
// or fewer per output element), so they match far more tightly than this bound requires; a
// single uniform bound is kept for comparability with the CUDA suite, not because any case
// is near it.
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
void ExpectElementwiseMatchesCPU(CPUBackend& cpu, HIPBackend& hip, ElementwiseOp op, unsigned seed) {
    std::vector<float> in = RandomVector(1000, seed);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(op, in.data(), cpu_out.data(), in.size());

    void* device_in = hip.allocate(in.size() * sizeof(float));
    void* device_out = hip.allocate(in.size() * sizeof(float));
    hip.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.elementwise(op, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> hip_out(in.size(), 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_in);
    hip.free(device_out);
}

class HipBackendEquivalenceTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    HIPBackend hip;
};

TEST_F(HipBackendEquivalenceTest, GemmMatchesCPUBackendOnRandomInput) {
    constexpr size_t m = 17, k = 23, n = 11;  // deliberately non-square, non-power-of-two
    std::vector<float> a = RandomVector(m * k, /*seed=*/1);
    std::vector<float> b = RandomVector(k * n, /*seed=*/2);

    std::vector<float> cpu_out(m * n, 0.0f);
    cpu.gemm(a.data(), b.data(), cpu_out.data(), m, k, n);

    void* device_a = hip.allocate(a.size() * sizeof(float));
    void* device_b = hip.allocate(b.size() * sizeof(float));
    void* device_out = hip.allocate(m * n * sizeof(float));
    hip.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), m, k,
             n);
    std::vector<float> hip_out(m * n, 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_a);
    hip.free(device_b);
    hip.free(device_out);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseReluMatchesCPUBackendOnRandomInput) {
    std::vector<float> in = RandomVector(1000, /*seed=*/3);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(ElementwiseOp::Relu, in.data(), cpu_out.data(), in.size());

    void* device_in = hip.allocate(in.size() * sizeof(float));
    void* device_out = hip.allocate(in.size() * sizeof(float));
    hip.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.elementwise(ElementwiseOp::Relu, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> hip_out(in.size(), 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_in);
    hip.free(device_out);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseNegMatchesCPUBackendOnRandomInput) {
    std::vector<float> in = RandomVector(1000, /*seed=*/4);

    std::vector<float> cpu_out(in.size(), 0.0f);
    cpu.elementwise(ElementwiseOp::Neg, in.data(), cpu_out.data(), in.size());

    void* device_in = hip.allocate(in.size() * sizeof(float));
    void* device_out = hip.allocate(in.size() * sizeof(float));
    hip.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.elementwise(ElementwiseOp::Neg, static_cast<float*>(device_in), static_cast<float*>(device_out), in.size());
    std::vector<float> hip_out(in.size(), 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_in);
    hip.free(device_out);
}

TEST_F(HipBackendEquivalenceTest, AddMatchesCPUBackendOnRandomInput) {
    std::vector<float> a = RandomVector(1000, /*seed=*/5);
    std::vector<float> b = RandomVector(1000, /*seed=*/6);

    std::vector<float> cpu_out(a.size(), 0.0f);
    cpu.add(a.data(), b.data(), cpu_out.data(), a.size());

    void* device_a = hip.allocate(a.size() * sizeof(float));
    void* device_b = hip.allocate(b.size() * sizeof(float));
    void* device_out = hip.allocate(a.size() * sizeof(float));
    hip.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.add(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), a.size());
    std::vector<float> hip_out(a.size(), 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_a);
    hip.free(device_b);
    hip.free(device_out);
}

// One case beyond Phase 1.5's four. `fill` is a DeviceBackend primitive and the charter's
// gate says "shared ops", so covering it closes a real, if small, gap in the CUDA suite --
// planned into the Stage 4 checklist up front, not discovered mid-objective.
TEST_F(HipBackendEquivalenceTest, FillMatchesCPUBackendOnRandomInput) {
    constexpr size_t n = 1000;
    constexpr float value = -0.375f;  // exactly representable; no rounding to argue about

    std::vector<float> cpu_out(n, 0.0f);
    cpu.fill(cpu_out.data(), value, n);

    void* device_out = hip.allocate(n * sizeof(float));
    hip.fill(device_out, value, n);
    std::vector<float> hip_out(n, 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_out);
}

TEST_F(HipBackendEquivalenceTest, MulMatchesCPUBackendOnRandomInput) {
    std::vector<float> a = RandomVector(1000, /*seed=*/7);
    std::vector<float> b = RandomVector(1000, /*seed=*/8);

    std::vector<float> cpu_out(a.size(), 0.0f);
    cpu.mul(a.data(), b.data(), cpu_out.data(), a.size());

    void* device_a = hip.allocate(a.size() * sizeof(float));
    void* device_b = hip.allocate(b.size() * sizeof(float));
    void* device_out = hip.allocate(a.size() * sizeof(float));
    hip.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    hip.mul(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), a.size());
    std::vector<float> hip_out(a.size(), 0.0f);
    hip.copy(hip_out.data(), device_out, hip_out.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        EXPECT_NEAR(cpu_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }

    hip.free(device_a);
    hip.free(device_b);
    hip.free(device_out);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseTanhMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, hip, ElementwiseOp::Tanh, /*seed=*/9);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseSigmoidMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, hip, ElementwiseOp::Sigmoid, /*seed=*/10);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseSiluMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, hip, ElementwiseOp::Silu, /*seed=*/11);
}

TEST_F(HipBackendEquivalenceTest, ElementwiseGeluTanhMatchesCPUBackendOnRandomInput) {
    ExpectElementwiseMatchesCPU(cpu, hip, ElementwiseOp::GeluTanh, /*seed=*/12);
}

// GPU-native-kernels Mission 1 primitives -- cases shared with the other GPU backend.
PULSATRIX_PRIMITIVE_EQUIVALENCE_TESTS(HipBackendEquivalenceTest, hip)

}  // namespace
}  // namespace pulsatrix
