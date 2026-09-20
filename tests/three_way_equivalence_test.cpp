#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/cuda_backend.hpp"
#include "exai/device_backend.hpp"
#include "exai/hip_backend.hpp"

// The direct CUDA <-> HIP arm of the charter's three-way gate ("CPU/CUDA/HIP must agree
// numerically on shared ops", charter line 106).
//
// STATUS: WRITTEN, NEVER RUN. This file is compiled only when BOTH EXAI_ENABLE_CUDA and
// EXAI_ENABLE_HIP are ON, and no machine available to this project carries both an NVIDIA
// and an AMD GPU -- the CUDA work of Phase 1.5 was verified on an RTX 3060 on a different
// host, and Phase 1.6's gfx1151 host has no NVIDIA device and no CUDA toolkit. Nothing
// below has ever executed. It is committed anyway so the gate is satisfiable the day such a
// node exists, without anyone having to reconstruct what the missing arm should have been.
//
// How the gate is actually closed today, per campaign Decision Point 3: CPUBackend is this
// codebase's declared reference implementation (see cpu_backend.hpp's class comment).
// CPU == CUDA is committed and real-hardware-verified from Phase 1.5
// (tests/backend_equivalence_test.cpp, RTX 3060 / sm_86). CPU == HIP is measured on real
// gfx1151 hardware in tests/backend_equivalence_hip_test.cpp. Equality within a shared
// tolerance against a common reference gives CUDA == HIP transitively, with the caveat that
// the composed bound is 2 * kBackendEquivalenceTolerance rather than the 1e-4 asserted
// below -- which is why this direct arm is worth keeping rather than declaring redundant.
//
// Deliberately uses the same helper, seeds, shapes and tolerance as the CPU-pivot suites, so
// a future run here is directly comparable to the numbers those recorded.
namespace exai {
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

// Runs one op on a device backend and returns the result on the host, so each test below
// reads as "same inputs, two vendors, compare" rather than repeating twelve lines of
// allocate/copy/free per arm.
std::vector<float> RunGemm(DeviceBackend& backend, const std::vector<float>& a, const std::vector<float>& b,
                           size_t m, size_t k, size_t n) {
    void* da = backend.allocate(a.size() * sizeof(float));
    void* db = backend.allocate(b.size() * sizeof(float));
    void* dout = backend.allocate(m * n * sizeof(float));
    backend.copy(da, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(db, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.gemm(static_cast<float*>(da), static_cast<float*>(db), static_cast<float*>(dout), m, k, n);
    std::vector<float> out(m * n, 0.0f);
    backend.copy(out.data(), dout, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    backend.free(da);
    backend.free(db);
    backend.free(dout);
    return out;
}

std::vector<float> RunElementwise(DeviceBackend& backend, ElementwiseOp op, const std::vector<float>& in) {
    void* din = backend.allocate(in.size() * sizeof(float));
    void* dout = backend.allocate(in.size() * sizeof(float));
    backend.copy(din, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.elementwise(op, static_cast<float*>(din), static_cast<float*>(dout), in.size());
    std::vector<float> out(in.size(), 0.0f);
    backend.copy(out.data(), dout, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    backend.free(din);
    backend.free(dout);
    return out;
}

std::vector<float> RunAdd(DeviceBackend& backend, const std::vector<float>& a, const std::vector<float>& b) {
    void* da = backend.allocate(a.size() * sizeof(float));
    void* db = backend.allocate(b.size() * sizeof(float));
    void* dout = backend.allocate(a.size() * sizeof(float));
    backend.copy(da, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(db, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.add(static_cast<float*>(da), static_cast<float*>(db), static_cast<float*>(dout), a.size());
    std::vector<float> out(a.size(), 0.0f);
    backend.copy(out.data(), dout, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    backend.free(da);
    backend.free(db);
    backend.free(dout);
    return out;
}

class ThreeWayEquivalenceTest : public ::testing::Test {
protected:
    CUDABackend cuda;
    HIPBackend hip;
};

TEST_F(ThreeWayEquivalenceTest, GemmAgreesBetweenCUDAAndHIP) {
    constexpr size_t m = 17, k = 23, n = 11;
    std::vector<float> a = RandomVector(m * k, /*seed=*/1);
    std::vector<float> b = RandomVector(k * n, /*seed=*/2);

    std::vector<float> cuda_out = RunGemm(cuda, a, b, m, k, n);
    std::vector<float> hip_out = RunGemm(hip, a, b, m, k, n);

    ASSERT_EQ(cuda_out.size(), hip_out.size());
    for (size_t i = 0; i < cuda_out.size(); ++i) {
        EXPECT_NEAR(cuda_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }
}

TEST_F(ThreeWayEquivalenceTest, ElementwiseAgreesBetweenCUDAAndHIP) {
    std::vector<float> in = RandomVector(1000, /*seed=*/3);

    for (ElementwiseOp op : {ElementwiseOp::Relu, ElementwiseOp::Neg}) {
        std::vector<float> cuda_out = RunElementwise(cuda, op, in);
        std::vector<float> hip_out = RunElementwise(hip, op, in);

        ASSERT_EQ(cuda_out.size(), hip_out.size());
        for (size_t i = 0; i < cuda_out.size(); ++i) {
            EXPECT_NEAR(cuda_out[i], hip_out[i], kBackendEquivalenceTolerance)
                << "mismatch at flat index " << i << " for op " << static_cast<int>(op);
        }
    }
}

TEST_F(ThreeWayEquivalenceTest, AddAgreesBetweenCUDAAndHIP) {
    std::vector<float> a = RandomVector(1000, /*seed=*/5);
    std::vector<float> b = RandomVector(1000, /*seed=*/6);

    std::vector<float> cuda_out = RunAdd(cuda, a, b);
    std::vector<float> hip_out = RunAdd(hip, a, b);

    ASSERT_EQ(cuda_out.size(), hip_out.size());
    for (size_t i = 0; i < cuda_out.size(); ++i) {
        EXPECT_NEAR(cuda_out[i], hip_out[i], kBackendEquivalenceTolerance) << "mismatch at flat index " << i;
    }
}

}  // namespace
}  // namespace exai
