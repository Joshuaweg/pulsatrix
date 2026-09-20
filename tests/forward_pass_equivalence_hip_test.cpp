#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/hip_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// Phase 1.6's closing exit-gate item: LinearModule/ReluModule forward passes produce
// numerically equivalent output on a CPU-backed vs. a genuinely HIP-backed Tensor, on real
// gfx1151 hardware. Mirrors forward_pass_equivalence_test.cpp (Phase 1.5 Mission 3) exactly.
//
// This is also the real test of whether Phase 1.5 Mission 3's two blocking-bug fixes were
// device-generic or quietly CUDA-specific: Tensor's copy and initializer-list constructors
// selecting CopyDirection from the device tag rather than hardcoding HostToHost, and
// LinearModule/ReluModule threading an explicit DeviceType to every internally-constructed
// Tensor member. Static reading says both are generic -- Tensor branches on
// `device_ == DeviceType::Cpu`, not on a vendor -- but a second vendor backend is the only
// thing that can actually distinguish "generic" from "happens to work for the one case
// that existed."
namespace exai {
namespace {

// Same bound as the primitive-level suite, for the same reason -- LinearModule::forward
// bottoms out in a single gemm call, so its deviation is that gemm's reduction-order
// deviation, nothing more.
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

// Writes host_values into a Tensor's buffer through its owning backend -- correct whether
// that buffer is host memory (CPUBackend) or device memory (HIPBackend), since it never
// dereferences the pointer directly. Tensor has no constructor for a runtime-sized buffer
// (only zero-init or a compile-time initializer_list), so every Tensor here is built
// zero-init then filled this way.
void WriteValues(DeviceBackend* backend, Tensor& t, const std::vector<float>& host_values) {
    backend->copy(t.data(), host_values.data(), host_values.size() * sizeof(float), CopyDirection::HostToDevice);
}

class HipForwardPassEquivalenceTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    HIPBackend hip;
};

TEST_F(HipForwardPassEquivalenceTest, LinearModuleForwardMatchesCPUBackendOnRandomInput) {
    constexpr int64_t in_features = 17;
    constexpr int64_t out_features = 11;

    LinearModule cpu_linear(in_features, out_features, &cpu);
    LinearModule hip_linear(in_features, out_features, &hip, DeviceType::Hip);

    std::vector<float> weight_values = RandomVector(static_cast<size_t>(in_features * out_features), /*seed=*/10);
    std::vector<float> bias_values = RandomVector(static_cast<size_t>(out_features), /*seed=*/11);

    auto cpu_params = cpu_linear.parameters();
    WriteValues(&cpu, *cpu_params[0].value, weight_values);
    WriteValues(&cpu, *cpu_params[1].value, bias_values);
    auto hip_params = hip_linear.parameters();
    WriteValues(&hip, *hip_params[0].value, weight_values);
    WriteValues(&hip, *hip_params[1].value, bias_values);

    std::vector<float> input_values = RandomVector(static_cast<size_t>(in_features), /*seed=*/12);
    Tensor cpu_input(Shape({in_features}), &cpu);
    WriteValues(&cpu, cpu_input, input_values);
    Tensor hip_input(Shape({in_features}), &hip, DeviceType::Hip);
    WriteValues(&hip, hip_input, input_values);

    Tensor cpu_output = cpu_linear.forward(cpu_input);
    Tensor hip_output = hip_linear.forward(hip_input);
    EXPECT_EQ(hip_output.device(), DeviceType::Hip);

    std::vector<float> hip_output_host(static_cast<size_t>(out_features), 0.0f);
    hip.copy(hip_output_host.data(), hip_output.data(), hip_output_host.size() * sizeof(float),
             CopyDirection::DeviceToHost);

    for (int64_t i = 0; i < out_features; ++i) {
        EXPECT_NEAR(cpu_output.data()[i], hip_output_host[static_cast<size_t>(i)], kBackendEquivalenceTolerance)
            << "mismatch at index " << i;
    }
}

TEST_F(HipForwardPassEquivalenceTest, ReluModuleForwardMatchesCPUBackendOnRandomInput) {
    ReluModule cpu_relu(&cpu);
    ReluModule hip_relu(&hip, DeviceType::Hip);

    std::vector<float> input_values = RandomVector(1000, /*seed=*/20);  // mix of positive/negative values
    Tensor cpu_input(Shape({1000}), &cpu);
    WriteValues(&cpu, cpu_input, input_values);
    Tensor hip_input(Shape({1000}), &hip, DeviceType::Hip);
    WriteValues(&hip, hip_input, input_values);

    Tensor cpu_output = cpu_relu.forward(cpu_input);
    Tensor hip_output = hip_relu.forward(hip_input);
    EXPECT_EQ(hip_output.device(), DeviceType::Hip);

    std::vector<float> hip_output_host(1000, 0.0f);
    hip.copy(hip_output_host.data(), hip_output.data(), hip_output_host.size() * sizeof(float),
             CopyDirection::DeviceToHost);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_NEAR(cpu_output.data()[i], hip_output_host[i], kBackendEquivalenceTolerance)
            << "mismatch at flat index " << i;
    }
}

}  // namespace
}  // namespace exai
