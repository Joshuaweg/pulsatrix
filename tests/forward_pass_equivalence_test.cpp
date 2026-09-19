#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/cuda_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

// Phase 1.5's closing exit-gate item: LinearModule/ReluModule forward passes produce
// numerically equivalent output on a CPU-backed vs. a genuinely CUDA-backed Tensor, on real
// hardware. See mission_forward_pass_equivalence.md, Objective 4.
namespace exai {
namespace {

// Same bound as Mission 1's backend_equivalence_test.cpp -- looser than exact equality to
// allow legitimate GPU reduction-order variance in LinearModule's underlying gemm call.
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
// that buffer is host memory (CPUBackend) or device memory (CUDABackend), since it never
// dereferences the pointer directly. Tensor has no constructor for a runtime-sized buffer
// (only zero-init or a compile-time initializer_list), so every Tensor here is built
// zero-init then filled this way.
void WriteValues(DeviceBackend* backend, Tensor& t, const std::vector<float>& host_values) {
    backend->copy(t.data(), host_values.data(), host_values.size() * sizeof(float), CopyDirection::HostToDevice);
}

class ForwardPassEquivalenceTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    CUDABackend cuda;
};

TEST_F(ForwardPassEquivalenceTest, LinearModuleForwardMatchesCPUBackendOnRandomInput) {
    constexpr int64_t in_features = 17;
    constexpr int64_t out_features = 11;

    LinearModule cpu_linear(in_features, out_features, &cpu);
    LinearModule cuda_linear(in_features, out_features, &cuda, DeviceType::Cuda);

    std::vector<float> weight_values = RandomVector(static_cast<size_t>(in_features * out_features), /*seed=*/10);
    std::vector<float> bias_values = RandomVector(static_cast<size_t>(out_features), /*seed=*/11);

    auto cpu_params = cpu_linear.parameters();
    WriteValues(&cpu, *cpu_params[0].value, weight_values);
    WriteValues(&cpu, *cpu_params[1].value, bias_values);
    auto cuda_params = cuda_linear.parameters();
    WriteValues(&cuda, *cuda_params[0].value, weight_values);
    WriteValues(&cuda, *cuda_params[1].value, bias_values);

    std::vector<float> input_values = RandomVector(static_cast<size_t>(in_features), /*seed=*/12);
    Tensor cpu_input(Shape({in_features}), &cpu);
    WriteValues(&cpu, cpu_input, input_values);
    Tensor cuda_input(Shape({in_features}), &cuda, DeviceType::Cuda);
    WriteValues(&cuda, cuda_input, input_values);

    Tensor cpu_output = cpu_linear.forward(cpu_input);
    Tensor cuda_output = cuda_linear.forward(cuda_input);
    EXPECT_EQ(cuda_output.device(), DeviceType::Cuda);

    std::vector<float> cuda_output_host(static_cast<size_t>(out_features), 0.0f);
    cuda.copy(cuda_output_host.data(), cuda_output.data(), cuda_output_host.size() * sizeof(float),
              CopyDirection::DeviceToHost);

    for (int64_t i = 0; i < out_features; ++i) {
        EXPECT_NEAR(cpu_output.data()[i], cuda_output_host[static_cast<size_t>(i)], kBackendEquivalenceTolerance)
            << "mismatch at index " << i;
    }
}

TEST_F(ForwardPassEquivalenceTest, ReluModuleForwardMatchesCPUBackendOnRandomInput) {
    ReluModule cpu_relu(&cpu);
    ReluModule cuda_relu(&cuda, DeviceType::Cuda);

    std::vector<float> input_values = RandomVector(1000, /*seed=*/20);  // mix of positive/negative values
    Tensor cpu_input(Shape({1000}), &cpu);
    WriteValues(&cpu, cpu_input, input_values);
    Tensor cuda_input(Shape({1000}), &cuda, DeviceType::Cuda);
    WriteValues(&cuda, cuda_input, input_values);

    Tensor cpu_output = cpu_relu.forward(cpu_input);
    Tensor cuda_output = cuda_relu.forward(cuda_input);
    EXPECT_EQ(cuda_output.device(), DeviceType::Cuda);

    std::vector<float> cuda_output_host(1000, 0.0f);
    cuda.copy(cuda_output_host.data(), cuda_output.data(), cuda_output_host.size() * sizeof(float),
              CopyDirection::DeviceToHost);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_NEAR(cpu_output.data()[i], cuda_output_host[i], kBackendEquivalenceTolerance)
            << "mismatch at flat index " << i;
    }
}

}  // namespace
}  // namespace exai
