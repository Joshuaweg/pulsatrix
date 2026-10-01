#include <gtest/gtest.h>

#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cuda_backend.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/tensor.hpp"

// Mirrors CPUBackendTest's exact test shape (allocate/free round-trip, zero-byte
// convention, copy round-trip, fill correctness) -- deliberate, so the two backends'
// test suites are structurally comparable, supporting Mission 1's numerical-equivalence
// suite. Run against real hardware (RTX 3060), not mocked.

namespace pulsatrix {
namespace {

class CUDABackendTest : public ::testing::Test {
protected:
    CUDABackend backend;
};

TEST_F(CUDABackendTest, AllocateReturnsNonNullForPositiveSize) {
    void* ptr = backend.allocate(64);
    ASSERT_NE(ptr, nullptr);
    backend.free(ptr);
}

TEST_F(CUDABackendTest, AllocateZeroBytesReturnsNullByConvention) {
    void* ptr = backend.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CUDABackendTest, FreeOnNullptrIsSafe) {
    EXPECT_NO_THROW(backend.free(nullptr));
}

TEST_F(CUDABackendTest, CopyRoundTripPreservesData) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    void* device_ptr = backend.allocate(src.size() * sizeof(float));
    ASSERT_NE(device_ptr, nullptr);

    backend.copy(device_ptr, src.data(), src.size() * sizeof(float), CopyDirection::HostToDevice);

    std::vector<float> dst(src.size(), 0.0f);
    backend.copy(dst.data(), device_ptr, src.size() * sizeof(float), CopyDirection::DeviceToHost);

    for (size_t i = 0; i < src.size(); ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
    backend.free(device_ptr);
}

TEST_F(CUDABackendTest, FillSetsEveryElementToValue) {
    constexpr size_t n = 5;
    void* device_ptr = backend.allocate(n * sizeof(float));
    ASSERT_NE(device_ptr, nullptr);

    backend.fill(device_ptr, 3.5f, n);

    std::vector<float> host(n, 0.0f);
    backend.copy(host.data(), device_ptr, n * sizeof(float), CopyDirection::DeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(host[i], 3.5f);
    }
    backend.free(device_ptr);
}

TEST_F(CUDABackendTest, FillOnZeroElementsIsSafe) {
    EXPECT_NO_THROW(backend.fill(nullptr, 1.0f, 0));
}

// Real-hardware proof that Tensor's initializer-list and copy constructors pick the right
// CopyDirection against a genuine CUDABackend (device_backend_test.cpp's mock only checks
// which enum value is chosen; this confirms the actual bytes round-trip correctly through
// real cudaMemcpy calls). Phase 1.5 Mission 3, Objective 1.
TEST_F(CUDABackendTest, TensorInitializerListConstructorRoundTripsThroughRealCUDA) {
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f}, DeviceType::Cuda);

    std::vector<float> host(3, 0.0f);
    backend.copy(host.data(), t.data(), host.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(host[0], 1.0f);
    EXPECT_FLOAT_EQ(host[1], 2.0f);
    EXPECT_FLOAT_EQ(host[2], 3.0f);
}

TEST_F(CUDABackendTest, TensorCopyConstructorRoundTripsThroughRealCUDA) {
    Tensor original(Shape({3}), &backend, {4.0f, 5.0f, 6.0f}, DeviceType::Cuda);
    Tensor copy(original);

    EXPECT_NE(copy.data(), original.data());  // independent device buffer

    std::vector<float> host(3, 0.0f);
    backend.copy(host.data(), copy.data(), host.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(host[0], 4.0f);
    EXPECT_FLOAT_EQ(host[1], 5.0f);
    EXPECT_FLOAT_EQ(host[2], 6.0f);
}

TEST_F(CUDABackendTest, GemmComputesHandVerified2x2Product) {
    // Mirrors CPUBackendTest::GemmComputesHandVerified2x2Product exactly.
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> A*B = [[19, 22], [43, 50]]
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};

    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(4 * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), 2,
                 2, 2);

    std::vector<float> out(4, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 19.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 43.0f);
    EXPECT_FLOAT_EQ(out[3], 50.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, GemmHandlesNonSquareDimensions) {
    // Mirrors CPUBackendTest::GemmHandlesNonSquareDimensions -- catches a transpose-only
    // (wrong) result, which a shape-only test would miss for a square case.
    // A (2x3) * B (3x2): A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]] -> [[58,64],[139,154]]
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {7, 8, 9, 10, 11, 12};

    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(4 * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.gemm(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out), 2,
                 3, 2);

    std::vector<float> out(4, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 58.0f);
    EXPECT_FLOAT_EQ(out[1], 64.0f);
    EXPECT_FLOAT_EQ(out[2], 139.0f);
    EXPECT_FLOAT_EQ(out[3], 154.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, ElementwiseReluClampsNegativeValuesToZero) {
    std::vector<float> in = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    void* device_in = backend.allocate(in.size() * sizeof(float));
    void* device_out = backend.allocate(in.size() * sizeof(float));
    backend.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.elementwise(ElementwiseOp::Relu, static_cast<float*>(device_in), static_cast<float*>(device_out),
                        in.size());

    std::vector<float> out(in.size(), 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 0.5f);
    EXPECT_FLOAT_EQ(out[4], 2.0f);

    backend.free(device_in);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, ElementwiseNegNegatesEveryElement) {
    std::vector<float> in = {1.0f, -1.0f, 0.0f, 3.5f};
    void* device_in = backend.allocate(in.size() * sizeof(float));
    void* device_out = backend.allocate(in.size() * sizeof(float));
    backend.copy(device_in, in.data(), in.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.elementwise(ElementwiseOp::Neg, static_cast<float*>(device_in), static_cast<float*>(device_out),
                        in.size());

    std::vector<float> out(in.size(), 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], -3.5f);

    backend.free(device_in);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, ElementwiseHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.elementwise(ElementwiseOp::Relu, nullptr, nullptr, 0));
}

TEST_F(CUDABackendTest, ElementwiseSupportsInPlaceAliasing) {
    std::vector<float> buf = {-1.0f, 2.0f, -3.0f};
    void* device_buf = backend.allocate(buf.size() * sizeof(float));
    backend.copy(device_buf, buf.data(), buf.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.elementwise(ElementwiseOp::Relu, static_cast<float*>(device_buf), static_cast<float*>(device_buf),
                        buf.size());

    backend.copy(buf.data(), device_buf, buf.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(buf[0], 0.0f);
    EXPECT_FLOAT_EQ(buf[1], 2.0f);
    EXPECT_FLOAT_EQ(buf[2], 0.0f);

    backend.free(device_buf);
}

TEST_F(CUDABackendTest, AddComputesElementwiseSum) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(a.size() * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.add(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out),
                a.size());

    std::vector<float> out(3, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 22.0f);
    EXPECT_FLOAT_EQ(out[2], 33.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, AddSupportsInPlaceAccumulation) {
    std::vector<float> acc = {1.0f, 2.0f, 3.0f};
    std::vector<float> delta = {0.5f, 0.5f, 0.5f};
    void* device_acc = backend.allocate(acc.size() * sizeof(float));
    void* device_delta = backend.allocate(delta.size() * sizeof(float));
    backend.copy(device_acc, acc.data(), acc.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_delta, delta.data(), delta.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.add(static_cast<float*>(device_acc), static_cast<float*>(device_delta), static_cast<float*>(device_acc),
                acc.size());  // out aliases a

    backend.copy(acc.data(), device_acc, acc.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(acc[0], 1.5f);
    EXPECT_FLOAT_EQ(acc[1], 2.5f);
    EXPECT_FLOAT_EQ(acc[2], 3.5f);

    backend.free(device_acc);
    backend.free(device_delta);
}

TEST_F(CUDABackendTest, AddHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.add(nullptr, nullptr, nullptr, 0));
}

TEST_F(CUDABackendTest, MulComputesElementwiseProduct) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    void* device_a = backend.allocate(a.size() * sizeof(float));
    void* device_b = backend.allocate(b.size() * sizeof(float));
    void* device_out = backend.allocate(a.size() * sizeof(float));
    backend.copy(device_a, a.data(), a.size() * sizeof(float), CopyDirection::HostToDevice);
    backend.copy(device_b, b.data(), b.size() * sizeof(float), CopyDirection::HostToDevice);

    backend.mul(static_cast<float*>(device_a), static_cast<float*>(device_b), static_cast<float*>(device_out),
                a.size());

    std::vector<float> out(3, 0.0f);
    backend.copy(out.data(), device_out, out.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 40.0f);
    EXPECT_FLOAT_EQ(out[2], 90.0f);

    backend.free(device_a);
    backend.free(device_b);
    backend.free(device_out);
}

TEST_F(CUDABackendTest, MulHandlesZeroLengthGracefully) {
    EXPECT_NO_THROW(backend.mul(nullptr, nullptr, nullptr, 0));
}

// Tensor::to(target, target_backend) on real CUDA hardware: the direction-selection logic is
// covered in tensor_test.cpp against a host-simulated backend; these prove the copies are
// valid against a genuine device allocator.
TEST_F(CUDABackendTest, TensorToDeviceAndBackRoundTripsValues) {
    CPUBackend cpu;
    Tensor t(Shape({4}), &cpu, {1.5f, -2.0f, 0.0f, 9.25f});

    t.to(DeviceType::Cuda, &backend);
    EXPECT_EQ(t.device(), DeviceType::Cuda);
    std::vector<float> on_device(4, 0.0f);
    backend.copy(on_device.data(), t.data(), on_device.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(on_device[3], 9.25f);

    t.to(DeviceType::Cpu, &cpu);
    EXPECT_EQ(t.device(), DeviceType::Cpu);
    EXPECT_FLOAT_EQ(t.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(t.data()[1], -2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 9.25f);
}

TEST_F(CUDABackendTest, TensorMovedToDeviceFeedsAModuleForwardPass) {
    CPUBackend cpu;
    Tensor input(Shape({2, 2}), &cpu, {-1.0f, 2.0f, 3.0f, -4.0f});
    ReluModule relu(&backend, DeviceType::Cuda);

    input.to(DeviceType::Cuda, &backend);
    Tensor output = relu.forward(input);
    output.to(DeviceType::Cpu, &cpu);

    EXPECT_FLOAT_EQ(output.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 0.0f);
}

TEST_F(CUDABackendTest, TensorToSecondBackendInstanceCopiesDeviceToDevice) {
    CUDABackend other;
    Tensor t(Shape({2}), &backend, {3.0f, 4.0f}, DeviceType::Cuda);

    t.to(DeviceType::Cuda, &other);

    std::vector<float> host(2, 0.0f);
    other.copy(host.data(), t.data(), host.size() * sizeof(float), CopyDirection::DeviceToHost);
    EXPECT_FLOAT_EQ(host[0], 3.0f);
    EXPECT_FLOAT_EQ(host[1], 4.0f);
}

TEST_F(CUDABackendTest, ReportsCudaDeviceAndTagsUntaggedTensors) {
    EXPECT_EQ(backend.device(), DeviceType::Cuda);
    EXPECT_EQ(Tensor(Shape({3}), &backend).device(), DeviceType::Cuda);
}

}  // namespace
}  // namespace pulsatrix
