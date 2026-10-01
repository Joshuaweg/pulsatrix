#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <type_traits>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {

class MockDeviceBackend : public DeviceBackend {
public:
    MOCK_METHOD(DeviceType, device, (), (const, noexcept, override));
    MOCK_METHOD(void*, allocate, (size_t bytes), (override));
    MOCK_METHOD(void, free, (void* ptr), (noexcept, override));
    MOCK_METHOD(void, copy, (void* dst, const void* src, size_t bytes, CopyDirection dir), (override));
    MOCK_METHOD(void, fill, (void* ptr, float value, size_t n), (override));
    MOCK_METHOD(void, gemm, (const float* a, const float* b, float* out, size_t m, size_t k, size_t n), (override));
    MOCK_METHOD(void, elementwise, (ElementwiseOp op, const float* in, float* out, size_t n), (override));
    MOCK_METHOD(void, add, (const float* a, const float* b, float* out, size_t n), (override));
    MOCK_METHOD(void, mul, (const float* a, const float* b, float* out, size_t n), (override));
};

TEST(DeviceBackendInterface, HasVirtualDestructor) {
    static_assert(std::has_virtual_destructor_v<DeviceBackend>,
                  "DeviceBackend must have a virtual destructor -- it is deleted via base pointer");
}

TEST(DeviceBackendInterface, IsCallableThroughBasePointer) {
    MockDeviceBackend mock;
    DeviceBackend& backend = mock;

    EXPECT_CALL(mock, allocate(128)).WillOnce(::testing::Return(reinterpret_cast<void*>(0x1)));
    void* ptr = backend.allocate(128);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(0x1));
}

TEST(DeviceBackendInterface, EveryPureVirtualIsMockable) {
    // If MockDeviceBackend compiled at all, every pure-virtual method above was
    // successfully overridden -- this test exists so the interface's completeness
    // is verified by the build itself, not just by inspection.
    MockDeviceBackend mock;
    EXPECT_CALL(mock, free(::testing::_)).Times(1);
    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, 0, CopyDirection::HostToHost)).Times(1);
    EXPECT_CALL(mock, fill(::testing::_, 0.0f, 0)).Times(1);
    EXPECT_CALL(mock, gemm(::testing::_, ::testing::_, ::testing::_, 0, 0, 0)).Times(1);
    EXPECT_CALL(mock, elementwise(ElementwiseOp::Relu, ::testing::_, ::testing::_, 0)).Times(1);
    EXPECT_CALL(mock, add(::testing::_, ::testing::_, ::testing::_, 0)).Times(1);

    mock.free(nullptr);
    mock.copy(nullptr, nullptr, 0, CopyDirection::HostToHost);
    mock.fill(nullptr, 0.0f, 0);
    mock.gemm(nullptr, nullptr, nullptr, 0, 0, 0);
    mock.elementwise(ElementwiseOp::Relu, nullptr, nullptr, 0);
    mock.add(nullptr, nullptr, nullptr, 0);
}

}  // namespace
}  // namespace pulsatrix
