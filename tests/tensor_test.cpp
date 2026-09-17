#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

#include "exai/cpu_backend.hpp"
#include "exai/device_backend.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace {

class MockDeviceBackend : public DeviceBackend {
public:
    MOCK_METHOD(void*, allocate, (size_t bytes), (override));
    MOCK_METHOD(void, free, (void* ptr), (noexcept, override));
    MOCK_METHOD(void, copy, (void* dst, const void* src, size_t bytes, CopyDirection dir), (override));
    MOCK_METHOD(void, fill, (void* ptr, float value, size_t n), (override));
    MOCK_METHOD(void, gemm, (const float* a, const float* b, float* out, size_t m, size_t k, size_t n), (override));
    MOCK_METHOD(void, elementwise, (ElementwiseOp op, const float* in, float* out, size_t n), (override));
};

class TensorTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST(TensorTypeTraits, MoveOperationsAreNoexcept) {
    static_assert(std::is_nothrow_move_constructible_v<Tensor>,
                  "Tensor move construction must be noexcept -- see cpp_style_guide error-handling table");
    static_assert(std::is_nothrow_move_assignable_v<Tensor>,
                  "Tensor move assignment must be noexcept");
    SUCCEED();
}

TEST_F(TensorTest, ConstructionZeroInitializesBuffer) {
    Tensor t(Shape({2, 2}), &backend);
    EXPECT_EQ(t.numel(), 4);
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
    }
}

TEST_F(TensorTest, ConstructionFromInitializerListCopiesData) {
    Tensor t(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 4.0f);
}

TEST_F(TensorTest, RankZeroScalarConstructsWithOneElement) {
    Tensor t(Shape({}), &backend, {42.0f});
    EXPECT_EQ(t.numel(), 1);
    EXPECT_FLOAT_EQ(t.data()[0], 42.0f);
}

TEST_F(TensorTest, ZeroElementTensorHasNullData) {
    Tensor t(Shape({0}), &backend);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.data(), nullptr);
}

TEST_F(TensorTest, DefaultDeviceIsCpu) {
    Tensor t(Shape({2}), &backend);
    EXPECT_EQ(t.device(), DeviceType::Cpu);
}

TEST_F(TensorTest, MoveConstructorTransfersDataPointer) {
    Tensor original(Shape({2}), &backend, {1.0f, 2.0f});
    const float* original_ptr = original.data();

    Tensor moved(std::move(original));

    EXPECT_EQ(moved.data(), original_ptr);
    EXPECT_FLOAT_EQ(moved.data()[0], 1.0f);
    EXPECT_EQ(original.data(), nullptr);  // moved-from: data pointer released, safe to destroy
}

TEST_F(TensorTest, MoveAssignmentTransfersDataPointer) {
    Tensor a(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor b(Shape({3}), &backend, {9.0f, 9.0f, 9.0f});
    const float* a_ptr = a.data();

    b = std::move(a);

    EXPECT_EQ(b.data(), a_ptr);
    EXPECT_EQ(b.numel(), 2);
    EXPECT_EQ(a.data(), nullptr);
}

TEST_F(TensorTest, CopyConstructorProducesIndependentBuffer) {
    Tensor original(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor copy(original);

    EXPECT_NE(copy.data(), original.data());
    EXPECT_FLOAT_EQ(copy.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(copy.data()[1], 2.0f);

    copy.data()[0] = 9.0f;
    EXPECT_FLOAT_EQ(original.data()[0], 1.0f);  // original unaffected by mutating the copy
}

TEST_F(TensorTest, CopyAssignmentProducesIndependentBuffer) {
    Tensor original(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor other(Shape({1}), &backend, {0.0f});

    other = original;

    EXPECT_NE(other.data(), original.data());
    EXPECT_EQ(other.numel(), 2);
    EXPECT_FLOAT_EQ(other.data()[0], 1.0f);
}

TEST_F(TensorTest, SelfCopyAssignmentIsSafe) {
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    t = t;
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
}

TEST(TensorDestructorTest, DestructorCallsBackendFreeExactlyOnce) {
    MockDeviceBackend mock;
    void* fake_ptr = reinterpret_cast<void*>(0x1234);
    EXPECT_CALL(mock, allocate(::testing::_)).WillOnce(::testing::Return(fake_ptr));
    EXPECT_CALL(mock, fill(fake_ptr, 0.0f, 2)).Times(1);
    EXPECT_CALL(mock, free(fake_ptr)).Times(1);

    { Tensor t(Shape({2}), &mock); }  // destructor runs at scope exit
}

}  // namespace
}  // namespace exai
