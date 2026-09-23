#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    MOCK_METHOD(void, add, (const float* a, const float* b, float* out, size_t n), (override));
    MOCK_METHOD(void, mul, (const float* a, const float* b, float* out, size_t n), (override));
};

class TensorTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Separately-named suite for death tests, per cpp_style_guide convention -- keeps the
// (slower, subprocess-forking) death tests excludable from the fast inner-loop run.
using TensorDeathTest = TensorTest;

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

// std::initializer_list can't be constructed from a runtime-sized buffer in standard
// C++ (no portable public (ptr, size) constructor) -- any caller with runtime-sized data
// (loading weights from a file, Phase 5's Python bindings marshalling a numpy array)
// needs this overload. Same semantics as the initializer_list ctor, just a different
// source container.
TEST_F(TensorTest, ConstructionFromVectorCopiesData) {
    std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t(Shape({2, 2}), &backend, values);
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(t.data()[3], 4.0f);
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// values-constructors are an external boundary (Tensor::from_values in
// bindings/exai_py.cpp passes an externally-supplied values list directly) -- escalated
// from EXAI_ASSERT-only to a real throw per Mission 0's classification table.
TEST_F(TensorTest, ConstructionFromInitializerListThrowsOnSizeMismatch) {
    EXPECT_THROW((Tensor(Shape({2, 2}), &backend, {1.0f, 2.0f})), std::invalid_argument);
}

TEST_F(TensorTest, ConstructionFromVectorThrowsOnSizeMismatch) {
    std::vector<float> values{1.0f, 2.0f};
    EXPECT_THROW((Tensor(Shape({2, 2}), &backend, values)), std::invalid_argument);
}

// The initializer-list ctor's source (std::initializer_list) always lives on the host;
// its destination may not. CopyDirection must reflect device(), not be hardcoded --
// otherwise a CUDA-backed Tensor would issue a HostToHost cudaMemcpy on what's actually a
// host-to-device transfer. Phase 1.5 Mission 3 (mission_forward_pass_equivalence.md), Obj 1.
// No real GPU needed: this only checks which CopyDirection enum value reaches the backend.
TEST_F(TensorTest, ConstructionFromInitializerListUsesHostToHostForCpuTensor) {
    ::testing::NiceMock<MockDeviceBackend> mock;
    ON_CALL(mock, allocate(::testing::_)).WillByDefault(::testing::Invoke(&std::malloc));
    ON_CALL(mock, free(::testing::_)).WillByDefault(::testing::Invoke([](void* p) { std::free(p); }));
    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, CopyDirection::HostToHost)).Times(1);

    Tensor t(Shape({2}), &mock, {1.0f, 2.0f});  // device defaults to Cpu
}

TEST_F(TensorTest, ConstructionFromInitializerListUsesHostToDeviceForNonCpuTensor) {
    ::testing::NiceMock<MockDeviceBackend> mock;
    ON_CALL(mock, allocate(::testing::_)).WillByDefault(::testing::Invoke(&std::malloc));
    ON_CALL(mock, free(::testing::_)).WillByDefault(::testing::Invoke([](void* p) { std::free(p); }));
    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, CopyDirection::HostToDevice)).Times(1);

    Tensor t(Shape({2}), &mock, {1.0f, 2.0f}, DeviceType::Cuda);
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

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// Tensor's own documented invariant is "data() == nullptr iff numel() == 0". Before this
// fix, the moved-from Tensor's shape_ was left in std::move's unspecified-but-valid state
// (an empty-dims Shape), which by the rank-0 scalar convention has numel() == 1 --
// inconsistent with data() == nullptr, so a subsequent at()/operator[] call computed a
// valid-looking flat index into a null buffer: an unconditional null-pointer dereference,
// not just "undefined behavior" in the abstract.
TEST_F(TensorTest, MoveConstructorLeavesSourceWithZeroNumel) {
    Tensor original(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor moved(std::move(original));
    EXPECT_EQ(original.numel(), 0);
}

TEST_F(TensorTest, MoveAssignmentLeavesSourceWithZeroNumel) {
    Tensor a(Shape({2}), &backend, {1.0f, 2.0f});
    Tensor b(Shape({3}), &backend, {9.0f, 9.0f, 9.0f});
    b = std::move(a);
    EXPECT_EQ(a.numel(), 0);
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

// Both sides of a copy ctor live on the SAME device (both buffers allocated by the same
// backend_) -- CopyDirection must reflect device(), not be hardcoded to HostToHost, or a
// CUDA-backed Tensor's copy ctor would issue cudaMemcpyHostToHost on two real device
// pointers, which is undefined/unreliable. Phase 1.5 Mission 3, Objective 1.
TEST_F(TensorTest, CopyConstructorUsesHostToHostForCpuTensor) {
    ::testing::NiceMock<MockDeviceBackend> mock;
    ON_CALL(mock, allocate(::testing::_)).WillByDefault(::testing::Invoke(&std::malloc));
    ON_CALL(mock, free(::testing::_)).WillByDefault(::testing::Invoke([](void* p) { std::free(p); }));
    ON_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, ::testing::_)).WillByDefault(::testing::Return());

    Tensor other(Shape({2}), &mock, {1.0f, 2.0f});  // device defaults to Cpu

    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, CopyDirection::HostToHost)).Times(1);
    Tensor copy(other);
}

TEST_F(TensorTest, CopyConstructorUsesDeviceToDeviceForNonCpuTensor) {
    ::testing::NiceMock<MockDeviceBackend> mock;
    ON_CALL(mock, allocate(::testing::_)).WillByDefault(::testing::Invoke(&std::malloc));
    ON_CALL(mock, free(::testing::_)).WillByDefault(::testing::Invoke([](void* p) { std::free(p); }));
    ON_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, ::testing::_)).WillByDefault(::testing::Return());

    Tensor other(Shape({2}), &mock, {1.0f, 2.0f}, DeviceType::Cuda);

    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, ::testing::_, CopyDirection::DeviceToDevice)).Times(1);
    Tensor copy(other);
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

TEST_F(TensorTest, AtReturnsElementAtMultiDimIndex) {
    // Row-major 2x3: [[1,2,3],[4,5,6]]
    Tensor t(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST_F(TensorTest, AtOnConstTensorReturnsConstReference) {
    const Tensor t(Shape({2}), &backend, {5.0f, 6.0f});
    EXPECT_FLOAT_EQ(t.at({1}), 6.0f);
}

TEST_F(TensorTest, AtAllowsMutation) {
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    t.at({0}) = 9.0f;
    EXPECT_FLOAT_EQ(t.data()[0], 9.0f);
}

TEST_F(TensorDeathTest, AtAbortsOnOutOfBoundsDimensionIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({2, 3}), &backend);
    EXPECT_DEATH({ (void)t.at({0, 5}); }, "EXAI_ASSERT failed");
}

TEST_F(TensorDeathTest, AtAbortsOnRankMismatch) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({2, 3}), &backend);
    EXPECT_DEATH({ (void)t.at({0}); }, "EXAI_ASSERT failed");
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// operator[] previously had zero bounds checking, not even assert-gated -- raw
// out-of-bounds buffer access, unconditional in every build. Internal invariant per
// Mission 0's classification table (hot path, indices computed internally by
// CPUBackend/module forward-backward loops) -- EXAI_ASSERT, not throw.
TEST_F(TensorDeathTest, IndexOperatorAbortsOnOutOfRangeFlatIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_DEATH({ (void)t[5]; }, "EXAI_ASSERT failed");
}

TEST_F(TensorDeathTest, IndexOperatorAbortsOnNegativeFlatIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_DEATH({ (void)t[-1]; }, "EXAI_ASSERT failed");
}

TEST_F(TensorTest, OperatorBracketFlatIndexesRegardlessOfRank) {
    Tensor t(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[3], 4.0f);
    t[1] = 20.0f;
    EXPECT_FLOAT_EQ(t.data()[1], 20.0f);
}

TEST_F(TensorTest, FillSetsEveryElement) {
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor& ref = t.fill(7.0f);

    EXPECT_EQ(&ref, &t);  // returns *this for chaining
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.data()[i], 7.0f);
    }
}

TEST_F(TensorTest, FillOnZeroElementTensorIsSafe) {
    Tensor t(Shape({0}), &backend);
    EXPECT_NO_THROW(t.fill(1.0f));
}

TEST_F(TensorTest, ReshapePreservesDataForCompatibleShape) {
    Tensor t(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t.reshape(Shape({3, 2}));

    EXPECT_EQ(t.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({2, 1}), 6.0f);
}

TEST_F(TensorTest, ReshapeThrowsOnElementCountMismatch) {
    Tensor t(Shape({2, 3}), &backend);
    EXPECT_THROW(t.reshape(Shape({4, 4})), std::invalid_argument);
}

TEST_F(TensorTest, ToSameDeviceIsNoOp) {
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    const float* ptr_before = t.data();
    t.to(DeviceType::Cpu);
    EXPECT_EQ(t.data(), ptr_before);
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
}

TEST_F(TensorTest, ToDifferentDeviceThrowsUntilThatBackendExists) {
    Tensor t(Shape({2}), &backend);
    EXPECT_THROW(t.to(DeviceType::Cuda), std::runtime_error);
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// escalated from EXAI_ASSERT-only to a real throw -- gradient-accumulation shapes trace
// back to module construction parameters, which can originate from external configuration.
TEST_F(TensorTest, AccumulateThrowsOnShapeMismatch) {
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor wrong_shape(Shape({2}), &backend, {0.5f, 0.5f});
    EXPECT_THROW(t.accumulate(wrong_shape), std::invalid_argument);
}

TEST_F(TensorTest, AccumulateAddsOtherIntoThisInPlace) {
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor delta(Shape({3}), &backend, {0.5f, 0.5f, 0.5f});

    Tensor& ref = t.accumulate(delta);

    EXPECT_EQ(&ref, &t);  // returns *this for chaining
    EXPECT_FLOAT_EQ(t.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.5f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.5f);
    EXPECT_FLOAT_EQ(delta.data()[0], 0.5f);  // delta unaffected
}

TEST_F(TensorTest, AccumulateCalledTwiceSumsBothContributions) {
    Tensor t(Shape({2}), &backend, {0.0f, 0.0f});
    Tensor a(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor b(Shape({2}), &backend, {2.0f, 2.0f});

    t.accumulate(a).accumulate(b);

    EXPECT_FLOAT_EQ(t.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 3.0f);
}

// Escalated from EXAI_ASSERT (death test) to a real throw -- see
// TensorTest.AccumulateThrowsOnShapeMismatch above (Mission 0's classification table).

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
