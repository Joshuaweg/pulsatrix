#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

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
    MOCK_METHOD(void, gemm_ex,
                (const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m, size_t k,
                 size_t n, float beta),
                (override));
    MOCK_METHOD(void, column_sums, (const float* in, float* out, size_t rows, size_t cols, float beta), (override));
    MOCK_METHOD(void, add_row_vector, (const float* in, const float* row, float* out, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, elementwise_backward,
                (ElementwiseOp op, const float* x, const float* grad_out, float* grad_in, size_t n), (override));
    MOCK_METHOD(void, axpby, (float alpha, const float* x, float beta, const float* y, float* out, size_t n),
                (override));
    MOCK_METHOD(float, dot, (const float* a, const float* b, size_t n), (override));
    MOCK_METHOD(void, softmax_rows, (const float* in, float* out, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, softmax_rows_backward, (const float* y, const float* dy, float* dx, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, logsumexp_rows, (const float* in, float* out, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, adam_step,
                (float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1, float beta2,
                 float eps, float bias_correction1, float bias_correction2),
                (override));
    MOCK_METHOD(float, sum, (const float* in, size_t n), (override));
    MOCK_METHOD(void, dropout_forward,
                (const float* in, float* out, float* mask, size_t n, float p, float scale, uint64_t seed,
                 uint64_t offset),
                (override));
    MOCK_METHOD(void, bce_with_logits, (const float* logits, const float* target, float* out, size_t n), (override));
    MOCK_METHOD(void, bce_with_logits_grad,
                (const float* logits, const float* target, float* grad, size_t n, float scale), (override));
    MOCK_METHOD(void, layer_norm_forward,
                (const float* in, const float* gamma, const float* beta, float* xhat, float* out, float* row_std,
                 size_t rows, size_t cols, float eps),
                (override));
    MOCK_METHOD(void, layer_norm_backward,
                (const float* grad_out, const float* gamma, const float* xhat, const float* row_std, float* grad_in,
                 size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, rms_norm_forward,
                (const float* in, const float* gamma, float* out, float* row_rms, size_t rows, size_t cols, float eps),
                (override));
    MOCK_METHOD(void, rms_norm_backward,
                (const float* grad_out, const float* gamma, const float* in, const float* row_rms, float* grad_in,
                 float* gamma_terms, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, rope_rotate,
                (const float* in, const float* cos_table, const float* sin_table, float* out, size_t num_slices,
                 size_t seq_len, size_t head_dim, bool inverse),
                (override));
    MOCK_METHOD(void, permute_0213, (const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3),
                (override));
    MOCK_METHOD(void, gather_rows, (const float* table, const float* indices, float* out, size_t count, size_t dim),
                (override));
    MOCK_METHOD(void, scatter_add_rows,
                (const float* src, const float* indices, float* table, size_t count, size_t dim), (override));
    MOCK_METHOD(void, tanh_gaussian_forward,
                (const float* mean, const float* log_std, const float* eps, float* action, float* std_cache,
                 float* log_prob, size_t rows, size_t cols, float stabilizer, double half_log_two_pi),
                (override));
    MOCK_METHOD(void, tanh_gaussian_backward,
                (const float* action, const float* std_cache, const float* eps, const float* grad_action,
                 const float* grad_log_prob, float* grad_mean, float* grad_log_std, size_t n, float stabilizer),
                (override));
    MOCK_METHOD(void, lrp_linear,
                (const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                 size_t in_features, size_t out_features, float eps),
                (override));
    MOCK_METHOD(void, lrp_residual_split,
                (const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n, float eps),
                (override));
    MOCK_METHOD(void, lrp_bilinear_elementwise,
                (const float* a, const float* b, const float* r, float* r_out, size_t n, float eps), (override));
    MOCK_METHOD(void, lrp_bilinear_matmul,
                (const float* a, const float* b, const float* o, const float* r_o, float* r_a, float* r_b,
                 size_t slices, size_t m, size_t p, size_t q, float eps, bool b_transposed),
                (override));
    MOCK_METHOD(void, lrp_softmax_rows,
                (const float* x, const float* y, const float* r, float* r_in, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, lrp_rope,
                (const float* x, const float* y, const float* r, const float* cos_table, const float* sin_table,
                 float* r_in, size_t slices, size_t seq_len, size_t head_dim, float eps),
                (override));
    MOCK_METHOD(void, logic_pointwise,
                (LogicOp op, int norm, const float* a, const float* b, const float* g_or_r, const float* y,
                 float* out_a, float* out_b, size_t n, float eps),
                (override));
    MOCK_METHOD(void, aggregator_forward,
                (const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p), (override));
    MOCK_METHOD(void, aggregator_backward,
                (const float* x, const float* mean_pow, const float* grad_out, float* grad_in, size_t n, size_t cols,
                 float p),
                (override));
    MOCK_METHOD(void, aggregator_lrp,
                (const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n, size_t cols,
                 float p, float eps),
                (override));
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
// bindings/pulsatrix_py.cpp passes an externally-supplied values list directly) -- escalated
// from PULSATRIX_ASSERT-only to a real throw per Mission 0's classification table.
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
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({2, 3}), &backend);
    EXPECT_DEATH({ (void)t.at({0, 5}); }, "PULSATRIX_ASSERT failed");
}

TEST_F(TensorDeathTest, AtAbortsOnRankMismatch) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({2, 3}), &backend);
    EXPECT_DEATH({ (void)t.at({0}); }, "PULSATRIX_ASSERT failed");
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// operator[] previously had zero bounds checking, not even assert-gated -- raw
// out-of-bounds buffer access, unconditional in every build. Internal invariant per
// Mission 0's classification table (hot path, indices computed internally by
// CPUBackend/module forward-backward loops) -- PULSATRIX_ASSERT, not throw.
TEST_F(TensorDeathTest, IndexOperatorAbortsOnOutOfRangeFlatIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_DEATH({ (void)t[5]; }, "PULSATRIX_ASSERT failed");
}

TEST_F(TensorDeathTest, IndexOperatorAbortsOnNegativeFlatIndex) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor t(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_DEATH({ (void)t[-1]; }, "PULSATRIX_ASSERT failed");
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

TEST_F(TensorTest, ToDifferentDeviceWithoutBackendThrows) {
    Tensor t(Shape({2}), &backend);
    EXPECT_THROW(t.to(DeviceType::Cuda), std::invalid_argument);
}

// Stands in for a GPU backend without needing one: host memory underneath (so results are
// inspectable), but records which CopyDirection each copy() was issued with, on which
// backend, and can be told to fail. Real-hardware coverage of the same paths lives in
// hip_backend_test.cpp / cuda_backend_test.cpp.
class RecordingBackend : public CPUBackend {
public:
    [[nodiscard]] DeviceType device() const noexcept override { return reported_device; }
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override {
        directions.push_back(dir);
        if (fail_copy) {
            throw std::runtime_error("RecordingBackend: injected copy failure");
        }
        CPUBackend::copy(dst, src, bytes, dir);
    }
    void free(void* ptr) noexcept override {
        if (ptr != nullptr) {
            ++frees;
        }
        CPUBackend::free(ptr);
    }

    std::vector<CopyDirection> directions;
    int frees = 0;
    bool fail_copy = false;
    DeviceType reported_device = DeviceType::Cpu;
};

// GPU-native-kernels Mission 0, O1: an untagged Tensor takes its backend's device, so a
// temporary allocated through a GPU backend can no longer be silently labelled Cpu.
TEST_F(TensorTest, UntaggedConstructorsTakeTheBackendsDevice) {
    RecordingBackend hip_like;
    hip_like.reported_device = DeviceType::Hip;

    EXPECT_EQ(Tensor(Shape({2}), &hip_like).device(), DeviceType::Hip);
    EXPECT_EQ(Tensor(Shape({2}), &hip_like, {1.0f, 2.0f}).device(), DeviceType::Hip);
    EXPECT_EQ(Tensor(Shape({2}), &hip_like, std::vector<float>{1.0f, 2.0f}).device(), DeviceType::Hip);
}

TEST_F(TensorTest, UntaggedDeviceTensorCopiesDeviceToDevice) {
    RecordingBackend hip_like;
    hip_like.reported_device = DeviceType::Hip;
    Tensor original(Shape({2}), &hip_like, {1.0f, 2.0f});  // HostToDevice, not HostToHost
    hip_like.directions.clear();

    Tensor copy(original);

    ASSERT_EQ(hip_like.directions.size(), 1u);
    EXPECT_EQ(hip_like.directions[0], CopyDirection::DeviceToDevice);
}

TEST_F(TensorTest, ExplicitTagStillOverridesTheBackendsDevice) {
    // The mislabelled-Tensor trick every host-guard death test relies on.
    EXPECT_EQ(Tensor(Shape({2}), &backend, DeviceType::Cuda).device(), DeviceType::Cuda);
}

TEST_F(TensorTest, ToNullBackendThrows) {
    Tensor t(Shape({2}), &backend);
    EXPECT_THROW(t.to(DeviceType::Cuda, nullptr), std::invalid_argument);
}

TEST_F(TensorTest, ToSameDeviceAndBackendIsNoOp) {
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    const float* ptr_before = t.data();
    t.to(DeviceType::Cpu, &backend);
    EXPECT_EQ(t.data(), ptr_before);
}

TEST_F(TensorTest, ToDeviceCopiesHostToDeviceThroughTargetBackend) {
    RecordingBackend source;
    RecordingBackend device;
    Tensor t(Shape({3}), &source, {1.0f, -2.0f, 3.0f});
    source.directions.clear();

    t.to(DeviceType::Hip, &device);

    EXPECT_EQ(t.device(), DeviceType::Hip);
    EXPECT_TRUE(source.directions.empty());
    ASSERT_EQ(device.directions.size(), 1u);
    EXPECT_EQ(device.directions[0], CopyDirection::HostToDevice);
    EXPECT_EQ(source.frees, 1);  // old buffer released through the backend that allocated it
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], -2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
}

TEST_F(TensorTest, ToHostCopiesDeviceToHostThroughSourceBackend) {
    RecordingBackend device;
    RecordingBackend host;
    Tensor t(Shape({2}), &device, {4.0f, 5.0f}, DeviceType::Cuda);
    device.directions.clear();

    t.to(DeviceType::Cpu, &host);

    EXPECT_EQ(t.device(), DeviceType::Cpu);
    ASSERT_EQ(device.directions.size(), 1u);
    EXPECT_EQ(device.directions[0], CopyDirection::DeviceToHost);
    EXPECT_TRUE(host.directions.empty());
    EXPECT_FLOAT_EQ(t.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 5.0f);
}

TEST_F(TensorTest, ToBetweenGpuVendorsStagesThroughHost) {
    RecordingBackend cuda_like;
    RecordingBackend hip_like;
    Tensor t(Shape({2}), &cuda_like, {6.0f, 7.0f}, DeviceType::Cuda);
    cuda_like.directions.clear();

    t.to(DeviceType::Hip, &hip_like);

    ASSERT_EQ(cuda_like.directions.size(), 1u);
    EXPECT_EQ(cuda_like.directions[0], CopyDirection::DeviceToHost);
    ASSERT_EQ(hip_like.directions.size(), 1u);
    EXPECT_EQ(hip_like.directions[0], CopyDirection::HostToDevice);
    EXPECT_FLOAT_EQ(t.data()[0], 6.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 7.0f);
}

TEST_F(TensorTest, ToSameDeviceTypeDifferentBackendCopiesDeviceToDevice) {
    RecordingBackend first;
    RecordingBackend second;
    Tensor t(Shape({1}), &first, {8.0f}, DeviceType::Hip);

    t.to(DeviceType::Hip, &second);

    ASSERT_EQ(second.directions.size(), 1u);
    EXPECT_EQ(second.directions[0], CopyDirection::DeviceToDevice);
    EXPECT_FLOAT_EQ(t.data()[0], 8.0f);
}

TEST_F(TensorTest, ToReroutesLaterOperationsThroughTargetBackend) {
    RecordingBackend device;
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    t.to(DeviceType::Hip, &device);
    device.directions.clear();

    Tensor copy(t);  // copy ctor must use the new backend and device's direction

    ASSERT_EQ(device.directions.size(), 1u);
    EXPECT_EQ(device.directions[0], CopyDirection::DeviceToDevice);
    EXPECT_EQ(copy.device(), DeviceType::Hip);
}

TEST_F(TensorTest, ToLeavesTensorUnchangedWhenCopyThrows) {
    RecordingBackend device;
    device.fail_copy = true;
    Tensor t(Shape({2}), &backend, {1.0f, 2.0f});
    const float* ptr_before = t.data();

    EXPECT_THROW(t.to(DeviceType::Cuda, &device), std::runtime_error);

    EXPECT_EQ(t.device(), DeviceType::Cpu);
    EXPECT_EQ(t.data(), ptr_before);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_EQ(device.frees, 1);  // the half-built target buffer was released, not leaked
}

TEST_F(TensorTest, ToOnEmptyTensorRetagsWithoutCopying) {
    RecordingBackend device;
    Tensor t(Shape({0}), &backend);

    t.to(DeviceType::Cuda, &device);

    EXPECT_EQ(t.device(), DeviceType::Cuda);
    EXPECT_EQ(t.data(), nullptr);
    EXPECT_TRUE(device.directions.empty());
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 0):
// escalated from PULSATRIX_ASSERT-only to a real throw -- gradient-accumulation shapes trace
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

// Escalated from PULSATRIX_ASSERT (death test) to a real throw -- see
// TensorTest.AccumulateThrowsOnShapeMismatch above (Mission 0's classification table).

TEST(TensorDestructorTest, DestructorCallsBackendFreeExactlyOnce) {
    MockDeviceBackend mock;
    void* fake_ptr = reinterpret_cast<void*>(0x1234);
    EXPECT_CALL(mock, allocate(::testing::_)).WillOnce(::testing::Return(fake_ptr));
    EXPECT_CALL(mock, fill(fake_ptr, 0.0f, 2)).Times(1);
    EXPECT_CALL(mock, free(fake_ptr)).Times(1);

    { Tensor t(Shape({2}), &mock); }  // destructor runs at scope exit
}

// campaign_exai_dl_library_data_pipeline, Mission 0: Tensor::Stack is the collate-time
// primitive that turns N independently-loaded Dataset samples into one batch Tensor.
// Three (1, 3) "rows" stacked into one (3, 3) batch, matching the batch-of-one-per-sample
// convention MnistIdxLoader already uses.
TEST_F(TensorTest, StackConcatenatesAlongLeadingDimension) {
    Tensor a(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor b(Shape({1, 3}), &backend, {4.0f, 5.0f, 6.0f});
    Tensor c(Shape({1, 3}), &backend, {7.0f, 8.0f, 9.0f});

    Tensor batch = Tensor::Stack({a, b, c}, &backend);

    EXPECT_EQ(batch.shape(), Shape({3, 3}));
    for (int64_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(batch.data()[i], static_cast<float>(i + 1));
    }
}

TEST_F(TensorTest, StackSumsUnequalLeadingDimensions) {
    Tensor a(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b(Shape({1, 2}), &backend, {5.0f, 6.0f});

    Tensor batch = Tensor::Stack({a, b}, &backend);

    EXPECT_EQ(batch.shape(), Shape({3, 2}));
    EXPECT_FLOAT_EQ(batch.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(batch.data()[3], 4.0f);
    EXPECT_FLOAT_EQ(batch.data()[4], 5.0f);
    EXPECT_FLOAT_EQ(batch.data()[5], 6.0f);
}

TEST_F(TensorTest, StackThrowsOnEmptyInput) {
    EXPECT_THROW(Tensor::Stack({}, &backend), std::invalid_argument);
}

TEST_F(TensorTest, StackThrowsOnRankMismatch) {
    Tensor a(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor b(Shape({1, 3, 1}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW(Tensor::Stack({a, b}, &backend), std::invalid_argument);
}

TEST_F(TensorTest, StackThrowsOnNonLeadingDimensionMismatch) {
    Tensor a(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor b(Shape({1, 4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW(Tensor::Stack({a, b}, &backend), std::invalid_argument);
}

TEST_F(TensorTest, StackThrowsOnDeviceMismatch) {
    Tensor a(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor b(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f}, DeviceType::Cuda);
    EXPECT_THROW(Tensor::Stack({a, b}, &backend), std::invalid_argument);
}

TEST_F(TensorTest, StackThrowsOnRankZeroTensor) {
    Tensor a(Shape({}), &backend, {1.0f});
    Tensor b(Shape({}), &backend, {2.0f});
    EXPECT_THROW(Tensor::Stack({a, b}, &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
