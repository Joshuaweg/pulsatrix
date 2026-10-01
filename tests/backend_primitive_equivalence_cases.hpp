// CPU-vs-GPU equivalence cases for the GPU-native-kernels Mission 1 primitives, shared by
// backend_equivalence_hip_test.cpp and backend_equivalence_test.cpp (CUDA) so both vendors are
// held to exactly the same cases -- the HIP run on gfx1151 is then direct evidence about the
// CUDA cases, which share the kernel source as well.
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace primitive_equivalence {

// Same bound as the original primitive suites: GPU reduction order and device libm
// (expf/tanhf vs std::) differ from CPUBackend at rounding level, far inside this.
constexpr float kTolerance = 1e-4f;

inline std::vector<float> Random(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

// Owns one device buffer, filled from / read back to the host.
class DeviceBuffer {
public:
    DeviceBuffer(DeviceBackend& backend, const std::vector<float>& host)
        : backend_(backend), size_(host.size()), ptr_(static_cast<float*>(backend.allocate(host.size() * sizeof(float)))) {
        backend_.copy(ptr_, host.data(), size_ * sizeof(float), CopyDirection::HostToDevice);
    }
    ~DeviceBuffer() { backend_.free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() { return ptr_; }
    std::vector<float> host() const {
        std::vector<float> out(size_);
        backend_.copy(out.data(), ptr_, size_ * sizeof(float), CopyDirection::DeviceToHost);
        return out;
    }

private:
    DeviceBackend& backend_;
    size_t size_;
    float* ptr_;
};

inline void ExpectNear(const std::vector<float>& cpu, const std::vector<float>& gpu, float tol = kTolerance) {
    ASSERT_EQ(cpu.size(), gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i) {
        EXPECT_NEAR(cpu[i], gpu[i], tol) << "mismatch at flat index " << i;
    }
}

inline void GemmEx(DeviceBackend& gpu, bool ta, bool tb, float beta) {
    CPUBackend cpu;
    const size_t m = 7, k = 5, n = 3;  // non-square: a transposed result cannot pass by accident
    std::vector<float> a = Random(m * k, 1), b = Random(k * n, 2), out = Random(m * n, 3);
    std::vector<float> cpu_out = out;
    cpu.gemm_ex(a.data(), ta, b.data(), tb, cpu_out.data(), m, k, n, beta);
    DeviceBuffer da(gpu, a), db(gpu, b), dout(gpu, out);
    gpu.gemm_ex(da.get(), ta, db.get(), tb, dout.get(), m, k, n, beta);
    ExpectNear(cpu_out, dout.host());
}

inline void ColumnSums(DeviceBackend& gpu, float beta) {
    CPUBackend cpu;
    const size_t rows = 37, cols = 11;
    std::vector<float> in = Random(rows * cols, 4), out = Random(cols, 5);
    std::vector<float> cpu_out = out;
    cpu.column_sums(in.data(), cpu_out.data(), rows, cols, beta);
    DeviceBuffer din(gpu, in), dout(gpu, out);
    gpu.column_sums(din.get(), dout.get(), rows, cols, beta);
    ExpectNear(cpu_out, dout.host());
}

inline void AddRowVectorInPlace(DeviceBackend& gpu) {
    CPUBackend cpu;
    const size_t rows = 9, cols = 13;
    std::vector<float> m = Random(rows * cols, 6), row = Random(cols, 7);
    std::vector<float> cpu_m = m;
    cpu.add_row_vector(cpu_m.data(), row.data(), cpu_m.data(), rows, cols);
    DeviceBuffer dm(gpu, m), drow(gpu, row);
    gpu.add_row_vector(dm.get(), drow.get(), dm.get(), rows, cols);
    ExpectNear(cpu_m, dm.host());
}

inline void ElementwiseBackward(DeviceBackend& gpu, ElementwiseOp op) {
    CPUBackend cpu;
    std::vector<float> x = Random(1000, 8, -4.0f, 4.0f), g = Random(1000, 9);
    x[0] = 0.0f;  // Relu's tie point must match exactly
    std::vector<float> cpu_out(x.size());
    cpu.elementwise_backward(op, x.data(), g.data(), cpu_out.data(), x.size());
    DeviceBuffer dx(gpu, x), dg(gpu, g), dout(gpu, std::vector<float>(x.size(), 0.0f));
    gpu.elementwise_backward(op, dx.get(), dg.get(), dout.get(), x.size());
    ExpectNear(cpu_out, dout.host());
}

inline void AxpbyInPlace(DeviceBackend& gpu) {
    CPUBackend cpu;
    std::vector<float> x = Random(1000, 10), y = Random(1000, 11);
    std::vector<float> cpu_y = y;
    cpu.axpby(-0.3f, x.data(), 0.7f, cpu_y.data(), cpu_y.data(), y.size());
    DeviceBuffer dx(gpu, x), dy(gpu, y);
    gpu.axpby(-0.3f, dx.get(), 0.7f, dy.get(), dy.get(), y.size());
    ExpectNear(cpu_y, dy.host());
}

inline void Dot(DeviceBackend& gpu, size_t n) {
    CPUBackend cpu;
    std::vector<float> a = Random(n, 12), b = Random(n, 13);
    const float expected = cpu.dot(a.data(), b.data(), n);
    if (n == 0) {
        EXPECT_FLOAT_EQ(gpu.dot(nullptr, nullptr, 0), 0.0f);
        return;
    }
    DeviceBuffer da(gpu, a), db(gpu, b);
    // Scale-aware bound: a sum of n terms of magnitude <= 1 accumulates order-dependent
    // rounding proportional to n.
    EXPECT_NEAR(expected, gpu.dot(da.get(), db.get(), n), kTolerance * static_cast<float>(1 + n / 100));
}

inline void SoftmaxFamily(DeviceBackend& gpu) {
    CPUBackend cpu;
    const size_t rows = 17, cols = 10;
    std::vector<float> logits = Random(rows * cols, 14, -6.0f, 6.0f);
    logits[0] = 1000.0f;  // max-subtraction must hold on device too
    std::vector<float> dy = Random(rows * cols, 15);

    std::vector<float> y(rows * cols), dx(rows * cols), lse(rows);
    cpu.softmax_rows(logits.data(), y.data(), rows, cols);
    cpu.softmax_rows_backward(y.data(), dy.data(), dx.data(), rows, cols);
    cpu.logsumexp_rows(logits.data(), lse.data(), rows, cols);

    DeviceBuffer dlogits(gpu, logits), ddy(gpu, dy), dyout(gpu, std::vector<float>(rows * cols)),
        ddx(gpu, std::vector<float>(rows * cols)), dlse(gpu, std::vector<float>(rows));
    gpu.softmax_rows(dlogits.get(), dyout.get(), rows, cols);
    gpu.softmax_rows_backward(dyout.get(), ddy.get(), ddx.get(), rows, cols);
    gpu.logsumexp_rows(dlogits.get(), dlse.get(), rows, cols);

    ExpectNear(y, dyout.host());
    ExpectNear(dx, ddx.host());
    // Row 0 is ~1000, where one float ulp is ~6e-5 -- still inside the shared bound.
    ExpectNear(lse, dlse.host());
}

inline void AdamThreeSteps(DeviceBackend& gpu) {
    CPUBackend cpu;
    const size_t n = 500;
    std::vector<float> p = Random(n, 16), m(n, 0.0f), v(n, 0.0f);
    std::vector<float> cp = p, cm = m, cv = v;
    DeviceBuffer dp(gpu, p), dm(gpu, m), dv(gpu, v);
    const float beta1 = 0.9f, beta2 = 0.999f;
    for (int t = 1; t <= 3; ++t) {
        std::vector<float> g = Random(n, 100 + static_cast<unsigned>(t));
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
        cpu.adam_step(cp.data(), g.data(), cm.data(), cv.data(), n, 0.01f, beta1, beta2, 1e-8f, bc1, bc2);
        DeviceBuffer dg(gpu, g);
        gpu.adam_step(dp.get(), dg.get(), dm.get(), dv.get(), n, 0.01f, beta1, beta2, 1e-8f, bc1, bc2);
    }
    ExpectNear(cp, dp.host());
    ExpectNear(cm, dm.host());
    ExpectNear(cv, dv.host());
}

// ---- Mission 1b ----------------------------------------------------------------------------

inline void ExpAndSum(DeviceBackend& gpu) {
    CPUBackend cpu;
    std::vector<float> x = Random(1000, 200, -5.0f, 5.0f), g = Random(1000, 201);
    std::vector<float> fwd(1000), bwd(1000);
    cpu.elementwise(ElementwiseOp::Exp, x.data(), fwd.data(), x.size());
    cpu.elementwise_backward(ElementwiseOp::Exp, x.data(), g.data(), bwd.data(), x.size());
    DeviceBuffer dx(gpu, x), dg(gpu, g), dfwd(gpu, std::vector<float>(1000)), dbwd(gpu, std::vector<float>(1000));
    gpu.elementwise(ElementwiseOp::Exp, dx.get(), dfwd.get(), x.size());
    gpu.elementwise_backward(ElementwiseOp::Exp, dx.get(), dg.get(), dbwd.get(), x.size());
    ExpectNear(fwd, dfwd.host(), 1e-3f);  // exp(5) ~ 148: 1e-4 relative, ~1e-3 absolute
    ExpectNear(bwd, dbwd.host(), 1e-3f);
    EXPECT_NEAR(cpu.sum(g.data(), g.size()), gpu.sum(dg.get(), g.size()), kTolerance * 10);
    EXPECT_FLOAT_EQ(gpu.sum(nullptr, 0), 0.0f);
}

inline void AxpbyZeroBetaIgnoresY(DeviceBackend& gpu) {
    std::vector<float> x = {1.0f, 2.0f}, y = {INFINITY, std::nanf("")};
    DeviceBuffer dx(gpu, x), dy(gpu, y), dout(gpu, std::vector<float>(2));
    gpu.axpby(3.0f, dx.get(), 0.0f, dy.get(), dout.get(), 2);
    EXPECT_EQ(dout.host(), (std::vector<float>{3.0f, 6.0f}));
}

// The property that makes Dropout testable CPU-vs-GPU: identical masks, bit for bit.
inline void DropoutMasksAreBitIdentical(DeviceBackend& gpu) {
    CPUBackend cpu;
    const size_t n = 10007;
    std::vector<float> in = Random(n, 202);
    std::vector<float> out(n), mask(n);
    cpu.dropout_forward(in.data(), out.data(), mask.data(), n, 0.4f, 1.0f / 0.6f, 12345, 777);
    DeviceBuffer din(gpu, in), dout(gpu, std::vector<float>(n)), dmask(gpu, std::vector<float>(n));
    gpu.dropout_forward(din.get(), dout.get(), dmask.get(), n, 0.4f, 1.0f / 0.6f, 12345, 777);
    EXPECT_EQ(mask, dmask.host());
    ExpectNear(out, dout.host());
}

inline void BceWithLogits(DeviceBackend& gpu) {
    CPUBackend cpu;
    std::vector<float> x = Random(1000, 203, -30.0f, 30.0f), y = Random(1000, 204, 0.0f, 1.0f);
    std::vector<float> terms(1000), grad(1000);
    cpu.bce_with_logits(x.data(), y.data(), terms.data(), x.size());
    cpu.bce_with_logits_grad(x.data(), y.data(), grad.data(), x.size(), 0.01f);
    DeviceBuffer dx(gpu, x), dy(gpu, y), dterms(gpu, std::vector<float>(1000)), dgrad(gpu, std::vector<float>(1000));
    gpu.bce_with_logits(dx.get(), dy.get(), dterms.get(), x.size());
    gpu.bce_with_logits_grad(dx.get(), dy.get(), dgrad.get(), x.size(), 0.01f);
    ExpectNear(terms, dterms.host(), 1e-3f);  // |x| up to 30: terms up to ~30
    ExpectNear(grad, dgrad.host());
}

// ---- Mission 2 -----------------------------------------------------------------------------

inline void PermuteGatherScatter(DeviceBackend& gpu) {
    CPUBackend cpu;
    const size_t a = 2, b = 5, c = 3, d = 4;
    std::vector<float> in = Random(a * b * c * d, 400), out(in.size());
    cpu.permute_0213(in.data(), out.data(), a, b, c, d);
    DeviceBuffer din(gpu, in), dout(gpu, std::vector<float>(in.size()));
    gpu.permute_0213(din.get(), dout.get(), a, b, c, d);
    EXPECT_EQ(out, dout.host());  // pure data movement: bit-exact

    const size_t rows = 7, dim = 9, count = 40;
    std::vector<float> table = Random(rows * dim, 401), src = Random(count * dim, 402), idx(count);
    for (size_t i = 0; i < count; ++i) {
        idx[i] = static_cast<float>((i * 3) % rows);  // every row hit several times
    }
    std::vector<float> gathered(count * dim), acc = table;
    cpu.gather_rows(table.data(), idx.data(), gathered.data(), count, dim);
    cpu.scatter_add_rows(src.data(), idx.data(), acc.data(), count, dim);
    DeviceBuffer dtable(gpu, table), didx(gpu, idx), dsrc(gpu, src), dgathered(gpu, std::vector<float>(count * dim));
    gpu.gather_rows(dtable.get(), didx.get(), dgathered.get(), count, dim);
    gpu.scatter_add_rows(dsrc.get(), didx.get(), dtable.get(), count, dim);
    EXPECT_EQ(gathered, dgathered.host());
    EXPECT_EQ(acc, dtable.host());  // same accumulation order: bit-exact, not merely close
}

}  // namespace primitive_equivalence
}  // namespace pulsatrix

// Instantiates every case for one GPU fixture. FIXTURE must expose the GPU backend as MEMBER.
#define PULSATRIX_PRIMITIVE_EQUIVALENCE_TESTS(FIXTURE, MEMBER)                                      \
    TEST_F(FIXTURE, GemmExAllTransposeAndBetaCombinationsMatchCPU) {                                \
        for (bool ta : {false, true})                                                               \
            for (bool tb : {false, true})                                                           \
                for (float beta : {0.0f, 1.0f})                                                     \
                    ::pulsatrix::primitive_equivalence::GemmEx(MEMBER, ta, tb, beta);              \
    }                                                                                               \
    TEST_F(FIXTURE, ColumnSumsMatchCPU) {                                                           \
        ::pulsatrix::primitive_equivalence::ColumnSums(MEMBER, 0.0f);                              \
        ::pulsatrix::primitive_equivalence::ColumnSums(MEMBER, 1.0f);                              \
    }                                                                                               \
    TEST_F(FIXTURE, AddRowVectorInPlaceMatchesCPU) {                                                \
        ::pulsatrix::primitive_equivalence::AddRowVectorInPlace(MEMBER);                           \
    }                                                                                               \
    TEST_F(FIXTURE, ElementwiseBackwardEveryOpMatchesCPU) {                                         \
        for (ElementwiseOp op : {ElementwiseOp::Relu, ElementwiseOp::Neg, ElementwiseOp::Tanh,      \
                                 ElementwiseOp::Sigmoid, ElementwiseOp::Silu})                      \
            ::pulsatrix::primitive_equivalence::ElementwiseBackward(MEMBER, op);                   \
    }                                                                                               \
    TEST_F(FIXTURE, AxpbyInPlaceMatchesCPU) { ::pulsatrix::primitive_equivalence::AxpbyInPlace(MEMBER); } \
    TEST_F(FIXTURE, DotMatchesCPUAcrossSizes) {                                                     \
        for (size_t n : {size_t{0}, size_t{1}, size_t{255}, size_t{256}, size_t{10007}})            \
            ::pulsatrix::primitive_equivalence::Dot(MEMBER, n);                                    \
    }                                                                                               \
    TEST_F(FIXTURE, SoftmaxRowsForwardBackwardAndLogsumexpMatchCPU) {                               \
        ::pulsatrix::primitive_equivalence::SoftmaxFamily(MEMBER);                                 \
    }                                                                                               \
    TEST_F(FIXTURE, AdamThreeStepsMatchCPU) { ::pulsatrix::primitive_equivalence::AdamThreeSteps(MEMBER); } \
    TEST_F(FIXTURE, ExpAndSumMatchCPU) { ::pulsatrix::primitive_equivalence::ExpAndSum(MEMBER); }       \
    TEST_F(FIXTURE, AxpbyZeroBetaIgnoresY) { ::pulsatrix::primitive_equivalence::AxpbyZeroBetaIgnoresY(MEMBER); } \
    TEST_F(FIXTURE, DropoutMasksBitIdenticalToCPU) {                                                \
        ::pulsatrix::primitive_equivalence::DropoutMasksAreBitIdentical(MEMBER);                   \
    }                                                                                               \
    TEST_F(FIXTURE, BceWithLogitsMatchesCPU) { ::pulsatrix::primitive_equivalence::BceWithLogits(MEMBER); } \
    TEST_F(FIXTURE, PermuteGatherScatterMatchCPUBitExactly) {                                       \
        ::pulsatrix::primitive_equivalence::PermuteGatherScatter(MEMBER);                          \
    }
