// CPU-vs-GPU equivalence for the modules, losses and optimizers GPU-native-kernels Mission 1
// made device-resident. Shared by forward_pass_equivalence_hip_test.cpp and
// forward_pass_equivalence_test.cpp (CUDA) via PULSATRIX_TRAINING_EQUIVALENCE_TESTS, so both
// vendors run identical cases.
#pragma once

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/softmax_module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace training_equivalence {

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

inline std::vector<float> ToHost(const Tensor& t) {
    std::vector<float> host(static_cast<size_t>(t.numel()));
    const CopyDirection dir = t.device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToHost;
    t.backend()->copy(host.data(), t.data(), host.size() * sizeof(float), dir);
    return host;
}

inline void ExpectNear(const Tensor& cpu, const Tensor& gpu, float tol = kTolerance) {
    ASSERT_EQ(cpu.shape(), gpu.shape());
    std::vector<float> a = ToHost(cpu), b = ToHost(gpu);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "mismatch at flat index " << i;
    }
}

// Random parameters on the CPU module, copied verbatim into the GPU module.
inline void RandomizeAndMirror(Module& cpu_m, Module& gpu_m, unsigned seed) {
    auto src = cpu_m.parameters();
    auto dst = gpu_m.parameters();
    ASSERT_EQ(src.size(), dst.size());
    for (size_t p = 0; p < src.size(); ++p) {
        std::vector<float> values = Random(static_cast<size_t>(src[p].value->numel()), seed + static_cast<unsigned>(p));
        *src[p].value = Tensor(src[p].value->shape(), src[p].value->backend(), values);
        *dst[p].value = Tensor(dst[p].value->shape(), dst[p].value->backend(), values);
    }
}

inline void ExpectParametersNear(Module& cpu_m, Module& gpu_m, float tol) {
    auto a = cpu_m.parameters();
    auto b = gpu_m.parameters();
    ASSERT_EQ(a.size(), b.size());
    for (size_t p = 0; p < a.size(); ++p) {
        ExpectNear(*a[p].value, *b[p].value, tol);
        ExpectNear(*a[p].grad, *b[p].grad, tol);
    }
}

inline void LinearBackwardAccumulates(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cl(6, 4, &cpu), gl(6, 4, &gpu);
    RandomizeAndMirror(cl, gl, 1);
    // Two backward calls: weight/bias gradients must accumulate (beta = 1) identically.
    for (unsigned call = 0; call < 2; ++call) {
        std::vector<float> x = Random(5 * 6, 10 + call), dy = Random(5 * 4, 20 + call);
        Tensor cx(Shape({5, 6}), &cpu, x), gx(Shape({5, 6}), &gpu, x);
        Tensor cdy(Shape({5, 4}), &cpu, dy), gdy(Shape({5, 4}), &gpu, dy);
        ExpectNear(cl.forward(cx), gl.forward(gx));
        ExpectNear(cl.backward(cdy), gl.backward(gdy));
    }
    ExpectParametersNear(cl, gl, kTolerance);
}

inline void ReluBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    ReluModule cr(&cpu), gr(&gpu);
    std::vector<float> x = Random(200, 30), dy = Random(200, 31);
    x[0] = 0.0f;  // tie point
    Tensor cx(Shape({10, 20}), &cpu, x), gx(Shape({10, 20}), &gpu, x);
    Tensor cdy(Shape({10, 20}), &cpu, dy), gdy(Shape({10, 20}), &gpu, dy);
    (void)cr.forward(cx);
    (void)gr.forward(gx);
    ExpectNear(cr.backward(cdy), gr.backward(gdy));
}

inline void SoftmaxForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    SoftmaxModule cs(&cpu), gs(&gpu);
    std::vector<float> x = Random(4 * 3 * 7, 40, -5.0f, 5.0f), dy = Random(4 * 3 * 7, 41);
    Tensor cx(Shape({4, 3, 7}), &cpu, x), gx(Shape({4, 3, 7}), &gpu, x);  // rank 3: 12 rows of 7
    Tensor cdy(Shape({4, 3, 7}), &cpu, dy), gdy(Shape({4, 3, 7}), &gpu, dy);
    ExpectNear(cs.forward(cx), gs.forward(gx));
    ExpectNear(cs.backward(cdy), gs.backward(gdy));
}

inline void MSEForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    MSELoss cl(&cpu), gl(&gpu);
    std::vector<float> p = Random(300, 50), t = Random(300, 51);
    Tensor cp(Shape({30, 10}), &cpu, p), gp(Shape({30, 10}), &gpu, p);
    Tensor ct(Shape({30, 10}), &cpu, t), gt(Shape({30, 10}), &gpu, t);
    EXPECT_NEAR(cl.forward(cp, ct), gl.forward(gp, gt), kTolerance);
    ExpectNear(cl.backward(), gl.backward());
}

inline void CrossEntropyForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    CrossEntropyLoss cl(&cpu), gl(&gpu);
    std::vector<float> logits = Random(10, 60, -3.0f, 3.0f);
    Tensor cz(Shape({10}), &cpu, logits), gz(Shape({10}), &gpu, logits);
    EXPECT_NEAR(cl.forward(cz, 7), gl.forward(gz, 7), kTolerance);
    ExpectNear(cl.backward(), gl.backward());
}

// The mission's integration gate: the same small MLP, same init, same data, trained for
// several steps on each backend, ends with the same parameters.
template <typename Optimizer>
inline void MlpTrainsToSameParameters(DeviceBackend& gpu, Optimizer& cpu_opt, Optimizer& gpu_opt) {
    CPUBackend cpu;
    LinearModule cl1(4, 8, &cpu), gl1(4, 8, &gpu);
    ReluModule cr(&cpu), gr(&gpu);
    LinearModule cl2(8, 3, &cpu), gl2(8, 3, &gpu);
    SequentialModule cnet({&cl1, &cr, &cl2});
    SequentialModule gnet({&gl1, &gr, &gl2});
    RandomizeAndMirror(cl1, gl1, 70);
    RandomizeAndMirror(cl2, gl2, 80);
    MSELoss closs(&cpu), gloss(&gpu);

    std::vector<float> x = Random(16 * 4, 90), y = Random(16 * 3, 91);
    Tensor cx(Shape({16, 4}), &cpu, x), gx(Shape({16, 4}), &gpu, x);
    Tensor cy(Shape({16, 3}), &cpu, y), gy(Shape({16, 3}), &gpu, y);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int step = 0; step < 25; ++step) {
        cpu_opt.zero_grad(cnet);
        gpu_opt.zero_grad(gnet);
        const float lc = closs.forward(cnet.forward(cx), cy);
        const float lg = gloss.forward(gnet.forward(gx), gy);
        EXPECT_NEAR(lc, lg, 1e-3f) << "loss diverged at step " << step;
        (void)cnet.backward(closs.backward());
        (void)gnet.backward(gloss.backward());
        cpu_opt.step(cnet);
        gpu_opt.step(gnet);
        if (step == 0) {
            first_loss = lc;
        }
        last_loss = lc;
    }
    EXPECT_LT(last_loss, first_loss) << "the network did not actually train";
    // 25 steps of compounding rounding differences: looser than the single-op bound.
    ExpectParametersNear(cl1, gl1, 1e-3f);
    ExpectParametersNear(cl2, gl2, 1e-3f);
}

}  // namespace training_equivalence
}  // namespace pulsatrix

// Instantiates every case for one GPU fixture. FIXTURE must expose the GPU backend as MEMBER.
#define PULSATRIX_TRAINING_EQUIVALENCE_TESTS(FIXTURE, MEMBER)                                        \
    TEST_F(FIXTURE, LinearBackwardAccumulatesLikeCPU) {                                              \
        ::pulsatrix::training_equivalence::LinearBackwardAccumulates(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, ReluBackwardMatchesCPU) { ::pulsatrix::training_equivalence::ReluBackward(MEMBER); } \
    TEST_F(FIXTURE, SoftmaxForwardBackwardMatchCPU) {                                                \
        ::pulsatrix::training_equivalence::SoftmaxForwardBackward(MEMBER);                           \
    }                                                                                                \
    TEST_F(FIXTURE, MSELossForwardBackwardMatchCPU) {                                                \
        ::pulsatrix::training_equivalence::MSEForwardBackward(MEMBER);                               \
    }                                                                                                \
    TEST_F(FIXTURE, CrossEntropyLossForwardBackwardMatchCPU) {                                       \
        ::pulsatrix::training_equivalence::CrossEntropyForwardBackward(MEMBER);                      \
    }                                                                                                \
    TEST_F(FIXTURE, MlpTrainedWithSGDEndsWithCPUParameters) {                                        \
        ::pulsatrix::SGDOptimizer cpu_opt(0.05f), gpu_opt(0.05f);                                    \
        ::pulsatrix::training_equivalence::MlpTrainsToSameParameters(MEMBER, cpu_opt, gpu_opt);      \
    }                                                                                                \
    TEST_F(FIXTURE, MlpTrainedWithAdamEndsWithCPUParameters) {                                       \
        ::pulsatrix::CPUBackend adam_cpu_backend;                                                    \
        ::pulsatrix::AdamOptimizer cpu_opt(0.01f, &adam_cpu_backend), gpu_opt(0.01f, &MEMBER);       \
        ::pulsatrix::training_equivalence::MlpTrainsToSameParameters(MEMBER, cpu_opt, gpu_opt);      \
    }
