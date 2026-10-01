// CPU-vs-GPU equivalence for the modules, losses and optimizers GPU-native-kernels Mission 1
// made device-resident. Shared by forward_pass_equivalence_hip_test.cpp and
// forward_pass_equivalence_test.cpp (CUDA) via PULSATRIX_TRAINING_EQUIVALENCE_TESTS, so both
// vendors run identical cases.
#pragma once

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/tanh_gaussian_policy.hpp"
#include "pulsatrix/transformer_block.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/noise_schedule.hpp"
#include "pulsatrix/reparameterize.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/softmax_module.hpp"
#include "pulsatrix/swiglu_module.hpp"
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

inline void ResidualBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cin(6, 6, &cpu), gin(6, 6, &gpu);
    ResidualModule cres(&cin, &cpu), gres(&gin, &gpu);
    RandomizeAndMirror(cin, gin, 110);
    std::vector<float> x = Random(4 * 6, 111), dy = Random(4 * 6, 112);
    Tensor cx(Shape({4, 6}), &cpu, x), gx(Shape({4, 6}), &gpu, x);
    Tensor cdy(Shape({4, 6}), &cpu, dy), gdy(Shape({4, 6}), &gpu, dy);
    ExpectNear(cres.forward(cx), gres.forward(gx));
    ExpectNear(cres.backward(cdy), gres.backward(gdy));
    ExpectParametersNear(cin, gin, kTolerance);
}

inline void SwiGLUBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    SwiGLUModule cs(6, 10, &cpu), gs(6, 10, &gpu);
    RandomizeAndMirror(cs, gs, 120);
    std::vector<float> x = Random(2 * 3 * 6, 121, -3.0f, 3.0f), dy = Random(2 * 3 * 6, 122);
    Tensor cx(Shape({2, 3, 6}), &cpu, x), gx(Shape({2, 3, 6}), &gpu, x);
    Tensor cdy(Shape({2, 3, 6}), &cpu, dy), gdy(Shape({2, 3, 6}), &gpu, dy);
    ExpectNear(cs.forward(cx), gs.forward(gx));
    ExpectNear(cs.backward(cdy), gs.backward(gdy));
    ExpectParametersNear(cs, gs, kTolerance);
}

// ---- Mission 1b ----------------------------------------------------------------------------

inline void NegationForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    NegationModule cn(&cpu), gn(&gpu);
    std::vector<float> x = Random(60, 130, 0.0f, 1.0f), dy = Random(60, 131);
    Tensor cx(Shape({6, 10}), &cpu, x), gx(Shape({6, 10}), &gpu, x);
    Tensor cdy(Shape({6, 10}), &cpu, dy), gdy(Shape({6, 10}), &gpu, dy);
    ExpectNear(cn.forward(cx), gn.forward(gx));
    ExpectNear(cn.backward(cdy), gn.backward(gdy));
}

// Same seed on both sides, two consecutive training passes (so the stream offset advances):
// identical masks, so forward and backward match.
inline void DropoutTrainingForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    DropoutModule cd(0.4f, &cpu, /*seed=*/2024), gd(0.4f, &gpu, /*seed=*/2024);
    for (unsigned pass = 0; pass < 2; ++pass) {
        std::vector<float> x = Random(500, 140 + pass), dy = Random(500, 150 + pass);
        Tensor cx(Shape({50, 10}), &cpu, x), gx(Shape({50, 10}), &gpu, x);
        Tensor cdy(Shape({50, 10}), &cpu, dy), gdy(Shape({50, 10}), &gpu, dy);
        ExpectNear(cd.forward(cx), gd.forward(gx));
        ExpectNear(cd.backward(cdy), gd.backward(gdy));
    }
}

inline void BCEForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    BCEWithLogitsLoss cl(&cpu), gl(&gpu);
    std::vector<float> x = Random(200, 160, -8.0f, 8.0f), y = Random(200, 161, 0.0f, 1.0f);
    Tensor cx(Shape({20, 10}), &cpu, x), gx(Shape({20, 10}), &gpu, x);
    Tensor cy(Shape({20, 10}), &cpu, y), gy(Shape({20, 10}), &gpu, y);
    EXPECT_NEAR(cl.forward(cx, cy), gl.forward(gx, gy), kTolerance);
    ExpectNear(cl.backward(), gl.backward());
}

inline void KLForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    KLDivergenceLoss cl(&cpu), gl(&gpu);
    std::vector<float> mu = Random(40, 170), ls = Random(40, 171, -1.0f, 1.0f);
    Tensor cmu(Shape({8, 5}), &cpu, mu), gmu(Shape({8, 5}), &gpu, mu);
    Tensor cls(Shape({8, 5}), &cpu, ls), gls(Shape({8, 5}), &gpu, ls);
    EXPECT_NEAR(cl.forward(cmu, cls), gl.forward(gmu, gls), kTolerance);
    ReparamGrad cg = cl.backward(), gg = gl.backward();
    ExpectNear(cg.grad_mu, gg.grad_mu);
    ExpectNear(cg.grad_log_sigma, gg.grad_log_sigma);
}

inline void CalibrationForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    CalibrationLoss cl(&cpu), gl(&gpu);
    std::vector<float> probs = Random(12 * 5, 180, 0.0f, 1.0f);
    std::vector<float> targets = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 2, 2};
    Tensor cp(Shape({12, 5}), &cpu, probs), gp(Shape({12, 5}), &gpu, probs);
    Tensor ct(Shape({12, 1}), &cpu, targets), gt(Shape({12, 1}), &gpu, targets);  // GPU targets read back for validation
    EXPECT_NEAR(cl.forward(cp, ct), gl.forward(gp, gt), kTolerance);
    ExpectNear(cl.backward(), gl.backward());
}

inline void ReparameterizeForwardBackward(DeviceBackend& gpu) {
    CPUBackend cpu;
    Reparameterize cr(&cpu), gr(&gpu);
    std::vector<float> mu = Random(30, 190), ls = Random(30, 191), eps = Random(30, 192), dz = Random(30, 193);
    Tensor cmu(Shape({30}), &cpu, mu), gmu(Shape({30}), &gpu, mu);
    Tensor cls(Shape({30}), &cpu, ls), gls(Shape({30}), &gpu, ls);
    Tensor ceps(Shape({30}), &cpu, eps), geps(Shape({30}), &gpu, eps);
    Tensor cdz(Shape({30}), &cpu, dz), gdz(Shape({30}), &gpu, dz);
    ExpectNear(cr.forward(cmu, cls, ceps), gr.forward(gmu, gls, geps));
    ReparamGrad cg = cr.backward(cdz), gg = gr.backward(gdz);
    ExpectNear(cg.grad_mu, gg.grad_mu);
    ExpectNear(cg.grad_log_sigma, gg.grad_log_sigma);
}

inline void NoiseScheduleAddAndDenoise(DeviceBackend& gpu) {
    CPUBackend cpu;
    NoiseSchedule schedule(50);
    std::vector<float> x0 = Random(64, 210), eps = Random(64, 211), z = Random(64, 212);
    Tensor cx0(Shape({8, 8}), &cpu, x0), gx0(Shape({8, 8}), &gpu, x0);
    Tensor ceps(Shape({8, 8}), &cpu, eps), geps(Shape({8, 8}), &gpu, eps);
    Tensor cz(Shape({8, 8}), &cpu, z), gz(Shape({8, 8}), &gpu, z);
    Tensor cxt = schedule.add_noise(cx0, ceps, 17);
    Tensor gxt = schedule.add_noise(gx0, geps, 17);
    ExpectNear(cxt, gxt);
    ExpectNear(schedule.denoise_step(cxt, ceps, cz, 17), schedule.denoise_step(gxt, geps, gz, 17));
}

// ---- Mission 2 -----------------------------------------------------------------------------

// Forward, backward and parameter gradients of a module on a (shape) input, CPU vs GPU.
// cpu must be the backend cm was built on: cm caches the input tensors, so their backend has to
// outlive it.
template <typename Module>
inline void ModuleForwardBackward(CPUBackend& cpu, DeviceBackend& gpu, Module& cm, Module& gm, const Shape& shape,
                                  unsigned seed, float tol = kTolerance) {
    RandomizeAndMirror(cm, gm, seed);
    const auto n = static_cast<size_t>(shape.numel());
    std::vector<float> x = Random(n, seed + 100), dy = Random(n, seed + 200);
    Tensor cx(shape, &cpu, x), gx(shape, &gpu, x);
    Tensor cdy(shape, &cpu, dy), gdy(shape, &gpu, dy);
    ExpectNear(cm.forward(cx), gm.forward(gx), tol);
    ExpectNear(cm.backward(cdy), gm.backward(gdy), tol);
    ExpectParametersNear(cm, gm, tol);
}

inline void LayerNormMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    LayerNormModule cm(12, &cpu), gm(12, &gpu);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({7, 12}), 300);
}

inline void RMSNormMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    RMSNormModule cm(12, &cpu), gm(12, &gpu);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({7, 12}), 310);
}

// Long enough that a device-side float angle would visibly drift; the tables are built in
// double on the host, so CPU and GPU see identical cos/sin.
inline void RoPEMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    RoPEModule cm(8, &cpu), gm(8, &gpu);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 3, 300, 8}), 320);
}

inline void AttentionMatches(DeviceBackend& gpu, bool use_rope, bool use_qk_norm) {
    CPUBackend cpu;
    MultiHeadAttentionModule cm(8, 2, &cpu, use_rope, use_qk_norm), gm(8, 2, &gpu, use_rope, use_qk_norm);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 5, 8}), 330);
}

inline void TransformerBlockMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    TransformerBlock cm(8, 2, 16, &cpu), gm(8, 2, 16, &gpu);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 5, 8}), 340);
}

// Repeated indices: the scatter-add must accumulate them, in token order.
inline void EmbeddingMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    EmbeddingModule cm(10, 6, &cpu), gm(10, 6, &gpu);
    RandomizeAndMirror(cm, gm, 350);
    std::vector<float> ids = {3, 7, 3, 0, 9, 3, 7, 1};
    std::vector<float> dy = Random(8 * 6, 351);
    Tensor cids(Shape({2, 4}), &cpu, ids), gids(Shape({2, 4}), &gpu, ids);  // GPU ids read back for validation
    Tensor cdy(Shape({2, 4, 6}), &cpu, dy), gdy(Shape({2, 4, 6}), &gpu, dy);
    ExpectNear(cm.forward(cids), gm.forward(gids));
    ExpectNear(cm.backward(cdy), gm.backward(gdy));
    ExpectParametersNear(cm, gm, kTolerance);
}

inline void TanhGaussianMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    TanhGaussianPolicy cp(&cpu), gp(&gpu);
    std::vector<float> mean = Random(12, 360), ls = Random(12, 361, -1.0f, 0.5f), eps = Random(12, 362);
    std::vector<float> ga = Random(12, 363), glp = Random(12, 364);
    Tensor cm(Shape({4, 3}), &cpu, mean), gm(Shape({4, 3}), &gpu, mean);
    Tensor cls(Shape({4, 3}), &cpu, ls), gls(Shape({4, 3}), &gpu, ls);
    Tensor ce(Shape({4, 3}), &cpu, eps), ge(Shape({4, 3}), &gpu, eps);
    TanhGaussianSample cs = cp.forward(cm, cls, ce), gs = gp.forward(gm, gls, ge);
    ExpectNear(cs.action, gs.action);
    ExpectNear(cs.log_prob, gs.log_prob);
    Tensor cga(Shape({4, 3}), &cpu, ga), gga(Shape({4, 3}), &gpu, ga);
    Tensor cglp(Shape({4, 3}), &cpu, glp), gglp(Shape({4, 3}), &gpu, glp);
    TanhGaussianGrad cg = cp.backward(cga, cglp), gg = gp.backward(gga, gglp);
    ExpectNear(cg.grad_mean, gg.grad_mean);
    ExpectNear(cg.grad_log_std, gg.grad_log_std);
}

// Mission 2 gate: a full transformer block (RMS/LayerNorm, RoPE attention, SwiGLU) trained
// with Adam on GPU ends with the CPU's parameters.
inline void TransformerBlockTrainsToSameParameters(DeviceBackend& gpu) {
    CPUBackend cpu;
    TransformerBlock cb(8, 2, 16, &cpu), gb(8, 2, 16, &gpu);
    RandomizeAndMirror(cb, gb, 370);
    AdamOptimizer copt(0.01f, &cpu), gopt(0.01f, &gpu);
    MSELoss closs(&cpu), gloss(&gpu);
    std::vector<float> x = Random(2 * 5 * 8, 371), y = Random(2 * 5 * 8, 372);
    Tensor cx(Shape({2, 5, 8}), &cpu, x), gx(Shape({2, 5, 8}), &gpu, x);
    Tensor cy(Shape({2, 5, 8}), &cpu, y), gy(Shape({2, 5, 8}), &gpu, y);
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 10; ++step) {
        copt.zero_grad(cb);
        gopt.zero_grad(gb);
        const float lc = closs.forward(cb.forward(cx), cy);
        const float lg = gloss.forward(gb.forward(gx), gy);
        EXPECT_NEAR(lc, lg, 1e-3f) << "loss diverged at step " << step;
        (void)cb.backward(closs.backward());
        (void)gb.backward(gloss.backward());
        copt.step(cb);
        gopt.step(gb);
        if (step == 0) {
            first_loss = lc;
        }
        last_loss = lc;
    }
    EXPECT_LT(last_loss, first_loss) << "the block did not actually train";
    ExpectParametersNear(cb, gb, 1e-3f);
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
    TEST_F(FIXTURE, ResidualBackwardMatchesCPU) { ::pulsatrix::training_equivalence::ResidualBackward(MEMBER); } \
    TEST_F(FIXTURE, SwiGLUBackwardMatchesCPU) { ::pulsatrix::training_equivalence::SwiGLUBackward(MEMBER); } \
    TEST_F(FIXTURE, NegationForwardBackwardMatchCPU) {                                               \
        ::pulsatrix::training_equivalence::NegationForwardBackward(MEMBER);                          \
    }                                                                                                \
    TEST_F(FIXTURE, DropoutTrainingForwardBackwardMatchCPU) {                                        \
        ::pulsatrix::training_equivalence::DropoutTrainingForwardBackward(MEMBER);                   \
    }                                                                                                \
    TEST_F(FIXTURE, BCEWithLogitsLossMatchesCPU) { ::pulsatrix::training_equivalence::BCEForwardBackward(MEMBER); } \
    TEST_F(FIXTURE, KLDivergenceLossMatchesCPU) { ::pulsatrix::training_equivalence::KLForwardBackward(MEMBER); } \
    TEST_F(FIXTURE, CalibrationLossMatchesCPU) {                                                     \
        ::pulsatrix::training_equivalence::CalibrationForwardBackward(MEMBER);                       \
    }                                                                                                \
    TEST_F(FIXTURE, ReparameterizeMatchesCPU) {                                                      \
        ::pulsatrix::training_equivalence::ReparameterizeForwardBackward(MEMBER);                    \
    }                                                                                                \
    TEST_F(FIXTURE, NoiseScheduleMatchesCPU) {                                                       \
        ::pulsatrix::training_equivalence::NoiseScheduleAddAndDenoise(MEMBER);                       \
    }                                                                                                \
    TEST_F(FIXTURE, LayerNormForwardBackwardMatchCPU) { ::pulsatrix::training_equivalence::LayerNormMatches(MEMBER); } \
    TEST_F(FIXTURE, RMSNormForwardBackwardMatchCPU) { ::pulsatrix::training_equivalence::RMSNormMatches(MEMBER); } \
    TEST_F(FIXTURE, RoPEForwardBackwardMatchCPU) { ::pulsatrix::training_equivalence::RoPEMatches(MEMBER); }    \
    TEST_F(FIXTURE, AttentionForwardBackwardMatchCPU) {                                              \
        ::pulsatrix::training_equivalence::AttentionMatches(MEMBER, false, false);                   \
        ::pulsatrix::training_equivalence::AttentionMatches(MEMBER, true, false);                    \
        ::pulsatrix::training_equivalence::AttentionMatches(MEMBER, true, true);                     \
    }                                                                                                \
    TEST_F(FIXTURE, TransformerBlockForwardBackwardMatchCPU) {                                       \
        ::pulsatrix::training_equivalence::TransformerBlockMatches(MEMBER);                          \
    }                                                                                                \
    TEST_F(FIXTURE, EmbeddingForwardBackwardMatchCPU) { ::pulsatrix::training_equivalence::EmbeddingMatches(MEMBER); } \
    TEST_F(FIXTURE, TanhGaussianPolicyMatchesCPU) { ::pulsatrix::training_equivalence::TanhGaussianMatches(MEMBER); } \
    TEST_F(FIXTURE, TransformerBlockTrainedWithAdamEndsWithCPUParameters) {                          \
        ::pulsatrix::training_equivalence::TransformerBlockTrainsToSameParameters(MEMBER);           \
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
