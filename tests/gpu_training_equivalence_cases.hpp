// CPU-vs-GPU equivalence for the modules, losses and optimizers GPU-native-kernels Mission 1
// made device-resident. Shared by forward_pass_equivalence_hip_test.cpp and
// forward_pass_equivalence_test.cpp (CUDA) via PULSATRIX_TRAINING_EQUIVALENCE_TESTS, so both
// vendors run identical cases.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/conjunction_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/tanh_gaussian_policy.hpp"
#include "pulsatrix/transformer_block.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"
#include "pulsatrix/neuro_symbolic_toy_kb.hpp"
#include "pulsatrix/noise_schedule.hpp"
#include "pulsatrix/reparameterize.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/rnn_module.hpp"
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

// ---- Mission 3: LRP and logic modules ------------------------------------------------------

// Campaign Decision Point 2: GPU relevance is judged by what LRP guarantees, not by 1e-4
// elementwise equality -- the epsilon rule divides by sums that can sit near zero, where
// rounding-level differences are amplified. Three checks:
//  1. conservation: total relevance matches the CPU's (relative to its magnitude);
//  2. sign agreement wherever the CPU's relevance is clearly away from zero;
//  3. elementwise closeness, relative to the relevance scale of the tensor.
// The shared per-output source (src/lrp_math.hpp) keeps CPU and GPU far inside these bounds in
// practice; the bounds are the contract, not the expectation.
constexpr float kLrpConservationTolerance = 1e-3f;  // relative to sum(|R|)
constexpr float kLrpElementTolerance = 1e-3f;       // relative to max(|R|)
constexpr float kLrpSignFloor = 1e-2f;              // |R| below this fraction of max(|R|) is "near zero"

inline void ExpectRelevanceAgrees(const Tensor& cpu, const Tensor& gpu) {
    ASSERT_EQ(cpu.shape(), gpu.shape());
    const std::vector<float> c = ToHost(cpu), g = ToHost(gpu);
    float max_abs = 0.0f, sum_abs = 0.0f, sum_c = 0.0f, sum_g = 0.0f;
    for (size_t i = 0; i < c.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(c[i]));
        sum_abs += std::fabs(c[i]);
        sum_c += c[i];
        sum_g += g[i];
    }
    EXPECT_NEAR(sum_c, sum_g, kLrpConservationTolerance * std::max(sum_abs, 1e-6f)) << "conservation";
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c[i], g[i], kLrpElementTolerance * std::max(max_abs, 1e-6f)) << "element " << i;
        if (std::fabs(c[i]) > kLrpSignFloor * max_abs) {
            EXPECT_EQ(c[i] > 0.0f, g[i] > 0.0f) << "sign flip at element " << i;
        }
    }
}

// Forward on both sides, then propagate_relevance with the same random relevance.
template <typename Module>
inline void ModuleRelevance(CPUBackend& cpu, DeviceBackend& gpu, Module& cm, Module& gm, const Shape& in_shape,
                            unsigned seed) {
    RandomizeAndMirror(cm, gm, seed);
    std::vector<float> x = Random(static_cast<size_t>(in_shape.numel()), seed + 100);
    Tensor cx(in_shape, &cpu, x), gx(in_shape, &gpu, x);
    Tensor cy = cm.forward(cx);
    Tensor gy = gm.forward(gx);
    std::vector<float> r = Random(static_cast<size_t>(cy.numel()), seed + 200);
    Tensor cr(cy.shape(), &cpu, r), gr(gy.shape(), &gpu, r);
    ExpectRelevanceAgrees(cm.propagate_relevance(cr, LRPRuleConfig{}), gm.propagate_relevance(gr, LRPRuleConfig{}));
}

inline void LinearRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cm(9, 6, &cpu), gm(9, 6, &gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({5, 9}), 500);

    // The conservation diagnostic itself runs on device tensors (Mission 3).
    std::vector<float> r = Random(30, 505);
    Tensor cr(Shape({5, 6}), &cpu, r), gr(Shape({5, 6}), &gpu, r);
    ConservationResult cc = ComputeConservation(cm.propagate_relevance(cr, LRPRuleConfig{}), cr);
    ConservationResult gc = ComputeConservation(gm.propagate_relevance(gr, LRPRuleConfig{}), gr);
    EXPECT_NEAR(cc.relevance_in_sum, gc.relevance_in_sum, 1e-3f);
    EXPECT_NEAR(cc.relevance_out_sum, gc.relevance_out_sum, 1e-4f);
}

inline void SoftmaxRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    SoftmaxModule cm(&cpu), gm(&gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({4, 3, 7}), 510);
}

inline void ResidualRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cin(6, 6, &cpu), gin(6, 6, &gpu);
    ResidualModule cm(&cin, &cpu), gm(&gin, &gpu);
    RandomizeAndMirror(cin, gin, 520);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({4, 6}), 521);
}

inline void SwiGLURelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    SwiGLUModule cm(6, 10, &cpu), gm(6, 10, &gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 3, 6}), 530);
}

inline void RoPERelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    RoPEModule cm(8, &cpu), gm(8, &gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 3, 40, 8}), 540);
}

inline void AttentionRelevance(DeviceBackend& gpu, bool use_rope, bool use_qk_norm) {
    CPUBackend cpu;
    MultiHeadAttentionModule cm(8, 2, &cpu, use_rope, use_qk_norm), gm(8, 2, &gpu, use_rope, use_qk_norm);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 5, 8}), 550);
}

inline void TransformerBlockRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    TransformerBlock cm(8, 2, 16, &cpu), gm(8, 2, 16, &gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 5, 8}), 560);
}

inline void EmbeddingRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    EmbeddingModule cm(10, 6, &cpu), gm(10, 6, &gpu);
    RandomizeAndMirror(cm, gm, 570);
    std::vector<float> ids = {3, 7, 3, 0, 9, 3, 7, 1};
    Tensor cids(Shape({2, 4}), &cpu, ids), gids(Shape({2, 4}), &gpu, ids);
    (void)cm.forward(cids);
    (void)gm.forward(gids);
    std::vector<float> r = Random(8 * 6, 571);
    Tensor cr(Shape({2, 4, 6}), &cpu, r), gr(Shape({2, 4, 6}), &gpu, r);
    ExpectRelevanceAgrees(cm.propagate_relevance(cr, LRPRuleConfig{}), gm.propagate_relevance(gr, LRPRuleConfig{}));
}

// Truth degrees in [0, 1], every norm, forward + backward + relevance.
template <typename Module, typename Norm>
inline void LogicModuleMatches(DeviceBackend& gpu, Norm norm, unsigned seed) {
    CPUBackend cpu;
    Module cm(&cpu, norm), gm(&gpu, norm);
    std::vector<float> a = Random(24, seed, 0.0f, 1.0f), b = Random(24, seed + 1, 0.0f, 1.0f);
    a[0] = b[0];  // a Godel tie: the tie convention must agree
    Tensor ca(Shape({4, 6}), &cpu, a), ga(Shape({4, 6}), &gpu, a);
    Tensor cb(Shape({4, 6}), &cpu, b), gb(Shape({4, 6}), &gpu, b);
    ExpectNear(cm.forward(ca, cb), gm.forward(ga, gb));
    std::vector<float> g = Random(24, seed + 2);
    Tensor cg(Shape({4, 6}), &cpu, g), gg(Shape({4, 6}), &gpu, g);
    ExpectNear(cm.backward(cg), gm.backward(gg));
    ExpectRelevanceAgrees(cm.propagate_relevance(cg, LRPRuleConfig{}), gm.propagate_relevance(gg, LRPRuleConfig{}));
}

inline void AggregatorMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    AggregatorModule cm(&cpu, 3.0f), gm(&gpu, 3.0f);
    std::vector<float> x = Random(5 * 7, 580, 0.05f, 1.0f);  // truth degrees, pow-safe
    Tensor cx(Shape({5, 7}), &cpu, x), gx(Shape({5, 7}), &gpu, x);
    ExpectNear(cm.forward(cx), gm.forward(gx));
    std::vector<float> g = Random(7, 581);
    Tensor cg(Shape({7}), &cpu, g), gg(Shape({7}), &gpu, g);
    ExpectNear(cm.backward(cg), gm.backward(gg));
    ExpectRelevanceAgrees(cm.propagate_relevance(cg, LRPRuleConfig{}), gm.propagate_relevance(gg, LRPRuleConfig{}));
}

// The neuro-symbolic pipelines end to end on GPU: the toy KB trains like the CPU one, and the
// datalog bridge (a deliberate host boundary) evaluates and attributes from a GPU tensor.
inline void NeuroSymbolicPipelines(DeviceBackend& gpu) {
    CPUBackend cpu;
    ToyKnowledgeBase ckb(&cpu), gkb(&gpu);
    SGDOptimizer copt(0.5f), gopt(0.5f);
    std::vector<float> x = Random(8, 590);
    Tensor cx(Shape({8, 1}), &cpu, x), gx(Shape({8, 1}), &gpu, x);
    for (int step = 0; step < 5; ++step) {
        EXPECT_NEAR(ckb.train_step(cx, copt), gkb.train_step(gx, gopt), 1e-4f) << "step " << step;
    }
    ExpectParametersNear(ckb.predicate_a(), gkb.predicate_a(), 1e-4f);

    datalog::NeuralPredicateDatalogBridge cbridge(&cpu), gbridge(&gpu);
    Tensor cq(Shape({1, 1}), &cpu, {0.3f}), gq(Shape({1, 1}), &gpu, {0.3f});
    datalog::NeuralPredicateQueryResult cres = cbridge.evaluate(cq), gres = gbridge.evaluate(gq);
    EXPECT_NEAR(cres.query_weight, gres.query_weight, 1e-6);
    EXPECT_NEAR(cres.grad_wrt_predicate_output, gres.grad_wrt_predicate_output, 1e-6);
    ExpectRelevanceAgrees(cbridge.propagate_relevance(1.0).relevance_wrt_x,
                          gbridge.propagate_relevance(1.0).relevance_wrt_x);
}

// ---- Mission 4: CNN ----------------------------------------------------------------------------

// Forward, backward, parameter gradients and relevance for one image-shaped module.
template <typename Module>
inline void ImageModuleMatches(CPUBackend& cpu, DeviceBackend& gpu, Module& cm, Module& gm, const Shape& in_shape,
                               unsigned seed) {
    RandomizeAndMirror(cm, gm, seed);
    std::vector<float> x = Random(static_cast<size_t>(in_shape.numel()), seed + 100);
    Tensor cx(in_shape, &cpu, x), gx(in_shape, &gpu, x);
    Tensor cy = cm.forward(cx);
    Tensor gy = gm.forward(gx);
    ExpectNear(cy, gy);
    std::vector<float> dy = Random(static_cast<size_t>(cy.numel()), seed + 200);
    Tensor cdy(cy.shape(), &cpu, dy), gdy(gy.shape(), &gpu, dy);
    ExpectNear(cm.backward(cdy), gm.backward(gdy));
    ExpectParametersNear(cm, gm, kTolerance);
    ExpectRelevanceAgrees(cm.propagate_relevance(cdy, LRPRuleConfig{}), gm.propagate_relevance(gdy, LRPRuleConfig{}));
}

// 3x3 kernels on 7x6 images: windows overlap heavily, so col2im's gather order matters.
inline void Conv2DMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    Conv2DModule cm(3, 4, 3, 3, &cpu), gm(3, 4, 3, 3, &gpu);
    ImageModuleMatches(cpu, gpu, cm, gm, Shape({2, 3, 7, 6}), 700);
}

// 7x6 with 2x2 windows leaves a trailing row/column outside every window (gradient 0 there),
// and a constant plane makes every window a tie: the first maximum must win on both sides.
inline void PoolingMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    MaxPool2DModule cmax(2, 2, &cpu), gmax(2, 2, &gpu);
    ImageModuleMatches(cpu, gpu, cmax, gmax, Shape({2, 3, 7, 6}), 710);
    AvgPool2DModule cavg(2, 2, &cpu), gavg(2, 2, &gpu);
    ImageModuleMatches(cpu, gpu, cavg, gavg, Shape({2, 3, 7, 6}), 720);

    std::vector<float> flat(36, 0.5f);
    Tensor cx(Shape({1, 1, 6, 6}), &cpu, flat), gx(Shape({1, 1, 6, 6}), &gpu, flat);
    (void)cmax.forward(cx);
    (void)gmax.forward(gx);
    std::vector<float> dy(9, 1.0f);
    Tensor cdy(Shape({1, 1, 3, 3}), &cpu, dy), gdy(Shape({1, 1, 3, 3}), &gpu, dy);
    EXPECT_EQ(ToHost(cmax.backward(cdy)), ToHost(gmax.backward(gdy)));  // same tie winner, exactly
}

inline void SpatialNormsMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    BatchNormModule cbn(4, &cpu), gbn(4, &gpu);
    ImageModuleMatches(cpu, gpu, cbn, gbn, Shape({3, 4, 5, 5}), 730);
    GroupNormModule cgn(2, 4, &cpu), ggn(2, 4, &gpu);
    ImageModuleMatches(cpu, gpu, cgn, ggn, Shape({3, 4, 5, 5}), 740);
}

// Mission 4 gate: Conv -> ReLU -> MaxPool -> Flatten -> Linear trained with Adam on GPU ends
// with the CPU's parameters.
inline void CnnTrainsToSameParameters(DeviceBackend& gpu) {
    CPUBackend cpu;
    Conv2DModule cc(1, 4, 3, 3, &cpu), gc(1, 4, 3, 3, &gpu);
    ReluModule cr(&cpu), gr(&gpu);
    MaxPool2DModule cp(2, 2, &cpu), gp(2, 2, &gpu);
    FlattenModule cf(&cpu), gf(&gpu);
    LinearModule cl(4 * 3 * 3, 3, &cpu), gl(4 * 3 * 3, 3, &gpu);
    SequentialModule cnet({&cc, &cr, &cp, &cf, &cl});
    SequentialModule gnet({&gc, &gr, &gp, &gf, &gl});
    RandomizeAndMirror(cnet, gnet, 750);
    AdamOptimizer copt(0.01f, &cpu), gopt(0.01f, &gpu);
    MSELoss closs(&cpu), gloss(&gpu);
    std::vector<float> x = Random(4 * 8 * 8, 751), y = Random(4 * 3, 752);
    Tensor cx(Shape({4, 1, 8, 8}), &cpu, x), gx(Shape({4, 1, 8, 8}), &gpu, x);
    Tensor cy(Shape({4, 3}), &cpu, y), gy(Shape({4, 3}), &gpu, y);
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 15; ++step) {
        copt.zero_grad(cnet);
        gopt.zero_grad(gnet);
        const float lc = closs.forward(cnet.forward(cx), cy);
        const float lg = gloss.forward(gnet.forward(gx), gy);
        EXPECT_NEAR(lc, lg, 1e-3f) << "loss diverged at step " << step;
        (void)cnet.backward(closs.backward());
        (void)gnet.backward(gloss.backward());
        copt.step(cnet);
        gopt.step(gnet);
        if (step == 0) {
            first_loss = lc;
        }
        last_loss = lc;
    }
    EXPECT_LT(last_loss, first_loss) << "the CNN did not actually train";
    ExpectParametersNear(cnet, gnet, 1e-3f);
}

// ---- Mission 5: recurrent networks --------------------------------------------------------------

// Forward, backward, parameter gradients and relevance over a (N, L, input) sequence. L = 6
// so the recurrence (and its backward / relevance unrolling) runs through several steps.
template <typename Module>
inline void RecurrentMatches(DeviceBackend& gpu, unsigned seed) {
    CPUBackend cpu;
    Module cm(5, 7, &cpu), gm(5, 7, &gpu);
    RandomizeAndMirror(cm, gm, seed);
    std::vector<float> x = Random(3 * 6 * 5, seed + 100);
    Tensor cx(Shape({3, 6, 5}), &cpu, x), gx(Shape({3, 6, 5}), &gpu, x);
    Tensor cy = cm.forward(cx);
    Tensor gy = gm.forward(gx);
    ExpectNear(cy, gy);
    std::vector<float> dy = Random(static_cast<size_t>(cy.numel()), seed + 200);
    Tensor cdy(cy.shape(), &cpu, dy), gdy(gy.shape(), &gpu, dy);
    ExpectNear(cm.backward(cdy), gm.backward(gdy));
    ExpectParametersNear(cm, gm, kTolerance);
    ExpectRelevanceAgrees(cm.propagate_relevance(cdy, LRPRuleConfig{}), gm.propagate_relevance(gdy, LRPRuleConfig{}));
}

// Mission 5 gate: an LSTM -> (last step) regression trained with Adam on GPU ends with the CPU's
// parameters -- BPTT through every timestep, every gate.
template <typename Module>
inline void RecurrentTrainsToSameParameters(DeviceBackend& gpu, unsigned seed) {
    CPUBackend cpu;
    Module cm(4, 6, &cpu), gm(4, 6, &gpu);
    RandomizeAndMirror(cm, gm, seed);
    AdamOptimizer copt(0.01f, &cpu), gopt(0.01f, &gpu);
    MSELoss closs(&cpu), gloss(&gpu);
    std::vector<float> x = Random(3 * 5 * 4, seed + 1), y = Random(3 * 5 * 6, seed + 2);
    Tensor cx(Shape({3, 5, 4}), &cpu, x), gx(Shape({3, 5, 4}), &gpu, x);
    Tensor cy(Shape({3, 5, 6}), &cpu, y), gy(Shape({3, 5, 6}), &gpu, y);
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 10; ++step) {
        copt.zero_grad(cm);
        gopt.zero_grad(gm);
        const float lc = closs.forward(cm.forward(cx), cy);
        const float lg = gloss.forward(gm.forward(gx), gy);
        EXPECT_NEAR(lc, lg, 1e-3f) << "loss diverged at step " << step;
        (void)cm.backward(closs.backward());
        (void)gm.backward(gloss.backward());
        copt.step(cm);
        gopt.step(gm);
        if (step == 0) {
            first_loss = lc;
        }
        last_loss = lc;
    }
    EXPECT_LT(last_loss, first_loss) << "the recurrent net did not actually train";
    ExpectParametersNear(cm, gm, 1e-3f);
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
    TEST_F(FIXTURE, LinearRelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::LinearRelevance(MEMBER); } \
    TEST_F(FIXTURE, SoftmaxRelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::SoftmaxRelevance(MEMBER); } \
    TEST_F(FIXTURE, ResidualRelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::ResidualRelevance(MEMBER); } \
    TEST_F(FIXTURE, SwiGLURelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::SwiGLURelevance(MEMBER); } \
    TEST_F(FIXTURE, RoPERelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::RoPERelevance(MEMBER); }    \
    TEST_F(FIXTURE, AttentionRelevanceAgreesWithCPU) {                                               \
        ::pulsatrix::training_equivalence::AttentionRelevance(MEMBER, false, false);                 \
        ::pulsatrix::training_equivalence::AttentionRelevance(MEMBER, true, false);                  \
        ::pulsatrix::training_equivalence::AttentionRelevance(MEMBER, true, true);                   \
    }                                                                                                \
    TEST_F(FIXTURE, TransformerBlockRelevanceAgreesWithCPU) {                                        \
        ::pulsatrix::training_equivalence::TransformerBlockRelevance(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, EmbeddingRelevanceAgreesWithCPU) { ::pulsatrix::training_equivalence::EmbeddingRelevance(MEMBER); } \
    TEST_F(FIXTURE, ConjunctionEveryTNormMatchesCPU) {                                               \
        using ::pulsatrix::ConjunctionModule;                                                        \
        for (auto norm : {ConjunctionModule::TNorm::Product, ConjunctionModule::TNorm::Lukasiewicz,  \
                          ConjunctionModule::TNorm::Godel})                                          \
            ::pulsatrix::training_equivalence::LogicModuleMatches<ConjunctionModule>(MEMBER, norm, 600); \
    }                                                                                                \
    TEST_F(FIXTURE, DisjunctionEveryTConormMatchesCPU) {                                             \
        using ::pulsatrix::DisjunctionModule;                                                        \
        for (auto norm : {DisjunctionModule::TConorm::Product, DisjunctionModule::TConorm::Lukasiewicz, \
                          DisjunctionModule::TConorm::Godel})                                        \
            ::pulsatrix::training_equivalence::LogicModuleMatches<DisjunctionModule>(MEMBER, norm, 610); \
    }                                                                                                \
    TEST_F(FIXTURE, AggregatorMatchesCPU) { ::pulsatrix::training_equivalence::AggregatorMatches(MEMBER); } \
    TEST_F(FIXTURE, NeuroSymbolicPipelinesMatchCPU) {                                                \
        ::pulsatrix::training_equivalence::NeuroSymbolicPipelines(MEMBER);                           \
    }                                                                                                \
    TEST_F(FIXTURE, Conv2DMatchesCPU) { ::pulsatrix::training_equivalence::Conv2DMatches(MEMBER); }      \
    TEST_F(FIXTURE, PoolingMatchesCPU) { ::pulsatrix::training_equivalence::PoolingMatches(MEMBER); }    \
    TEST_F(FIXTURE, SpatialNormsMatchCPU) { ::pulsatrix::training_equivalence::SpatialNormsMatch(MEMBER); } \
    TEST_F(FIXTURE, CnnTrainedWithAdamEndsWithCPUParameters) {                                       \
        ::pulsatrix::training_equivalence::CnnTrainsToSameParameters(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, RNNMatchesCPU) { ::pulsatrix::training_equivalence::RecurrentMatches<::pulsatrix::RNNModule>(MEMBER, 900); } \
    TEST_F(FIXTURE, LSTMMatchesCPU) { ::pulsatrix::training_equivalence::RecurrentMatches<::pulsatrix::LSTMModule>(MEMBER, 910); } \
    TEST_F(FIXTURE, GRUMatchesCPU) { ::pulsatrix::training_equivalence::RecurrentMatches<::pulsatrix::GRUModule>(MEMBER, 920); } \
    TEST_F(FIXTURE, RecurrentNetsTrainedWithAdamEndWithCPUParameters) {                              \
        ::pulsatrix::training_equivalence::RecurrentTrainsToSameParameters<::pulsatrix::RNNModule>(MEMBER, 930); \
        ::pulsatrix::training_equivalence::RecurrentTrainsToSameParameters<::pulsatrix::LSTMModule>(MEMBER, 940); \
        ::pulsatrix::training_equivalence::RecurrentTrainsToSameParameters<::pulsatrix::GRUModule>(MEMBER, 950); \
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
