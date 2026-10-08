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

#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/cartpole_env.hpp"
#include "pulsatrix/categorical_policy_agent.hpp"
#include "pulsatrix/conjunction_module.hpp"
#include "pulsatrix/continuous_cartpole_env.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/dqn_agent.hpp"
#include "pulsatrix/dqn_loss.hpp"
#include "pulsatrix/dqn_target.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/gae.hpp"
#include "pulsatrix/gflownet_forward_policy.hpp"
#include "pulsatrix/gflownet_trajectory.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/hypergrid_env.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mamba_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/tanh_gaussian_policy.hpp"
#include "pulsatrix/encoder_block.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_contacts.hpp"
#include "pulsatrix/protein_explanations.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_training.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"
#include "pulsatrix/jumprelu_sparse_autoencoder.hpp"
#include "pulsatrix/transformer_block.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"
#include "pulsatrix/neuro_symbolic_toy_kb.hpp"
#include "pulsatrix/noise_schedule.hpp"
#include "pulsatrix/pdp.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"
#include "pulsatrix/polyak_update.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"
#include "pulsatrix/replay_buffer.hpp"
#include "pulsatrix/reparameterize.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rollout_buffer.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/sinusoidal_timestep_embedding.hpp"
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

// LLM-1: a Qwen3-shaped layer -- 4 query heads on 2 K/V heads, head_dim 6 (query width 24 vs.
// d_model 8), Q/K/V biases only, rotate-half RoPE from position 9, QK-Norm, causal, and a
// padding mask with a fully masked row (sequence 1 is left-padded).
inline AttentionConfig LlmAttentionConfig() {
    AttentionConfig c;
    c.d_model = 8;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 6;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 1000.0f;
    c.use_qk_norm = true;
    c.qkv_bias = true;
    c.out_bias = false;
    c.causal = true;
    return c;
}

inline void PrepareLlmAttention(CPUBackend& cpu, DeviceBackend& gpu, MultiHeadAttentionModule& cm,
                                MultiHeadAttentionModule& gm) {
    const std::vector<float> keep = {1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
    cm.set_key_padding_mask(Tensor(Shape({2, 5}), &cpu, keep));
    gm.set_key_padding_mask(Tensor(Shape({2, 5}), &gpu, keep));
    cm.set_position_offset(9);
    gm.set_position_offset(9);
}

inline void LlmAttentionMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    MultiHeadAttentionModule cm(LlmAttentionConfig(), &cpu), gm(LlmAttentionConfig(), &gpu);
    PrepareLlmAttention(cpu, gpu, cm, gm);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 5, 8}), 335);
}

// LLM-5: a cached pass in pieces on the GPU matches the CPU's, piece by piece.
inline void CachedAttentionMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    MultiHeadAttentionModule cm(LlmAttentionConfig(), &cpu), gm(LlmAttentionConfig(), &gpu);
    RandomizeAndMirror(cm, gm, 336);
    KVCache cc = cm.MakeKVCache(2, 8), gc = gm.MakeKVCache(2, 8);
    std::vector<float> x = Random(2 * 6 * 8, 337);
    int64_t at = 0;
    for (int64_t piece : {4, 1, 1}) {
        std::vector<float> part;
        for (int64_t n = 0; n < 2; ++n) {
            part.insert(part.end(), x.begin() + (n * 6 + at) * 8, x.begin() + (n * 6 + at + piece) * 8);
        }
        Tensor cx(Shape({2, piece, 8}), &cpu, part), gx(Shape({2, piece, 8}), &gpu, part);
        ExpectNear(cm.forward_cached(cx, cc), gm.forward_cached(gx, gc));
        at += piece;
    }
}

inline void RotateHalfRoPEMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    RoPEModule cm(8, &cpu, 500000.0f, RoPELayout::RotateHalf), gm(8, &gpu, 500000.0f, RoPELayout::RotateHalf);
    cm.set_position_offset(1000);
    gm.set_position_offset(1000);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 3, 50, 8}), 325);
}

inline void TransformerBlockMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    TransformerBlock cm(8, 2, 16, &cpu), gm(8, 2, 16, &gpu);
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 5, 8}), 340);
}

// PLM-1: ESM-2's layout (pre-LayerNorm, rotate-half RoPE, exact-GELU MLP) and BERT's (post-LayerNorm).
inline EncoderBlockOptions EncoderLayout(bool post) {
    EncoderBlockOptions o;
    if (post) o.norm_position = NormPosition::Post;
    return o;
}

inline AttentionConfig EncoderAttention() {
    AttentionConfig a;
    a.d_model = 8;
    a.num_heads = 2;
    a.rope_layout = RoPELayout::RotateHalf;
    return a;
}

inline void EncoderBlockMatches(DeviceBackend& gpu, bool post) {
    CPUBackend cpu;
    EncoderBlock cm(EncoderAttention(), 16, &cpu, EncoderLayout(post)), gm(EncoderAttention(), 16, &gpu, EncoderLayout(post));
    ModuleForwardBackward(cpu, gpu, cm, gm, Shape({2, 5, 8}), post ? 342 : 341);
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

// LRP campaign Mission 1: whole-model LRP::explain on the GPU, every seed mode and a contrast.
inline void LRPExplainerMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cl1(5, 8, &cpu), gl1(5, 8, &gpu);
    ReluModule cr(&cpu), gr(&gpu);
    LinearModule cl2(8, 3, &cpu), gl2(8, 3, &gpu);
    RandomizeAndMirror(cl1, gl1, 1200);
    RandomizeAndMirror(cl2, gl2, 1201);
    ExplainerContext cctx({&cl1, &cr, &cl2}), gctx({&gl1, &gr, &gl2});
    std::vector<float> x = Random(4 * 5, 1202);
    Tensor cx(Shape({4, 5}), &cpu, x), gx(Shape({4, 5}), &gpu, x);
    for (const LRPTarget& t : {LRPTarget{{2}}, LRPTarget{{0, 1, 2, 0}, {}, LRPSeed::OneHot},
                               LRPTarget{{1}, {2}}}) {
        Attribution ca = LRP().explain(cctx, cx, t, &cpu);
        Attribution ga = LRP().explain(gctx, gx, t, &gpu);
        EXPECT_EQ(ga.values.device(), gpu.device());
        ExpectRelevanceAgrees(ca.values, ga.values);
    }
}

// The six post-hoc explainers on a GPU network: the forward/backward runs on the device and
// the explainers' own bookkeeping crosses a host boundary, so every result must match the CPU
// and stay on the device. Gradient methods compare at kTolerance; the surrogate fits (LIME,
// KernelSHAP) solve a small regression whose conditioning amplifies the forward's rounding, so
// they get 1e-3.
inline void PosthocExplainersMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cl1(4, 6, &cpu), gl1(4, 6, &gpu), cl2(6, 3, &cpu), gl2(6, 3, &gpu);
    ReluModule cr(&cpu), gr(&gpu);
    RandomizeAndMirror(cl1, gl1, 1200);
    RandomizeAndMirror(cl2, gl2, 1210);
    ExplainerContext cctx({&cl1, &cr, &cl2}), gctx({&gl1, &gr, &gl2});
    auto expect_match = [&](const Attribution& c, const Attribution& g, float tol) {
        EXPECT_EQ(c.method, g.method);
        EXPECT_EQ(g.values.device(), gpu.device()) << g.method << " result left the device";
        ExpectNear(c.values, g.values, tol);
    };

    std::vector<float> x = Random(2 * 4, 1220);
    Tensor cx(Shape({2, 4}), &cpu, x), gx(Shape({2, 4}), &gpu, x);
    Tensor cb(Shape({2, 4}), &cpu), gb(Shape({2, 4}), &gpu);
    expect_match(Saliency{}.explain(cctx, cx, 1, &cpu), Saliency{}.explain(gctx, gx, 1, &gpu), kTolerance);
    expect_match(IntegratedGradients{}.explain(cctx, cx, cb, 2, 16, &cpu),
                 IntegratedGradients{}.explain(gctx, gx, gb, 2, 16, &gpu), kTolerance);

    // Surrogates on one example (flat target index into a (1, 3) output), same seeds/params.
    std::vector<float> x1 = Random(4, 1230);
    Tensor cx1(Shape({1, 4}), &cpu, x1), gx1(Shape({1, 4}), &gpu, x1);
    Tensor cb1(Shape({1, 4}), &cpu), gb1(Shape({1, 4}), &gpu);
    auto cpredict = [&](const Tensor& t) { return cctx.forward_pass(t); };
    auto gpredict = [&](const Tensor& t) { return gctx.forward_pass(t); };
    expect_match(LIME{}.explain(cpredict, cx1, 0, 64, 0.5f, 0.01f, 7u, &cpu),
                 LIME{}.explain(gpredict, gx1, 0, 64, 0.5f, 0.01f, 7u, &gpu), 1e-3f);
    expect_match(KernelSHAP{}.explain(cpredict, cx1, cb1, 2, &cpu),
                 KernelSHAP{}.explain(gpredict, gx1, gb1, 2, &gpu), 1e-3f);
    std::vector<Tensor> cbackground, gbackground;
    for (unsigned b = 0; b < 3; ++b) {
        std::vector<float> v = Random(4, 1240 + b);
        cbackground.emplace_back(Shape({1, 4}), &cpu, v);
        gbackground.emplace_back(Shape({1, 4}), &gpu, v);
    }
    expect_match(PDP{}.explain(cpredict, cbackground, 2, 1, -1.0f, 1.0f, 5, &cpu),
                 PDP{}.explain(gpredict, gbackground, 2, 1, -1.0f, 1.0f, 5, &gpu), kTolerance);

    // Grad-CAM over a Conv2D -> ReLU -> Flatten -> Linear model (the Captum benchmark's shape).
    Conv2DModule cconv(1, 2, 2, 2, &cpu), gconv(1, 2, 2, 2, &gpu);
    ReluModule ccr(&cpu), gcr(&gpu);
    FlattenModule cf(&cpu), gf(&gpu);
    LinearModule chead(2 * 3 * 3, 3, &cpu), ghead(2 * 3 * 3, 3, &gpu);
    RandomizeAndMirror(cconv, gconv, 1250);
    RandomizeAndMirror(chead, ghead, 1260);
    ExplainerContext ccnn({&cconv, &ccr, &cf, &chead}), gcnn({&gconv, &gcr, &gf, &ghead});
    std::vector<float> img = Random(2 * 4 * 4, 1270);
    Tensor cimg(Shape({2, 1, 4, 4}), &cpu, img), gimg(Shape({2, 1, 4, 4}), &gpu, img);
    expect_match(GradCAM{}.explain(ccnn, cimg, 1, &cpu), GradCAM{}.explain(gcnn, gimg, 1, &gpu), kTolerance);
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

// LLM-2: the tied head's logits, input gradient, shared-table gradient and relevance. The
// table is mirrored through the embedding, since the head owns no parameters.
// PLM-2: the tiny ESM-2 end to end -- token dropout, padding, the biased tied head -- on both backends.
inline void EncoderLMMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm";
    std::unique_ptr<EncoderLM> cm = LoadEncoderLM(dir, &cpu), gm = LoadEncoderLM(dir, &gpu);
    const std::vector<float> ids = {0, 20, 15, 32, 5, 19, 2, 1, 0, 6, 32, 32, 9, 2, 1, 1}, keep = {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0};
    cm->set_padding_mask(Tensor(Shape({2, 8}), &cpu, keep));
    gm->set_padding_mask(Tensor(Shape({2, 8}), &gpu, keep));
    Tensor cx(Shape({2, 8}), &cpu, ids), gx(Shape({2, 8}), &gpu, ids);
    ExpectNear(cm->forward(cx), gm->forward(gx), 1e-4f);
    const std::vector<float> dy = Random(2 * 8 * 33, 380);
    (void)cm->backward(Tensor(Shape({2, 8, 33}), &cpu, dy));
    (void)gm->backward(Tensor(Shape({2, 8, 33}), &gpu, dy));
    ExpectParametersNear(*cm, *gm, 1e-3f);
    const std::vector<float> r = Random(2 * 8 * 33, 381);
    ExpectRelevanceAgrees(cm->propagate_relevance(Tensor(Shape({2, 8, 33}), &cpu, r), LRPRuleConfig{}),
                          gm->propagate_relevance(Tensor(Shape({2, 8, 33}), &gpu, r), LRPRuleConfig{}));
}

// PLM-4: contacts from the tiny ESM-2's attention, read layer by layer off the device.
inline void ContactsMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm";
    std::unique_ptr<EncoderLM> cm = LoadEncoderLM(dir, &cpu), gm = LoadEncoderLM(dir, &gpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    const EsmContactHead head = LoadEsmContactHead(dir, cm->config());
    ContactPredictor cp(*cm, tok, &cpu), gp(*gm, tok, &gpu);
    const std::string seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ";
    const ContactMap c = cp.predict(seq, head), g = gp.predict(seq, head);
    const ContactMap ca = cp.average_heads(seq, {{0, 1}, {1, 2}}), ga = gp.average_heads(seq, {{0, 1}, {1, 2}});
    ASSERT_EQ(c.values.size(), g.values.size());
    for (size_t k = 0; k < c.values.size(); ++k) {
        EXPECT_NEAR(c.values[k], g.values[k], 1e-5f) << k;
        EXPECT_NEAR(ca.values[k], ga.values[k], 1e-5f) << k;
    }
}

// PLM-6: AttnLRP on the tiny ESM-2, LayerNorm's detached-std rule included, for each kind of target.
inline void EncoderExplanationsMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm";
    std::unique_ptr<EncoderLM> cm = LoadEncoderLM(dir, &cpu), gm = LoadEncoderLM(dir, &gpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    EncoderExplainer ce(*cm, tok, &cpu), ge(*gm, tok, &gpu);
    const std::string seq = "MQIFVKTLTGKTITLEVEPS";
    const LinearHead head{Random(cm->config().hidden_size, 390), 0.1f};
    for (const EncoderTarget& t : {EncoderTarget::MaskedToken(6, 'L'), EncoderTarget::ForMutation(ParseMutations("K6R")[0]),
                                   EncoderTarget::ProteinHead(head), EncoderTarget::ResidueHead(head, 3)}) {
        const ResidueRelevance c = ce.explain(seq, t), g = ge.explain(seq, t);
        EXPECT_NEAR(c.value, g.value, 1e-4f) << c.target;
        double scale = 0;
        for (float v : c.residues) scale = std::max(scale, std::abs(static_cast<double>(v)));
        for (size_t i = 0; i < c.residues.size(); ++i) EXPECT_NEAR(c.residues[i], g.residues[i], 1e-3 * scale + 1e-6) << c.target << " " << i;
    }
}

// PLM-7: a masked-LM step with AdamW, and a mean-pooled head trained into the encoder.
inline void MaskedLMTrainingMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm";
    std::unique_ptr<EncoderLM> cm = LoadEncoderLM(dir, &cpu), gm = LoadEncoderLM(dir, &gpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    MaskedLMCollator cc(tok, &cpu, {}, 4), gc(tok, &gpu, {}, 4);
    const std::vector<std::string> seqs = {"MKTAYIAKQRQISFVKSHFSRQLEER", "MQIFVKTLTGKTITLEV", "GSHMLEDPKK"};
    const MaskedLMBatch cb = cc.collate(seqs), gb = gc.collate(seqs);
    TokenCrossEntropyLoss cl(&cpu), gl(&gpu);
    AdamWOptimizer co(1e-3f, &cpu, 0.01f, 0.9f, 0.98f), go(1e-3f, &gpu, 0.01f, 0.9f, 0.98f);
    for (int step = 0; step < 2; ++step) {
        co.zero_grad(*cm);
        go.zero_grad(*gm);
        EXPECT_NEAR(MaskedLMForwardBackward(*cm, cb, cl), MaskedLMForwardBackward(*gm, gb, gl), 1e-4f) << "step " << step;
        co.step(*cm);
        go.step(*gm);
    }
    ExpectParametersNear(*cm, *gm, 1e-3f);

    SequenceHead ch(cm->config().hidden_size, 3, SequenceHead::Pooling::Mean, &cpu), gh(gm->config().hidden_size, 3, SequenceHead::Pooling::Mean, &gpu);
    RandomizeAndMirror(ch, gh, 391);
    ch.set_residue_mask(ResidueMask(cb.ids, &cb.keep, tok, &cpu));
    gh.set_residue_mask(ResidueMask(gb.ids, &gb.keep, tok, &gpu));
    cm->set_padding_mask(cb.keep);
    gm->set_padding_mask(gb.keep);
    (void)cm->forward(cb.ids);
    (void)gm->forward(gb.ids);
    ExpectNear(ch.forward(cm->last_hidden_state()), gh.forward(gm->last_hidden_state()), 1e-4f);
    const std::vector<float> dy = Random(3 * 3, 392);
    ExpectNear(cm->backward_hidden(ch.backward(Tensor(Shape({3, 3}), &cpu, dy))), gm->backward_hidden(gh.backward(Tensor(Shape({3, 3}), &gpu, dy))),
               1e-3f);
}

// FEAT-1: a sparse autoencoder trained through the Featurizer interface, unit-norm decoders included.
inline void FeaturizerTrainingMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    SparseAutoencoder c(6, 16, 0.01f, &cpu, 4u), g(6, 16, 0.01f, &gpu, 4u);
    const std::vector<float> x = Random(32 * 6, 393);
    const Tensor cx(Shape({32, 6}), &cpu, x), gx(Shape({32, 6}), &gpu, x);
    AdamOptimizer co(0.01f, &cpu), go(0.01f, &gpu);
    for (int step = 0; step < 5; ++step) {
        const FeaturizerLoss cl = TrainFeaturizer(c, cx, co), gl = TrainFeaturizer(g, gx, go);
        EXPECT_NEAR(cl.total, gl.total, 1e-4f) << "step " << step;
    }
    ExpectParametersNear(c, g, 1e-3f);
    ExpectNear(c.encode(cx), g.encode(gx), 1e-4f);
}

// FEAT-2: a TopK sparse autoencoder with dead latents, so the auxiliary loss runs too.
inline void TopKFeaturizerTrainingMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    TopKSaeOptions o;
    o.k = 3;
    o.dead_after = 32;
    TopKSparseAutoencoder c(6, 16, &cpu, o), g(6, 16, &gpu, o);
    const std::vector<float> x = Random(32 * 6, 394);
    const Tensor cx(Shape({32, 6}), &cpu, x), gx(Shape({32, 6}), &gpu, x);
    c.initialize_bias(cx);
    g.initialize_bias(gx);
    AdamOptimizer co(0.01f, &cpu), go(0.01f, &gpu);
    for (int step = 0; step < 5; ++step) {
        const FeaturizerLoss cl = TrainFeaturizer(c, cx, co), gl = TrainFeaturizer(g, gx, go);
        EXPECT_NEAR(cl.total, gl.total, 1e-4f) << "step " << step;
        EXPECT_NEAR(cl.sparsity, gl.sparsity, 1e-5f) << "step " << step;
    }
    EXPECT_EQ(c.dead_latents(), g.dead_latents());
    ExpectParametersNear(c, g, 1e-3f);
    ExpectNear(c.encode(cx), g.encode(gx), 1e-4f);
}

// FEAT-4: Matryoshka BatchTopK (its threshold too) and JumpReLU (its thresholds' straight-through
// gradient too).
inline void SaeVariantsTrainingMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::vector<float> x = Random(32 * 6, 395);
    const Tensor cx(Shape({32, 6}), &cpu, x), gx(Shape({32, 6}), &gpu, x);
    TopKSaeOptions o;
    o.k = 3;
    o.dead_after = 32;
    o.batch_topk = true;
    o.matryoshka_prefixes = {4, 10};
    TopKSparseAutoencoder c(6, 16, &cpu, o), g(6, 16, &gpu, o);
    AdamOptimizer co(0.01f, &cpu), go(0.01f, &gpu);
    for (int step = 0; step < 5; ++step) {
        const FeaturizerLoss cl = TrainFeaturizer(c, cx, co), gl = TrainFeaturizer(g, gx, go);
        EXPECT_NEAR(cl.total, gl.total, 1e-4f) << "step " << step;
    }
    ExpectParametersNear(c, g, 1e-3f);
    EXPECT_NEAR(c.threshold(), g.threshold(), 1e-4f);
    ExpectNear(c.encode(cx), g.encode(gx), 1e-4f);

    JumpReLUSaeOptions j;
    j.bandwidth = 0.5f;
    j.initial_threshold = 0.2f;
    j.l0_coefficient = 0.01f;
    JumpReLUSparseAutoencoder cj(6, 16, &cpu, j), gj(6, 16, &gpu, j);
    AdamOptimizer cjo(0.01f, &cpu), gjo(0.01f, &gpu);
    for (int step = 0; step < 5; ++step) {
        const FeaturizerLoss cl = TrainFeaturizer(cj, cx, cjo), gl = TrainFeaturizer(gj, gx, gjo);
        EXPECT_NEAR(cl.total, gl.total, 1e-4f) << "step " << step;
    }
    ExpectParametersNear(cj, gj, 1e-3f);
    ExpectNear(cj.encode(cx), gj.encode(gx), 1e-4f);
}

inline void TiedLMHeadMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    EmbeddingModule ce(11, 6, &cpu), ge(11, 6, &gpu);
    RandomizeAndMirror(ce, ge, 370);
    TiedLMHeadModule ch(ce, &cpu), gh(ge, &gpu);
    std::vector<float> x = Random(2 * 3 * 6, 371), dy = Random(2 * 3 * 11, 372), r = Random(2 * 3 * 11, 373);
    Tensor cx(Shape({2, 3, 6}), &cpu, x), gx(Shape({2, 3, 6}), &gpu, x);
    ExpectNear(ch.forward(cx), gh.forward(gx));
    Tensor cdy(Shape({2, 3, 11}), &cpu, dy), gdy(Shape({2, 3, 11}), &gpu, dy);
    ExpectNear(ch.backward(cdy), gh.backward(gdy));
    ExpectParametersNear(ce, ge, kTolerance);
    Tensor cr(Shape({2, 3, 11}), &cpu, r), gr(Shape({2, 3, 11}), &gpu, r);
    ExpectRelevanceAgrees(ch.propagate_relevance(cr, LRPRuleConfig{}), gh.propagate_relevance(gr, LRPRuleConfig{}));
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

inline void LlmAttentionRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    MultiHeadAttentionModule cm(LlmAttentionConfig(), &cpu), gm(LlmAttentionConfig(), &gpu);
    PrepareLlmAttention(cpu, gpu, cm, gm);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 5, 8}), 555);
}

inline void RotateHalfRoPERelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    RoPEModule cm(8, &cpu, 10000.0f, RoPELayout::RotateHalf), gm(8, &gpu, 10000.0f, RoPELayout::RotateHalf);
    cm.set_position_offset(7);
    gm.set_position_offset(7);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 3, 40, 8}), 545);
}

inline void TransformerBlockRelevance(DeviceBackend& gpu) {
    CPUBackend cpu;
    TransformerBlock cm(8, 2, 16, &cpu), gm(8, 2, 16, &gpu);
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 5, 8}), 560);
}

inline void EncoderBlockRelevance(DeviceBackend& gpu, bool post) {
    CPUBackend cpu;
    EncoderBlock cm(EncoderAttention(), 16, &cpu, EncoderLayout(post)), gm(EncoderAttention(), 16, &gpu, EncoderLayout(post));
    ModuleRelevance(cpu, gpu, cm, gm, Shape({2, 5, 8}), post ? 562 : 561);
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
                               unsigned seed, float tol = kTolerance) {
    RandomizeAndMirror(cm, gm, seed);
    std::vector<float> x = Random(static_cast<size_t>(in_shape.numel()), seed + 100);
    Tensor cx(in_shape, &cpu, x), gx(in_shape, &gpu, x);
    Tensor cy = cm.forward(cx);
    Tensor gy = gm.forward(gx);
    ExpectNear(cy, gy, tol);
    std::vector<float> dy = Random(static_cast<size_t>(cy.numel()), seed + 200);
    Tensor cdy(cy.shape(), &cpu, dy), gdy(gy.shape(), &gpu, dy);
    ExpectNear(cm.backward(cdy), gm.backward(gdy), tol);
    ExpectParametersNear(cm, gm, tol);
    ExpectRelevanceAgrees(cm.propagate_relevance(cdy, LRPRuleConfig{}), gm.propagate_relevance(gdy, LRPRuleConfig{}));
}

// 3x3 kernels on 7x6 images: windows overlap heavily, so col2im's gather order matters.
inline void Conv2DMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    Conv2DModule cm(3, 4, 3, 3, &cpu), gm(3, 4, 3, 3, &gpu);
    ImageModuleMatches(cpu, gpu, cm, gm, Shape({2, 3, 7, 6}), 700);
    // FND-6: stride 2, padding 1 -- padded taps read zero, strided windows skip columns.
    Conv2DModule cs(3, 4, 3, 3, &cpu, 2, 1), gs(3, 4, 3, 3, &gpu, 2, 1);
    ImageModuleMatches(cpu, gpu, cs, gs, Shape({2, 3, 7, 6}), 705);
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


// ---- Mission 6: SSM / linear-attention scans ---------------------------------------------------

// Forward, backward, parameter gradients and relevance over an (N, L, d_model) sequence. L = 7
// so each scan carries state across several steps, both directions.
// cpu must be the backend cm was built on (cm caches tensors owned by it).
template <typename Module>
inline void ScanModuleMatches(CPUBackend& cpu, DeviceBackend& gpu, Module& cm, Module& gm, unsigned seed) {
    RandomizeAndMirror(cm, gm, seed);
    std::vector<float> x = Random(2 * 7 * 6, seed + 100);
    Tensor cx(Shape({2, 7, 6}), &cpu, x), gx(Shape({2, 7, 6}), &gpu, x);
    Tensor cy = cm.forward(cx);
    Tensor gy = gm.forward(gx);
    ExpectNear(cy, gy);
    std::vector<float> dy = Random(static_cast<size_t>(cy.numel()), seed + 200);
    Tensor cdy(cy.shape(), &cpu, dy), gdy(gy.shape(), &gpu, dy);
    ExpectNear(cm.backward(cdy), gm.backward(gdy));
    ExpectParametersNear(cm, gm, kTolerance);
    ExpectRelevanceAgrees(cm.propagate_relevance(cdy, LRPRuleConfig{}), gm.propagate_relevance(gdy, LRPRuleConfig{}));
}

// Mission 6 gate: each scan model trained with Adam on GPU ends with the CPU's parameters.
template <typename Module>
inline void ScanModuleTrainsToSameParameters(CPUBackend& cpu, DeviceBackend& gpu, Module& cm, Module& gm,
                                             unsigned seed) {
    RandomizeAndMirror(cm, gm, seed);
    AdamOptimizer copt(0.01f, &cpu), gopt(0.01f, &gpu);
    MSELoss closs(&cpu), gloss(&gpu);
    std::vector<float> x = Random(2 * 5 * 6, seed + 1), y = Random(2 * 5 * 6, seed + 2);
    Tensor cx(Shape({2, 5, 6}), &cpu, x), gx(Shape({2, 5, 6}), &gpu, x);
    Tensor cy(Shape({2, 5, 6}), &cpu, y), gy(Shape({2, 5, 6}), &gpu, y);
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
    EXPECT_LT(last_loss, first_loss) << "the scan model did not actually train";
    ExpectParametersNear(cm, gm, 1e-3f);
}


// ---- Mission 7: reinforcement learning -----------------------------------------------------------

// Whole-number action indices in [0, action_dim), as floats.
inline std::vector<float> RandomActions(size_t n, int64_t action_dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, action_dim - 1);
    std::vector<float> v(n);
    for (auto& a : v) {
        a = static_cast<float>(dist(rng));
    }
    return v;
}

// DQN / policy-gradient / PPO losses: forward value and gradient. N = 37 rows spans several
// rl_rows lanes; PPO's old log-probs are spread so some rows clip on each side.
inline void RlLossesMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    const int64_t n = 37, a = 5;
    std::vector<float> q = Random(n * a, 1100, -2.0f, 2.0f), actions = RandomActions(n, a, 1101);
    std::vector<float> targets = Random(n, 1102, -2.0f, 2.0f), weights = Random(n, 1103, -1.5f, 1.5f);
    std::vector<float> old_log_probs = Random(n, 1104, -3.0f, -0.2f);
    Tensor cq(Shape({n, a}), &cpu, q), gq(Shape({n, a}), &gpu, q);
    Tensor ca(Shape({n, 1}), &cpu, actions), ga(Shape({n, 1}), &gpu, actions);  // read back for validation
    Tensor ct(Shape({n, 1}), &cpu, targets), gt(Shape({n, 1}), &gpu, targets);
    Tensor cw(Shape({n, 1}), &cpu, weights), gw(Shape({n, 1}), &gpu, weights);
    Tensor co(Shape({n, 1}), &cpu, old_log_probs), go(Shape({n, 1}), &gpu, old_log_probs);

    DQNLoss cd(&cpu), gd(&gpu);
    EXPECT_NEAR(cd.forward(cq, ca, ct), gd.forward(gq, ga, gt), kTolerance);
    ExpectNear(cd.backward(), gd.backward());

    PolicyGradientLoss cp(&cpu), gp(&gpu);
    EXPECT_NEAR(cp.forward(cq, ca, cw), gp.forward(gq, ga, gw), kTolerance);
    ExpectNear(cp.backward(), gp.backward());

    PPOClippedLoss cc(&cpu), gc(&gpu);
    EXPECT_NEAR(cc.forward(cq, ca, co, cw, 0.2f), gc.forward(gq, ga, go, gw, 0.2f), kTolerance);
    ExpectNear(cc.backward(), gc.backward());
}

// Vanilla and Double DQN targets, with terminal rows.
inline void DqnTargetsMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    const int64_t n = 41, a = 4;
    std::vector<float> online = Random(n * a, 1110), target = Random(n * a, 1111), rewards = Random(n, 1112);
    std::vector<float> dones(n);
    for (int64_t i = 0; i < n; ++i) {
        dones[static_cast<size_t>(i)] = (i % 3 == 0) ? 1.0f : 0.0f;
    }
    Tensor con(Shape({n, a}), &cpu, online), gon(Shape({n, a}), &gpu, online);
    Tensor cta(Shape({n, a}), &cpu, target), gta(Shape({n, a}), &gpu, target);
    Tensor cr(Shape({n, 1}), &cpu, rewards), gr(Shape({n, 1}), &gpu, rewards);
    Tensor cd(Shape({n, 1}), &cpu, dones), gd(Shape({n, 1}), &gpu, dones);
    ExpectNear(ComputeDQNTarget(cta, cr, cd, 0.9f, &cpu), ComputeDQNTarget(gta, gr, gd, 0.9f, &gpu));
    ExpectNear(ComputeDoubleDQNTarget(con, cta, cr, cd, 0.9f, &cpu),
               ComputeDoubleDQNTarget(gon, gta, gr, gd, 0.9f, &gpu));
}

// PolyakUpdate and SyncTargetNetwork through GPU parameters, and a CPU -> GPU sync.
inline void TargetNetworkUpdatesMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cs(5, 3, &cpu), gs(5, 3, &gpu), cd(5, 3, &cpu), gd(5, 3, &gpu);
    RandomizeAndMirror(cs, gs, 1120);
    RandomizeAndMirror(cd, gd, 1121);
    PolyakUpdate(cs, cd, 0.3f);
    PolyakUpdate(gs, gd, 0.3f);
    auto cp = cd.parameters(), gp = gd.parameters();
    for (size_t i = 0; i < cp.size(); ++i) {
        ExpectNear(*cp[i].value, *gp[i].value);
    }
    LinearModule gfresh(5, 3, &gpu);
    SyncTargetNetwork(gs, gfresh);  // device -> device
    SyncTargetNetwork(cd, gd);      // host -> device
    auto sp = cs.parameters(), fp = gfresh.parameters(), dp = gd.parameters();
    for (size_t i = 0; i < sp.size(); ++i) {
        ExpectNear(*sp[i].value, *fp[i].value, 0.0f);
        ExpectNear(*cp[i].value, *dp[i].value, 0.0f);
    }
}

// The host-boundary classes accept device tensors and hand back tensors on their own device,
// with the CPU's values: replay/rollout buffers, GAE, the environments, the timestep embedding.
inline void HostBoundariesMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    ReplayBuffer creplay(8, 3, 1, &cpu, 7), greplay(8, 3, 1, &gpu, 7);
    RolloutBuffer croll(10, 3, 1, &cpu), groll(10, 3, 1, &gpu);
    for (int i = 0; i < 10; ++i) {
        std::vector<float> obs = Random(3, 1130 + static_cast<unsigned>(i)), act = {static_cast<float>(i % 2)};
        Tensor co(Shape({1, 3}), &cpu, obs), go(Shape({1, 3}), &gpu, obs);
        Tensor ca(Shape({1, 1}), &cpu, act), ga(Shape({1, 1}), &gpu, act);
        creplay.add(co, ca, 0.5f * static_cast<float>(i), co, i % 4 == 3);
        greplay.add(go, ga, 0.5f * static_cast<float>(i), go, i % 4 == 3);
        croll.add(co, ca, 0.5f * static_cast<float>(i), -0.1f * static_cast<float>(i), i % 4 == 3);
        groll.add(go, ga, 0.5f * static_cast<float>(i), -0.1f * static_cast<float>(i), i % 4 == 3);
    }
    const ReplayBatch cb = creplay.sample(6), gb = greplay.sample(6);
    EXPECT_EQ(gb.observations.device(), gpu.device());
    ExpectNear(cb.observations, gb.observations, 0.0f);
    ExpectNear(cb.actions, gb.actions, 0.0f);
    ExpectNear(cb.rewards, gb.rewards, 0.0f);
    ExpectNear(cb.next_observations, gb.next_observations, 0.0f);
    ExpectNear(cb.dones, gb.dones, 0.0f);
    const RolloutBatch crb = croll.compute_returns(0.9f), grb = groll.compute_returns(0.9f);
    EXPECT_EQ(grb.returns.device(), gpu.device());
    ExpectNear(crb.observations, grb.observations, 0.0f);
    ExpectNear(crb.returns, grb.returns, 0.0f);
    ExpectNear(crb.log_probs, grb.log_probs, 0.0f);

    std::vector<float> values = Random(10, 1140);
    Tensor cv(Shape({10, 1}), &cpu, values), gv(Shape({10, 1}), &gpu, values);
    const GAEResult cg = ComputeGAE(croll.rewards(), croll.dones(), cv, 0.25f, 0.95f, 0.9f, &cpu);
    const GAEResult gg = ComputeGAE(groll.rewards(), groll.dones(), gv, 0.25f, 0.95f, 0.9f, &gpu);
    ExpectNear(cg.advantages, gg.advantages, 0.0f);
    ExpectNear(cg.returns, gg.returns, 0.0f);

    CartPoleEnv ccart(&cpu, 50, 3), gcart(&gpu, 50, 3);
    ContinuousCartPoleEnv ccont(&cpu, 50, 3), gcont(&gpu, 50, 3);
    std::vector<float> state = {0.01f, -0.02f, 0.03f, 0.0f};
    ExpectNear(ccart.reset(Tensor(Shape({1, 4}), &cpu, state)), gcart.reset(Tensor(Shape({1, 4}), &gpu, state)), 0.0f);
    (void)ccont.reset();
    (void)gcont.reset();
    for (int i = 0; i < 5; ++i) {
        std::vector<float> discrete = {static_cast<float>(i % 2)}, force = {0.4f - 0.2f * static_cast<float>(i)};
        const StepResult c1 = ccart.step(Tensor(Shape({1, 1}), &cpu, discrete));
        const StepResult g1 = gcart.step(Tensor(Shape({1, 1}), &gpu, discrete));
        EXPECT_EQ(g1.observation.device(), gpu.device());
        ExpectNear(c1.observation, g1.observation, 0.0f);
        ExpectNear(ccont.step(Tensor(Shape({1, 1}), &cpu, force)).observation,
                   gcont.step(Tensor(Shape({1, 1}), &gpu, force)).observation, 0.0f);
    }

    const Tensor ge = SinusoidalTimestepEmbedding(17, 8, &gpu);
    EXPECT_EQ(ge.device(), gpu.device());
    ExpectNear(SinusoidalTimestepEmbedding(17, 8, &cpu), ge, 0.0f);
}

// The agents and the GFlowNet sampler acting from networks on the GPU (action selection stays
// on the host): same actions and log-probabilities as the CPU for the same seeds.
inline void RlAgentsMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cnet(3, 4, &cpu), gnet(3, 4, &gpu);
    RandomizeAndMirror(cnet, gnet, 1150);
    DQNAgent cdqn(&cnet, 4, 0.3f, &cpu, 5), gdqn(&gnet, 4, 0.3f, &gpu, 5);
    CategoricalPolicyAgent ccat(&cnet, 4, &cpu, 6), gcat(&gnet, 4, &gpu, 6);
    for (int i = 0; i < 8; ++i) {
        std::vector<float> obs = Random(3, 1160 + static_cast<unsigned>(i));
        Tensor co(Shape({1, 3}), &cpu, obs), go(Shape({1, 3}), &gpu, obs);
        const Tensor ga = gdqn.act(go);
        EXPECT_EQ(ga.device(), gpu.device());
        ExpectNear(cdqn.act(co), ga, 0.0f);
        ExpectNear(ccat.act(co), gcat.act(go), 0.0f);
        EXPECT_NEAR(ccat.log_prob(), gcat.log_prob(), kTolerance);
        ExpectNear(ccat.act_greedy(co), gcat.act_greedy(go), 0.0f);
    }

    HyperGridEnv cenv(&cpu, 2, 6), genv(&gpu, 2, 6);
    LinearModule cpol(2, 3, &cpu), gpol(2, 3, &gpu);
    RandomizeAndMirror(cpol, gpol, 1170);
    GFlowNetForwardPolicy cfp(&cpol, 3, &cpu, 9), gfp(&gpol, 3, &gpu, 9);
    for (int episode = 0; episode < 4; ++episode) {
        const GFlowNetTrajectory ct = sample_gflownet_trajectory(cenv, cfp);
        const GFlowNetTrajectory gt = sample_gflownet_trajectory(genv, gfp);
        EXPECT_EQ(ct.actions, gt.actions);
        EXPECT_NEAR(ct.sum_log_pf, gt.sum_log_pf, kTolerance);
        EXPECT_FLOAT_EQ(ct.sum_log_pb, gt.sum_log_pb);
        EXPECT_FLOAT_EQ(ct.terminal_reward, gt.terminal_reward);
    }
}

// Mission 7 gate: a DQN Q-network (Double-DQN targets from a Polyak-tracked target network)
// trained with Adam on GPU ends with the CPU's parameters.
inline void DqnTrainsToSameParameters(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cq(4, 3, &cpu), gq(4, 3, &gpu), ct(4, 3, &cpu), gt(4, 3, &gpu);
    RandomizeAndMirror(cq, gq, 1180);
    SyncTargetNetwork(cq, ct);
    SyncTargetNetwork(gq, gt);
    AdamOptimizer copt(0.01f, &cpu), gopt(0.01f, &gpu);
    DQNLoss closs(&cpu), gloss(&gpu);
    const int64_t n = 16;
    std::vector<float> obs = Random(n * 4, 1181), next = Random(n * 4, 1182), rewards = Random(n, 1183);
    std::vector<float> actions = RandomActions(n, 3, 1184), dones(n, 0.0f);
    dones[3] = dones[9] = 1.0f;
    Tensor co(Shape({n, 4}), &cpu, obs), go(Shape({n, 4}), &gpu, obs);
    Tensor cn(Shape({n, 4}), &cpu, next), gn(Shape({n, 4}), &gpu, next);
    Tensor cr(Shape({n, 1}), &cpu, rewards), gr(Shape({n, 1}), &gpu, rewards);
    Tensor ca(Shape({n, 1}), &cpu, actions), ga(Shape({n, 1}), &gpu, actions);
    Tensor cd(Shape({n, 1}), &cpu, dones), gd(Shape({n, 1}), &gpu, dones);
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 15; ++step) {
        const Tensor c_target = ComputeDoubleDQNTarget(cq.forward(cn), ct.forward(cn), cr, cd, 0.9f, &cpu);
        const Tensor g_target = ComputeDoubleDQNTarget(gq.forward(gn), gt.forward(gn), gr, gd, 0.9f, &gpu);
        copt.zero_grad(cq);
        gopt.zero_grad(gq);
        const float lc = closs.forward(cq.forward(co), ca, c_target);
        const float lg = gloss.forward(gq.forward(go), ga, g_target);
        EXPECT_NEAR(lc, lg, 1e-3f) << "loss diverged at step " << step;
        (void)cq.backward(closs.backward());
        (void)gq.backward(gloss.backward());
        copt.step(cq);
        gopt.step(gq);
        PolyakUpdate(cq, ct, 0.1f);
        PolyakUpdate(gq, gt, 0.1f);
        if (step == 0) {
            first_loss = lc;
        }
        last_loss = lc;
    }
    EXPECT_LT(last_loss, first_loss) << "the Q-network did not actually train";
    ExpectParametersNear(cq, gq, 1e-3f);
}


// Mission 7 gate: a full DQN loop -- CartPole environment, epsilon-greedy agent, replay buffer,
// Bellman targets, DQN loss, Adam -- run in lockstep on CPU and on the GPU backend (every
// component built on that backend) ends with the same network. Same seeds on both sides, so
// the environment, exploration and replay sampling streams agree.
inline void DqnLoopTrainsToSameParameters(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cl1(4, 16, &cpu), gl1(4, 16, &gpu);
    ReluModule cr1(&cpu), gr1(&gpu);
    LinearModule cl2(16, 2, &cpu), gl2(16, 2, &gpu);
    SequentialModule cnet({&cl1, &cr1, &cl2}), gnet({&gl1, &gr1, &gl2});
    RandomizeAndMirror(cnet, gnet, 1120);
    LinearModule ct1(4, 16, &cpu), gt1(4, 16, &gpu);
    ReluModule ctr(&cpu), gtr(&gpu);
    LinearModule ct2(16, 2, &cpu), gt2(16, 2, &gpu);
    SequentialModule ctarget({&ct1, &ctr, &ct2}), gtarget({&gt1, &gtr, &gt2});
    SyncTargetNetwork(cnet, ctarget);
    SyncTargetNetwork(gnet, gtarget);

    CartPoleEnv cenv(&cpu, 200, 7), genv(&gpu, 200, 7);
    DQNAgent cagent(&cnet, 2, 0.3f, &cpu, 11), gagent(&gnet, 2, 0.3f, &gpu, 11);
    ReplayBuffer cbuf(256, 4, 1, &cpu, 13), gbuf(256, 4, 1, &gpu, 13);
    AdamOptimizer copt(1e-3f, &cpu), gopt(1e-3f, &gpu);
    DQNLoss closs(&cpu), gloss(&gpu);

    Tensor cobs = cenv.reset(), gobs = genv.reset();
    EXPECT_EQ(gobs.device(), gpu.device());
    for (int step = 0; step < 60; ++step) {
        Tensor ca = cagent.act(cobs), ga = gagent.act(gobs);
        ASSERT_EQ(ToHost(ca), ToHost(ga)) << "agents chose different actions at step " << step;
        StepResult cs = cenv.step(ca), gs = genv.step(ga);
        ASSERT_EQ(cs.done, gs.done);
        cbuf.add(cobs, ca, cs.reward, cs.observation, cs.done);
        gbuf.add(gobs, ga, gs.reward, gs.observation, gs.done);
        cobs = cs.done ? cenv.reset() : cs.observation;
        gobs = gs.done ? genv.reset() : gs.observation;

        if (cbuf.size() >= 16) {
            ReplayBatch cb = cbuf.sample(16), gb = gbuf.sample(16);
            ExpectNear(cb.observations, gb.observations, 1e-3f);
            Tensor cy = ComputeDQNTarget(ctarget.forward(cb.next_observations), cb.rewards, cb.dones, 0.99f, &cpu);
            Tensor gy = ComputeDQNTarget(gtarget.forward(gb.next_observations), gb.rewards, gb.dones, 0.99f, &gpu);
            copt.zero_grad(cnet);
            gopt.zero_grad(gnet);
            const float lc = closs.forward(cnet.forward(cb.observations), cb.actions, cy);
            const float lg = gloss.forward(gnet.forward(gb.observations), gb.actions, gy);
            EXPECT_NEAR(lc, lg, 1e-3f) << "TD loss diverged at step " << step;
            (void)cnet.backward(closs.backward());
            (void)gnet.backward(gloss.backward());
            copt.step(cnet);
            gopt.step(gnet);
        }
    }
    ExpectParametersNear(cnet, gnet, 1e-3f);
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

// LRP-rules Mission 2: every rule on Linear and Conv2D, and each Zennit preset through
// LRP::explain on a small conv net, CPU vs GPU.
inline void LRPRulesMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    std::vector<LRPRuleConfig> rules(6);
    rules[0].epsilon_bias_in_denominator = true;
    rules[1].rule = LRPRule::AlphaBeta;  // ZPlus
    rules[2].rule = LRPRule::AlphaBeta;
    rules[2].alpha = 2.0f;
    rules[2].beta = 1.0f;
    rules[3].rule = LRPRule::Gamma;
    rules[4].rule = LRPRule::ZBox;
    rules[4].low = -1.0f;
    rules[5].rule = LRPRule::Epsilon;  // the original pre-bias rule, for completeness

    LinearModule cl(6, 5, &cpu), gl(6, 5, &gpu);
    RandomizeAndMirror(cl, gl, 1300);
    Conv2DModule cc(2, 3, 2, 2, &cpu), gc(2, 3, 2, 2, &gpu);
    RandomizeAndMirror(cc, gc, 1301);
    const std::vector<float> lx = Random(4 * 6, 1302), cx = Random(2 * 2 * 5 * 5, 1303);
    const std::vector<float> lr = Random(4 * 5, 1304), cr = Random(2 * 3 * 4 * 4, 1305);
    (void)cl.forward(Tensor(Shape({4, 6}), &cpu, lx));
    (void)gl.forward(Tensor(Shape({4, 6}), &gpu, lx));
    (void)cc.forward(Tensor(Shape({2, 2, 5, 5}), &cpu, cx));
    (void)gc.forward(Tensor(Shape({2, 2, 5, 5}), &gpu, cx));
    for (const LRPRuleConfig& config : rules) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        Tensor g_lin = gl.propagate_relevance(Tensor(Shape({4, 5}), &gpu, lr), config);
        EXPECT_EQ(g_lin.device(), gpu.device());
        ExpectRelevanceAgrees(cl.propagate_relevance(Tensor(Shape({4, 5}), &cpu, lr), config), g_lin);
        ExpectRelevanceAgrees(cc.propagate_relevance(Tensor(Shape({2, 3, 4, 4}), &cpu, cr), config),
                              gc.propagate_relevance(Tensor(Shape({2, 3, 4, 4}), &gpu, cr), config));
    }
    // FND-6: every rule through a strided, padded Conv2D (ZBox fills its bounds per image, so
    // padded taps get zero bounds on both backends).
    Conv2DModule cp(2, 3, 3, 3, &cpu, 2, 1), gp(2, 3, 3, 3, &gpu, 2, 1);
    RandomizeAndMirror(cp, gp, 1306);
    (void)cp.forward(Tensor(Shape({2, 2, 5, 5}), &cpu, cx));
    (void)gp.forward(Tensor(Shape({2, 2, 5, 5}), &gpu, cx));
    const std::vector<float> pr = Random(2 * 3 * 3 * 3, 1307);
    for (const LRPRuleConfig& config : rules) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        ExpectRelevanceAgrees(cp.propagate_relevance(Tensor(Shape({2, 3, 3, 3}), &cpu, pr), config),
                              gp.propagate_relevance(Tensor(Shape({2, 3, 3, 3}), &gpu, pr), config));
    }

    // conv -> relu -> conv -> relu -> flatten -> linear, through each preset.
    Conv2DModule cc1(1, 3, 2, 2, &cpu), gc1(1, 3, 2, 2, &gpu), cc2(3, 2, 2, 2, &cpu), gc2(3, 2, 2, 2, &gpu);
    ReluModule cr1(&cpu), gr1(&gpu), cr2(&cpu), gr2(&gpu);
    FlattenModule cf(&cpu), gf(&gpu);
    LinearModule cl2(8, 3, &cpu), gl2(8, 3, &gpu);
    RandomizeAndMirror(cc1, gc1, 1306);
    RandomizeAndMirror(cc2, gc2, 1307);
    RandomizeAndMirror(cl2, gl2, 1308);
    ExplainerContext cctx({&cc1, &cr1, &cc2, &cr2, &cf, &cl2}), gctx({&gc1, &gr1, &gc2, &gr2, &gf, &gl2});
    const std::vector<float> x = Random(2 * 1 * 4 * 4, 1309, 0.0f, 1.0f);
    Tensor cxt(Shape({2, 1, 4, 4}), &cpu, x), gxt(Shape({2, 1, 4, 4}), &gpu, x);
    for (const LRP& lrp : {LRP::epsilon_plus(), LRP::epsilon_alpha2_beta1(), LRP::epsilon_gamma_box(0.0f, 1.0f)}) {
        Attribution ca = lrp.explain(cctx, cxt, 1, &cpu);
        Attribution ga = lrp.explain(gctx, gxt, 1, &gpu);
        SCOPED_TRACE(ca.metadata.at("rule"));
        EXPECT_EQ(ca.metadata.at("rules"), ga.metadata.at("rules"));
        EXPECT_EQ(ga.values.device(), gpu.device());
        ExpectRelevanceAgrees(ca.values, ga.values);
    }
}

// FND-8 on real hardware: a tensor on the wrong device must throw before any kernel sees it.
// Before, a CPU input to a GPU module was an uncatchable HSA fault on gfx1151 (and a sticky
// illegal-address error on CUDA), so reaching the assertions below at all is the test.
inline void DeviceMismatchThrowsInsteadOfFaulting(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule on_gpu(3, 2, &gpu), on_cpu(3, 2, &cpu);
    const std::vector<float> values = Random(2 * 3, 1500);
    Tensor x_cpu(Shape({2, 3}), &cpu, values), x_gpu(Shape({2, 3}), &gpu, values);
    EXPECT_THROW((void)on_gpu.forward(x_cpu), std::invalid_argument);
    EXPECT_THROW((void)on_cpu.forward(x_gpu), std::invalid_argument);
    Tensor y = on_gpu.forward(x_gpu);
    EXPECT_THROW((void)on_gpu.backward(Tensor(y.shape(), &cpu, std::vector<float>(4, 1.0f))), std::invalid_argument);
    MSELoss loss(&gpu);
    EXPECT_THROW((void)loss.forward(x_cpu, x_cpu), std::invalid_argument);

    // gpu_review #2: CPU samples collated onto the GPU land on the GPU, with their values.
    std::vector<Tensor> rows = {Tensor(Shape({3}), &cpu, {1, 2, 3}), Tensor(Shape({3}), &cpu, {4, 5, 6})};
    Tensor stacked = Tensor::Stack(rows, &gpu);
    EXPECT_EQ(stacked.device(), gpu.device());
    EXPECT_EQ(ToHost(stacked), (std::vector<float>{1, 2, 3, 4, 5, 6}));
    Tensor back = Tensor::Stack({stacked}, &cpu);
    EXPECT_EQ(back.device(), DeviceType::Cpu);
    EXPECT_EQ(ToHost(back), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

// TRN-3: global-norm clipping on the device (per-parameter dot products, then one scale).
inline void ClipGradNormMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    LinearModule cm(5, 3, &cpu), gm(5, 3, &gpu);
    const std::vector<float> gw = Random(15, 1600), gb = Random(3, 1601);
    *cm.parameters()[0].grad = Tensor(Shape({5, 3}), &cpu, gw);
    *gm.parameters()[0].grad = Tensor(Shape({5, 3}), &gpu, gw);
    *cm.parameters()[1].grad = Tensor(Shape({3}), &cpu, gb);
    *gm.parameters()[1].grad = Tensor(Shape({3}), &gpu, gb);
    const float cn = ClipGradNorm(cm, 0.5f), gn = ClipGradNorm(gm, 0.5f);
    EXPECT_NEAR(cn, gn, 1e-5f);
    ExpectNear(*cm.parameters()[0].grad, *gm.parameters()[0].grad);
    ExpectNear(*cm.parameters()[1].grad, *gm.parameters()[1].grad);
}

// TRN-5: token cross-entropy (rl_rows PgLoss/PgGrad with 0/1 token weights), ignored tokens
// and an explicit accumulation normalizer.
inline void TokenCrossEntropyMatches(DeviceBackend& gpu) {
    CPUBackend cpu;
    const std::vector<float> logits = Random(6 * 7, 1700);
    const std::vector<float> targets = {3, -100, 0, 6, -100, 2};
    TokenCrossEntropyLoss cl(&cpu), gl(&gpu);
    const float cv = cl.forward(Tensor(Shape({2, 3, 7}), &cpu, logits), Tensor(Shape({2, 3}), &cpu, targets), 9.0f);
    const float gv = gl.forward(Tensor(Shape({2, 3, 7}), &gpu, logits), Tensor(Shape({2, 3}), &gpu, targets), 9.0f);
    EXPECT_NEAR(cv, gv, 1e-5f);
    ExpectNear(cl.backward(), gl.backward());
}

// HIP-2: BatchNorm with few channels and large planes, so one channel spans many blocks; then
// eval mode and the running statistics; then run-to-run determinism on the GPU.
inline void BatchNormLargePlanesMatch(DeviceBackend& gpu) {
    CPUBackend cpu;
    for (const Shape& shape : {Shape({8, 3, 64, 64}), Shape({2, 1, 5, 7}), Shape({4, 16, 33, 31})}) {
        const int64_t c = shape.dim(1);
        // Per-channel sums over m = N*H*W elements: the CPU reference adds them one at a time in
        // float, the GPU by a tree, so they differ by rounding that grows with m (32,768 here).
        const int64_t m = shape.numel() / c;
        const float tol = kTolerance * (1.0f + static_cast<float>(m) / 4096.0f);
        BatchNormModule cbn(c, &cpu), gbn(c, &gpu);
        ImageModuleMatches(cpu, gpu, cbn, gbn, shape, 900, tol);
        ExpectNear(cbn.running_mean(), gbn.running_mean(), tol);
        ExpectNear(cbn.running_var(), gbn.running_var(), tol);

        cbn.set_training(false);
        gbn.set_training(false);
        std::vector<float> x = Random(static_cast<size_t>(shape.numel()), 901);
        Tensor cy = cbn.forward(Tensor(shape, &cpu, x)), gy = gbn.forward(Tensor(shape, &gpu, x));
        ExpectNear(cy, gy, tol);
        std::vector<float> dy = Random(static_cast<size_t>(shape.numel()), 902);
        ExpectNear(cbn.backward(Tensor(shape, &cpu, dy)), gbn.backward(Tensor(shape, &gpu, dy)), tol);
    }
    // Same input twice on the GPU: bit-identical (fixed partition and order, no atomics).
    BatchNormModule a(3, &gpu);
    const Shape shape({8, 3, 64, 64});
    std::vector<float> x = Random(static_cast<size_t>(shape.numel()), 903);
    const std::vector<float> first = ToHost(a.forward(Tensor(shape, &gpu, x)));
    EXPECT_EQ(ToHost(a.forward(Tensor(shape, &gpu, x))), first);
}

}  // namespace training_equivalence
}  // namespace pulsatrix

// Instantiates every case for one GPU fixture. FIXTURE must expose the GPU backend as MEMBER.
#define PULSATRIX_TRAINING_EQUIVALENCE_TESTS(FIXTURE, MEMBER)                                        \
    TEST_F(FIXTURE, LRPExplainerMatchesCPU) { ::pulsatrix::training_equivalence::LRPExplainerMatches(MEMBER); } \
    TEST_F(FIXTURE, PosthocExplainersMatchCPU) {                                                     \
        ::pulsatrix::training_equivalence::PosthocExplainersMatch(MEMBER);                           \
    }                                                                                                \
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
    TEST_F(FIXTURE, LlmAttentionForwardBackwardMatchCPU) {                                           \
        ::pulsatrix::training_equivalence::LlmAttentionMatches(MEMBER);                              \
    }                                                                                                \
    TEST_F(FIXTURE, CachedAttentionMatchesCPU) {                                                     \
        ::pulsatrix::training_equivalence::CachedAttentionMatches(MEMBER);                           \
    }                                                                                                \
    TEST_F(FIXTURE, RotateHalfRoPEForwardBackwardMatchCPU) {                                         \
        ::pulsatrix::training_equivalence::RotateHalfRoPEMatches(MEMBER);                            \
    }                                                                                                \
    TEST_F(FIXTURE, TransformerBlockForwardBackwardMatchCPU) {                                       \
        ::pulsatrix::training_equivalence::TransformerBlockMatches(MEMBER);                          \
    }                                                                                                \
    TEST_F(FIXTURE, EncoderBlockForwardBackwardMatchCPU) {                                           \
        ::pulsatrix::training_equivalence::EncoderBlockMatches(MEMBER, false);                       \
        ::pulsatrix::training_equivalence::EncoderBlockMatches(MEMBER, true);                        \
    }                                                                                                \
    TEST_F(FIXTURE, EmbeddingForwardBackwardMatchCPU) { ::pulsatrix::training_equivalence::EmbeddingMatches(MEMBER); } \
    TEST_F(FIXTURE, TiedLMHeadMatchesCPU) { ::pulsatrix::training_equivalence::TiedLMHeadMatches(MEMBER); } \
    TEST_F(FIXTURE, EncoderLMMatchesCPU) { ::pulsatrix::training_equivalence::EncoderLMMatches(MEMBER); } \
    TEST_F(FIXTURE, ContactsMatchCPU) { ::pulsatrix::training_equivalence::ContactsMatch(MEMBER); } \
    TEST_F(FIXTURE, EncoderExplanationsMatchCPU) { ::pulsatrix::training_equivalence::EncoderExplanationsMatch(MEMBER); } \
    TEST_F(FIXTURE, MaskedLMTrainingMatchesCPU) { ::pulsatrix::training_equivalence::MaskedLMTrainingMatches(MEMBER); } \
    TEST_F(FIXTURE, FeaturizerTrainingMatchesCPU) { ::pulsatrix::training_equivalence::FeaturizerTrainingMatches(MEMBER); } \
    TEST_F(FIXTURE, TopKFeaturizerTrainingMatchesCPU) { ::pulsatrix::training_equivalence::TopKFeaturizerTrainingMatches(MEMBER); } \
    TEST_F(FIXTURE, SaeVariantsTrainingMatchCPU) { ::pulsatrix::training_equivalence::SaeVariantsTrainingMatch(MEMBER); } \
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
    TEST_F(FIXTURE, LlmAttentionRelevanceAgreesWithCPU) {                                            \
        ::pulsatrix::training_equivalence::LlmAttentionRelevance(MEMBER);                            \
    }                                                                                                \
    TEST_F(FIXTURE, RotateHalfRoPERelevanceAgreesWithCPU) {                                          \
        ::pulsatrix::training_equivalence::RotateHalfRoPERelevance(MEMBER);                          \
    }                                                                                                \
    TEST_F(FIXTURE, TransformerBlockRelevanceAgreesWithCPU) {                                        \
        ::pulsatrix::training_equivalence::TransformerBlockRelevance(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, EncoderBlockRelevanceAgreesWithCPU) {                                            \
        ::pulsatrix::training_equivalence::EncoderBlockRelevance(MEMBER, false);                     \
        ::pulsatrix::training_equivalence::EncoderBlockRelevance(MEMBER, true);                      \
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
    TEST_F(FIXTURE, ClipGradNormMatchesCPU) { ::pulsatrix::training_equivalence::ClipGradNormMatches(MEMBER); } \
    TEST_F(FIXTURE, BatchNormLargePlanesMatchCPU) {                                                 \
        ::pulsatrix::training_equivalence::BatchNormLargePlanesMatch(MEMBER);                      \
    }                                                                                               \
    TEST_F(FIXTURE, TokenCrossEntropyMatchesCPU) {                                                  \
        ::pulsatrix::training_equivalence::TokenCrossEntropyMatches(MEMBER);                       \
    }                                                                                               \
    TEST_F(FIXTURE, DeviceMismatchThrowsInsteadOfFaulting) {                                       \
        ::pulsatrix::training_equivalence::DeviceMismatchThrowsInsteadOfFaulting(MEMBER);         \
    }                                                                                               \
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
    TEST_F(FIXTURE, MambaMatchesCPU) {                                                               \
        ::pulsatrix::CPUBackend cpu;                                                                 \
        ::pulsatrix::MambaModule cm(6, 4, &cpu), gm(6, 4, &MEMBER);                                  \
        ::pulsatrix::training_equivalence::ScanModuleMatches(cpu, MEMBER, cm, gm, 1000);             \
    }                                                                                                \
    TEST_F(FIXTURE, RWKVMatchesCPU) {                                                                \
        ::pulsatrix::CPUBackend cpu;                                                                 \
        ::pulsatrix::RWKVModule cm(6, &cpu), gm(6, &MEMBER);                                         \
        ::pulsatrix::training_equivalence::ScanModuleMatches(cpu, MEMBER, cm, gm, 1010);             \
    }                                                                                                \
    TEST_F(FIXTURE, RetNetMatchesCPU) {                                                              \
        ::pulsatrix::CPUBackend cpu;                                                                 \
        ::pulsatrix::RetNetModule cm(6, 4, 0.8f, &cpu), gm(6, 4, 0.8f, &MEMBER);                     \
        ::pulsatrix::training_equivalence::ScanModuleMatches(cpu, MEMBER, cm, gm, 1020);             \
    }                                                                                                \
    TEST_F(FIXTURE, ScanModelsTrainedWithAdamEndWithCPUParameters) {                                 \
        using namespace ::pulsatrix;                                                                 \
        CPUBackend cpu;                                                                              \
        MambaModule cm(6, 4, &cpu), gm(6, 4, &MEMBER);                                               \
        training_equivalence::ScanModuleTrainsToSameParameters(cpu, MEMBER, cm, gm, 1030);           \
        RWKVModule cr(6, &cpu), gr(6, &MEMBER);                                                      \
        training_equivalence::ScanModuleTrainsToSameParameters(cpu, MEMBER, cr, gr, 1040);           \
        RetNetModule ct(6, 4, 0.8f, &cpu), gt(6, 4, 0.8f, &MEMBER);                                  \
        training_equivalence::ScanModuleTrainsToSameParameters(cpu, MEMBER, ct, gt, 1050);           \
    }                                                                                                \
    TEST_F(FIXTURE, RlLossesMatchCPU) { ::pulsatrix::training_equivalence::RlLossesMatch(MEMBER); }   \
    TEST_F(FIXTURE, DqnTargetsMatchCPU) { ::pulsatrix::training_equivalence::DqnTargetsMatch(MEMBER); } \
    TEST_F(FIXTURE, TargetNetworkUpdatesMatchCPU) {                                                  \
        ::pulsatrix::training_equivalence::TargetNetworkUpdatesMatch(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, RlHostBoundariesMatchCPU) { ::pulsatrix::training_equivalence::HostBoundariesMatch(MEMBER); } \
    TEST_F(FIXTURE, RlAgentsMatchCPU) { ::pulsatrix::training_equivalence::RlAgentsMatch(MEMBER); }  \
    TEST_F(FIXTURE, DqnTrainedWithAdamEndsWithCPUParameters) {                                       \
        ::pulsatrix::training_equivalence::DqnTrainsToSameParameters(MEMBER);                        \
    }                                                                                                \
    TEST_F(FIXTURE, DqnLoopOnGpuEndsWithCPUParameters) {                                             \
        ::pulsatrix::training_equivalence::DqnLoopTrainsToSameParameters(MEMBER);                    \
    }                                                                                                \
    TEST_F(FIXTURE, MlpTrainedWithSGDEndsWithCPUParameters) {                                        \
        ::pulsatrix::SGDOptimizer cpu_opt(0.05f), gpu_opt(0.05f);                                    \
        ::pulsatrix::training_equivalence::MlpTrainsToSameParameters(MEMBER, cpu_opt, gpu_opt);      \
    }                                                                                                \
    TEST_F(FIXTURE, MlpTrainedWithAdamEndsWithCPUParameters) {                                       \
        ::pulsatrix::CPUBackend adam_cpu_backend;                                                    \
        ::pulsatrix::AdamOptimizer cpu_opt(0.01f, &adam_cpu_backend), gpu_opt(0.01f, &MEMBER);       \
        ::pulsatrix::training_equivalence::MlpTrainsToSameParameters(MEMBER, cpu_opt, gpu_opt);      \
    }                                                                                               \
    TEST_F(FIXTURE, LRPRulesMatchCPU) { ::pulsatrix::training_equivalence::LRPRulesMatch(MEMBER); }
