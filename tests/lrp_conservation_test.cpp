#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>

#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conjunction_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mamba_module.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/swiglu_module.hpp"

// Phase 4 Mission 0: charter's "not just a spot-check on one architecture" LRP
// completeness requirement, turned into a systematic TEST_P suite over every module type
// this codebase has, plus a genuinely new end-to-end conservation test. Reuses each
// module's existing, already-proven-correct weight/input/relevance setups from Phase
// 1-3's own per-module tests (LinearModuleTest, Conv2DModuleTest, ReluModuleTest,
// FlattenModuleTest) -- this mission doesn't re-derive their math, it systematizes it.
namespace pulsatrix {
namespace {

struct ConservationCase {
    std::string name;
    std::function<std::pair<float, float>()> run;  // returns {sum(relevance_in), sum(relevance_out)}
};

class LRPConservationTest : public ::testing::TestWithParam<ConservationCase> {};

TEST_P(LRPConservationTest, RelevanceIsConserved) {
    auto [sum_in, sum_out] = GetParam().run();
    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

std::vector<ConservationCase> AllModuleTypeCases() {
    std::vector<ConservationCase> cases;

    cases.push_back({"LinearModule", [] {
                          CPUBackend backend;
                          LinearModule linear(3, 2, &backend);
                          linear.set_weight({1.0f, -2.0f, 3.0f, 0.5f, 2.5f, -1.0f});
                          linear.set_bias({0.1f, -0.2f});
                          Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
                          (void)linear.forward(x);
                          Tensor relevance_out(Shape({1, 2}), &backend, {4.0f, 6.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = linear.propagate_relevance(relevance_out, config);
                          float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
                          float sum_out = relevance_out.data()[0] + relevance_out.data()[1];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"Conv2DModule", [] {
                          CPUBackend backend;
                          Conv2DModule conv(1, 2, 2, 2, &backend);
                          conv.set_kernel({1.0f, -0.5f, 0.5f, 2.0f, -1.0f, 1.5f, 0.5f, -0.5f});
                          conv.set_bias({0.1f, -0.2f});
                          Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
                          (void)conv.forward(input);
                          Tensor relevance_out(Shape({1, 2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = conv.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"ReluModule", [] {
                          CPUBackend backend;
                          ReluModule relu(&backend);
                          Tensor x(Shape({3}), &backend, {-1.0f, 2.0f, 3.0f});
                          (void)relu.forward(x);
                          Tensor relevance_out(Shape({3}), &backend, {0.1f, 5.0f, -2.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = relu.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"FlattenModule", [] {
                          CPUBackend backend;
                          FlattenModule flatten(&backend);
                          Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
                          (void)flatten.forward(input);
                          Tensor relevance_out(Shape({8}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = flatten.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"MaxPool2DModule", [] {
                          CPUBackend backend;
                          MaxPool2DModule pool(2, 2, &backend);
                          Tensor input(Shape({1, 1, 4, 4}), &backend,
                                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                        13.0f, 14.0f, 15.0f, 16.0f});
                          (void)pool.forward(input);
                          Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = pool.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"AvgPool2DModule", [] {
                          CPUBackend backend;
                          AvgPool2DModule pool(2, 2, &backend);
                          Tensor input(Shape({1, 1, 4, 4}), &backend,
                                       {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                        13.0f, 14.0f, 15.0f, 16.0f});
                          (void)pool.forward(input);
                          Tensor relevance_out(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = pool.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"EmbeddingModule", [] {
                          CPUBackend backend;
                          EmbeddingModule emb(3, 3, &backend);
                          emb.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
                          Tensor input(Shape({1, 2}), &backend, {0.0f, 2.0f});
                          (void)emb.forward(input);
                          Tensor relevance_out(Shape({1, 2, 3}), &backend, {1.0f, 2.0f, 3.0f, 0.5f, 0.5f, 1.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = emb.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"SequentialModule", [] {
                          CPUBackend backend;
                          LinearModule linear1(3, 4, &backend);
                          linear1.set_weight(
                              {0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
                          linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
                          ReluModule relu(&backend);
                          LinearModule linear2(4, 2, &backend);
                          linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
                          linear2.set_bias({0.05f, -0.05f});
                          SequentialModule seq({&linear1, &relu, &linear2});

                          Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
                          (void)seq.forward(input);
                          Tensor relevance_out(Shape({1, 2}), &backend, {4.0f, 6.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = seq.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"RNNModule", [] {
                          CPUBackend backend;
                          RNNModule rnn(2, 2, &backend);
                          rnn.set_weight_xh({0.1f, -0.2f, 0.3f, 0.15f});
                          rnn.set_weight_hh({0.05f, -0.1f, 0.2f, 0.05f});
                          rnn.set_bias({0.01f, -0.02f});
                          Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
                          (void)rnn.forward(input);
                          Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = rnn.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"LSTMModule", [] {
                          CPUBackend backend;
                          LSTMModule lstm(2, 2, &backend);
                          lstm.set_weight_xi({0.10f, -0.20f, 0.30f, 0.15f});
                          lstm.set_weight_hi({0.05f, -0.10f, 0.20f, 0.05f});
                          lstm.set_bias_i({0.01f, -0.02f});
                          lstm.set_weight_xf({-0.25f, 0.35f, 0.05f, -0.15f});
                          lstm.set_weight_hf({0.12f, 0.08f, -0.18f, 0.22f});
                          lstm.set_bias_f({0.30f, -0.05f});
                          lstm.set_weight_xg({0.40f, 0.10f, -0.30f, 0.25f});
                          lstm.set_weight_hg({-0.07f, 0.14f, 0.09f, -0.11f});
                          lstm.set_bias_g({-0.03f, 0.04f});
                          lstm.set_weight_xo({0.22f, -0.33f, 0.18f, 0.27f});
                          lstm.set_weight_ho({0.06f, 0.13f, -0.09f, 0.16f});
                          lstm.set_bias_o({0.05f, 0.02f});
                          Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
                          (void)lstm.forward(input);
                          Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = lstm.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"GRUModule", [] {
                          CPUBackend backend;
                          GRUModule gru(2, 2, &backend);
                          gru.set_weight_xz({0.10f, -0.20f, 0.30f, 0.15f});
                          gru.set_weight_hz({0.05f, -0.10f, 0.20f, 0.05f});
                          gru.set_bias_z({0.01f, -0.02f});
                          gru.set_weight_xr({-0.25f, 0.35f, 0.05f, -0.15f});
                          gru.set_weight_hr({0.12f, 0.08f, -0.18f, 0.22f});
                          gru.set_bias_r({0.30f, -0.05f});
                          gru.set_weight_xn({0.40f, 0.10f, -0.30f, 0.25f});
                          gru.set_weight_hn({-0.07f, 0.14f, 0.09f, -0.11f});
                          gru.set_bias_n({-0.03f, 0.04f});
                          Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
                          (void)gru.forward(input);
                          Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = gru.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // RoPEModule is a fixed, bias-free linear map per feature pair propagated with the
    // standard weighted-connection epsilon rule, so unlike SoftmaxModule (deliberately
    // excluded -- AttnLRP Eq. 13 does not conserve) it belongs in this systematic sweep.
    // Shape (1, 3, 4): one (L=3, head_dim=4) slice, i.e. 3 positions x 2 feature pairs,
    // exercising both the pos=0 identity rotation and two genuinely rotated positions.
    cases.push_back({"RoPEModule", [] {
                          CPUBackend backend;
                          RoPEModule rope(4, &backend);
                          Tensor input(Shape({1, 3, 4}), &backend,
                                       {0.5f, -1.2f, 2.0f, 0.1f, 0.7f, -0.3f, -0.9f, 1.4f, 0.25f, -0.6f, 1.1f,
                                        0.05f});
                          (void)rope.forward(input);
                          Tensor relevance_out(Shape({1, 3, 4}), &backend,
                                               {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -0.25f, 1.2f, 0.6f, 0.9f,
                                                -1.1f, 0.4f});
                          LRPRuleConfig config;
                          Tensor relevance_in = rope.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // MambaModule's MambaLRP rule detaches Abar_t/Bbar_t/C_t and redistributes over plain
    // weighted sums, so (with h_0 == 0) it conserves up to the epsilon stabilizers only --
    // measured gap 2.5e-5 against a sum(R_out) of 5.3 in mamba_module_test.cpp's
    // PropagateRelevanceConservationGapIsMeasuredNotAssumed, comfortably inside this
    // suite's 1e-2 tolerance, so it belongs in the systematic sweep rather than getting
    // SoftmaxModule's dedicated-measurement-test treatment.
    cases.push_back({"MambaModule", [] {
                          CPUBackend backend;
                          MambaModule mamba(2, 2, &backend);
                          mamba.set_W_delta({0.37f, -0.62f, 0.18f, 0.45f});
                          mamba.set_bias_delta({-0.21f, 0.33f});
                          mamba.set_W_B({0.54f, -0.28f, 0.41f, 0.66f});
                          mamba.set_W_C({-0.35f, 0.72f, 0.59f, -0.16f});
                          mamba.set_A({-0.85f, -0.30f, -0.55f, -1.20f});
                          mamba.set_D({0.62f, -0.41f});
                          Tensor input(Shape({1, 3, 2}), &backend, {0.30f, -0.20f, 0.60f, 0.10f, -0.40f, 0.50f});
                          (void)mamba.forward(input);
                          Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = mamba.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // RetNetModule's original derived LRP rule (2026-09-27 follow-on to Decision Point 2 --
    // see retnet_module.hpp's class-level note): the retention recurrence unrolls into a
    // causal, gamma-decay-gated weighted sum Y = G @ V with G[t,s] = gamma^(t-s)*(Q_t.K_s),
    // structurally identical to MultiHeadAttentionModule's own Q@K^T -> Attn@V shape but
    // with NO softmax in the middle -- so unlike MultiHeadAttentionModule, every composed
    // step (two AttnLRP Eq. 15 bilinear splits, one exact gamma-constant pass-through, three
    // no-bias linear projections' own z-rule) conserves exactly or near-exactly, gated only
    // by the usual epsilon stabilizers. It belongs in this systematic sweep, not in
    // SoftmaxModule/MultiHeadAttentionModule's dedicated-measurement-test treatment.
    cases.push_back({"RetNetModule", [] {
                          CPUBackend backend;
                          RetNetModule retnet(2, 3, 0.7f, &backend);
                          retnet.set_W_Q({0.37f, -0.62f, 0.18f, 0.45f, 0.83f, -0.26f});
                          retnet.set_W_K({0.54f, -0.28f, 0.41f, 0.66f, -0.73f, 0.19f});
                          retnet.set_W_V({-0.35f, 0.72f, 0.59f, -0.16f});
                          Tensor input(Shape({1, 4, 2}), &backend,
                                       {0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f, 0.40f, -1.30f});
                          (void)retnet.forward(input);
                          Tensor relevance_out(Shape({1, 4, 2}), &backend,
                                               {1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f, 0.70f, -0.40f});
                          LRPRuleConfig config;
                          Tensor relevance_in = retnet.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // RWKVModule's original derived LRP rule (2026-09-27 follow-on reassessment -- see
    // rwkv_module.hpp's class-level note): the WKV quotient wkv_t = num_t/den_t unrolls
    // into a two-term weighted sum of the carried state A[t] and v_t (weights 1/den_t and
    // e_t/den_t, summing to exactly 1), structurally MambaModule's own Abar/Bbar shape, not
    // softmax's cross-normalizing one -- so the SAME MambaLRP detach-the-gate technique
    // applies (detaching r_t, e_t, kk_t, decay as constants) and the rule conserves
    // near-exactly, unlike SoftmaxModule/MultiHeadAttentionModule.
    cases.push_back({"RWKVModule", [] {
                          CPUBackend backend;
                          RWKVModule rwkv(2, &backend);
                          rwkv.set_W_r({0.37f, -0.62f, 0.18f, 0.45f});
                          rwkv.set_W_k({0.54f, -0.28f, 0.41f, 0.66f});
                          rwkv.set_W_v({-0.35f, 0.72f, 0.59f, -0.16f});
                          rwkv.set_W_o({0.62f, -0.41f, 0.25f, 0.88f});
                          rwkv.set_w({0.30f, 0.75f});
                          rwkv.set_u({-0.20f, 0.45f});
                          rwkv.set_mu_r({0.65f, 0.35f});
                          rwkv.set_mu_k({0.40f, 0.80f});
                          rwkv.set_mu_v({0.55f, 0.25f});
                          Tensor input(Shape({1, 4, 2}), &backend,
                                       {0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f, 0.40f, -1.30f});
                          (void)rwkv.forward(input);
                          Tensor relevance_out(Shape({1, 4, 2}), &backend,
                                               {1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f, 0.70f, -0.40f});
                          LRPRuleConfig config;
                          Tensor relevance_in = rwkv.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"AggregatorModule", [] {
                          CPUBackend backend;
                          AggregatorModule agg(&backend, 2.0f);
                          Tensor input(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
                          Tensor output = agg.forward(input);
                          Tensor relevance_out(Shape({}), &backend, {output.data()[0]});
                          LRPRuleConfig config;
                          Tensor relevance_in = agg.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = relevance_out.data()[0];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // BatchNormModule: identity LRP pass-through (Montavon et al. 2019 -- normalization
    // layers pass relevance through unchanged), same fixture as
    // BatchNormModuleTest.PropagateRelevanceConservesTotalRelevance.
    cases.push_back({"BatchNormModule", [] {
                          CPUBackend backend;
                          BatchNormModule norm(2, &backend);
                          Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
                          (void)norm.forward(x);
                          Tensor relevance_out(Shape({2, 2, 1, 1}), &backend, {4.0f, -1.0f, 2.5f, 0.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = norm.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // LayerNormModule: identity LRP pass-through (AttnLRP, Achtibat et al. 2024), same
    // fixture as LayerNormModuleTest.PropagateRelevanceConservesTotalRelevance.
    cases.push_back({"LayerNormModule", [] {
                          CPUBackend backend;
                          LayerNormModule norm(3, &backend);
                          Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
                          (void)norm.forward(x);
                          Tensor relevance_out(Shape({1, 3}), &backend, {4.0f, -1.0f, 2.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = norm.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // GroupNormModule: identity LRP pass-through, same fixture as
    // GroupNormModuleTest.PropagateRelevanceConservesTotalRelevance.
    cases.push_back({"GroupNormModule", [] {
                          CPUBackend backend;
                          GroupNormModule norm(2, 4, &backend);
                          Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
                          (void)norm.forward(x);
                          Tensor relevance_out(Shape({1, 4, 1, 2}), &backend,
                                                {4.0f, -1.0f, 2.5f, 0.0f, 1.0f, 1.0f, -2.0f, 3.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = norm.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // RMSNormModule: identity LRP pass-through (AttnLRP, Achtibat et al. 2024), same
    // fixture as RMSNormModuleTest.PropagateRelevanceConservesTotalRelevance.
    cases.push_back({"RMSNormModule", [] {
                          CPUBackend backend;
                          RMSNormModule norm(3, &backend);
                          Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
                          (void)norm.forward(x);
                          Tensor relevance_out(Shape({1, 3}), &backend, {4.0f, -1.0f, 2.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = norm.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // DropoutModule: propagate_relevance is an UNCONDITIONAL identity pass-through --
    // per dropout_module.hpp's class note, it does not depend on the training-mode mask at
    // all (unlike backward(), which does). set_training(false) is applied anyway so this
    // shared sweep's forward() call is itself deterministic (no RNG draw), matching this
    // mission's determinism requirement rather than relying on the rule's own
    // mask-independence to paper over a nondeterministic forward. Same fixture shape as
    // DropoutModuleTest.PropagateRelevanceIsUnconditionalIdentity.
    cases.push_back({"DropoutModule", [] {
                          CPUBackend backend;
                          DropoutModule d(0.5f, &backend, /*seed=*/7);
                          d.set_training(false);
                          const int64_t n = 50;
                          std::vector<float> values(static_cast<size_t>(n), 1.0f);
                          Tensor input(Shape({n}), &backend, values);
                          (void)d.forward(input);
                          std::vector<float> relevance_values(static_cast<size_t>(n));
                          for (int64_t i = 0; i < n; ++i) relevance_values[static_cast<size_t>(i)] = static_cast<float>(i) + 1.0f;
                          Tensor relevance_out(Shape({n}), &backend, relevance_values);
                          LRPRuleConfig config;
                          Tensor relevance_in = d.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // ResidualModule: y = x + inner->forward(x), a real ResNet-style bottleneck block
    // (Conv2D(1x1) -> ReLU -> Conv2D(1x1), in==out channels). Reuses
    // ResidualModuleTest.PropagateRelevanceConservesNearExactlyAcrossRealResidualBlock's
    // exact bias values -- chosen there specifically to keep every position's
    // pre-activation positive in every channel, avoiding the "dead ReLU row" edge case
    // (a fully-clipped ReLU row traps incoming relevance and collapses the following
    // epsilon-rule denominator to a bare epsilon, a real property of the rule family, not
    // a bug -- but not what this near-exact-conservation case is meant to measure).
    cases.push_back({"ResidualModule", [] {
                          CPUBackend backend;
                          Conv2DModule conv1(2, 2, 1, 1, &backend);
                          conv1.set_kernel({0.4f, -0.2f, 0.3f, 0.5f});
                          conv1.set_bias({0.5f, 0.7f});
                          ReluModule relu(&backend);
                          Conv2DModule conv2(2, 2, 1, 1, &backend);
                          conv2.set_kernel({0.6f, -0.1f, 0.2f, 0.4f});
                          conv2.set_bias({-0.05f, 0.1f});
                          SequentialModule inner({&conv1, &relu, &conv2});
                          ResidualModule residual(&inner, &backend);

                          Tensor x(Shape({1, 2, 2, 2}), &backend,
                                   {0.6f, -0.9f, 1.1f, -0.7f, 0.4f, -0.7f, 0.8f, 0.6f});
                          (void)residual.forward(x);
                          Tensor relevance_out(Shape({1, 2, 2, 2}), &backend,
                                                {1.0f, -0.5f, 1.0f, -0.5f, 1.0f, -0.5f, 1.0f, -0.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = residual.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // SwiGLUModule: gate_proj/up_proj/down_proj composed with SiLU's exact identity
    // pass-through and the diagonal Eq. 15 split, conserving near-exactly. Reuses
    // SwiGLUModuleTest.PropagateRelevanceConservesNearExactly's exact input
    // (0.65f/-0.85f/1.15f, not the "nicer" 0.7f/-0.9f/1.3f) -- that test's own close-out
    // found the nicer values drive one of up_proj's pre-bias outputs to exactly 0.0,
    // which blows up the epsilon-rule denominator; these values keep every pre-activation
    // safely away from zero.
    cases.push_back({"SwiGLUModule", [] {
                          CPUBackend backend;
                          SwiGLUModule m(3, 4, &backend);
                          m.gate_proj().set_weight(
                              {0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f, 0.4f});
                          m.gate_proj().set_bias({0.1f, -0.1f, 0.05f, 0.2f});
                          m.up_proj().set_weight(
                              {-0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f, 0.1f, 0.2f});
                          m.up_proj().set_bias({0.05f, 0.1f, -0.05f, 0.15f});
                          m.down_proj().set_weight(
                              {0.6f, -0.2f, 0.3f, 0.1f, -0.5f, 0.4f, 0.2f, -0.3f, 0.1f, 0.3f, -0.4f, 0.2f});
                          m.down_proj().set_bias({-0.1f, 0.2f, 0.05f});

                          Tensor x(Shape({1, 3}), &backend, {0.65f, -0.85f, 1.15f});
                          (void)m.forward(x);
                          Tensor relevance_out(Shape({1, 3}), &backend, {2.0f, -1.0f, 0.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = m.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // NegationModule: pass-through (y = 1 - x), same fixture as
    // NegationModuleTest.PropagateRelevancePassesThroughUnchanged.
    cases.push_back({"NegationModule", [] {
                          CPUBackend backend;
                          NegationModule negation(&backend);
                          Tensor x(Shape({3}), &backend, {0.1f, 0.4f, 0.9f});
                          (void)negation.forward(x);
                          Tensor relevance_out(Shape({3}), &backend, {0.3f, -1.0f, 2.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = negation.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // ConjunctionModule (Product t-norm): bilinear split (AttnLRP Eq. 15), conserving
    // near-exactly. Two-operand forward(a, b); relevance_in is the stacked (2, N) tensor,
    // so its total sum (both operands' shares) is compared against sum(y) -- the canonical
    // LRP setup, relevance_out == y itself. Same fixture as
    // ConjunctionModuleTest.ProductPropagateRelevanceConservesNearExactly.
    cases.push_back({"ConjunctionModule", [] {
                          CPUBackend backend;
                          ConjunctionModule conj(&backend);
                          Tensor a(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
                          Tensor b(Shape({4}), &backend, {0.6f, 0.2f, 0.5f, 0.1f});
                          Tensor y = conj.forward(a, b);
                          LRPRuleConfig config;
                          Tensor relevance_in = conj.propagate_relevance(y, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < y.numel(); ++i) sum_out += y.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // DisjunctionModule (Product t-conorm): averaged dual-decomposition rule, conserving
    // near-exactly. Same fixture as
    // DisjunctionModuleTest.ProductPropagateRelevanceConservesNearExactly.
    cases.push_back({"DisjunctionModule", [] {
                          CPUBackend backend;
                          DisjunctionModule disj(&backend);
                          Tensor a(Shape({4}), &backend, {0.3f, 0.7f, 0.5f, 0.9f});
                          Tensor b(Shape({4}), &backend, {0.6f, 0.2f, 0.5f, 0.1f});
                          Tensor y = disj.forward(a, b);
                          LRPRuleConfig config;
                          Tensor relevance_in = disj.propagate_relevance(y, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < y.numel(); ++i) sum_out += y.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    // Deliberately NOT here: SoftmaxModule and MultiHeadAttentionModule.
    //
    // SoftmaxModule's rule is AttnLRP Eq. 13, a first-order DTD approximation with a known
    // residual "hidden bias term" -- it does not conserve by construction (see
    // softmax_module.hpp). MultiHeadAttentionModule composes that same softmax step into the
    // middle of its pipeline, so it inherits the gap: measured at 2.3546 against a
    // sum(R_out) of 4.75 (~50% of the output relevance) in
    // multihead_attention_module_test.cpp's PropagateRelevanceConservationGapIsMeasuredNot-
    // AssumedZero, which reports the number and decomposes it stage by stage. Every *other*
    // stage of that module conserves -- including AttnLRP Eq. 15, whose factor-2 denominator
    // makes each matmul's two operand shares sum back to R_O exactly.
    //
    // Adding either here would force this suite's 1e-2 tolerance up by two-plus orders of
    // magnitude for every one of the thirteen modules above that genuinely meets it, turning a
    // real invariant into a rubber stamp. The per-module measurement tests are the right home
    // for a rule that is known not to conserve; this suite stays the home for rules that do.
    return cases;
}

INSTANTIATE_TEST_SUITE_P(AllModuleTypes, LRPConservationTest, ::testing::ValuesIn(AllModuleTypeCases()),
                          [](const ::testing::TestParamInfo<ConservationCase>& info) { return info.param.name; });

// The mission's actual new deliverable: no end-to-end conservation test existed anywhere
// in this codebase before this mission (grep-confirmed at campaign activation). Chains
// propagate_relevance backward through a real multi-module network -- LinearModule ->
// ReluModule -> LinearModule, the same network shape mission_module_graph_wiring.md and
// mission_explainer_context.md already used -- mirroring Module::forward_traced's
// chaining shape (Phase 2 Mission 0) but for relevance instead of activations.
TEST(LRPConservationEndToEndTest, ConservationHoldsAcrossFullNetworkChain) {
    CPUBackend backend;

    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor h1 = linear1.forward(input);
    Tensor h2 = relu.forward(h1);
    (void)linear2.forward(h2);

    Tensor relevance_seed(Shape({1, 2}), &backend, {4.0f, 6.0f});
    LRPRuleConfig config;

    Tensor relevance_h2 = linear2.propagate_relevance(relevance_seed, config);
    Tensor relevance_h1 = relu.propagate_relevance(relevance_h2, config);
    Tensor relevance_input = linear1.propagate_relevance(relevance_h1, config);

    float sum_seed = relevance_seed.data()[0] + relevance_seed.data()[1];
    float sum_input = 0.0f;
    for (int64_t i = 0; i < relevance_input.numel(); ++i) {
        sum_input += relevance_input.data()[i];
    }

    EXPECT_NEAR(sum_input, sum_seed, 1e-2f)
        << "end-to-end LRP conservation violated across the full network chain";
}

}  // namespace
}  // namespace pulsatrix
