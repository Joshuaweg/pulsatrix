#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>

#include "exai/avg_pool2d_module.hpp"
#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/embedding_module.hpp"
#include "exai/flatten_module.hpp"
#include "exai/gru_module.hpp"
#include "exai/linear_module.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/lstm_module.hpp"
#include "exai/mamba_module.hpp"
#include "exai/max_pool2d_module.hpp"
#include "exai/relu_module.hpp"
#include "exai/rnn_module.hpp"
#include "exai/rope_module.hpp"
#include "exai/sequential_module.hpp"

// Phase 4 Mission 0: charter's "not just a spot-check on one architecture" LRP
// completeness requirement, turned into a systematic TEST_P suite over every module type
// this codebase has, plus a genuinely new end-to-end conservation test. Reuses each
// module's existing, already-proven-correct weight/input/relevance setups from Phase
// 1-3's own per-module tests (LinearModuleTest, Conv2DModuleTest, ReluModuleTest,
// FlattenModuleTest) -- this mission doesn't re-derive their math, it systematizes it.
namespace exai {
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
}  // namespace exai
