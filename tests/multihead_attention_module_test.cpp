#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/multihead_attention_module.hpp"

namespace exai {
namespace {

// The shared "general case" configuration used by the reference-value, finite-difference,
// conservation and toggle tests below: d_model=4, num_heads=2 (so head_dim=2), L=2, N=1.
// Small enough that every intermediate is inspectable, large enough that all four heads'
// worth of index arithmetic (2 heads x 2 positions x 2 features) is genuinely exercised.
const std::vector<float> kWq{0.10f, -0.20f, 0.30f, 0.40f, 0.50f, 0.60f,  -0.70f, 0.80f,
                             -0.90f, 1.00f, 0.11f, -0.12f, 0.13f, -0.14f, 0.15f, 0.16f};
const std::vector<float> kBq{0.01f, -0.02f, 0.03f, -0.04f};
const std::vector<float> kWk{0.20f, 0.10f, -0.30f, 0.15f, -0.25f, 0.35f, 0.05f, -0.45f,
                             0.55f, -0.65f, 0.75f, 0.85f, -0.05f, 0.95f, -0.15f, 0.25f};
const std::vector<float> kBk{-0.01f, 0.02f, -0.03f, 0.04f};
const std::vector<float> kWv{0.30f, -0.40f, 0.50f, 0.60f, 0.70f, 0.80f,  -0.90f, 0.21f,
                             -0.31f, 0.41f, -0.51f, 0.61f, 0.71f, -0.81f, 0.91f, -0.22f};
const std::vector<float> kBv{0.05f, -0.05f, 0.10f, -0.10f};
const std::vector<float> kWo{0.12f, 0.22f, -0.32f, 0.42f, -0.52f, 0.62f, 0.72f, -0.82f,
                             0.92f, -0.13f, 0.23f, 0.33f, -0.43f, 0.53f, -0.63f, 0.73f};
const std::vector<float> kBo{0.02f, 0.04f, -0.06f, 0.08f};
const std::vector<float> kX{0.5f, -1.2f, 2.0f, 0.1f, 0.7f, -0.3f, -0.9f, 1.4f};

void ConfigureGeneralCase(MultiHeadAttentionModule& mha) {
    mha.q_proj().set_weight(kWq);
    mha.q_proj().set_bias(kBq);
    mha.k_proj().set_weight(kWk);
    mha.k_proj().set_bias(kBk);
    mha.v_proj().set_weight(kWv);
    mha.v_proj().set_bias(kBv);
    mha.out_proj().set_weight(kWo);
    mha.out_proj().set_bias(kBo);
}

class MultiHeadAttentionModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor GeneralInput() { return Tensor(Shape({1, 2, 4}), &backend, kX); }
};

// ---------------------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------------------

// Fully hand-traceable, no reference implementation involved. The construction pins every
// step of the 9-step pipeline to arithmetic that can be done on paper:
//
//   d_model=4, num_heads=2 (head_dim=2), L=2, N=1, no RoPE, no QK-Norm.
//   x            = [[1, 2, 3, 4], [5, 6, 7, 8]]
//   q_proj       = identity, zero bias        -> Q = x
//   k_proj       = ZERO weight, bias all ones -> K = all-ones, every row identical
//   v_proj       = identity, zero bias        -> V = x
//   out_proj     = identity, bias [0.5, -0.5, 1, -1]
//
//   Because every row of K is identical, every entry of a given row of Q@K^T is the same
//   number (Q_i . (1,1)), so the 1/sqrt(2) scale is irrelevant and softmax returns exactly
//   [0.5, 0.5] for all four rows -- attention degenerates to "average the values".
//
//   head 0 sees features 0-1:  V0 = [[1,2],[5,6]]  -> context rows = [3, 4]
//   head 1 sees features 2-3:  V1 = [[3,4],[7,8]]  -> context rows = [5, 6]
//   merged context (both positions) = [3, 4, 5, 6]
//   out = merged + [0.5, -0.5, 1, -1]              = [3.5, 3.5, 6, 5]
//
// A head-split/merge bug (e.g. treating the permutation as a plain reshape) would pair
// feature 0 with feature 2 and produce [4, 5, 4, 5]-shaped context instead, so this example
// is not just an arithmetic check -- it pins the permutation too.
TEST_F(MultiHeadAttentionModuleTest, ForwardMatchesHandTracedUniformAttentionExample) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    mha.q_proj().set_weight({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    mha.q_proj().set_bias({0, 0, 0, 0});
    mha.k_proj().set_weight({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    mha.k_proj().set_bias({1, 1, 1, 1});
    mha.v_proj().set_weight({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    mha.v_proj().set_bias({0, 0, 0, 0});
    mha.out_proj().set_weight({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    mha.out_proj().set_bias({0.5f, -0.5f, 1.0f, -1.0f});

    Tensor x(Shape({1, 2, 4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    Tensor y = mha.forward(x);

    ASSERT_EQ(y.shape(), Shape({1, 2, 4}));
    const std::vector<float> expected{3.5f, 3.5f, 6.0f, 5.0f, 3.5f, 3.5f, 6.0f, 5.0f};
    for (int64_t i = 0; i < y.numel(); ++i) {
        EXPECT_NEAR(y.data()[i], expected[static_cast<size_t>(i)], 1e-5f) << "element " << i;
    }

    // Every attention weight is exactly 0.5 -- the premise the hand trace rests on.
    const Tensor& attn = mha.last_attention_weights();
    ASSERT_EQ(attn.shape(), Shape({1, 2, 2, 2}));
    for (int64_t i = 0; i < attn.numel(); ++i) {
        EXPECT_NEAR(attn.data()[i], 0.5f, 1e-6f) << "attention weight " << i;
    }
}

// The general case with RoPE on and nontrivial projections, checked against an independent
// float64 NumPy transcription of the same formulas (a separate implementation of the spec,
// not a re-run of this one). Values below are that reference's output.
TEST_F(MultiHeadAttentionModuleTest, ForwardMatchesIndependentReferenceWithRoPE) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);

    Tensor x = GeneralInput();
    Tensor y = mha.forward(x);

    const std::vector<float> expected{3.04503019f,  -1.25478873f,  -1.47368502f,  2.88494607f,
                                      0.539065447f, -0.177510492f, -0.402203872f, 0.907438209f};
    for (int64_t i = 0; i < y.numel(); ++i) {
        EXPECT_NEAR(y.data()[i], expected[static_cast<size_t>(i)], 1e-5f) << "element " << i;
    }
}

// Same weights/input, RoPE off -- also an independent-reference value, and the counterpart
// the use_rope toggle test compares against.
TEST_F(MultiHeadAttentionModuleTest, ForwardMatchesIndependentReferenceWithoutRoPE) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);

    Tensor y = mha.forward(GeneralInput());

    const std::vector<float> expected{2.30812805f,  -0.913773196f, -1.91607766f,  3.06154651f,
                                      0.551352418f, -0.184035184f, -0.368611522f, 0.87825571f};
    for (int64_t i = 0; i < y.numel(); ++i) {
        EXPECT_NEAR(y.data()[i], expected[static_cast<size_t>(i)], 1e-5f) << "element " << i;
    }
}

// Every (n, h, i) row of the attention matrix is a probability distribution over the L key
// positions -- the softmax axis is the last one, not some other axis.
TEST_F(MultiHeadAttentionModuleTest, AttentionWeightRowsEachSumToOne) {
    MultiHeadAttentionModule mha(6, 3, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    std::vector<float> w(36);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.07f * static_cast<float>(i % 11) - 0.31f;
    }
    mha.q_proj().set_weight(w);
    mha.k_proj().set_weight(w);
    mha.v_proj().set_weight(w);
    mha.out_proj().set_weight(w);

    std::vector<float> xs(2 * 4 * 6);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = 0.13f * static_cast<float>(i) - 1.7f;
    }
    Tensor x(Shape({2, 4, 6}), &backend, xs);
    Tensor y = mha.forward(x);
    ASSERT_EQ(y.shape(), Shape({2, 4, 6}));

    const Tensor& attn = mha.last_attention_weights();
    ASSERT_EQ(attn.shape(), Shape({2, 3, 4, 4}));
    for (int64_t row = 0; row < attn.numel() / 4; ++row) {
        float sum = 0.0f;
        for (int64_t j = 0; j < 4; ++j) {
            sum += attn.data()[row * 4 + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "attention row " << row;
    }
}

TEST_F(MultiHeadAttentionModuleTest, ForwardPreservesShapeForMultiExampleBatch) {
    MultiHeadAttentionModule mha(8, 4, &backend);
    Tensor x(Shape({3, 5, 8}), &backend);
    x.fill(0.25f);
    Tensor y = mha.forward(x);
    EXPECT_EQ(y.shape(), Shape({3, 5, 8}));
}

// ---------------------------------------------------------------------------------------
// Backward -- central finite differences
// ---------------------------------------------------------------------------------------

namespace {
// f(params, x) = sum_k grad_out[k] * mha(x)[k]; its gradient w.r.t. anything is exactly what
// backward(grad_out) computes.
float ScalarObjective(MultiHeadAttentionModule& mha, const Tensor& x, const std::vector<float>& grad_out_values) {
    Tensor y = mha.forward(x);
    float total = 0.0f;
    for (int64_t i = 0; i < y.numel(); ++i) {
        total += grad_out_values[static_cast<size_t>(i)] * y.data()[i];
    }
    return total;
}
}  // namespace

// Every path back to the input: out_proj -> merge -> (Attn@V matmul) -> softmax ->
// (Q@K^T matmul) -> RoPE -> QK-Norm -> three projections, with the three contributions
// summing into one input gradient. RoPE and QK-Norm are both on so no branch is skipped.
TEST_F(MultiHeadAttentionModuleTest, BackwardMatchesCentralFiniteDifferencesOverInput) {
    const std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f, 0.75f, -0.6f};

    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/true);
    ConfigureGeneralCase(mha);

    Tensor x = GeneralInput();
    (void)mha.forward(x);
    Tensor grad_out(Shape({1, 2, 4}), &backend, grad_out_values);
    Tensor grad_in = mha.backward(grad_out);
    ASSERT_EQ(grad_in.shape(), Shape({1, 2, 4}));

    const float h = 1e-3f;
    for (size_t i = 0; i < kX.size(); ++i) {
        std::vector<float> plus = kX;
        std::vector<float> minus = kX;
        plus[i] += h;
        minus[i] -= h;

        MultiHeadAttentionModule probe(4, 2, &backend, true, true);
        ConfigureGeneralCase(probe);
        Tensor xp(Shape({1, 2, 4}), &backend, plus);
        Tensor xm(Shape({1, 2, 4}), &backend, minus);
        const float fp = ScalarObjective(probe, xp, grad_out_values);
        const float fm = ScalarObjective(probe, xm, grad_out_values);
        const float numeric = (fp - fm) / (2.0f * h);

        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 5e-3f) << "input element " << i;
    }
}

// Every learnable parameter of every sub-module at once (4 x (16 weights + 4 biases) for the
// projections, plus the two head_dim-sized QK-Norm gammas = 84 parameters), driven through
// the aggregated parameters() view -- which also proves parameters() actually reaches every
// sub-module's storage rather than returning copies.
TEST_F(MultiHeadAttentionModuleTest, BackwardMatchesCentralFiniteDifferencesOverAllSubModuleParameters) {
    const std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.5f, 0.75f, -0.6f};

    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/true);
    ConfigureGeneralCase(mha);

    Tensor x = GeneralInput();
    (void)mha.forward(x);
    Tensor grad_out(Shape({1, 2, 4}), &backend, grad_out_values);
    (void)mha.backward(grad_out);

    std::vector<ParamRef> params = mha.parameters();
    ASSERT_EQ(params.size(), 10u);  // 4 projections x {weight, bias} + q_norm gamma + k_norm gamma

    // Snapshot the analytic gradients before the finite-difference forwards run (they don't
    // touch gradient storage, but snapshotting keeps the comparison unambiguous).
    std::vector<std::vector<float>> analytic;
    int64_t total_params = 0;
    for (const ParamRef& p : params) {
        std::vector<float> g(static_cast<size_t>(p.grad->numel()));
        for (int64_t i = 0; i < p.grad->numel(); ++i) {
            g[static_cast<size_t>(i)] = p.grad->data()[i];
        }
        total_params += p.value->numel();
        analytic.push_back(std::move(g));
    }
    EXPECT_EQ(total_params, 84);

    const float h = 1e-3f;
    for (size_t pi = 0; pi < params.size(); ++pi) {
        for (int64_t i = 0; i < params[pi].value->numel(); ++i) {
            const float original = params[pi].value->data()[i];

            params[pi].value->data()[i] = original + h;
            const float fp = ScalarObjective(mha, x, grad_out_values);
            params[pi].value->data()[i] = original - h;
            const float fm = ScalarObjective(mha, x, grad_out_values);
            params[pi].value->data()[i] = original;

            const float numeric = (fp - fm) / (2.0f * h);
            EXPECT_NEAR(analytic[pi][static_cast<size_t>(i)], numeric, 5e-3f)
                << "parameter tensor " << pi << ", element " << i;
        }
    }
}

// The no-RoPE / no-QK-Norm branch has its own backward path (two `if`s skipped); finite
// differences must hold there too, not just in the all-features configuration.
TEST_F(MultiHeadAttentionModuleTest, BackwardMatchesCentralFiniteDifferencesWithOptionalPathsDisabled) {
    const std::vector<float> grad_out_values{0.4f, 1.1f, -0.7f, 2.2f, -0.3f, 0.9f, 1.6f, -1.2f};

    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);

    Tensor x = GeneralInput();
    (void)mha.forward(x);
    Tensor grad_out(Shape({1, 2, 4}), &backend, grad_out_values);
    Tensor grad_in = mha.backward(grad_out);

    const float h = 1e-3f;
    for (size_t i = 0; i < kX.size(); ++i) {
        std::vector<float> plus = kX;
        std::vector<float> minus = kX;
        plus[i] += h;
        minus[i] -= h;

        MultiHeadAttentionModule probe(4, 2, &backend, false, false);
        ConfigureGeneralCase(probe);
        Tensor xp(Shape({1, 2, 4}), &backend, plus);
        Tensor xm(Shape({1, 2, 4}), &backend, minus);
        const float numeric =
            (ScalarObjective(probe, xp, grad_out_values) - ScalarObjective(probe, xm, grad_out_values)) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 5e-3f) << "input element " << i;
    }
}

// ---------------------------------------------------------------------------------------
// LRP
// ---------------------------------------------------------------------------------------

TEST_F(MultiHeadAttentionModuleTest, PropagateRelevanceReturnsInputShapedFiniteRelevance) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);
    (void)mha.forward(GeneralInput());

    Tensor relevance_out(Shape({1, 2, 4}), &backend, {1.0f, 0.5f, -0.25f, 2.0f, 0.75f, 0.0f, 1.25f, -0.5f});
    Tensor relevance_in = mha.propagate_relevance(relevance_out, LRPRuleConfig{});

    ASSERT_EQ(relevance_in.shape(), Shape({1, 2, 4}));
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(relevance_in.data()[i])) << "element " << i;
    }
}

// Element-wise check of the whole composed rule chain (out_proj epsilon rule -> merge
// permutation -> Eq. 15 on Attn@V -> Eq. 13 softmax -> Eq. 15 on Q@K^T -> RoPE epsilon rule
// -> split permutation -> three projections' epsilon rules, summed) against an independent
// float64 NumPy transcription of those same formulas. A sum-only conservation check cannot
// catch a transposed R_K or a swapped R_A/R_B in Eq. 15 -- both of those preserve the total
// while scrambling the attribution. This can.
TEST_F(MultiHeadAttentionModuleTest, PropagateRelevanceMatchesIndependentReferenceElementwise) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);
    (void)mha.forward(GeneralInput());

    Tensor relevance_out(Shape({1, 2, 4}), &backend, {1.0f, 0.5f, -0.25f, 2.0f, 0.75f, 0.0f, 1.25f, -0.5f});
    Tensor relevance_in = mha.propagate_relevance(relevance_out, LRPRuleConfig{});

    const std::vector<float> expected{0.572878888f, 1.357619978f, -1.276327973f, 0.177238465f,
                                      0.335377065f, 0.160526884f, 0.178630640f,  0.889485984f};
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        EXPECT_NEAR(relevance_in.data()[i], expected[static_cast<size_t>(i)], 1e-4f) << "element " << i;
    }
}

// Mission requirement: measure the *composed* block's end-to-end conservation gap and report
// it as a concrete number rather than assuming it from the parts.
//
// This module is deliberately NOT in tests/lrp_conservation_test.cpp's AllModuleTypeCases()
// -- for exactly the reason SoftmaxModule is not. The pipeline routes relevance through
// SoftmaxModule's AttnLRP Eq. 13, a first-order DTD approximation with a known residual
// "hidden bias term" (softmax_module_test.cpp measures a gap of ~6.84 on a sum(R_out)=4
// seed, i.e. ~1.7x of output, in isolation). Forcing this case into the shared suite would
// require relaxing a 1e-2 tolerance that eleven genuinely-conserving modules currently meet.
//
// Every *other* stage here does conserve: LinearModule's epsilon rule conserves up to eps,
// RoPEModule's does too, RMSNormModule's identity rule is exact, the head split/merge is a
// pure permutation, and AttnLRP Eq. 15's factor-2 denominator is what makes each matmul's
// two operand shares sum back to R_O exactly (sum_j A[i,j]B[j,k] == O[i,k], so each operand
// receives O/(2O) = 1/2 of R_O and the pair receives all of it). So the composed gap is the
// softmax stage's gap, carried through the stages that surround it -- amplified or damped by
// them, not created by them.
TEST_F(MultiHeadAttentionModuleTest, PropagateRelevanceConservationGapIsMeasuredNotAssumedZero) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    ConfigureGeneralCase(mha);
    (void)mha.forward(GeneralInput());

    Tensor relevance_out(Shape({1, 2, 4}), &backend, {1.0f, 0.5f, -0.25f, 2.0f, 0.75f, 0.0f, 1.25f, -0.5f});
    Tensor relevance_in = mha.propagate_relevance(relevance_out, LRPRuleConfig{});

    double sum_out = 0.0;
    double sum_in = 0.0;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) {
        sum_out += relevance_out.data()[i];
        sum_in += relevance_in.data()[i];
    }
    const double gap = sum_out - sum_in;

    std::cout << "[ MHA      ] composed AttnLRP conservation gap on (N=1, L=2, d_model=4, heads=2, RoPE on): "
              << "sum(R_out)=" << sum_out << ", sum(R_in)=" << sum_in << ", gap=" << gap
              << ", gap/sum(R_out)=" << (gap / sum_out) << std::endl;
    RecordProperty("lrp_conservation_gap", std::to_string(gap));

    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(relevance_in.data()[i])) << "element " << i;
    }
    // Characterization of the measured values at the time of writing, cross-checked against
    // an independent float64 NumPy transcription of the same composed rule chain:
    //   sum(R_out) = 4.75
    //   sum(R_in)  = 2.395430   (NumPy float64: 2.3954299308)
    //   gap        = 2.354570   (NumPy float64: 2.3545700692), i.e. 49.6% of sum(R_out)
    // That same reference also decomposes the gap stage by stage and confirms the hypothesis
    // above exactly:
    //   sum(R_attn) = sum(R_v) = 2.3749959  -- Eq. 15 splits Attn@V's relevance into two
    //                                          *equal* halves summing to 4.7499918 ~= 4.75,
    //                                          which is the factor-2 denominator doing its job;
    //   sum(R_scores) = 0.0204369            -- after SoftmaxModule's Eq. 13;
    //   sum(R_attn) - sum(R_scores) = 2.3545590.
    // The composed block's gap and the softmax stage's own gap agree to 5 decimals: the
    // residual 1.1e-5 is the epsilon stabilizers in the surrounding (otherwise exactly
    // conserving) rules. The composed module does not amplify or dampen SoftmaxModule's
    // non-conservation -- it transports it unchanged, on whatever relevance mass Eq. 15
    // routes into the softmax branch (here half the total, hence gap ~= 0.5 * sum(R_out)
    // for this configuration rather than SoftmaxModule's isolated ~1.7x).
    EXPECT_NEAR(sum_out, 4.75, 1e-6);
    EXPECT_NEAR(sum_in, 2.39543, 2e-4);
    EXPECT_NEAR(gap, 2.35457, 2e-4);
    // Positive assertion that the composed rule genuinely does NOT conserve, so that a
    // future change which accidentally "fixes" it is caught rather than silently accepted
    // (same guard SoftmaxModuleTest uses).
    EXPECT_GT(std::abs(gap), 1e-2) << "the composed chain routes relevance through AttnLRP Eq. 13, which does "
                                      "not conserve; a ~0 gap means a composed rule was changed";
}

// ---------------------------------------------------------------------------------------
// Optional-path toggles
// ---------------------------------------------------------------------------------------

// use_rope changes behavior on its own: same weights, same input, same use_qk_norm.
TEST_F(MultiHeadAttentionModuleTest, UseRopeToggleIndependentlyChangesForwardOutput) {
    MultiHeadAttentionModule with_rope(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    MultiHeadAttentionModule without_rope(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    ConfigureGeneralCase(with_rope);
    ConfigureGeneralCase(without_rope);

    EXPECT_TRUE(with_rope.uses_rope());
    EXPECT_FALSE(without_rope.uses_rope());

    Tensor a = with_rope.forward(GeneralInput());
    Tensor b = without_rope.forward(GeneralInput());

    float max_abs_diff = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(a.data()[i] - b.data()[i]));
    }
    EXPECT_GT(max_abs_diff, 1e-3f) << "use_rope=true produced the same output as use_rope=false -- "
                                      "the RoPE branch is not engaging";

    // Not just "different" -- different in the specific way the independent reference says.
    EXPECT_NEAR(a.data()[0], 3.04503019f, 1e-5f);
    EXPECT_NEAR(b.data()[0], 2.30812805f, 1e-5f);
}

// use_qk_norm changes behavior on its own, with use_rope held fixed in BOTH states, so the
// two toggles are verified genuinely independently rather than only in combination.
TEST_F(MultiHeadAttentionModuleTest, UseQkNormToggleIndependentlyChangesForwardOutput) {
    MultiHeadAttentionModule rope_only(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    MultiHeadAttentionModule rope_and_norm(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/true);
    MultiHeadAttentionModule norm_only(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/true);
    MultiHeadAttentionModule neither(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    ConfigureGeneralCase(rope_only);
    ConfigureGeneralCase(rope_and_norm);
    ConfigureGeneralCase(norm_only);
    ConfigureGeneralCase(neither);

    EXPECT_FALSE(rope_only.uses_qk_norm());
    EXPECT_TRUE(rope_and_norm.uses_qk_norm());

    Tensor a = rope_only.forward(GeneralInput());
    Tensor b = rope_and_norm.forward(GeneralInput());
    Tensor c = neither.forward(GeneralInput());
    Tensor d = norm_only.forward(GeneralInput());

    float diff_with_rope = 0.0f;
    float diff_without_rope = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        diff_with_rope = std::max(diff_with_rope, std::abs(a.data()[i] - b.data()[i]));
        diff_without_rope = std::max(diff_without_rope, std::abs(c.data()[i] - d.data()[i]));
    }
    EXPECT_GT(diff_with_rope, 1e-3f) << "QK-Norm did not engage with use_rope=true";
    EXPECT_GT(diff_without_rope, 1e-3f) << "QK-Norm did not engage with use_rope=false";

    // Independent-reference values for the two QK-Norm-on configurations.
    EXPECT_NEAR(b.data()[0], 2.80526795f, 1e-5f);
    EXPECT_NEAR(d.data()[0], 1.95519068f, 1e-5f);
}

// QK-Norm's gamma is exposed and defaults to ones (not RMSNormModule's own zero default) --
// see the header's rationale. Q and K get independent instances.
TEST_F(MultiHeadAttentionModuleTest, QkNormGammaDefaultsToOnesAndIsPerSideIndependent) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/true);
    ASSERT_NE(mha.q_norm(), nullptr);
    ASSERT_NE(mha.k_norm(), nullptr);
    EXPECT_NE(mha.q_norm(), mha.k_norm());
    for (int64_t i = 0; i < mha.q_norm()->gamma().numel(); ++i) {
        EXPECT_FLOAT_EQ(mha.q_norm()->gamma().data()[i], 1.0f);
        EXPECT_FLOAT_EQ(mha.k_norm()->gamma().data()[i], 1.0f);
    }

    MultiHeadAttentionModule off(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    EXPECT_EQ(off.q_norm(), nullptr);
    EXPECT_EQ(off.k_norm(), nullptr);
}

// ---------------------------------------------------------------------------------------
// Constructor validation / metadata
// ---------------------------------------------------------------------------------------

TEST_F(MultiHeadAttentionModuleTest, ConstructorRejectsNonPositiveDModel) {
    EXPECT_THROW({ MultiHeadAttentionModule mha(0, 2, &backend); }, std::invalid_argument);
    EXPECT_THROW({ MultiHeadAttentionModule mha(-8, 2, &backend); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, ConstructorRejectsNonPositiveNumHeads) {
    EXPECT_THROW({ MultiHeadAttentionModule mha(8, 0, &backend); }, std::invalid_argument);
    EXPECT_THROW({ MultiHeadAttentionModule mha(8, -2, &backend); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, ConstructorRejectsIndivisibleDModel) {
    EXPECT_THROW({ MultiHeadAttentionModule mha(6, 4, &backend); }, std::invalid_argument);
}

// RoPE rotates adjacent feature pairs, so an odd head_dim has no valid pairing. Checked in
// this constructor (not left to RoPEModule's) so the message names d_model/num_heads.
TEST_F(MultiHeadAttentionModuleTest, ConstructorRejectsOddHeadDimWhenRopeEnabled) {
    EXPECT_THROW({ MultiHeadAttentionModule mha(6, 2, &backend, /*use_rope=*/true); }, std::invalid_argument);
    // ...and accepts exactly the same geometry with RoPE off, proving the rejection is
    // RoPE's constraint rather than a general one.
    EXPECT_NO_THROW({ MultiHeadAttentionModule mha(6, 2, &backend, /*use_rope=*/false); });
}

TEST_F(MultiHeadAttentionModuleTest, GeometryAccessorsReportConstructorArguments) {
    MultiHeadAttentionModule mha(12, 3, &backend);
    EXPECT_EQ(mha.d_model(), 12);
    EXPECT_EQ(mha.num_heads(), 3);
    EXPECT_EQ(mha.head_dim(), 4);
}

TEST_F(MultiHeadAttentionModuleTest, OpTypeIsAttention) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    EXPECT_EQ(mha.op_type(), OpType::Attention);
}

TEST_F(MultiHeadAttentionModuleTest, ParametersCoverEverySubModuleThatHasAny) {
    MultiHeadAttentionModule plain(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/false);
    EXPECT_EQ(plain.parameters().size(), 8u);  // 4 projections x {weight, bias}; RoPE/softmax have none

    MultiHeadAttentionModule normed(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/true);
    EXPECT_EQ(normed.parameters().size(), 10u);  // + q_norm gamma, k_norm gamma
}

TEST_F(MultiHeadAttentionModuleTest, SetTrainingCascadesToSubModules) {
    MultiHeadAttentionModule mha(4, 2, &backend, true, true);
    mha.set_training(false);
    EXPECT_FALSE(mha.is_training());
    EXPECT_FALSE(mha.q_proj().is_training());
    EXPECT_FALSE(mha.out_proj().is_training());
    EXPECT_FALSE(mha.q_norm()->is_training());
}

// ---------------------------------------------------------------------------------------
// Adversarial / sequencing boundaries
// ---------------------------------------------------------------------------------------

TEST_F(MultiHeadAttentionModuleTest, ForwardRejectsWrongRankInput) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor rank2(Shape({2, 4}), &backend);
    rank2.fill(1.0f);
    EXPECT_THROW({ (void)mha.forward(rank2); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, ForwardRejectsWrongFeatureWidth) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 5}), &backend);
    x.fill(1.0f);
    EXPECT_THROW({ (void)mha.forward(x); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, ForwardRejectsEmptyInput) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor empty(Shape({0, 2, 4}), &backend);
    EXPECT_THROW({ (void)mha.forward(empty); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, BackwardBeforeForwardThrows) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor grad_out(Shape({1, 2, 4}), &backend);
    grad_out.fill(1.0f);
    EXPECT_THROW({ (void)mha.backward(grad_out); }, std::logic_error);
}

TEST_F(MultiHeadAttentionModuleTest, PropagateRelevanceBeforeForwardThrows) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor relevance_out(Shape({1, 2, 4}), &backend);
    relevance_out.fill(1.0f);
    EXPECT_THROW({ (void)mha.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(MultiHeadAttentionModuleTest, BackwardRejectsShapeMismatchAgainstCachedForward) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 4}), &backend);
    x.fill(0.5f);
    (void)mha.forward(x);

    Tensor wrong_seq(Shape({1, 3, 4}), &backend);
    wrong_seq.fill(1.0f);
    EXPECT_THROW({ (void)mha.backward(wrong_seq); }, std::invalid_argument);

    Tensor wrong_rank(Shape({2, 4}), &backend);
    wrong_rank.fill(1.0f);
    EXPECT_THROW({ (void)mha.backward(wrong_rank); }, std::invalid_argument);
}

TEST_F(MultiHeadAttentionModuleTest, PropagateRelevanceRejectsShapeMismatchAgainstCachedForward) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 4}), &backend);
    x.fill(0.5f);
    (void)mha.forward(x);

    Tensor wrong_batch(Shape({2, 2, 4}), &backend);
    wrong_batch.fill(1.0f);
    EXPECT_THROW({ (void)mha.propagate_relevance(wrong_batch, LRPRuleConfig{}); }, std::invalid_argument);
}

using MultiHeadAttentionModuleDeathTest = MultiHeadAttentionModuleTest;

// Exactly three death tests -- forward/backward/propagate_relevance, the three entry points
// that dereference Tensor::data() in raw host loops. Same mislabeled-Tensor pattern as
// SoftmaxModuleDeathTest/RNNModuleDeathTest: DeviceType::Cuda over real CPUBackend memory
// trips the guard with no GPU involved.
TEST_F(MultiHeadAttentionModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 4}), &backend, kX, DeviceType::Cuda);
    EXPECT_DEATH({ (void)mha.forward(x); }, "EXAI_ASSERT failed");
}

TEST_F(MultiHeadAttentionModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 4}), &backend, kX);
    (void)mha.forward(x);

    Tensor grad_out(Shape({1, 2, 4}), &backend, kX, DeviceType::Cuda);
    EXPECT_DEATH({ (void)mha.backward(grad_out); }, "EXAI_ASSERT failed");
}

TEST_F(MultiHeadAttentionModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    MultiHeadAttentionModule mha(4, 2, &backend);
    Tensor x(Shape({1, 2, 4}), &backend, kX);
    (void)mha.forward(x);

    Tensor relevance_out(Shape({1, 2, 4}), &backend, kX, DeviceType::Cuda);
    EXPECT_DEATH({ (void)mha.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
