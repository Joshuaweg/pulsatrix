#include "exai/transformer_block.hpp"

#include <gtest/gtest.h>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace {

class TransformerBlockTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using TransformerBlockDeathTest = TransformerBlockTest;

// ---------------------------------------------------------------------------
// Constructor validation propagates from sub-modules (no redundant re-validation).
// ---------------------------------------------------------------------------

TEST_F(TransformerBlockTest, ConstructorPropagatesMHAsDModelValidation) {
    EXPECT_THROW({ TransformerBlock block(0, 1, 4, &backend); }, std::invalid_argument);
}

TEST_F(TransformerBlockTest, ConstructorPropagatesSwiGLUsDFFValidation) {
    EXPECT_THROW({ TransformerBlock block(2, 1, 0, &backend); }, std::invalid_argument);
}

TEST_F(TransformerBlockTest, ConstructorPropagatesDModelNotDivisibleByNumHeads) {
    EXPECT_THROW({ TransformerBlock block(3, 2, 4, &backend); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

// Every sub-module (RMSNorm's gamma, every LinearModule's weight/bias) is zero-initialized
// by default, so both residual branches collapse to exactly zero regardless of what
// softmax/attention computes internally (attn_out = attn_weights @ V, V == 0 -- the
// softmax's own nonuniform output doesn't matter, it's multiplied by zero). This makes the
// whole block an exact identity function without needing to hand-trace softmax's
// arithmetic -- a fully provable correctness anchor, not an approximation.
TEST_F(TransformerBlockTest, ForwardWithZeroInitializedParametersActsAsIdentity) {
    TransformerBlock block(4, 2, 6, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    std::vector<float> x_values{0.3f, -0.7f, 1.1f, -0.2f, 0.5f, -0.4f, 0.9f, 1.3f};
    Tensor x(Shape({1, 2, 4}), &backend, x_values);

    Tensor out = block.forward(x);

    ASSERT_EQ(out.shape(), Shape({1, 2, 4}));
    for (size_t i = 0; i < x_values.size(); ++i) {
        EXPECT_NEAR(out.data()[static_cast<int64_t>(i)], x_values[i], 1e-6f) << "element " << i;
    }
}

TEST_F(TransformerBlockTest, ForwardRejectsRankOneInput) {
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)block.forward(x); }, std::invalid_argument);
}

TEST_F(TransformerBlockTest, ForwardRejectsWrongFinalDimension) {
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW({ (void)block.forward(x); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Backward -- central finite differences over the input, plus a representative
// parameter from each of the four sub-modules (not exhaustive over every scalar --
// mission_transformer_block.md's own proportionate-coverage decision, since the block's
// full parameter count across 4 sub-modules is large relative to what MultiHeadAttentionModule's
// and SwiGLUModule's own missions already independently verified exhaustively).
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] TransformerBlock BuildBlock(DeviceBackend* backend, const std::vector<float>& gamma1,
                                           const std::vector<float>& q_w, const std::vector<float>& out_w,
                                           const std::vector<float>& gamma2, const std::vector<float>& gate_w) {
    TransformerBlock block(2, 1, 2, backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    block.norm1().set_gamma(gamma1);
    block.mha().q_proj().set_weight(q_w);
    block.mha().out_proj().set_weight(out_w);
    block.norm2().set_gamma(gamma2);
    block.swiglu().gate_proj().set_weight(gate_w);
    return block;
}

}  // namespace

TEST_F(TransformerBlockTest, BackwardMatchesCentralFiniteDifferences) {
    const std::vector<float> gamma1{0.6f, 0.4f};
    const std::vector<float> q_w{0.3f, -0.2f, 0.5f, 0.1f};
    const std::vector<float> out_w{0.4f, -0.3f, 0.2f, 0.6f};
    const std::vector<float> gamma2{0.5f, 0.7f};
    const std::vector<float> gate_w{0.2f, -0.4f, 0.3f, 0.1f};
    std::vector<float> x_values{0.6f, -1.1f};
    const std::vector<float> grad_out_values{1.2f, -0.5f};

    TransformerBlock block = BuildBlock(&backend, gamma1, q_w, out_w, gamma2, gate_w);
    Tensor x(Shape({1, 1, 2}), &backend, x_values);
    (void)block.forward(x);
    Tensor grad_out(Shape({1, 1, 2}), &backend, grad_out_values);
    Tensor grad_in = block.backward(grad_out);

    auto loss_of = [&](const std::vector<float>& g1, const std::vector<float>& qw, const std::vector<float>& dw,
                        const std::vector<float>& g2, const std::vector<float>& gw, const std::vector<float>& xv) {
        TransformerBlock probe = BuildBlock(&backend, g1, qw, dw, g2, gw);
        Tensor xp(Shape({1, 1, 2}), &backend, xv);
        Tensor out = probe.forward(xp);
        float loss = 0.0f;
        for (size_t k = 0; k < grad_out_values.size(); ++k) {
            loss += grad_out_values[k] * out.data()[static_cast<int64_t>(k)];
        }
        return loss;
    };

    const float h = 1e-3f;

    // Input gradients.
    for (size_t i = 0; i < x_values.size(); ++i) {
        std::vector<float> xp = x_values;
        std::vector<float> xm = x_values;
        xp[i] += h;
        xm[i] -= h;
        const float fp = loss_of(gamma1, q_w, out_w, gamma2, gate_w, xp);
        const float fm = loss_of(gamma1, q_w, out_w, gamma2, gate_w, xm);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 5e-2f) << "input element " << i;
    }

    // One representative parameter per sub-module: norm1's gamma[0], mha's q_proj weight[0],
    // norm2's gamma[0], swiglu's gate_proj weight[0].
    {
        std::vector<float> gp = gamma1, gm = gamma1;
        gp[0] += h;
        gm[0] -= h;
        const float fp = loss_of(gp, q_w, out_w, gamma2, gate_w, x_values);
        const float fm = loss_of(gm, q_w, out_w, gamma2, gate_w, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(block.norm1().gamma_grad().data()[0], numeric, 5e-2f) << "norm1 gamma[0]";
    }
    {
        std::vector<float> qp = q_w, qm = q_w;
        qp[0] += h;
        qm[0] -= h;
        const float fp = loss_of(gamma1, qp, out_w, gamma2, gate_w, x_values);
        const float fm = loss_of(gamma1, qm, out_w, gamma2, gate_w, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(block.mha().q_proj().weight_grad().data()[0], numeric, 5e-2f) << "mha q_proj weight[0]";
    }
    {
        std::vector<float> g2p = gamma2, g2m = gamma2;
        g2p[0] += h;
        g2m[0] -= h;
        const float fp = loss_of(gamma1, q_w, out_w, g2p, gate_w, x_values);
        const float fm = loss_of(gamma1, q_w, out_w, g2m, gate_w, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(block.norm2().gamma_grad().data()[0], numeric, 5e-2f) << "norm2 gamma[0]";
    }
    {
        std::vector<float> gwp = gate_w, gwm = gate_w;
        gwp[0] += h;
        gwm[0] -= h;
        const float fp = loss_of(gamma1, q_w, out_w, gamma2, gwp, x_values);
        const float fm = loss_of(gamma1, q_w, out_w, gamma2, gwm, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(block.swiglu().gate_proj().weight_grad().data()[0], numeric, 5e-2f) << "swiglu gate_proj weight[0]";
    }
}

TEST_F(TransformerBlockTest, BackwardBeforeForwardThrows) {
    TransformerBlock block(2, 1, 2, &backend);
    Tensor grad_out(Shape({1, 1, 2}), &backend);
    EXPECT_THROW({ (void)block.backward(grad_out); }, std::logic_error);
}

TEST_F(TransformerBlockTest, BackwardRejectsShapeMismatch) {
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({1, 1, 2}), &backend);
    (void)block.forward(x);
    Tensor wrong(Shape({1, 1, 3}), &backend);
    EXPECT_THROW({ (void)block.backward(wrong); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP conservation
// ---------------------------------------------------------------------------

// Predicted (mission_transformer_block.md Recon): dominated by MultiHeadAttentionModule's
// own known large by-design gap (~49.6% of its own output, mission_multihead_attention.md),
// propagated through by the two near-exact residual splits and SwiGLUModule's own
// near-exact contribution -- not amplified into something categorically larger. Confirmed
// as a substantial (not near-zero) gap, consistent with that prediction; a full
// stage-by-stage algebraic decomposition (as mission_multihead_attention.md's and
// mission_swiglu.md's own missions did via an independent NumPy reference) was not
// re-derived here -- logged as a scope simplification in the close-out, not silently
// skipped.
TEST_F(TransformerBlockTest, PropagateRelevanceMeasuresSubstantialGapConsistentWithAttentionSubBlock) {
    TransformerBlock block(4, 2, 6, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    block.norm1().set_gamma({0.6f, 0.4f, 0.5f, 0.3f});
    block.mha().q_proj().set_weight({0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.4f, 0.2f, 0.1f, -0.3f, 0.4f,
                                      -0.2f, 0.3f, 0.1f, -0.1f});
    block.mha().k_proj().set_weight({0.2f, 0.1f, -0.3f, 0.4f, 0.1f, -0.2f, 0.3f, 0.2f, -0.1f, 0.4f, 0.2f, -0.3f,
                                      0.3f, -0.1f, 0.4f, 0.2f});
    block.mha().v_proj().set_weight({0.4f, -0.1f, 0.2f, 0.3f, -0.2f, 0.3f, 0.1f, -0.4f, 0.3f, 0.2f, -0.1f, 0.4f,
                                      0.1f, -0.3f, 0.2f, 0.4f});
    block.mha().out_proj().set_weight({0.5f, -0.2f, 0.3f, 0.1f, -0.3f, 0.4f, 0.2f, -0.1f, 0.1f, 0.3f, -0.4f, 0.2f,
                                        -0.2f, 0.1f, 0.3f, 0.4f});
    block.norm2().set_gamma({0.5f, 0.7f, 0.4f, 0.6f});

    Tensor x(Shape({1, 3, 4}), &backend);
    std::vector<float> x_values{0.6f, -0.9f, 1.1f, -0.3f, 0.4f, -0.7f, 0.8f, 0.2f, -0.5f, 0.6f, -0.2f, 0.9f};
    x = Tensor(Shape({1, 3, 4}), &backend, x_values);
    (void)block.forward(x);

    Tensor relevance_out(Shape({1, 3, 4}), &backend);
    std::vector<float> r_values(12, 0.0f);
    for (size_t i = 0; i < r_values.size(); ++i) {
        r_values[i] = (i % 2 == 0) ? 1.0f : -0.5f;
    }
    relevance_out = Tensor(Shape({1, 3, 4}), &backend, r_values);

    Tensor relevance_in = block.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        sum_in += relevance_in.data()[i];
    }
    for (int64_t i = 0; i < relevance_out.numel(); ++i) {
        sum_out += relevance_out.data()[i];
    }
    const float gap = sum_in - sum_out;
    // Substantial (not epsilon-residual-only) -- consistent with attention's known large gap
    // dominating; not a tight bound, a regression guard against a future change silently
    // making this near-exact (which would be a real, worth-investigating surprise).
    EXPECT_GT(std::abs(gap), 1e-2f);
}

TEST_F(TransformerBlockTest, PropagateRelevanceBeforeForwardThrows) {
    TransformerBlock block(2, 1, 2, &backend);
    Tensor relevance_out(Shape({1, 1, 2}), &backend);
    EXPECT_THROW({ (void)block.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

TEST_F(TransformerBlockTest, ParametersExposesAllFourSubModules) {
    TransformerBlock block(2, 1, 3, &backend);
    auto params = block.parameters();
    // norm1 (1) + mha (4 linear x2 = 8) + norm2 (1) + swiglu (3 linear x2 = 6) = 16.
    ASSERT_EQ(params.size(), 16u);
}

// ---------------------------------------------------------------------------
// Independent AttnLRP reference validation (Phase 3 Mission 6, Phase 3's own exit gate:
// "validated against the AttnLRP reference implementation"). Reference values generated by
// what/docs/research/transformer_block_attnlrp_reference.py in the knowledge vault -- an
// independent NumPy transcription of Eq. 13 (softmax), Eq. 15 (bilinear, including its
// diagonal specialization for SwiGLU's gate multiply), and Eq. 19 (normalization identity)
// directly from the paper, not the reference repo's code, plus this codebase's own resolved
// residual-split rule. d_model=4, num_heads=2 (head_dim=2), d_ff=6, L=2, N=1, RoPE/QK-Norm
// both off.
TEST_F(TransformerBlockTest, ValidatesAgainstIndependentAttnLRPReference) {
    TransformerBlock block(4, 2, 6, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    block.norm1().set_gamma({0.6f, 0.4f, 0.5f, 0.3f});
    block.mha().q_proj().set_weight({0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.4f, 0.2f, 0.1f, -0.3f, 0.4f,
                                      -0.2f, 0.3f, 0.1f, -0.1f});
    block.mha().k_proj().set_weight({0.2f, 0.1f, -0.3f, 0.4f, 0.1f, -0.2f, 0.3f, 0.2f, -0.1f, 0.4f, 0.2f, -0.3f,
                                      0.3f, -0.1f, 0.4f, 0.2f});
    block.mha().v_proj().set_weight({0.4f, -0.1f, 0.2f, 0.3f, -0.2f, 0.3f, 0.1f, -0.4f, 0.3f, 0.2f, -0.1f, 0.4f,
                                      0.1f, -0.3f, 0.2f, 0.4f});
    block.mha().out_proj().set_weight({0.5f, -0.2f, 0.3f, 0.1f, -0.3f, 0.4f, 0.2f, -0.1f, 0.1f, 0.3f, -0.4f, 0.2f,
                                        -0.2f, 0.1f, 0.3f, 0.4f});
    block.norm2().set_gamma({0.5f, 0.7f, 0.4f, 0.6f});
    block.swiglu().gate_proj().set_weight({0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f,
                                            0.4f, -0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f,
                                            0.1f, 0.2f});
    block.swiglu().up_proj().set_weight({-0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f, 0.1f, 0.2f,
                                          0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f, 0.4f});
    block.swiglu().down_proj().set_weight({0.6f, -0.2f, 0.3f, 0.1f, -0.5f, 0.4f, 0.2f, -0.3f, 0.1f, 0.3f, -0.4f, 0.2f,
                                            0.4f, -0.1f, 0.3f, 0.2f, -0.2f, 0.3f, 0.1f, -0.4f, 0.2f, -0.3f, 0.4f, 0.1f});

    // (L, d_model), N=1 -- not the "nicer"-looking values first tried, which happened to
    // make x+attn_out nearly cancel at one index, blowing up the residual-split epsilon
    // rule's denominator (the same class of coincidence mission_swiglu.md's own close-out
    // documents). See the reference script's own comment for the numeric trace.
    Tensor x(Shape({1, 2, 4}), &backend,
             {0.6f, -0.9f, 1.1f, -0.7f, 0.4f, -0.7f, 0.8f, 0.6f});
    Tensor out = block.forward(x);

    const std::vector<float> expected_y2{0.720711f, -0.964979f, 1.392442f, -0.429377f,
                                          0.480444f, -0.761009f, 1.012949f, 0.897909f};
    for (size_t i = 0; i < expected_y2.size(); ++i) {
        EXPECT_NEAR(out.data()[static_cast<int64_t>(i)], expected_y2[i], 1e-4f) << "forward element " << i;
    }

    Tensor relevance_out(Shape({1, 2, 4}), &backend,
                          {1.0f, -0.5f, 1.0f, -0.5f, 1.0f, -0.5f, 1.0f, -0.5f});
    Tensor relevance_in = block.propagate_relevance(relevance_out, LRPRuleConfig{});

    const std::vector<float> expected_r_x{0.923577f, -0.423664f, 0.888385f, -0.815141f,
                                           0.937874f, -0.490247f, 0.857730f, -0.364326f};
    for (size_t i = 0; i < expected_r_x.size(); ++i) {
        EXPECT_NEAR(relevance_in.data()[static_cast<int64_t>(i)], expected_r_x[i], 1e-4f) << "relevance element " << i;
    }
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition (device guards)
// ---------------------------------------------------------------------------

TEST_F(TransformerBlockDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({1, 1, 2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)block.forward(x); }, "EXAI_ASSERT failed");
}

TEST_F(TransformerBlockDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({1, 1, 2}), &backend, {1.0f, 2.0f});
    (void)block.forward(x);
    Tensor grad_out(Shape({1, 1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)block.backward(grad_out); }, "EXAI_ASSERT failed");
}

TEST_F(TransformerBlockDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    TransformerBlock block(2, 1, 2, &backend);
    Tensor x(Shape({1, 1, 2}), &backend, {1.0f, 2.0f});
    (void)block.forward(x);
    Tensor relevance_out(Shape({1, 1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)block.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
