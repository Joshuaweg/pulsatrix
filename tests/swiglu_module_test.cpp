#include "exai/swiglu_module.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace {

class SwiGLUModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using SwiGLUModuleDeathTest = SwiGLUModuleTest;

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

TEST_F(SwiGLUModuleTest, ConstructorRejectsNonPositiveDModel) {
    EXPECT_THROW({ SwiGLUModule m(0, 4, &backend); }, std::invalid_argument);
    EXPECT_THROW({ SwiGLUModule m(-1, 4, &backend); }, std::invalid_argument);
}

TEST_F(SwiGLUModuleTest, ConstructorRejectsNonPositiveDFF) {
    EXPECT_THROW({ SwiGLUModule m(4, 0, &backend); }, std::invalid_argument);
    EXPECT_THROW({ SwiGLUModule m(4, -1, &backend); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

// Identity weights on all three projections collapse the pipeline to
// out = silu(x) * x (elementwise) -- fully hand-traceable via sigmoid(1)/sigmoid(2).
TEST_F(SwiGLUModuleTest, ForwardMatchesHandDerivedExample) {
    SwiGLUModule swiglu(2, 2, &backend);
    swiglu.gate_proj().set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    swiglu.gate_proj().set_bias({0.0f, 0.0f});
    swiglu.up_proj().set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    swiglu.up_proj().set_bias({0.0f, 0.0f});
    swiglu.down_proj().set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    swiglu.down_proj().set_bias({0.0f, 0.0f});

    Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor out = swiglu.forward(x);

    const float sigmoid1 = 1.0f / (1.0f + std::exp(-1.0f));
    const float sigmoid2 = 1.0f / (1.0f + std::exp(-2.0f));
    const float expected0 = (1.0f * sigmoid1) * 1.0f;
    const float expected1 = (2.0f * sigmoid2) * 2.0f;

    ASSERT_EQ(out.shape(), Shape({1, 2}));
    EXPECT_NEAR(out.data()[0], expected0, 1e-5f);
    EXPECT_NEAR(out.data()[1], expected1, 1e-5f);
}

TEST_F(SwiGLUModuleTest, ForwardPreservesLeadingRankThreeShape) {
    SwiGLUModule swiglu(2, 3, &backend);
    Tensor x(Shape({2, 4, 2}), &backend);
    x.fill(0.5f);
    Tensor out = swiglu.forward(x);
    EXPECT_EQ(out.shape(), Shape({2, 4, 2}));
}

TEST_F(SwiGLUModuleTest, ForwardRejectsRankOneInput) {
    SwiGLUModule swiglu(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)swiglu.forward(x); }, std::invalid_argument);
}

TEST_F(SwiGLUModuleTest, ForwardRejectsWrongFinalDimension) {
    SwiGLUModule swiglu(2, 2, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_THROW({ (void)swiglu.forward(x); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Backward -- central finite differences over every scalar parameter + input
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] SwiGLUModule BuildModule(DeviceBackend* backend, const std::vector<float>& gate_w,
                                        const std::vector<float>& gate_b, const std::vector<float>& up_w,
                                        const std::vector<float>& up_b, const std::vector<float>& down_w,
                                        const std::vector<float>& down_b) {
    SwiGLUModule m(2, 2, backend);
    m.gate_proj().set_weight(gate_w);
    m.gate_proj().set_bias(gate_b);
    m.up_proj().set_weight(up_w);
    m.up_proj().set_bias(up_b);
    m.down_proj().set_weight(down_w);
    m.down_proj().set_bias(down_b);
    return m;
}

}  // namespace

TEST_F(SwiGLUModuleTest, BackwardMatchesCentralFiniteDifferences) {
    const std::vector<float> gate_w{0.4f, -0.3f, 0.2f, 0.5f};
    const std::vector<float> gate_b{0.1f, -0.2f};
    const std::vector<float> up_w{-0.2f, 0.6f, 0.3f, -0.1f};
    const std::vector<float> up_b{0.05f, 0.15f};
    const std::vector<float> down_w{0.7f, -0.4f, -0.5f, 0.9f};
    const std::vector<float> down_b{-0.1f, 0.2f};
    std::vector<float> x_values{0.8f, -1.1f};
    const std::vector<float> grad_out_values{1.3f, -0.7f};

    SwiGLUModule m = BuildModule(&backend, gate_w, gate_b, up_w, up_b, down_w, down_b);
    Tensor x(Shape({1, 2}), &backend, x_values);
    (void)m.forward(x);
    Tensor grad_out(Shape({1, 2}), &backend, grad_out_values);
    Tensor grad_in = m.backward(grad_out);

    auto loss_of = [&](const std::vector<float>& gw, const std::vector<float>& gb, const std::vector<float>& uw,
                        const std::vector<float>& ub, const std::vector<float>& dw, const std::vector<float>& db,
                        const std::vector<float>& xv) {
        SwiGLUModule probe = BuildModule(&backend, gw, gb, uw, ub, dw, db);
        Tensor xp(Shape({1, 2}), &backend, xv);
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
        const float fp = loss_of(gate_w, gate_b, up_w, up_b, down_w, down_b, xp);
        const float fm = loss_of(gate_w, gate_b, up_w, up_b, down_w, down_b, xm);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 1e-2f) << "input element " << i;
    }

    // gate_proj weight/bias gradients.
    for (size_t i = 0; i < gate_w.size(); ++i) {
        std::vector<float> wp = gate_w;
        std::vector<float> wm = gate_w;
        wp[i] += h;
        wm[i] -= h;
        const float fp = loss_of(wp, gate_b, up_w, up_b, down_w, down_b, x_values);
        const float fm = loss_of(wm, gate_b, up_w, up_b, down_w, down_b, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.gate_proj().weight_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f)
            << "gate_proj weight " << i;
    }
    for (size_t i = 0; i < gate_b.size(); ++i) {
        std::vector<float> bp = gate_b;
        std::vector<float> bm = gate_b;
        bp[i] += h;
        bm[i] -= h;
        const float fp = loss_of(gate_w, bp, up_w, up_b, down_w, down_b, x_values);
        const float fm = loss_of(gate_w, bm, up_w, up_b, down_w, down_b, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.gate_proj().bias_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f)
            << "gate_proj bias " << i;
    }

    // up_proj weight/bias gradients.
    for (size_t i = 0; i < up_w.size(); ++i) {
        std::vector<float> wp = up_w;
        std::vector<float> wm = up_w;
        wp[i] += h;
        wm[i] -= h;
        const float fp = loss_of(gate_w, gate_b, wp, up_b, down_w, down_b, x_values);
        const float fm = loss_of(gate_w, gate_b, wm, up_b, down_w, down_b, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.up_proj().weight_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f) << "up_proj weight "
                                                                                                 << i;
    }
    for (size_t i = 0; i < up_b.size(); ++i) {
        std::vector<float> bp = up_b;
        std::vector<float> bm = up_b;
        bp[i] += h;
        bm[i] -= h;
        const float fp = loss_of(gate_w, gate_b, up_w, bp, down_w, down_b, x_values);
        const float fm = loss_of(gate_w, gate_b, up_w, bm, down_w, down_b, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.up_proj().bias_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f) << "up_proj bias " << i;
    }

    // down_proj weight/bias gradients.
    for (size_t i = 0; i < down_w.size(); ++i) {
        std::vector<float> wp = down_w;
        std::vector<float> wm = down_w;
        wp[i] += h;
        wm[i] -= h;
        const float fp = loss_of(gate_w, gate_b, up_w, up_b, wp, down_b, x_values);
        const float fm = loss_of(gate_w, gate_b, up_w, up_b, wm, down_b, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.down_proj().weight_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f)
            << "down_proj weight " << i;
    }
    for (size_t i = 0; i < down_b.size(); ++i) {
        std::vector<float> bp = down_b;
        std::vector<float> bm = down_b;
        bp[i] += h;
        bm[i] -= h;
        const float fp = loss_of(gate_w, gate_b, up_w, up_b, down_w, bp, x_values);
        const float fm = loss_of(gate_w, gate_b, up_w, up_b, down_w, bm, x_values);
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(m.down_proj().bias_grad().data()[static_cast<int64_t>(i)], numeric, 1e-2f)
            << "down_proj bias " << i;
    }
}

TEST_F(SwiGLUModuleTest, BackwardBeforeForwardThrows) {
    SwiGLUModule m(2, 2, &backend);
    Tensor grad_out(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)m.backward(grad_out); }, std::logic_error);
}

TEST_F(SwiGLUModuleTest, BackwardRejectsShapeMismatch) {
    SwiGLUModule m(2, 2, &backend);
    Tensor x(Shape({1, 2}), &backend);
    (void)m.forward(x);
    Tensor wrong(Shape({1, 3}), &backend);
    EXPECT_THROW({ (void)m.backward(wrong); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP conservation
// ---------------------------------------------------------------------------

// Predicted (mission_swiglu.md Recon) to conserve near-exactly, unlike SoftmaxModule's/
// MultiHeadAttentionModule's large by-design gap: the diagonal Eq. 15 split satisfies
// R_a+R_b ~= R_c up to an epsilon residual, and SiLU's identity pass-through is exact.
// Confirmed here, not just assumed.
TEST_F(SwiGLUModuleTest, PropagateRelevanceConservesNearExactly) {
    SwiGLUModule m(3, 4, &backend);
    m.gate_proj().set_weight({0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f, 0.4f});
    m.gate_proj().set_bias({0.1f, -0.1f, 0.05f, 0.2f});
    m.up_proj().set_weight({-0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f, 0.1f, 0.2f});
    m.up_proj().set_bias({0.05f, 0.1f, -0.05f, 0.15f});
    m.down_proj().set_weight({0.6f, -0.2f, 0.3f, 0.1f, -0.5f, 0.4f, 0.2f, -0.3f, 0.1f, 0.3f, -0.4f, 0.2f});
    m.down_proj().set_bias({-0.1f, 0.2f, 0.05f});

    // 0.65/-0.85/1.15, not 0.7/-0.9/1.3 -- the "nicer" values happen to make up_proj's
    // pre-bias output for one out_feature exactly 0.0 given these weights, which blows up
    // the epsilon-rule denominator (a genuine degenerate-input case, not a bug in the rule
    // itself -- see mission_swiglu.md's close-out for the numeric trace). These values keep
    // every pre-activation safely away from zero so the near-exact-conservation prediction
    // is actually being tested, not accidentally defeated by an unrelated coincidence.
    Tensor x(Shape({1, 3}), &backend, {0.65f, -0.85f, 1.15f});
    (void)m.forward(x);

    Tensor relevance_out(Shape({1, 3}), &backend, {2.0f, -1.0f, 0.5f});
    Tensor relevance_in = m.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        sum_in += relevance_in.data()[i];
    }
    for (int64_t i = 0; i < relevance_out.numel(); ++i) {
        sum_out += relevance_out.data()[i];
    }
    EXPECT_NEAR(sum_in, sum_out, 1e-3f);
}

TEST_F(SwiGLUModuleTest, PropagateRelevanceBeforeForwardThrows) {
    SwiGLUModule m(2, 2, &backend);
    Tensor relevance_out(Shape({1, 2}), &backend);
    EXPECT_THROW({ (void)m.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

TEST_F(SwiGLUModuleTest, ParametersExposesAllThreeProjections) {
    SwiGLUModule m(2, 3, &backend);
    auto params = m.parameters();
    // weight+bias per projection x 3 projections.
    ASSERT_EQ(params.size(), 6u);
    EXPECT_EQ(params[0].value, &m.gate_proj().weight());
    EXPECT_EQ(params[2].value, &m.up_proj().weight());
    EXPECT_EQ(params[4].value, &m.down_proj().weight());
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition (device guards)
// ---------------------------------------------------------------------------

// Not yet backend-generic -- raw host loops in forward_impl/backward/propagate_relevance
// dereference Tensor::data() directly. Same mislabeled-Tensor pattern as
// RoPEModuleDeathTest/SoftmaxModuleDeathTest: no real GPU needed, DeviceType::Cuda tagged
// over real CPUBackend memory trips the guard identically.
TEST_F(SwiGLUModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    SwiGLUModule m(2, 2, &backend);
    Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)m.forward(x); }, "EXAI_ASSERT failed");
}

TEST_F(SwiGLUModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    SwiGLUModule m(2, 2, &backend);
    Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    (void)m.forward(x);
    Tensor grad_out(Shape({1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)m.backward(grad_out); }, "EXAI_ASSERT failed");
}

TEST_F(SwiGLUModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    SwiGLUModule m(2, 2, &backend);
    Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    (void)m.forward(x);
    Tensor relevance_out(Shape({1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)m.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
