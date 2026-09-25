#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/lstm_module.hpp"

namespace pulsatrix {
namespace {

float sigmoid_ref(float z) { return 1.0f / (1.0f + std::exp(-z)); }

class LSTMModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(LSTMModuleTest, ConstructionThrowsOnZeroInputSize) {
    EXPECT_THROW({ LSTMModule lstm(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(LSTMModuleTest, ConstructionThrowsOnNegativeHiddenSize) {
    EXPECT_THROW({ LSTMModule lstm(2, -1, &backend); }, std::invalid_argument);
}

TEST_F(LSTMModuleTest, ForwardThrowsOnWrongRank) {
    LSTMModule lstm(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)lstm.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(LSTMModuleTest, ForwardThrowsOnMismatchedInputSize) {
    LSTMModule lstm(2, 2, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)lstm.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(LSTMModuleTest, BackwardThrowsIfCalledBeforeForward) {
    LSTMModule lstm(1, 1, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)lstm.backward(grad); }, std::logic_error);
}

TEST_F(LSTMModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    LSTMModule lstm(1, 1, &backend);
    Tensor relevance(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)lstm.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(LSTMModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    LSTMModule lstm(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)lstm.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)lstm.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(LSTMModuleTest, PropagateRelevanceThrowsOnShapeMismatch) {
    LSTMModule lstm(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)lstm.forward(input);
    Tensor wrong_shape_relevance(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)lstm.propagate_relevance(wrong_shape_relevance, LRPRuleConfig{}); },
                 std::invalid_argument);
}

// Independent reference recomputation of the full 4-gate recurrence (h_0 = c_0 = 0, tied
// weights across timesteps) -- catches gate-wiring/indexing bugs, not just formula bugs
// baked into both sides. Distinct per-gate weights so a swapped i/f/g/o would fail.
TEST_F(LSTMModuleTest, ForwardMatchesReferenceRecurrence) {
    LSTMModule lstm(1, 1, &backend);
    lstm.set_weight_xi({1.0f});
    lstm.set_weight_hi({0.5f});
    lstm.set_bias_i({0.1f});
    lstm.set_weight_xf({-0.7f});
    lstm.set_weight_hf({0.2f});
    lstm.set_bias_f({0.3f});
    lstm.set_weight_xg({2.0f});
    lstm.set_weight_hg({-0.4f});
    lstm.set_bias_g({-0.2f});
    lstm.set_weight_xo({0.6f});
    lstm.set_weight_ho({0.9f});
    lstm.set_bias_o({0.05f});

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    Tensor output = lstm.forward(input);

    float h = 0.0f;
    float c = 0.0f;
    std::array<float, 2> x = {1.0f, 0.5f};
    std::array<float, 2> h_expected = {0.0f, 0.0f};
    for (int t = 0; t < 2; ++t) {
        float i_t = sigmoid_ref(x[static_cast<size_t>(t)] * 1.0f + h * 0.5f + 0.1f);
        float f_t = sigmoid_ref(x[static_cast<size_t>(t)] * -0.7f + h * 0.2f + 0.3f);
        float g_t = std::tanh(x[static_cast<size_t>(t)] * 2.0f + h * -0.4f + -0.2f);
        float o_t = sigmoid_ref(x[static_cast<size_t>(t)] * 0.6f + h * 0.9f + 0.05f);
        c = f_t * c + i_t * g_t;
        h = o_t * std::tanh(c);
        h_expected[static_cast<size_t>(t)] = h;
    }

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), h_expected[0], 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), h_expected[1], 1e-5f);
}

// N=2 -- proves each batch row is threaded through its own independent h/c carry.
TEST_F(LSTMModuleTest, ForwardHandlesMultiBatch) {
    LSTMModule lstm(1, 1, &backend);
    lstm.set_weight_xi({1.0f});
    lstm.set_weight_hi({0.5f});
    lstm.set_bias_i({0.1f});
    lstm.set_weight_xf({-0.7f});
    lstm.set_weight_hf({0.2f});
    lstm.set_bias_f({0.3f});
    lstm.set_weight_xg({2.0f});
    lstm.set_weight_hg({-0.4f});
    lstm.set_bias_g({-0.2f});
    lstm.set_weight_xo({0.6f});
    lstm.set_weight_ho({0.9f});
    lstm.set_bias_o({0.05f});

    Tensor input(Shape({2, 2, 1}), &backend, {1.0f, 0.5f, -0.3f, 0.8f});
    Tensor output = lstm.forward(input);

    std::array<std::array<float, 2>, 2> rows = {std::array<float, 2>{1.0f, 0.5f},
                                                 std::array<float, 2>{-0.3f, 0.8f}};
    for (size_t n = 0; n < 2; ++n) {
        float h = 0.0f;
        float c = 0.0f;
        for (size_t t = 0; t < 2; ++t) {
            float x = rows[n][t];
            float i_t = sigmoid_ref(x * 1.0f + h * 0.5f + 0.1f);
            float f_t = sigmoid_ref(x * -0.7f + h * 0.2f + 0.3f);
            float g_t = std::tanh(x * 2.0f + h * -0.4f + -0.2f);
            float o_t = sigmoid_ref(x * 0.6f + h * 0.9f + 0.05f);
            c = f_t * c + i_t * g_t;
            h = o_t * std::tanh(c);
            EXPECT_NEAR(output.at({static_cast<int64_t>(n), static_cast<int64_t>(t), 0}), h, 1e-5f)
                << "row " << n << " step " << t;
        }
    }
}

// ---------------------------------------------------------------------------
// BPTT finite-difference harness. 12 parameter buffers (8 weight matrices + 4 biases)
// flattened into one canonical index order so a single loop can perturb every scalar and
// compare against the matching gradient slot -- the same central-difference discipline
// RNNModuleTest used, scaled to the gated cell rather than hand-derived per gate.
// ---------------------------------------------------------------------------
struct LSTMParams {
    std::array<float, 4> wxi, whi, wxf, whf, wxg, whg, wxo, who;
    std::array<float, 2> bi, bf, bg, bo;
};

void configure(LSTMModule& m, const LSTMParams& p) {
    m.set_weight_xi({p.wxi[0], p.wxi[1], p.wxi[2], p.wxi[3]});
    m.set_weight_hi({p.whi[0], p.whi[1], p.whi[2], p.whi[3]});
    m.set_bias_i({p.bi[0], p.bi[1]});
    m.set_weight_xf({p.wxf[0], p.wxf[1], p.wxf[2], p.wxf[3]});
    m.set_weight_hf({p.whf[0], p.whf[1], p.whf[2], p.whf[3]});
    m.set_bias_f({p.bf[0], p.bf[1]});
    m.set_weight_xg({p.wxg[0], p.wxg[1], p.wxg[2], p.wxg[3]});
    m.set_weight_hg({p.whg[0], p.whg[1], p.whg[2], p.whg[3]});
    m.set_bias_g({p.bg[0], p.bg[1]});
    m.set_weight_xo({p.wxo[0], p.wxo[1], p.wxo[2], p.wxo[3]});
    m.set_weight_ho({p.who[0], p.who[1], p.who[2], p.who[3]});
    m.set_bias_o({p.bo[0], p.bo[1]});
}

// Canonical flattened order: (Wx*, Wh*, b*) per gate, gates in i, f, g, o order.
std::vector<float*> param_slots(LSTMParams& p) {
    std::vector<float*> slots;
    auto push4 = [&slots](std::array<float, 4>& a) {
        for (float& v : a) slots.push_back(&v);
    };
    auto push2 = [&slots](std::array<float, 2>& a) {
        for (float& v : a) slots.push_back(&v);
    };
    push4(p.wxi);
    push4(p.whi);
    push2(p.bi);
    push4(p.wxf);
    push4(p.whf);
    push2(p.bf);
    push4(p.wxg);
    push4(p.whg);
    push2(p.bg);
    push4(p.wxo);
    push4(p.who);
    push2(p.bo);
    return slots;
}

std::vector<float> analytic_grads(const LSTMModule& m) {
    std::vector<float> g;
    auto append = [&g](const Tensor& t) {
        for (int64_t i = 0; i < t.numel(); ++i) g.push_back(t.data()[i]);
    };
    append(m.weight_xi_grad());
    append(m.weight_hi_grad());
    append(m.bias_i_grad());
    append(m.weight_xf_grad());
    append(m.weight_hf_grad());
    append(m.bias_f_grad());
    append(m.weight_xg_grad());
    append(m.weight_hg_grad());
    append(m.bias_g_grad());
    append(m.weight_xo_grad());
    append(m.weight_ho_grad());
    append(m.bias_o_grad());
    return g;
}

float compute_loss(CPUBackend& backend, const LSTMParams& p, const Tensor& input, const Tensor& grad_seed) {
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, p);
    Tensor out = lstm.forward(input);
    float total = 0.0f;
    for (int64_t i = 0; i < out.numel(); ++i) total += out.data()[i] * grad_seed.data()[i];
    return total;
}

// Deliberately well-separated per-gate weights and biases: near-zero, near-identical
// parameters drive every gate to sigmoid(0)=0.5 with derivative 0.25, which makes an
// absolute-tolerance gradient check blind to a swapped gate derivative (confirmed by
// mutation testing during this mission -- a wrong o-gate derivative passed under the small
// weights first tried here). These values keep the four gates at genuinely different
// operating points.
LSTMParams reference_params() {
    LSTMParams p;
    p.wxi = {0.80f, -1.20f, 0.60f, 0.90f};
    p.whi = {0.50f, -0.40f, 0.70f, 0.30f};
    p.bi = {-1.10f, 0.90f};
    p.wxf = {-1.00f, 0.70f, 1.30f, -0.50f};
    p.whf = {0.40f, 0.60f, -0.80f, 0.50f};
    p.bf = {1.40f, -0.70f};
    p.wxg = {1.10f, 0.50f, -0.90f, 0.80f};
    p.whg = {-0.30f, 0.60f, 0.40f, -0.50f};
    p.bg = {-0.40f, 0.50f};
    p.wxo = {0.70f, -1.40f, 0.90f, 1.20f};
    p.who = {0.30f, 0.80f, -0.60f, 0.50f};
    p.bo = {0.90f, -1.30f};
    return p;
}

// Central differences in float32 on an O(1) loss with h=1e-3 carries roughly 1e-4 of
// absolute noise; everything above that floor is checked relatively so the test's
// sensitivity does not silently degrade with gradient magnitude.
float fd_tolerance(float numeric) { return 1e-3f + 2e-2f * std::fabs(numeric); }

// BPTT correctness through four gates and a cell carry cannot be reliably hand-derived --
// central finite differences over every one of the 40 scalar parameters is this mission's
// actual proof, same discipline RNNModule's Mission 0 used.
TEST_F(LSTMModuleTest, BackwardGradientsMatchFiniteDifferences) {
    const float h = 1e-3f;
    Tensor input(Shape({1, 3, 2}), &backend, {0.9f, -1.2f, 1.4f, 0.6f, -0.8f, 1.1f});
    Tensor grad_seed(Shape({1, 3, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f, -0.7f, 0.4f});

    LSTMParams params = reference_params();
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, params);
    (void)lstm.forward(input);
    (void)lstm.backward(grad_seed);

    std::vector<float> analytic = analytic_grads(lstm);
    ASSERT_EQ(analytic.size(), 40u);

    for (size_t idx = 0; idx < 40; ++idx) {
        LSTMParams plus = params;
        LSTMParams minus = params;
        *param_slots(plus)[idx] += h;
        *param_slots(minus)[idx] -= h;
        float numeric = (compute_loss(backend, plus, input, grad_seed) -
                         compute_loss(backend, minus, input, grad_seed)) /
                        (2 * h);
        EXPECT_NEAR(analytic[idx], numeric, fd_tolerance(numeric)) << "parameter slot " << idx;
    }
}

// Gradient w.r.t. the input sequence itself (the value backward() returns), also by
// central differences -- the BPTT path that feeds an upstream module.
TEST_F(LSTMModuleTest, BackwardInputGradientMatchesFiniteDifferences) {
    const float h = 1e-3f;
    std::vector<float> x = {0.9f, -1.2f, 1.4f, 0.6f, -0.8f, 1.1f};
    Tensor input(Shape({1, 3, 2}), &backend, x);
    Tensor grad_seed(Shape({1, 3, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f, -0.7f, 0.4f});

    LSTMParams params = reference_params();
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, params);
    (void)lstm.forward(input);
    Tensor grad_input = lstm.backward(grad_seed);

    for (size_t idx = 0; idx < x.size(); ++idx) {
        std::vector<float> x_plus = x, x_minus = x;
        x_plus[idx] += h;
        x_minus[idx] -= h;
        Tensor in_plus(Shape({1, 3, 2}), &backend, x_plus);
        Tensor in_minus(Shape({1, 3, 2}), &backend, x_minus);
        float numeric = (compute_loss(backend, params, in_plus, grad_seed) -
                         compute_loss(backend, params, in_minus, grad_seed)) /
                        (2 * h);
        EXPECT_NEAR(grad_input.data()[idx], numeric, fd_tolerance(numeric)) << "input element " << idx;
    }
}

// propagate_relevance: end-to-end conservation on a deliberately ASYMMETRIC multi-timestep
// input (distinct per-step, per-feature magnitudes AND distinct per-gate weights) -- the
// degenerate-symmetric-input trap Mission 0 (RMSNorm) documented would let a wrong
// redistribution still sum correctly. Numeric tolerance check, not EXPECT_TRUE.
TEST_F(LSTMModuleTest, PropagateRelevanceConservesOnAsymmetricSequence) {
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, reference_params());

    Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
    (void)lstm.forward(input);

    Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor relevance_in = lstm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

// Conservation must hold per batch row independently, not merely in aggregate -- an
// implementation that leaked relevance across rows could still balance globally.
TEST_F(LSTMModuleTest, PropagateRelevanceConservesPerBatchRow) {
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, reference_params());

    Tensor input(Shape({2, 3, 2}), &backend,
                 {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f, 0.9f, 0.7f, -0.8f, 0.2f, 0.4f, -0.6f});
    (void)lstm.forward(input);

    Tensor relevance_out(Shape({2, 3, 2}), &backend,
                         {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -1.0f, 0.4f, 2.5f, 0.3f, 0.6f, 1.1f});
    Tensor relevance_in = lstm.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t n = 0; n < 2; ++n) {
        float sum_in = 0.0f;
        float sum_out = 0.0f;
        for (int64_t k = 0; k < 6; ++k) {
            sum_in += relevance_in.data()[n * 6 + k];
            sum_out += relevance_out.data()[n * 6 + k];
        }
        EXPECT_NEAR(sum_in, sum_out, 1e-2f) << "batch row " << n;
    }
}

// Arras et al. 2019's gate-signal split is *not* the trivial "pass everything through"
// rule: relevance must actually be distributed across timesteps in proportion to the
// signal terms. A nonzero relevance seeded only at the LAST timestep must still reach
// earlier timesteps' inputs through the cell carry -- proof the c_{t-1} branch is live.
TEST_F(LSTMModuleTest, RelevanceSeededAtLastStepReachesEarlierTimesteps) {
    LSTMModule lstm(2, 2, &backend);
    configure(lstm, reference_params());

    Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
    (void)lstm.forward(input);

    Tensor relevance_out(Shape({1, 3, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f});
    Tensor relevance_in = lstm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float early = std::fabs(relevance_in.at({0, 0, 0})) + std::fabs(relevance_in.at({0, 0, 1})) +
                  std::fabs(relevance_in.at({0, 1, 0})) + std::fabs(relevance_in.at({0, 1, 1}));
    EXPECT_GT(early, 1e-6f);
}

using LSTMModuleDeathTest = LSTMModuleTest;

// forward_impl/backward/propagate_relevance all dereference Tensor::data() in raw host
// loops (including the sigmoid/tanh workarounds) -- undefined behavior on a CUDA-backed
// Tensor. See RNNModuleDeathTest for the mislabeled-Tensor testing pattern this reuses.
// Written from the start of this mission, not deferred -- the exact discipline Mission 7's
// remediation established.
TEST_F(LSTMModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LSTMModule lstm(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)lstm.forward(input); }, "PULSATRIX_ASSERT failed");
}

TEST_F(LSTMModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LSTMModule lstm(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)lstm.forward(input);

    Tensor grad_output(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)lstm.backward(grad_output); }, "PULSATRIX_ASSERT failed");
}

TEST_F(LSTMModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LSTMModule lstm(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)lstm.forward(input);

    Tensor relevance_out(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)lstm.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
