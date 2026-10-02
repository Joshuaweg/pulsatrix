#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

float sigmoid_ref(float z) { return 1.0f / (1.0f + std::exp(-z)); }

class GRUModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(GRUModuleTest, ConstructionThrowsOnZeroInputSize) {
    EXPECT_THROW({ GRUModule gru(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(GRUModuleTest, ConstructionThrowsOnNegativeHiddenSize) {
    EXPECT_THROW({ GRUModule gru(2, -1, &backend); }, std::invalid_argument);
}

TEST_F(GRUModuleTest, ForwardThrowsOnWrongRank) {
    GRUModule gru(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)gru.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(GRUModuleTest, ForwardThrowsOnMismatchedInputSize) {
    GRUModule gru(2, 2, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)gru.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(GRUModuleTest, BackwardThrowsIfCalledBeforeForward) {
    GRUModule gru(1, 1, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)gru.backward(grad); }, std::logic_error);
}

TEST_F(GRUModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    GRUModule gru(1, 1, &backend);
    Tensor relevance(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)gru.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(GRUModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    GRUModule gru(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)gru.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)gru.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(GRUModuleTest, PropagateRelevanceThrowsOnShapeMismatch) {
    GRUModule gru(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)gru.forward(input);
    Tensor wrong_shape_relevance(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)gru.propagate_relevance(wrong_shape_relevance, LRPRuleConfig{}); },
                 std::invalid_argument);
}

// Independent reference recomputation of the full Cho et al. 2014 recurrence (h_0 = 0, tied
// weights across timesteps) -- catches gate-wiring/indexing bugs, not just formula bugs
// baked into both sides. Distinct per-gate weights so a swapped z/r/n would fail, and the
// reset gate is applied to the *projected* h_{t-1} (h * W_hn), not to h_{t-1} itself, which
// is the structural detail that distinguishes GRU from LSTM here.
TEST_F(GRUModuleTest, ForwardMatchesReferenceRecurrence) {
    GRUModule gru(1, 1, &backend);
    gru.set_weight_xz({1.0f});
    gru.set_weight_hz({0.5f});
    gru.set_bias_z({0.1f});
    gru.set_weight_xr({-0.7f});
    gru.set_weight_hr({0.2f});
    gru.set_bias_r({0.3f});
    gru.set_weight_xn({2.0f});
    gru.set_weight_hn({-0.4f});
    gru.set_bias_n({-0.2f});

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    Tensor output = gru.forward(input);

    float h = 0.0f;
    std::array<float, 2> x = {1.0f, 0.5f};
    std::array<float, 2> h_expected = {0.0f, 0.0f};
    for (int t = 0; t < 2; ++t) {
        const float xt = x[static_cast<size_t>(t)];
        float z_t = sigmoid_ref(xt * 1.0f + h * 0.5f + 0.1f);
        float r_t = sigmoid_ref(xt * -0.7f + h * 0.2f + 0.3f);
        float hn_prev = h * -0.4f;
        float n_t = std::tanh(xt * 2.0f + r_t * hn_prev + -0.2f);
        h = (1.0f - z_t) * h + z_t * n_t;
        h_expected[static_cast<size_t>(t)] = h;
    }

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), h_expected[0], 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), h_expected[1], 1e-5f);
}

// N=2 -- proves each batch row is threaded through its own independent hidden carry.
TEST_F(GRUModuleTest, ForwardHandlesMultiBatch) {
    GRUModule gru(1, 1, &backend);
    gru.set_weight_xz({1.0f});
    gru.set_weight_hz({0.5f});
    gru.set_bias_z({0.1f});
    gru.set_weight_xr({-0.7f});
    gru.set_weight_hr({0.2f});
    gru.set_bias_r({0.3f});
    gru.set_weight_xn({2.0f});
    gru.set_weight_hn({-0.4f});
    gru.set_bias_n({-0.2f});

    Tensor input(Shape({2, 2, 1}), &backend, {1.0f, 0.5f, -0.3f, 0.8f});
    Tensor output = gru.forward(input);

    std::array<std::array<float, 2>, 2> rows = {std::array<float, 2>{1.0f, 0.5f},
                                                std::array<float, 2>{-0.3f, 0.8f}};
    for (size_t n = 0; n < 2; ++n) {
        float h = 0.0f;
        for (size_t t = 0; t < 2; ++t) {
            float x = rows[n][t];
            float z_t = sigmoid_ref(x * 1.0f + h * 0.5f + 0.1f);
            float r_t = sigmoid_ref(x * -0.7f + h * 0.2f + 0.3f);
            float hn_prev = h * -0.4f;
            float n_t = std::tanh(x * 2.0f + r_t * hn_prev + -0.2f);
            h = (1.0f - z_t) * h + z_t * n_t;
            EXPECT_NEAR(output.at({static_cast<int64_t>(n), static_cast<int64_t>(t), 0}), h, 1e-5f)
                << "row " << n << " step " << t;
        }
    }
}

// ---------------------------------------------------------------------------
// BPTT finite-difference harness. 9 parameter buffers (6 weight matrices + 3 biases)
// flattened into one canonical index order so a single loop can perturb every scalar and
// compare against the matching gradient slot -- the same central-difference discipline
// RNNModuleTest/LSTMModuleTest used.
// ---------------------------------------------------------------------------
struct GRUParams {
    std::array<float, 4> wxz, whz, wxr, whr, wxn, whn;
    std::array<float, 2> bz, br, bn;
};

void configure(GRUModule& m, const GRUParams& p) {
    m.set_weight_xz({p.wxz[0], p.wxz[1], p.wxz[2], p.wxz[3]});
    m.set_weight_hz({p.whz[0], p.whz[1], p.whz[2], p.whz[3]});
    m.set_bias_z({p.bz[0], p.bz[1]});
    m.set_weight_xr({p.wxr[0], p.wxr[1], p.wxr[2], p.wxr[3]});
    m.set_weight_hr({p.whr[0], p.whr[1], p.whr[2], p.whr[3]});
    m.set_bias_r({p.br[0], p.br[1]});
    m.set_weight_xn({p.wxn[0], p.wxn[1], p.wxn[2], p.wxn[3]});
    m.set_weight_hn({p.whn[0], p.whn[1], p.whn[2], p.whn[3]});
    m.set_bias_n({p.bn[0], p.bn[1]});
}

// Canonical flattened order: (Wx*, Wh*, b*) per gate, gates in z, r, n order.
std::vector<float*> param_slots(GRUParams& p) {
    std::vector<float*> slots;
    auto push4 = [&slots](std::array<float, 4>& a) {
        for (float& v : a) slots.push_back(&v);
    };
    auto push2 = [&slots](std::array<float, 2>& a) {
        for (float& v : a) slots.push_back(&v);
    };
    push4(p.wxz);
    push4(p.whz);
    push2(p.bz);
    push4(p.wxr);
    push4(p.whr);
    push2(p.br);
    push4(p.wxn);
    push4(p.whn);
    push2(p.bn);
    return slots;
}

std::vector<float> analytic_grads(const GRUModule& m) {
    std::vector<float> g;
    auto append = [&g](const Tensor& t) {
        for (int64_t i = 0; i < t.numel(); ++i) g.push_back(t.data()[i]);
    };
    append(m.weight_xz_grad());
    append(m.weight_hz_grad());
    append(m.bias_z_grad());
    append(m.weight_xr_grad());
    append(m.weight_hr_grad());
    append(m.bias_r_grad());
    append(m.weight_xn_grad());
    append(m.weight_hn_grad());
    append(m.bias_n_grad());
    return g;
}

float compute_loss(CPUBackend& backend, const GRUParams& p, const Tensor& input, const Tensor& grad_seed) {
    GRUModule gru(2, 2, &backend);
    configure(gru, p);
    Tensor out = gru.forward(input);
    float total = 0.0f;
    for (int64_t i = 0; i < out.numel(); ++i) total += out.data()[i] * grad_seed.data()[i];
    return total;
}

// Deliberately well-separated per-gate weights and biases, applied from the START of this
// mission rather than discovered by mutation testing afterwards (LSTMModule's close-out
// finding): near-zero, near-identical parameters drive every sigmoid gate to
// sigmoid(0)=0.5 with derivative 0.25, which makes a flat-absolute-tolerance gradient check
// blind to a swapped gate derivative. These values keep z_t and r_t at genuinely different,
// non-saturated operating points (z biased low/negative, r biased high/positive).
GRUParams reference_params() {
    GRUParams p;
    p.wxz = {0.80f, -1.20f, 0.60f, 0.90f};
    p.whz = {0.50f, -0.40f, 0.70f, 0.30f};
    p.bz = {-1.10f, 0.90f};
    p.wxr = {-1.00f, 0.70f, 1.30f, -0.50f};
    p.whr = {0.40f, 0.60f, -0.80f, 0.50f};
    p.br = {1.40f, -0.70f};
    p.wxn = {1.10f, 0.50f, -0.90f, 0.80f};
    p.whn = {-0.30f, 0.60f, 0.40f, -0.50f};
    p.bn = {-0.40f, 0.50f};
    return p;
}

// Central differences in float32 on an O(1) loss with h=1e-3 carries roughly 1e-4 of
// absolute noise; everything above that floor is checked relatively so the test's
// sensitivity does not silently degrade with gradient magnitude.
float fd_tolerance(float numeric) { return 1e-3f + 2e-2f * std::fabs(numeric); }

// BPTT correctness through two gates, a reset-gated candidate and the (1-z)/z convex
// carry cannot be reliably hand-derived -- central finite differences over every one of the
// 30 scalar parameters is this mission's actual proof.
TEST_F(GRUModuleTest, BackwardGradientsMatchFiniteDifferences) {
    const float h = 1e-3f;
    Tensor input(Shape({1, 3, 2}), &backend, {0.9f, -1.2f, 1.4f, 0.6f, -0.8f, 1.1f});
    Tensor grad_seed(Shape({1, 3, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f, -0.7f, 0.4f});

    GRUParams params = reference_params();
    GRUModule gru(2, 2, &backend);
    configure(gru, params);
    (void)gru.forward(input);
    (void)gru.backward(grad_seed);

    std::vector<float> analytic = analytic_grads(gru);
    ASSERT_EQ(analytic.size(), 30u);

    for (size_t idx = 0; idx < 30; ++idx) {
        GRUParams plus = params;
        GRUParams minus = params;
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
TEST_F(GRUModuleTest, BackwardInputGradientMatchesFiniteDifferences) {
    const float h = 1e-3f;
    std::vector<float> x = {0.9f, -1.2f, 1.4f, 0.6f, -0.8f, 1.1f};
    Tensor input(Shape({1, 3, 2}), &backend, x);
    Tensor grad_seed(Shape({1, 3, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f, -0.7f, 0.4f});

    GRUParams params = reference_params();
    GRUModule gru(2, 2, &backend);
    configure(gru, params);
    (void)gru.forward(input);
    Tensor grad_input = gru.backward(grad_seed);

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
// degenerate-symmetric-input trap a symmetric setup would hide. GRU has more split points
// than LSTM (the h_{t-1} accumulator is fed by two separate paths), so this is the single
// most load-bearing numeric check in this file.
TEST_F(GRUModuleTest, PropagateRelevanceConservesOnAsymmetricSequence) {
    GRUModule gru(2, 2, &backend);
    configure(gru, reference_params());

    Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
    (void)gru.forward(input);

    Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor relevance_in = gru.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

// Conservation must hold per batch row independently, not merely in aggregate -- an
// implementation that leaked relevance across rows could still balance globally.
TEST_F(GRUModuleTest, PropagateRelevanceConservesPerBatchRow) {
    GRUModule gru(2, 2, &backend);
    configure(gru, reference_params());

    Tensor input(Shape({2, 3, 2}), &backend,
                 {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f, 0.9f, 0.7f, -0.8f, 0.2f, 0.4f, -0.6f});
    (void)gru.forward(input);

    Tensor relevance_out(Shape({2, 3, 2}), &backend,
                         {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -1.0f, 0.4f, 2.5f, 0.3f, 0.6f, 1.1f});
    Tensor relevance_in = gru.propagate_relevance(relevance_out, LRPRuleConfig{});

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

// The gate-signal split is *not* the trivial "pass everything through" rule: relevance must
// actually be distributed across timesteps in proportion to the signal terms. A nonzero
// relevance seeded only at the LAST timestep must still reach earlier timesteps' inputs
// through the carried h_{t-1} accumulator -- proof that BOTH contributions to that
// accumulator (the direct (1-z_t)*h_{t-1} split and the candidate path through
// hn_prev_t = h_{t-1} @ W_hn) are live.
TEST_F(GRUModuleTest, RelevanceSeededAtLastStepReachesEarlierTimesteps) {
    GRUModule gru(2, 2, &backend);
    configure(gru, reference_params());

    Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
    (void)gru.forward(input);

    Tensor relevance_out(Shape({1, 3, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f});
    Tensor relevance_in = gru.propagate_relevance(relevance_out, LRPRuleConfig{});

    float early = std::fabs(relevance_in.at({0, 0, 0})) + std::fabs(relevance_in.at({0, 0, 1})) +
                  std::fabs(relevance_in.at({0, 1, 0})) + std::fabs(relevance_in.at({0, 1, 1}));
    EXPECT_GT(early, 1e-6f);

    // Conservation must still hold with an all-but-last-step-zero seed: this is the case
    // that fails loudly if only one of the two paths into the carried h_{t-1} accumulator
    // is wired up (a single-path implementation silently drops the candidate branch's share
    // of the earlier-timestep relevance).
    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    EXPECT_NEAR(sum_in, 3.0f, 1e-2f);
}

}  // namespace
}  // namespace pulsatrix
