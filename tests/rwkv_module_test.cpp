#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/rwkv_module.hpp"

namespace exai {
namespace {

class RWKVModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(RWKVModuleTest, ConstructionThrowsOnZeroDModel) {
    EXPECT_THROW({ RWKVModule rwkv(0, &backend); }, std::invalid_argument);
}

TEST_F(RWKVModuleTest, ConstructionThrowsOnNegativeDModel) {
    EXPECT_THROW({ RWKVModule rwkv(-3, &backend); }, std::invalid_argument);
}

TEST_F(RWKVModuleTest, ForwardThrowsOnWrongRank) {
    RWKVModule rwkv(2, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)rwkv.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(RWKVModuleTest, ForwardThrowsOnMismatchedDModel) {
    RWKVModule rwkv(2, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)rwkv.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(RWKVModuleTest, BackwardThrowsIfCalledBeforeForward) {
    RWKVModule rwkv(1, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)rwkv.backward(grad); }, std::logic_error);
}

TEST_F(RWKVModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    RWKVModule rwkv(1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)rwkv.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)rwkv.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(RWKVModuleTest, ParametersExposesAllNineTensors) {
    RWKVModule rwkv(3, &backend);
    auto params = rwkv.parameters();
    ASSERT_EQ(params.size(), 9u);
    EXPECT_EQ(params[0].value->shape(), Shape({3, 3}));  // W_r
    EXPECT_EQ(params[1].value->shape(), Shape({3, 3}));  // W_k
    EXPECT_EQ(params[2].value->shape(), Shape({3, 3}));  // W_v
    EXPECT_EQ(params[3].value->shape(), Shape({3, 3}));  // W_o
    EXPECT_EQ(params[4].value->shape(), Shape({3}));     // w
    EXPECT_EQ(params[5].value->shape(), Shape({3}));     // u
    EXPECT_EQ(params[6].value->shape(), Shape({3}));     // mu_r
    EXPECT_EQ(params[7].value->shape(), Shape({3}));     // mu_k
    EXPECT_EQ(params[8].value->shape(), Shape({3}));     // mu_v
    for (const auto& p : params) {
        EXPECT_EQ(p.grad->shape(), p.value->shape());
    }
}

TEST_F(RWKVModuleTest, OpTypeIsRecurrent) {
    RWKVModule rwkv(2, &backend);
    EXPECT_EQ(rwkv.op_type(), OpType::Recurrent);
}

// Independent reference recomputation of the WKV recurrence, written straight from the
// equations in scalar form (d_model = 1) rather than reusing the module's own loop/indexing
// -- catches shape/indexing bugs, not just formula bugs baked into both. Same discipline as
// RNNModuleTest::ForwardMatchesReferenceRecurrence and MambaModuleTest's own.
TEST_F(RWKVModuleTest, ForwardMatchesReferenceRecurrence) {
    RWKVModule rwkv(1, &backend);
    rwkv.set_W_r({0.5f});
    rwkv.set_W_k({0.8f});
    rwkv.set_W_v({1.5f});
    rwkv.set_W_o({0.3f});
    rwkv.set_w({0.7f});
    rwkv.set_u({-0.2f});
    rwkv.set_mu_r({0.6f});
    rwkv.set_mu_k({0.4f});
    rwkv.set_mu_v({0.9f});

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, -0.5f});
    Tensor output = rwkv.forward(input);

    auto sigmoid_ref = [](float z) { return 1.0f / (1.0f + std::exp(-z)); };
    const float decay = std::exp(-0.7f);

    // t = 0: x_{-1} = 0, a_0 = b_0 = 0.
    const float x0 = 1.0f;
    const float xr0 = 0.6f * x0;
    const float xk0 = 0.4f * x0;
    const float xv0 = 0.9f * x0;
    const float r0 = sigmoid_ref(xr0 * 0.5f);
    const float k0 = xk0 * 0.8f;
    const float v0 = xv0 * 1.5f;
    const float e0 = std::exp(-0.2f + k0);
    const float wkv0 = (0.0f + e0 * v0) / (0.0f + e0);
    const float o0 = (r0 * wkv0) * 0.3f;
    const float a0 = decay * 0.0f + std::exp(k0) * v0;
    const float b0 = decay * 0.0f + std::exp(k0);

    // t = 1: the token-shift now mixes in x_0.
    const float x1 = -0.5f;
    const float xr1 = 0.6f * x1 + 0.4f * x0;
    const float xk1 = 0.4f * x1 + 0.6f * x0;
    const float xv1 = 0.9f * x1 + 0.1f * x0;
    const float r1 = sigmoid_ref(xr1 * 0.5f);
    const float k1 = xk1 * 0.8f;
    const float v1 = xv1 * 1.5f;
    const float e1 = std::exp(-0.2f + k1);
    const float wkv1 = (a0 + e1 * v1) / (b0 + e1);
    const float o1 = (r1 * wkv1) * 0.3f;

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), o0, 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), o1, 1e-5f);
}

// N = 2 -- proves each batch row is scanned independently (neither the a/b state carry nor
// the token-shift ever bleeds across rows).
TEST_F(RWKVModuleTest, ForwardHandlesMultiBatch) {
    auto configure = [](RWKVModule& m) {
        m.set_W_r({0.5f});
        m.set_W_k({0.8f});
        m.set_W_v({1.5f});
        m.set_W_o({0.3f});
        m.set_w({0.7f});
        m.set_u({-0.2f});
        m.set_mu_r({0.6f});
        m.set_mu_k({0.4f});
        m.set_mu_v({0.9f});
    };

    RWKVModule rwkv(1, &backend);
    configure(rwkv);
    Tensor two_row(Shape({2, 2, 1}), &backend, {1.0f, -0.5f, 0.4f, 0.9f});
    Tensor two_row_out = rwkv.forward(two_row);

    RWKVModule single(1, &backend);
    configure(single);
    Tensor row1(Shape({1, 2, 1}), &backend, {0.4f, 0.9f});
    Tensor row1_out = single.forward(row1);

    EXPECT_NEAR(two_row_out.at({1, 0, 0}), row1_out.at({0, 0, 0}), 1e-6f);
    EXPECT_NEAR(two_row_out.at({1, 1, 0}), row1_out.at({0, 1, 0}), 1e-6f);
}

// mu = 0 makes every token-shift mix select the PREVIOUS token outright, and x_{-1} == 0
// (this module's documented zero-init convention). So at t = 0 all three projections see an
// all-zero input: k_0 = v_0 = 0, wkv_0 = (e_0*0)/e_0 = 0, and the output is exactly zero --
// a hand-checkable degenerate case that pins the token-shift's orientation (mu weights the
// CURRENT token, 1-mu the previous one) and the zero-init convention at once.
TEST_F(RWKVModuleTest, ZeroMixRatiosSelectPreviousTokenSoFirstStepOutputIsZero) {
    RWKVModule rwkv(2, &backend);
    rwkv.set_W_r({0.37f, -0.62f, 0.18f, 0.45f});
    rwkv.set_W_k({0.54f, -0.28f, 0.41f, 0.66f});
    rwkv.set_W_v({-0.35f, 0.72f, 0.59f, -0.16f});
    rwkv.set_W_o({0.62f, -0.41f, 0.25f, 0.88f});
    rwkv.set_w({0.30f, 0.75f});
    rwkv.set_u({-0.20f, 0.45f});
    rwkv.set_mu_r({0.0f, 0.0f});
    rwkv.set_mu_k({0.0f, 0.0f});
    rwkv.set_mu_v({0.0f, 0.0f});

    Tensor input(Shape({1, 3, 2}), &backend, {0.30f, -0.20f, 0.60f, 0.10f, -0.40f, 0.50f});
    Tensor output = rwkv.forward(input);

    EXPECT_NEAR(output.at({0, 0, 0}), 0.0f, 1e-6f);
    EXPECT_NEAR(output.at({0, 0, 1}), 0.0f, 1e-6f);
    // Later timesteps are NOT zero -- otherwise the test above would pass on a module that
    // simply returns zeros everywhere.
    EXPECT_GT(std::fabs(output.at({0, 1, 0})) + std::fabs(output.at({0, 1, 1})), 1e-4f);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient checks.
//
// This BPTT differentiates through the sigmoid receptance gate, through both exp()
// nonlinearities, through the num/den quotient, through the decayed a/b state carry and
// through three separate token-shift mixes; it cannot be reliably hand-derived past L = 1,
// so finite differences are this mission's actual proof of the backward pass.
//
// The weights below are deliberately WELL-SEPARATED and non-uniform, not RNN-scale uniform
// values -- LSTMModule's/GRUModule's missions established that uniform weights park every
// channel at the same operating point and make the check vacuous (it passes even against a
// wrong-but-symmetric gradient). Tolerance is relative (1e-3 + 2e-2*|numeric|) for the same
// established reason: a fixed absolute tolerance is either vacuous for small gradients or
// spuriously tight for large ones.
// ---------------------------------------------------------------------------------------
namespace {

constexpr int64_t kD = 2;

struct RWKVParams {
    std::vector<float> w_r{0.37f, -0.62f, 0.18f, 0.45f};
    std::vector<float> w_k{0.54f, -0.28f, 0.41f, 0.66f};
    std::vector<float> w_v{-0.35f, 0.72f, 0.59f, -0.16f};
    std::vector<float> w_o{0.62f, -0.41f, 0.25f, 0.88f};
    std::vector<float> w{0.30f, 0.75f};
    std::vector<float> u{-0.20f, 0.45f};
    std::vector<float> mu_r{0.65f, 0.35f};
    std::vector<float> mu_k{0.40f, 0.80f};
    std::vector<float> mu_v{0.55f, 0.25f};
};

void apply_params(RWKVModule& m, const RWKVParams& p) {
    m.set_W_r(p.w_r);
    m.set_W_k(p.w_k);
    m.set_W_v(p.w_v);
    m.set_W_o(p.w_o);
    m.set_w(p.w);
    m.set_u(p.u);
    m.set_mu_r(p.mu_r);
    m.set_mu_k(p.mu_k);
    m.set_mu_v(p.mu_v);
}

// L = 4 with large, sign-alternating, well-separated tokens. Deliberately NOT the small
// 0.1-0.6-scale values MambaModuleTest uses: this module's decay/bonus gradients (dw, du)
// only accumulate through the a/b state carry, which is exactly zero at t = 0 and stays
// tiny under small k_t -- at small scale those two checks pass vacuously against the
// tolerance floor. Verified by mutation probe: a blanket 1.2x scaling of the incoming
// gradient fails all ten checks at this scale, but left dw and dmu_k green at the smaller
// one.
Tensor fd_input(CPUBackend& backend) {
    return Tensor(Shape({1, 4, kD}), &backend, {0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f, 0.40f, -1.30f});
}

Tensor fd_grad_seed(CPUBackend& backend) {
    return Tensor(Shape({1, 4, kD}), &backend, {1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f, 0.70f, -0.40f});
}

float scalar_loss(CPUBackend& backend, const RWKVParams& p, const Tensor& input, const Tensor& grad_seed) {
    RWKVModule m(kD, &backend);
    apply_params(m, p);
    Tensor out = m.forward(input);
    float total = 0.0f;
    for (int64_t i = 0; i < out.numel(); ++i) {
        total += out.data()[i] * grad_seed.data()[i];
    }
    return total;
}

float relative_tolerance(float numeric) {
    return 1e-3f + 2e-2f * std::fabs(numeric);
}

// Runs the analytic-vs-numeric comparison over every element of one parameter tensor.
void check_param_gradient(CPUBackend& backend, std::vector<float> RWKVParams::* field,
                          const Tensor& analytic_grad, const char* name) {
    const float h = 1e-3f;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    RWKVParams base;

    const size_t n = (base.*field).size();
    ASSERT_EQ(static_cast<int64_t>(n), analytic_grad.numel()) << name;
    for (size_t idx = 0; idx < n; ++idx) {
        RWKVParams plus = base;
        RWKVParams minus = base;
        (plus.*field)[idx] += h;
        (minus.*field)[idx] -= h;
        float numeric = (scalar_loss(backend, plus, input, grad_seed) -
                         scalar_loss(backend, minus, input, grad_seed)) /
                        (2.0f * h);
        EXPECT_NEAR(analytic_grad.data()[idx], numeric, relative_tolerance(numeric)) << name << " index " << idx;
    }
}

}  // namespace

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForWR) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::w_r, rwkv.W_r_grad(), "W_r");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForWK) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::w_k, rwkv.W_k_grad(), "W_k");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForWV) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::w_v, rwkv.W_v_grad(), "W_v");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForWO) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::w_o, rwkv.W_o_grad(), "W_o");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForDecayW) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::w, rwkv.w_grad(), "w");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForBonusU) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::u, rwkv.u_grad(), "u");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForMuR) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::mu_r, rwkv.mu_r_grad(), "mu_r");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForMuK) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::mu_k, rwkv.mu_k_grad(), "mu_k");
}

TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForMuV) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    (void)rwkv.forward(fd_input(backend));
    (void)rwkv.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RWKVParams::mu_v, rwkv.mu_v_grad(), "mu_v");
}

// The input gradient threads all three token-shift mixes at BOTH the t and t-1 slots (six
// separate contributions to the same grad_input element), so it gets its own check rather
// than being assumed correct from the parameter checks.
TEST_F(RWKVModuleTest, BackwardGradientMatchesFiniteDifferenceForInput) {
    const float h = 1e-3f;
    RWKVParams p;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, p);
    (void)rwkv.forward(input);
    Tensor grad_input = rwkv.backward(grad_seed);

    std::vector<float> base_values(static_cast<size_t>(input.numel()));
    for (int64_t i = 0; i < input.numel(); ++i) {
        base_values[static_cast<size_t>(i)] = input.data()[i];
    }

    for (int64_t idx = 0; idx < input.numel(); ++idx) {
        std::vector<float> plus = base_values;
        std::vector<float> minus = base_values;
        plus[static_cast<size_t>(idx)] += h;
        minus[static_cast<size_t>(idx)] -= h;
        Tensor input_plus(input.shape(), &backend, plus);
        Tensor input_minus(input.shape(), &backend, minus);
        float numeric = (scalar_loss(backend, p, input_plus, grad_seed) -
                         scalar_loss(backend, p, input_minus, grad_seed)) /
                        (2.0f * h);
        EXPECT_NEAR(grad_input.data()[idx], numeric, relative_tolerance(numeric)) << "input index " << idx;
    }
}

TEST_F(RWKVModuleTest, BackwardAccumulatesGradientsAcrossCalls) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    (void)rwkv.forward(input);
    (void)rwkv.backward(grad_seed);
    std::vector<float> after_one(static_cast<size_t>(rwkv.W_k_grad().numel()));
    for (int64_t i = 0; i < rwkv.W_k_grad().numel(); ++i) {
        after_one[static_cast<size_t>(i)] = rwkv.W_k_grad().data()[i];
    }

    (void)rwkv.forward(input);
    (void)rwkv.backward(grad_seed);
    for (int64_t i = 0; i < rwkv.W_k_grad().numel(); ++i) {
        EXPECT_NEAR(rwkv.W_k_grad().data()[i], 2.0f * after_one[static_cast<size_t>(i)], 1e-5f) << "W_k index " << i;
    }
}

// ---------------------------------------------------------------------------------------
// The deliberate, logged absence of an LRP rule.
// ---------------------------------------------------------------------------------------

// This module ships without a relevance rule under the campaign's Phase 5 charter deviation
// (operator-directed, see the class-level note). The throw is the *contract*, not a gap: it
// is tested here so the deviation stays visible and intentional, and so that quietly
// replacing the throw with zeros or an approximation would break the suite. Also confirms
// the throw is unconditional -- it fires on a perfectly well-formed, post-forward call.
TEST_F(RWKVModuleTest, PropagateRelevanceThrowsNotYetImplemented) {
    RWKVModule rwkv(kD, &backend);
    apply_params(rwkv, RWKVParams{});
    Tensor input = fd_input(backend);
    (void)rwkv.forward(input);

    Tensor relevance_out(Shape({1, 4, kD}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -1.2f, 0.7f});
    try {
        (void)rwkv.propagate_relevance(relevance_out, LRPRuleConfig{});
        FAIL() << "propagate_relevance returned instead of throwing -- this module has no LRP rule by design";
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(),
                     "RWKVModule::propagate_relevance: LRP rule not yet implemented -- see campaign Decision "
                     "Point 2");
    }
}

// ...and it fires even before any forward() call, since there is no cached state it could
// possibly need. A rule-bearing module would throw the "called before any forward()"
// logic_error here instead; the message assertion pins which of the two this is.
TEST_F(RWKVModuleTest, PropagateRelevanceThrowsNotYetImplementedEvenBeforeForward) {
    RWKVModule rwkv(1, &backend);
    Tensor relevance_out(Shape({1, 2, 1}), &backend, {1.0f, 1.0f});
    try {
        (void)rwkv.propagate_relevance(relevance_out, LRPRuleConfig{});
        FAIL() << "propagate_relevance returned instead of throwing";
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(),
                     "RWKVModule::propagate_relevance: LRP rule not yet implemented -- see campaign Decision "
                     "Point 2");
    }
}

using RWKVModuleDeathTest = RWKVModuleTest;

// forward_impl/backward/propagate_relevance all inspect Tensor::device() and (the first two)
// dereference Tensor::data() in raw host loops (exp/sigmoid have no DeviceBackend
// primitive) -- undefined behavior on a CUDA-backed Tensor. See RNNModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses. Written from the start of this mission, not
// deferred.
TEST_F(RWKVModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RWKVModule rwkv(1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)rwkv.forward(input); }, "EXAI_ASSERT failed");
}

TEST_F(RWKVModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RWKVModule rwkv(1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)rwkv.forward(input);

    Tensor grad_output(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)rwkv.backward(grad_output); }, "EXAI_ASSERT failed");
}

// The device guard sits AHEAD of propagate_relevance's unconditional throw (see the header's
// note), so a CUDA-backed relevance tensor must abort rather than throw -- this test would
// fail with "threw std::logic_error" if the guard were ever reordered behind the throw.
TEST_F(RWKVModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RWKVModule rwkv(1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)rwkv.forward(input);

    Tensor relevance_out(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)rwkv.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
