#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/mamba_module.hpp"

namespace pulsatrix {
namespace {

class MambaModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(MambaModuleTest, ConstructionThrowsOnZeroDModel) {
    EXPECT_THROW({ MambaModule mamba(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(MambaModuleTest, ConstructionThrowsOnNegativeStateSize) {
    EXPECT_THROW({ MambaModule mamba(2, -1, &backend); }, std::invalid_argument);
}

TEST_F(MambaModuleTest, ForwardThrowsOnWrongRank) {
    MambaModule mamba(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)mamba.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(MambaModuleTest, ForwardThrowsOnMismatchedDModel) {
    MambaModule mamba(2, 2, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)mamba.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(MambaModuleTest, BackwardThrowsIfCalledBeforeForward) {
    MambaModule mamba(1, 1, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)mamba.backward(grad); }, std::logic_error);
}

TEST_F(MambaModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    MambaModule mamba(1, 1, &backend);
    Tensor relevance(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)mamba.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(MambaModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    MambaModule mamba(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)mamba.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)mamba.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(MambaModuleTest, PropagateRelevanceThrowsOnShapeMismatch) {
    MambaModule mamba(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)mamba.forward(input);
    Tensor wrong_shape_relevance(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)mamba.propagate_relevance(wrong_shape_relevance, LRPRuleConfig{}); },
                 std::invalid_argument);
}

TEST_F(MambaModuleTest, ParametersExposesAllSixTensors) {
    MambaModule mamba(3, 4, &backend);
    auto params = mamba.parameters();
    ASSERT_EQ(params.size(), 6u);
    EXPECT_EQ(params[0].value->shape(), Shape({3, 3}));   // W_delta
    EXPECT_EQ(params[1].value->shape(), Shape({3}));      // bias_delta
    EXPECT_EQ(params[2].value->shape(), Shape({3, 4}));   // W_B
    EXPECT_EQ(params[3].value->shape(), Shape({3, 4}));   // W_C
    EXPECT_EQ(params[4].value->shape(), Shape({3, 4}));   // A
    EXPECT_EQ(params[5].value->shape(), Shape({3}));      // D
    for (const auto& p : params) {
        EXPECT_EQ(p.grad->shape(), p.value->shape());
    }
}

TEST_F(MambaModuleTest, OpTypeIsRecurrent) {
    MambaModule mamba(2, 2, &backend);
    EXPECT_EQ(mamba.op_type(), OpType::Recurrent);
}

// Independent reference recomputation of the selective scan, written straight from the
// equations in scalar form (d_model = state_size = 1) rather than reusing the module's own
// loop/indexing -- catches shape/indexing bugs, not just formula bugs baked into both.
// Same discipline as RNNModuleTest::ForwardMatchesReferenceRecurrence.
TEST_F(MambaModuleTest, ForwardMatchesReferenceRecurrence) {
    MambaModule mamba(1, 1, &backend);
    mamba.set_W_delta({0.5f});
    mamba.set_bias_delta({0.2f});
    mamba.set_W_B({0.8f});
    mamba.set_W_C({1.5f});
    mamba.set_A({-0.7f});
    mamba.set_D({0.3f});

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, -0.5f});
    Tensor output = mamba.forward(input);

    auto softplus_ref = [](float z) { return std::log(1.0f + std::exp(z)); };

    // t = 0, h_{-1} = 0.
    float x0 = 1.0f;
    float delta0 = softplus_ref(x0 * 0.5f + 0.2f);
    float b0 = x0 * 0.8f;
    float c0 = x0 * 1.5f;
    float abar0 = std::exp(delta0 * -0.7f);
    float bbar0 = delta0 * b0;
    float h0 = abar0 * 0.0f + bbar0 * x0;
    float y0 = c0 * h0 + 0.3f * x0;

    // t = 1.
    float x1 = -0.5f;
    float delta1 = softplus_ref(x1 * 0.5f + 0.2f);
    float b1 = x1 * 0.8f;
    float c1 = x1 * 1.5f;
    float abar1 = std::exp(delta1 * -0.7f);
    float bbar1 = delta1 * b1;
    float h1 = abar1 * h0 + bbar1 * x1;
    float y1 = c1 * h1 + 0.3f * x1;

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), y0, 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), y1, 1e-5f);
}

// N = 2 -- proves each batch row is scanned independently (the state carry never bleeds
// across rows).
TEST_F(MambaModuleTest, ForwardHandlesMultiBatch) {
    MambaModule mamba(1, 1, &backend);
    mamba.set_W_delta({0.5f});
    mamba.set_bias_delta({0.2f});
    mamba.set_W_B({0.8f});
    mamba.set_W_C({1.5f});
    mamba.set_A({-0.7f});
    mamba.set_D({0.3f});

    Tensor two_row(Shape({2, 2, 1}), &backend, {1.0f, -0.5f, 0.4f, 0.9f});
    Tensor two_row_out = mamba.forward(two_row);

    MambaModule single(1, 1, &backend);
    single.set_W_delta({0.5f});
    single.set_bias_delta({0.2f});
    single.set_W_B({0.8f});
    single.set_W_C({1.5f});
    single.set_A({-0.7f});
    single.set_D({0.3f});
    Tensor row1(Shape({1, 2, 1}), &backend, {0.4f, 0.9f});
    Tensor row1_out = single.forward(row1);

    EXPECT_NEAR(two_row_out.at({1, 0, 0}), row1_out.at({0, 0, 0}), 1e-6f);
    EXPECT_NEAR(two_row_out.at({1, 1, 0}), row1_out.at({0, 1, 0}), 1e-6f);
}

// A zero A matrix makes Abar = exp(0) = 1, i.e. a pure running sum of Bbar*x -- a hand-
// checkable degenerate case that pins the discretization's orientation (Abar multiplies
// the *previous* state, Bbar the *current* input).
TEST_F(MambaModuleTest, ZeroStateMatrixGivesRunningSumRecurrence) {
    MambaModule mamba(1, 1, &backend);
    mamba.set_W_delta({0.0f});
    mamba.set_bias_delta({0.0f});  // Delta = softplus(0) = ln 2, constant across timesteps
    mamba.set_W_B({1.0f});
    mamba.set_W_C({1.0f});
    mamba.set_A({0.0f});
    mamba.set_D({0.0f});

    Tensor input(Shape({1, 3, 1}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor output = mamba.forward(input);

    const float delta = std::log(2.0f);
    // h_t = h_{t-1} + (Delta*B_t)*x_t, with B_t = x_t, so h_t = Delta * sum_{u<=t} x_u^2,
    // and y_t = C_t * h_t = x_t * h_t.
    float h = 0.0f;
    for (int64_t t = 0; t < 3; ++t) {
        float x = static_cast<float>(t + 1);
        h += delta * x * x;
        EXPECT_NEAR(output.at({0, t, 0}), x * h, 1e-4f) << "t = " << t;
    }
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient checks.
//
// This BPTT differentiates through softplus, through exp(Delta*A) and through all three
// selective projections; it cannot be reliably hand-derived past L = 1, so finite
// differences are this mission's actual proof of the backward pass.
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
constexpr int64_t kS = 2;

struct MambaParams {
    std::vector<float> w_delta{0.37f, -0.62f, 0.18f, 0.45f};
    std::vector<float> bias_delta{-0.21f, 0.33f};
    std::vector<float> w_b{0.54f, -0.28f, 0.41f, 0.66f};
    std::vector<float> w_c{-0.35f, 0.72f, 0.59f, -0.16f};
    std::vector<float> a{-0.85f, -0.30f, -0.55f, -1.20f};
    std::vector<float> d{0.62f, -0.41f};
};

void apply_params(MambaModule& m, const MambaParams& p) {
    m.set_W_delta(p.w_delta);
    m.set_bias_delta(p.bias_delta);
    m.set_W_B(p.w_b);
    m.set_W_C(p.w_c);
    m.set_A(p.a);
    m.set_D(p.d);
}

Tensor fd_input(CPUBackend& backend) {
    return Tensor(Shape({1, 3, kD}), &backend, {0.30f, -0.20f, 0.60f, 0.10f, -0.40f, 0.50f});
}

Tensor fd_grad_seed(CPUBackend& backend) {
    return Tensor(Shape({1, 3, kD}), &backend, {1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f});
}

float scalar_loss(CPUBackend& backend, const MambaParams& p, const Tensor& input, const Tensor& grad_seed) {
    MambaModule m(kD, kS, &backend);
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
void check_param_gradient(CPUBackend& backend, std::vector<float> MambaParams::* field,
                          const Tensor& analytic_grad, const char* name) {
    const float h = 1e-3f;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    MambaParams base;

    const size_t n = (base.*field).size();
    ASSERT_EQ(static_cast<int64_t>(n), analytic_grad.numel()) << name;
    for (size_t idx = 0; idx < n; ++idx) {
        MambaParams plus = base;
        MambaParams minus = base;
        (plus.*field)[idx] += h;
        (minus.*field)[idx] -= h;
        float numeric = (scalar_loss(backend, plus, input, grad_seed) -
                         scalar_loss(backend, minus, input, grad_seed)) /
                        (2.0f * h);
        EXPECT_NEAR(analytic_grad.data()[idx], numeric, relative_tolerance(numeric))
            << name << " index " << idx;
    }
}

}  // namespace

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForWDelta) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::w_delta, mamba.W_delta_grad(), "W_delta");
}

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForBiasDelta) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::bias_delta, mamba.bias_delta_grad(), "bias_delta");
}

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForWB) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::w_b, mamba.W_B_grad(), "W_B");
}

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForWC) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::w_c, mamba.W_C_grad(), "W_C");
}

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForA) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::a, mamba.A_grad(), "A");
}

TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForD) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    check_param_gradient(backend, &MambaParams::d, mamba.D_grad(), "D");
}

// The input gradient threads BOTH the skip path (D*x_t) and all three selective
// projections plus Bbar_t*x_t -- four separate contributions to the same dx_t element, so
// it gets its own check rather than being assumed correct from the parameter checks.
TEST_F(MambaModuleTest, BackwardGradientMatchesFiniteDifferenceForInput) {
    const float h = 1e-3f;
    MambaParams p;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, p);
    (void)mamba.forward(input);
    Tensor grad_input = mamba.backward(grad_seed);

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

TEST_F(MambaModuleTest, BackwardAccumulatesGradientsAcrossCalls) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    std::vector<float> after_one(static_cast<size_t>(mamba.A_grad().numel()));
    for (int64_t i = 0; i < mamba.A_grad().numel(); ++i) {
        after_one[static_cast<size_t>(i)] = mamba.A_grad().data()[i];
    }

    (void)mamba.forward(input);
    (void)mamba.backward(grad_seed);
    for (int64_t i = 0; i < mamba.A_grad().numel(); ++i) {
        EXPECT_NEAR(mamba.A_grad().data()[i], 2.0f * after_one[static_cast<size_t>(i)], 1e-5f) << "A index " << i;
    }
}

// ---------------------------------------------------------------------------------------
// MambaLRP relevance propagation.
// ---------------------------------------------------------------------------------------

// Stage 2's prediction was that this module's conservation gap lands in the "near-exact,
// epsilon-stabilizer-only" category (like RoPEModule/SwiGLUModule), NOT SoftmaxModule's
// large by-design DTD gap -- restoring conservation that naive LRP breaks is MambaLRP's
// whole point, so a large gap here would be an implementation bug, not a property. This
// test quantifies the gap rather than merely asserting pass/fail, per this project's
// "quantify, don't just assert" standard.
TEST_F(MambaModuleTest, PropagateRelevanceConservationGapIsMeasuredNotAssumed) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});

    // Deliberately asymmetric across both timesteps and channels -- a symmetric input can
    // make an unconserved rule look conserved by cancellation.
    Tensor input = fd_input(backend);
    (void)mamba.forward(input);

    Tensor relevance_out(Shape({1, 3, kD}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor relevance_in = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];

    const float gap = std::fabs(sum_in - sum_out);
    RecordProperty("conservation_gap", std::to_string(gap));
    RecordProperty("sum_relevance_out", std::to_string(sum_out));

    // Near-exact category, as Stage 2 predicted: the measured gap is 2.5e-5 against a
    // sum(R_out) of 5.3 -- 4.7e-6 relative, epsilon-stabilizer residual only. Contrast
    // MultiHeadAttentionModule's measured 2.3546 against 4.75 (~50%, by design). The guard
    // below is set two orders of magnitude above the measured value: tight enough that a
    // genuine conservation break trips it, loose enough not to be a float-noise tripwire.
    EXPECT_LT(gap, 1e-3f) << "measured conservation gap " << gap << " against sum(R_out) " << sum_out
                          << " -- Stage 2 predicted the near-exact category; a large gap here means a bug";
}

// Per-batch-row conservation: a rule that conserves only in aggregate (by cancellation
// across rows) would pass the total-sum test and fail this one.
TEST_F(MambaModuleTest, PropagateRelevanceConservesPerBatchRow) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});

    Tensor input(Shape({2, 3, kD}), &backend,
                 {0.30f, -0.20f, 0.60f, 0.10f, -0.40f, 0.50f, 0.85f, 0.15f, -0.70f, 0.25f, 0.45f, -0.35f});
    (void)mamba.forward(input);

    Tensor relevance_out(Shape({2, 3, kD}), &backend,
                         {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -1.2f, 0.7f, 2.2f, 0.4f, -0.6f, 1.1f});
    Tensor relevance_in = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t b = 0; b < 2; ++b) {
        float row_in = 0.0f;
        float row_out = 0.0f;
        for (int64_t t = 0; t < 3; ++t) {
            for (int64_t d = 0; d < kD; ++d) {
                row_in += relevance_in.at({b, t, d});
                row_out += relevance_out.at({b, t, d});
            }
        }
        EXPECT_NEAR(row_in, row_out, 1e-2f) << "batch row " << b;
    }
}

// Seeding relevance only at the LAST timestep must still reach earlier timesteps -- that is
// the entire point of threading the state-relevance accumulator backward through the scan.
// A rule that dropped the carry would return zeros everywhere except t = L-1.
TEST_F(MambaModuleTest, RelevanceSeededAtLastStepReachesEarlierTimesteps) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});

    Tensor input = fd_input(backend);
    (void)mamba.forward(input);

    Tensor relevance_out(Shape({1, 3, kD}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f});
    Tensor relevance_in = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    float earlier_mass = 0.0f;
    for (int64_t t = 0; t < 2; ++t) {
        for (int64_t d = 0; d < kD; ++d) {
            earlier_mass += std::fabs(relevance_in.at({0, t, d}));
        }
    }
    EXPECT_GT(earlier_mass, 1e-3f) << "relevance seeded at t = 2 never reached t = 0/1 -- the carried "
                                      "state-relevance accumulator is not being threaded";

    // And it still conserves under this sparse seed.
    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    EXPECT_NEAR(sum_in, 3.0f, 1e-2f);
}

// MambaLRP's central claim, made behavioral: the selective projections' weights
// (W_delta/bias_delta/W_B/W_C) are never consulted during relevance redistribution --
// Delta_t/B_t/C_t are pure conductors, entering only through the cached, DETACHED
// Abar_t/Bbar_t/C_t forward values. Mutating all four after forward() must therefore leave
// propagate_relevance's output bit-identical. (Grep-confirmed in the source as well, per
// the mission's verification checklist; this is the runtime half of that check.)
TEST_F(MambaModuleTest, SelectiveProjectionWeightsNeverInfluenceRelevance) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});

    Tensor input = fd_input(backend);
    (void)mamba.forward(input);

    Tensor relevance_out(Shape({1, 3, kD}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor before = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    // Wildly different selective-projection weights, applied AFTER the forward pass.
    mamba.set_W_delta({-9.0f, 4.0f, 7.0f, -2.0f});
    mamba.set_bias_delta({5.0f, -6.0f});
    mamba.set_W_B({3.0f, -8.0f, 1.0f, 2.0f});
    mamba.set_W_C({-4.0f, 6.0f, -3.0f, 9.0f});

    Tensor after = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < before.numel(); ++i) {
        EXPECT_FLOAT_EQ(after.data()[i], before.data()[i]) << "index " << i;
    }
}

// The complement of the test above: D *is* a genuine weighted connection from x_t to y_t
// (the skip path), so it must be consulted -- this pins that the test above is proving
// detachment, not just that propagate_relevance ignores its parameters wholesale.
TEST_F(MambaModuleTest, SkipWeightDoesInfluenceRelevance) {
    MambaModule mamba(kD, kS, &backend);
    apply_params(mamba, MambaParams{});

    Tensor input = fd_input(backend);
    (void)mamba.forward(input);

    Tensor relevance_out(Shape({1, 3, kD}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor before = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    mamba.set_D({5.0f, -7.0f});
    Tensor after = mamba.propagate_relevance(relevance_out, LRPRuleConfig{});

    float max_delta = 0.0f;
    for (int64_t i = 0; i < before.numel(); ++i) {
        max_delta = std::max(max_delta, std::fabs(after.data()[i] - before.data()[i]));
    }
    EXPECT_GT(max_delta, 1e-4f);
}

}  // namespace
}  // namespace pulsatrix
