#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/retnet_module.hpp"

namespace exai {
namespace {

class RetNetModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(RetNetModuleTest, ConstructionThrowsOnZeroDModel) {
    EXPECT_THROW({ RetNetModule retnet(0, 2, 0.7f, &backend); }, std::invalid_argument);
}

TEST_F(RetNetModuleTest, ConstructionThrowsOnNegativeDModel) {
    EXPECT_THROW({ RetNetModule retnet(-3, 2, 0.7f, &backend); }, std::invalid_argument);
}

TEST_F(RetNetModuleTest, ConstructionThrowsOnZeroKeyDim) {
    EXPECT_THROW({ RetNetModule retnet(2, 0, 0.7f, &backend); }, std::invalid_argument);
}

TEST_F(RetNetModuleTest, ConstructionThrowsOnNegativeKeyDim) {
    EXPECT_THROW({ RetNetModule retnet(2, -1, 0.7f, &backend); }, std::invalid_argument);
}

// gamma is deliberately unconstrained (documented scope cut, mirroring MambaModule's A):
// out-of-range values construct fine rather than throwing.
TEST_F(RetNetModuleTest, ConstructionAcceptsOutOfRangeGamma) {
    EXPECT_NO_THROW({ RetNetModule retnet(2, 2, 1.5f, &backend); });
    EXPECT_NO_THROW({ RetNetModule retnet(2, 2, -0.3f, &backend); });
    RetNetModule retnet(2, 2, 1.5f, &backend);
    EXPECT_FLOAT_EQ(retnet.gamma(), 1.5f);
}

TEST_F(RetNetModuleTest, ForwardThrowsOnWrongRank) {
    RetNetModule retnet(2, 3, 0.7f, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)retnet.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(RetNetModuleTest, ForwardThrowsOnMismatchedDModel) {
    RetNetModule retnet(2, 3, 0.7f, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)retnet.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(RetNetModuleTest, BackwardThrowsIfCalledBeforeForward) {
    RetNetModule retnet(1, 2, 0.7f, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)retnet.backward(grad); }, std::logic_error);
}

TEST_F(RetNetModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    RetNetModule retnet(1, 2, 0.7f, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)retnet.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)retnet.backward(wrong_shape_grad); }, std::invalid_argument);
}

// Three parameter tensors only -- gamma is a plain float hyperparameter, NOT a learned
// Tensor, so it is deliberately absent here (this module's parameter set is strictly
// smaller than RWKVModule's or MambaModule's; see the header's note).
TEST_F(RetNetModuleTest, ParametersExposesExactlyThreeTensorsAndNotGamma) {
    RetNetModule retnet(2, 3, 0.7f, &backend);
    auto params = retnet.parameters();
    ASSERT_EQ(params.size(), 3u);
    EXPECT_EQ(params[0].value->shape(), Shape({2, 3}));  // W_Q
    EXPECT_EQ(params[1].value->shape(), Shape({2, 3}));  // W_K
    EXPECT_EQ(params[2].value->shape(), Shape({2, 2}));  // W_V
    for (const auto& p : params) {
        EXPECT_EQ(p.grad->shape(), p.value->shape());
    }
}

TEST_F(RetNetModuleTest, OpTypeIsRecurrent) {
    RetNetModule retnet(2, 3, 0.7f, &backend);
    EXPECT_EQ(retnet.op_type(), OpType::Recurrent);
}

// Independent reference recomputation of the retention recurrence, written straight from the
// equations in fully unrolled scalar form rather than reusing the module's own loop/indexing
// -- catches shape/indexing bugs, not just formula bugs baked into both. d_model = 1 with
// key_dim = 2 deliberately: the two dimensions differ, so a transposed or conflated
// (d_model, key_dim) index shows up here instead of being masked by a square shape. Same
// discipline as RNNModuleTest::ForwardMatchesReferenceRecurrence and RWKVModuleTest's own.
TEST_F(RetNetModuleTest, ForwardMatchesReferenceRecurrence) {
    const float gamma = 0.7f;
    RetNetModule retnet(1, 2, gamma, &backend);
    retnet.set_W_Q({0.5f, -0.3f});   // (d_model=1, key_dim=2)
    retnet.set_W_K({0.8f, 0.25f});   // (d_model=1, key_dim=2)
    retnet.set_W_V({1.5f});          // (d_model=1, d_model=1)

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, -0.5f});
    Tensor output = retnet.forward(input);

    // t = 0: S_0 = 0.
    const float x0 = 1.0f;
    const float q0_0 = x0 * 0.5f;
    const float q0_1 = x0 * -0.3f;
    const float k0_0 = x0 * 0.8f;
    const float k0_1 = x0 * 0.25f;
    const float v0 = x0 * 1.5f;
    const float s0_0 = gamma * 0.0f + k0_0 * v0;
    const float s0_1 = gamma * 0.0f + k0_1 * v0;
    const float o0 = q0_0 * s0_0 + q0_1 * s0_1;

    // t = 1: the gamma-decayed carry of S_0 plus this token's own outer product.
    const float x1 = -0.5f;
    const float q1_0 = x1 * 0.5f;
    const float q1_1 = x1 * -0.3f;
    const float k1_0 = x1 * 0.8f;
    const float k1_1 = x1 * 0.25f;
    const float v1 = x1 * 1.5f;
    const float s1_0 = gamma * s0_0 + k1_0 * v1;
    const float s1_1 = gamma * s0_1 + k1_1 * v1;
    const float o1 = q1_0 * s1_0 + q1_1 * s1_1;

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), o0, 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), o1, 1e-5f);
}

// N = 2 -- proves each batch row is scanned independently (the retention state carry never
// bleeds across rows).
TEST_F(RetNetModuleTest, ForwardHandlesMultiBatch) {
    auto configure = [](RetNetModule& m) {
        m.set_W_Q({0.5f, -0.3f});
        m.set_W_K({0.8f, 0.25f});
        m.set_W_V({1.5f});
    };

    RetNetModule retnet(1, 2, 0.7f, &backend);
    configure(retnet);
    Tensor two_row(Shape({2, 2, 1}), &backend, {1.0f, -0.5f, 0.4f, 0.9f});
    Tensor two_row_out = retnet.forward(two_row);

    RetNetModule single(1, 2, 0.7f, &backend);
    configure(single);
    Tensor row1(Shape({1, 2, 1}), &backend, {0.4f, 0.9f});
    Tensor row1_out = single.forward(row1);

    EXPECT_NEAR(two_row_out.at({1, 0, 0}), row1_out.at({0, 0, 0}), 1e-6f);
    EXPECT_NEAR(two_row_out.at({1, 1, 0}), row1_out.at({0, 1, 0}), 1e-6f);
}

// gamma = 0 kills the carry outright, so S_t collapses to the bare outer product
// K_t (x) V_t and the readout becomes o_t[j] = (Q_t . K_t) * V_t[j] -- a closed form with no
// recurrence left in it. At d_model = 2, key_dim = 3 (deliberately non-square, deliberately
// key_dim > d_model) this pins the contraction orientation: the Q/K dot product must run
// over the key index i and V must be indexed by the value index j. A module that transposed
// those two would fail here even though the square-shaped reference test above could not
// see it.
TEST_F(RetNetModuleTest, ZeroGammaCollapsesToQDotKTimesV) {
    RetNetModule retnet(2, 3, 0.0f, &backend);
    retnet.set_W_Q({0.37f, -0.62f, 0.18f, 0.45f, 0.83f, -0.26f});
    retnet.set_W_K({0.54f, -0.28f, 0.41f, 0.66f, -0.73f, 0.19f});
    retnet.set_W_V({-0.35f, 0.72f, 0.59f, -0.16f});

    Tensor input(Shape({1, 2, 2}), &backend, {0.80f, -0.60f, 1.20f, 0.50f});
    Tensor output = retnet.forward(input);

    const float wq[2][3] = {{0.37f, -0.62f, 0.18f}, {0.45f, 0.83f, -0.26f}};
    const float wk[2][3] = {{0.54f, -0.28f, 0.41f}, {0.66f, -0.73f, 0.19f}};
    const float wv[2][2] = {{-0.35f, 0.72f}, {0.59f, -0.16f}};
    const float x[2][2] = {{0.80f, -0.60f}, {1.20f, 0.50f}};

    for (int t = 0; t < 2; ++t) {
        float qk = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const float q_i = x[t][0] * wq[0][i] + x[t][1] * wq[1][i];
            const float k_i = x[t][0] * wk[0][i] + x[t][1] * wk[1][i];
            qk += q_i * k_i;
        }
        for (int j = 0; j < 2; ++j) {
            const float v_j = x[t][0] * wv[0][j] + x[t][1] * wv[1][j];
            EXPECT_NEAR(output.at({0, static_cast<int64_t>(t), static_cast<int64_t>(j)}), qk * v_j, 1e-5f)
                << "t " << t << " j " << j;
        }
    }
    // ...and the collapsed outputs are not all trivially zero, so the check above cannot be
    // satisfied by a module that simply returns zeros.
    EXPECT_GT(std::fabs(output.at({0, 0, 0})) + std::fabs(output.at({0, 0, 1})), 1e-4f);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient checks.
//
// This BPTT threads a (key_dim, d_model) state-gradient accumulator backwards through the
// gamma-decayed carry and splits it across three separate projections that all land on the
// same grad_input slot; past L = 1 it cannot be reliably hand-derived, so finite differences
// are this mission's actual proof of the backward pass.
//
// The fixture below is deliberately GENEROUS and non-uniform from the start rather than
// small-and-uniform: RWKVModule's mission found the hard way that a small-scale, uniform
// fixture can make individual parameter checks pass *vacuously* against the tolerance floor
// (a wrong gradient still lands inside 1e-3 when every gradient is ~1e-3). So: L = 4,
// key_dim (3) != d_model (2), sign-alternating tokens of magnitude 0.4-1.3, well-separated
// weights, and a sign-alternating grad seed. Tolerance is relative (1e-3 + 2e-2*|numeric|)
// for the same established reason -- a fixed absolute tolerance is either vacuous for small
// gradients or spuriously tight for large ones. Non-vacuity is verified by mutation probe
// (see the mission's close-out evidence): scaling any single gradient term in backward()
// fails these checks.
// ---------------------------------------------------------------------------------------
namespace {

constexpr int64_t kD = 2;
constexpr int64_t kKey = 3;
constexpr float kGamma = 0.7f;

struct RetNetParams {
    std::vector<float> w_q{0.37f, -0.62f, 0.18f, 0.45f, 0.83f, -0.26f};
    std::vector<float> w_k{0.54f, -0.28f, 0.41f, 0.66f, -0.73f, 0.19f};
    std::vector<float> w_v{-0.35f, 0.72f, 0.59f, -0.16f};
};

void apply_params(RetNetModule& m, const RetNetParams& p) {
    m.set_W_Q(p.w_q);
    m.set_W_K(p.w_k);
    m.set_W_V(p.w_v);
}

Tensor fd_input(CPUBackend& backend) {
    return Tensor(Shape({1, 4, kD}), &backend, {0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f, 0.40f, -1.30f});
}

Tensor fd_grad_seed(CPUBackend& backend) {
    return Tensor(Shape({1, 4, kD}), &backend, {1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f, 0.70f, -0.40f});
}

float scalar_loss(CPUBackend& backend, const RetNetParams& p, const Tensor& input, const Tensor& grad_seed) {
    RetNetModule m(kD, kKey, kGamma, &backend);
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
void check_param_gradient(CPUBackend& backend, std::vector<float> RetNetParams::* field,
                          const Tensor& analytic_grad, const char* name) {
    const float h = 1e-3f;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);
    RetNetParams base;

    const size_t n = (base.*field).size();
    ASSERT_EQ(static_cast<int64_t>(n), analytic_grad.numel()) << name;
    for (size_t idx = 0; idx < n; ++idx) {
        RetNetParams plus = base;
        RetNetParams minus = base;
        (plus.*field)[idx] += h;
        (minus.*field)[idx] -= h;
        float numeric =
            (scalar_loss(backend, plus, input, grad_seed) - scalar_loss(backend, minus, input, grad_seed)) /
            (2.0f * h);
        EXPECT_NEAR(analytic_grad.data()[idx], numeric, relative_tolerance(numeric)) << name << " index " << idx;
    }
}

}  // namespace

TEST_F(RetNetModuleTest, BackwardGradientMatchesFiniteDifferenceForWQ) {
    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, RetNetParams{});
    (void)retnet.forward(fd_input(backend));
    (void)retnet.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RetNetParams::w_q, retnet.W_Q_grad(), "W_Q");
}

TEST_F(RetNetModuleTest, BackwardGradientMatchesFiniteDifferenceForWK) {
    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, RetNetParams{});
    (void)retnet.forward(fd_input(backend));
    (void)retnet.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RetNetParams::w_k, retnet.W_K_grad(), "W_K");
}

TEST_F(RetNetModuleTest, BackwardGradientMatchesFiniteDifferenceForWV) {
    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, RetNetParams{});
    (void)retnet.forward(fd_input(backend));
    (void)retnet.backward(fd_grad_seed(backend));
    check_param_gradient(backend, &RetNetParams::w_v, retnet.W_V_grad(), "W_V");
}

// All three projections contribute to the same grad_input element, and the state carry makes
// every earlier token's gradient depend on every later one, so the input gradient gets its
// own check rather than being assumed correct from the parameter checks.
TEST_F(RetNetModuleTest, BackwardGradientMatchesFiniteDifferenceForInput) {
    const float h = 1e-3f;
    RetNetParams p;
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, p);
    (void)retnet.forward(input);
    Tensor grad_input = retnet.backward(grad_seed);

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

// Multi-batch backward -- the state-gradient carry is per batch row, so a row-index bug in
// the reverse scan would show up as a cross-row contamination the single-row finite-
// difference checks above cannot see.
TEST_F(RetNetModuleTest, BackwardGradientIsPerBatchRowIndependent) {
    RetNetParams p;

    RetNetModule two_row(kD, kKey, kGamma, &backend);
    apply_params(two_row, p);
    Tensor batched(Shape({2, 2, kD}), &backend, {0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f, 0.40f, -1.30f});
    (void)two_row.forward(batched);
    Tensor batched_grad_in =
        two_row.backward(Tensor(Shape({2, 2, kD}), &backend, {1.00f, -0.50f, 0.30f, 0.80f, 0.0f, 0.0f, 0.0f, 0.0f}));

    // Row 1's grad seed is all zeros above, so its input gradient must be exactly zero and
    // row 0's must equal what the same row produces on its own.
    RetNetModule single(kD, kKey, kGamma, &backend);
    apply_params(single, p);
    Tensor row0(Shape({1, 2, kD}), &backend, {0.80f, -0.60f, 1.20f, 0.50f});
    (void)single.forward(row0);
    Tensor row0_grad_in = single.backward(Tensor(Shape({1, 2, kD}), &backend, {1.00f, -0.50f, 0.30f, 0.80f}));

    for (int64_t i = 0; i < row0_grad_in.numel(); ++i) {
        EXPECT_NEAR(batched_grad_in.data()[i], row0_grad_in.data()[i], 1e-5f) << "row 0 index " << i;
        EXPECT_NEAR(batched_grad_in.data()[row0_grad_in.numel() + i], 0.0f, 1e-6f) << "row 1 index " << i;
    }
}

TEST_F(RetNetModuleTest, BackwardAccumulatesGradientsAcrossCalls) {
    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, RetNetParams{});
    Tensor input = fd_input(backend);
    Tensor grad_seed = fd_grad_seed(backend);

    (void)retnet.forward(input);
    (void)retnet.backward(grad_seed);
    std::vector<float> after_one(static_cast<size_t>(retnet.W_K_grad().numel()));
    for (int64_t i = 0; i < retnet.W_K_grad().numel(); ++i) {
        after_one[static_cast<size_t>(i)] = retnet.W_K_grad().data()[i];
    }

    (void)retnet.forward(input);
    (void)retnet.backward(grad_seed);
    for (int64_t i = 0; i < retnet.W_K_grad().numel(); ++i) {
        EXPECT_NEAR(retnet.W_K_grad().data()[i], 2.0f * after_one[static_cast<size_t>(i)], 1e-5f)
            << "W_K index " << i;
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
TEST_F(RetNetModuleTest, PropagateRelevanceThrowsNotYetImplemented) {
    RetNetModule retnet(kD, kKey, kGamma, &backend);
    apply_params(retnet, RetNetParams{});
    Tensor input = fd_input(backend);
    (void)retnet.forward(input);

    Tensor relevance_out(Shape({1, 4, kD}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f, -1.2f, 0.7f});
    try {
        (void)retnet.propagate_relevance(relevance_out, LRPRuleConfig{});
        FAIL() << "propagate_relevance returned instead of throwing -- this module has no LRP rule by design";
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(),
                     "RetNetModule::propagate_relevance: LRP rule not yet implemented -- see campaign Decision "
                     "Point 2");
    }
}

// ...and it fires even before any forward() call, since there is no cached state it could
// possibly need. A rule-bearing module would throw the "called before any forward()"
// logic_error here instead; the message assertion pins which of the two this is.
TEST_F(RetNetModuleTest, PropagateRelevanceThrowsNotYetImplementedEvenBeforeForward) {
    RetNetModule retnet(1, 2, kGamma, &backend);
    Tensor relevance_out(Shape({1, 2, 1}), &backend, {1.0f, 1.0f});
    try {
        (void)retnet.propagate_relevance(relevance_out, LRPRuleConfig{});
        FAIL() << "propagate_relevance returned instead of throwing";
    } catch (const std::logic_error& e) {
        EXPECT_STREQ(e.what(),
                     "RetNetModule::propagate_relevance: LRP rule not yet implemented -- see campaign Decision "
                     "Point 2");
    }
}

using RetNetModuleDeathTest = RetNetModuleTest;

// forward_impl/backward/propagate_relevance all inspect Tensor::device() and (the first two)
// dereference Tensor::data() in raw host loops for the state recurrence -- undefined
// behavior on a CUDA-backed Tensor. See RNNModuleDeathTest for the mislabeled-Tensor testing
// pattern this reuses. Written from the start of this mission, not deferred.
TEST_F(RetNetModuleDeathTest, ForwardAbortsOnNonCpuInput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RetNetModule retnet(1, 2, kGamma, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)retnet.forward(input); }, "EXAI_ASSERT failed");
}

TEST_F(RetNetModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RetNetModule retnet(1, 2, kGamma, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)retnet.forward(input);

    Tensor grad_output(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)retnet.backward(grad_output); }, "EXAI_ASSERT failed");
}

// The device guard sits AHEAD of propagate_relevance's unconditional throw (see the header's
// note), so a CUDA-backed relevance tensor must abort rather than throw -- this test would
// fail with "threw std::logic_error" if the guard were ever reordered behind the throw.
TEST_F(RetNetModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RetNetModule retnet(1, 2, kGamma, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)retnet.forward(input);

    Tensor relevance_out(Shape({1, 2, 1}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)retnet.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
