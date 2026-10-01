#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/rope_module.hpp"

namespace pulsatrix {
namespace {

class RoPEModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

// head_dim must be even (the rotation pairs adjacent components) and positive. Both are
// external-boundary constructor arguments -- std::invalid_argument, same convention as
// every other module's constructor checks.
TEST_F(RoPEModuleTest, ConstructorRejectsOddOrNonPositiveHeadDim) {
    EXPECT_THROW({ RoPEModule rope(3, &backend); }, std::invalid_argument);
    EXPECT_THROW({ RoPEModule rope(1, &backend); }, std::invalid_argument);
    EXPECT_THROW({ RoPEModule rope(0, &backend); }, std::invalid_argument);
    EXPECT_THROW({ RoPEModule rope(-2, &backend); }, std::invalid_argument);
    EXPECT_THROW({ RoPEModule rope(-3, &backend); }, std::invalid_argument);
    EXPECT_NO_THROW({ RoPEModule rope(2, &backend); });
    EXPECT_NO_THROW({ RoPEModule rope(8, &backend); });
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// The clean sanity check the mission calls out explicitly: at sequence position 0,
// theta_i = 0 * base^(-2i/head_dim) = 0 for *every* pair, so cos = 1 / sin = 0 and the
// rotation is exactly the identity. Every element must come back bit-identical.
TEST_F(RoPEModuleTest, ForwardAtPositionZeroIsIdentity) {
    RoPEModule rope(6, &backend);
    // A single (L=1, head_dim=6) slice -- position index 0 is the only position present.
    const std::vector<float> values{1.0f, -2.0f, 3.5f, 0.25f, -0.75f, 10.0f};
    Tensor x(Shape({1, 1, 6}), &backend, values);
    Tensor y = rope.forward(x);

    ASSERT_EQ(y.shape(), x.shape());
    for (int64_t i = 0; i < y.numel(); ++i) {
        EXPECT_FLOAT_EQ(y.data()[i], values[static_cast<size_t>(i)]) << "element " << i;
    }
}

// Same identity property must hold for row 0 of a multi-position slice (i.e. the position
// index really is read off the L axis, not hardcoded or shared across the slice).
TEST_F(RoPEModuleTest, ForwardLeavesRowZeroUntouchedInMultiPositionSlice) {
    RoPEModule rope(4, &backend);
    Tensor x(Shape({1, 3, 4}), &backend,
             {1.0f, 2.0f, 3.0f, 4.0f,     // pos 0 -- identity
              1.0f, 2.0f, 3.0f, 4.0f,     // pos 1 -- rotated
              1.0f, 2.0f, 3.0f, 4.0f});   // pos 2 -- rotated further
    Tensor y = rope.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(y.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(y.data()[3], 4.0f);
    // Positions 1 and 2 must have actually changed (otherwise "identity at pos 0" would be
    // trivially satisfied by a do-nothing implementation).
    EXPECT_NE(y.data()[4], 1.0f);
    EXPECT_NE(y.data()[8], 1.0f);
}

// Hand-computed against the RoPE definition, head_dim = 4, base = 10000, pos = 1:
//   pair 0: theta_0 = 1 * 10000^(0)    = 1.0   -> cos = 0.54030230587, sin = 0.84147098481
//   pair 1: theta_1 = 1 * 10000^(-1/2) = 0.01  -> cos = 0.99995000042, sin = 0.00999983333
// x = [1, 2, 3, 4]:
//   y[0] = 1*0.54030230587 - 2*0.84147098481 = -1.14263966375
//   y[1] = 1*0.84147098481 + 2*0.54030230587 =  1.92207559654
//   y[2] = 3*0.99995000042 - 4*0.00999983333 =  2.95985066791
//   y[3] = 3*0.00999983333 + 4*0.99995000042 =  4.02979950167
TEST_F(RoPEModuleTest, ForwardMatchesHandComputedRotationAtPositionOne) {
    RoPEModule rope(4, &backend);
    Tensor x(Shape({1, 2, 4}), &backend,
             {0.0f, 0.0f, 0.0f, 0.0f,    // pos 0 (unused by the assertions below)
              1.0f, 2.0f, 3.0f, 4.0f});  // pos 1
    Tensor y = rope.forward(x);

    EXPECT_NEAR(y.data()[4], -1.14263966375f, 1e-5f);
    EXPECT_NEAR(y.data()[5], 1.92207559654f, 1e-5f);
    EXPECT_NEAR(y.data()[6], 2.95985066791f, 1e-5f);
    EXPECT_NEAR(y.data()[7], 4.02979950167f, 1e-5f);
}

// A rotation preserves the norm of each pair -- an independent, coefficient-free check on
// the forward formula at several positions at once.
TEST_F(RoPEModuleTest, ForwardPreservesPerPairNorm) {
    RoPEModule rope(4, &backend);
    std::vector<float> values(5 * 4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.37f * static_cast<float>(i) - 1.5f;
    }
    Tensor x(Shape({1, 5, 4}), &backend, values);
    Tensor y = rope.forward(x);

    for (int64_t p = 0; p < 10; ++p) {  // 5 positions * 2 pairs
        const float xa = x.data()[2 * p];
        const float xb = x.data()[2 * p + 1];
        const float ya = y.data()[2 * p];
        const float yb = y.data()[2 * p + 1];
        EXPECT_NEAR(ya * ya + yb * yb, xa * xa + xb * xb, 1e-4f) << "pair " << p;
    }
}

// Rank-agnostic over leading dims: the last two axes are (L, head_dim) and everything in
// front is just "more independent slices". A (2, 2, 3, 4) input and a (4, 3, 4) input over
// the same buffer must produce identical numbers -- both are 4 independent (3, 4) slices.
TEST_F(RoPEModuleTest, ForwardIsRankAgnosticOverLeadingDimensions) {
    std::vector<float> values(4 * 3 * 4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.11f * static_cast<float>(i) - 2.0f;
    }

    RoPEModule rope4(4, &backend);
    Tensor x4(Shape({2, 2, 3, 4}), &backend, values);
    Tensor y4 = rope4.forward(x4);

    RoPEModule rope3(4, &backend);
    Tensor x3(Shape({4, 3, 4}), &backend, values);
    Tensor y3 = rope3.forward(x3);

    ASSERT_EQ(y4.numel(), y3.numel());
    for (int64_t i = 0; i < y4.numel(); ++i) {
        EXPECT_FLOAT_EQ(y4.data()[i], y3.data()[i]) << "element " << i;
    }
}

// Slices are independent: the position index restarts at 0 for each (L, head_dim) slice,
// so every slice's row 0 is the identity, not just the very first one.
TEST_F(RoPEModuleTest, ForwardRestartsPositionIndexPerSlice) {
    RoPEModule rope(2, &backend);
    Tensor x(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});
    Tensor y = rope.forward(x);

    // Slice 0, pos 0 and slice 1, pos 0 are both identity rows.
    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(y.data()[4], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[5], 2.0f);
    // Both slices' pos-1 rows get the identical (pos=1) rotation.
    EXPECT_FLOAT_EQ(y.data()[2], y.data()[6]);
    EXPECT_FLOAT_EQ(y.data()[3], y.data()[7]);
}

// A non-default base changes the per-pair frequencies (pair 0 is always base^0 = 1 and so
// is unaffected; pair 1 is base^(-1/2) and must differ).
TEST_F(RoPEModuleTest, ForwardHonoursNonDefaultBase) {
    RoPEModule rope_default(4, &backend);
    RoPEModule rope_other(4, &backend, 100.0f);
    Tensor x(Shape({1, 2, 4}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    Tensor yd = rope_default.forward(x);
    Tensor yo = rope_other.forward(x);

    // Pair 0 is base-independent (theta_0 = pos * base^0 = pos).
    EXPECT_FLOAT_EQ(yd.data()[4], yo.data()[4]);
    EXPECT_FLOAT_EQ(yd.data()[5], yo.data()[5]);
    // Pair 1 is not: 10000^(-1/2) = 0.01 vs 100^(-1/2) = 0.1.
    EXPECT_NE(yd.data()[6], yo.data()[6]);
}

TEST_F(RoPEModuleTest, ForwardRejectsEmptyInput) {
    RoPEModule rope(4, &backend);
    Tensor empty(Shape({0}), &backend);
    EXPECT_THROW({ (void)rope.forward(empty); }, std::invalid_argument);
}

TEST_F(RoPEModuleTest, ForwardRejectsRankBelowTwoOrWrongLastDimension) {
    RoPEModule rope(4, &backend);
    Tensor rank1(Shape({4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW({ (void)rope.forward(rank1); }, std::invalid_argument);

    Tensor wrong_last(Shape({1, 2, 6}), &backend);
    EXPECT_THROW({ (void)rope.forward(wrong_last); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

// The mission explicitly refuses to accept the orthogonality argument on its own: verify
// the analytic inverse-rotation backward against central finite differences of
// f(x) = sum_k(grad_out[k] * rope(x)[k]).
TEST_F(RoPEModuleTest, BackwardMatchesCentralFiniteDifferences) {
    const std::vector<float> base_values{0.5f,  -1.2f, 2.0f,  0.1f,  0.7f, -0.3f,
                                         -0.9f, 1.4f,  0.25f, -0.6f, 1.1f, 0.05f};
    const std::vector<float> grad_out_values{1.0f, -2.0f, 0.5f,  0.25f, 3.0f,  -1.5f,
                                             0.8f, 0.2f,  -0.4f, 1.7f,  -0.1f, 2.2f};
    const Shape shape({1, 3, 4});

    RoPEModule rope(4, &backend);
    Tensor x(shape, &backend, base_values);
    (void)rope.forward(x);
    Tensor grad_out(shape, &backend, grad_out_values);
    Tensor grad_in = rope.backward(grad_out);

    const float h = 1e-3f;
    for (size_t i = 0; i < base_values.size(); ++i) {
        std::vector<float> plus = base_values;
        std::vector<float> minus = base_values;
        plus[i] += h;
        minus[i] -= h;

        RoPEModule probe(4, &backend);
        Tensor xp(shape, &backend, plus);
        Tensor yp = probe.forward(xp);
        Tensor xm(shape, &backend, minus);
        Tensor ym = probe.forward(xm);

        float fp = 0.0f;
        float fm = 0.0f;
        for (size_t k = 0; k < base_values.size(); ++k) {
            fp += grad_out_values[k] * yp.data()[static_cast<int64_t>(k)];
            fm += grad_out_values[k] * ym.data()[static_cast<int64_t>(k)];
        }
        const float numeric = (fp - fm) / (2.0f * h);
        EXPECT_NEAR(grad_in.data()[static_cast<int64_t>(i)], numeric, 1e-3f) << "element " << i;
    }
}

// Structural consequence of the same fact, checked independently: backward applied to a
// forward output recovers the original input (R^T R = I).
TEST_F(RoPEModuleTest, BackwardOfForwardOutputRecoversInput) {
    RoPEModule rope(4, &backend);
    std::vector<float> values(3 * 4);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.29f * static_cast<float>(i) - 1.3f;
    }
    Tensor x(Shape({1, 3, 4}), &backend, values);
    Tensor y = rope.forward(x);
    Tensor recovered = rope.backward(y);

    for (int64_t i = 0; i < recovered.numel(); ++i) {
        EXPECT_NEAR(recovered.data()[i], values[static_cast<size_t>(i)], 1e-5f) << "element " << i;
    }
}

TEST_F(RoPEModuleTest, BackwardBeforeForwardThrows) {
    RoPEModule rope(4, &backend);
    Tensor grad_out(Shape({1, 2, 4}), &backend);
    EXPECT_THROW({ (void)rope.backward(grad_out); }, std::logic_error);
}

TEST_F(RoPEModuleTest, BackwardRejectsShapeMismatch) {
    RoPEModule rope(4, &backend);
    Tensor x(Shape({1, 2, 4}), &backend);
    (void)rope.forward(x);
    Tensor grad_out(Shape({1, 3, 4}), &backend);
    EXPECT_THROW({ (void)rope.backward(grad_out); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP
// ---------------------------------------------------------------------------

// Hand-worked epsilon rule for a single pair, head_dim = 2, pos = 1 (theta = 1.0),
// x = [1, 2], R_out = [1.0, 0.5], epsilon = 1e-6:
//   c = 0.54030230587, s = 0.84147098481
//   y[0] = 1*c - 2*s = -1.14263966375  -> denom0 = y[0] - 1e-6
//   y[1] = 1*s + 2*c =  1.92207559654  -> denom1 = y[1] + 1e-6
//   R_x[0] = (x0*c  / denom0)*R_out[0] + (x0*s / denom1)*R_out[1] = -0.25395776598
//   R_x[1] = (-x1*s / denom0)*R_out[0] + (x1*c / denom1)*R_out[1] =  1.75395663068
// Note both R_x entries are two-term sums -- an implementation that overwrote instead of
// accumulating would land on one of the two terms alone and fail this test outright.
TEST_F(RoPEModuleTest, PropagateRelevanceMatchesHandWorkedEpsilonRule) {
    RoPEModule rope(2, &backend);
    Tensor x(Shape({1, 2, 2}), &backend, {0.0f, 0.0f, 1.0f, 2.0f});
    (void)rope.forward(x);

    Tensor relevance_out(Shape({1, 2, 2}), &backend, {0.0f, 0.0f, 1.0f, 0.5f});
    Tensor relevance_in = rope.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_NEAR(relevance_in.data()[2], -0.25395776598f, 1e-5f);
    EXPECT_NEAR(relevance_in.data()[3], 1.75395663068f, 1e-5f);
}

// Targeted structural proof of the two-contribution accumulation, independent of any
// hand-computed constant: the rule is linear in R_out (the denominators depend only on the
// cached forward pass), so propagating [a, 0] and [0, b] separately and summing must equal
// propagating [a, b] in one go. An implementation where the second redistribution
// *overwrote* the first would give R_x = (contribution from R_out[1] only) for the combined
// call, which does not equal the sum of the two separate calls.
TEST_F(RoPEModuleTest, PropagateRelevanceAccumulatesBothOutputContributions) {
    const Shape shape({1, 2, 2});
    const std::vector<float> x_values{0.0f, 0.0f, 1.0f, 2.0f};

    RoPEModule rope_a(2, &backend);
    Tensor xa(shape, &backend, x_values);
    (void)rope_a.forward(xa);
    Tensor only_first(shape, &backend, {0.0f, 0.0f, 1.0f, 0.0f});
    Tensor r_first = rope_a.propagate_relevance(only_first, LRPRuleConfig{});

    RoPEModule rope_b(2, &backend);
    Tensor xb(shape, &backend, x_values);
    (void)rope_b.forward(xb);
    Tensor only_second(shape, &backend, {0.0f, 0.0f, 0.0f, 0.5f});
    Tensor r_second = rope_b.propagate_relevance(only_second, LRPRuleConfig{});

    RoPEModule rope_c(2, &backend);
    Tensor xc(shape, &backend, x_values);
    (void)rope_c.forward(xc);
    Tensor both(shape, &backend, {0.0f, 0.0f, 1.0f, 0.5f});
    Tensor r_both = rope_c.propagate_relevance(both, LRPRuleConfig{});

    // Both single-source runs must be individually nonzero on BOTH x-components -- i.e.
    // each output component genuinely feeds both inputs of its pair.
    EXPECT_GT(std::abs(r_first.data()[2]), 1e-4f);
    EXPECT_GT(std::abs(r_first.data()[3]), 1e-4f);
    EXPECT_GT(std::abs(r_second.data()[2]), 1e-4f);
    EXPECT_GT(std::abs(r_second.data()[3]), 1e-4f);

    EXPECT_NEAR(r_both.data()[2], r_first.data()[2] + r_second.data()[2], 1e-5f);
    EXPECT_NEAR(r_both.data()[3], r_first.data()[3] + r_second.data()[3], 1e-5f);
}

// Conservation: this is a plain weighted-connection epsilon rule over a bias-free linear
// map, so sum(R_in) == sum(R_out) up to the epsilon stabilizer -- checked here on a
// multi-slice, multi-position, multi-pair tensor.
TEST_F(RoPEModuleTest, PropagateRelevanceConservesRelevance) {
    RoPEModule rope(4, &backend);
    std::vector<float> x_values(2 * 3 * 4);
    std::vector<float> r_values(2 * 3 * 4);
    for (size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = 0.41f * static_cast<float>(i) - 3.0f;
        r_values[i] = 0.17f * static_cast<float>(i) - 1.0f;
    }
    Tensor x(Shape({2, 3, 4}), &backend, x_values);
    (void)rope.forward(x);
    Tensor relevance_out(Shape({2, 3, 4}), &backend, r_values);
    Tensor relevance_in = rope.propagate_relevance(relevance_out, LRPRuleConfig{});

    double sum_in = 0.0;
    double sum_out = 0.0;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        sum_in += relevance_in.data()[i];
        sum_out += relevance_out.data()[i];
    }
    EXPECT_NEAR(sum_in, sum_out, 1e-3);
}

// Conservation is per-pair, not merely global: each pair's two inputs receive exactly that
// pair's two outputs' relevance (nothing leaks across pairs or positions).
TEST_F(RoPEModuleTest, PropagateRelevanceConservesWithinEachPair) {
    RoPEModule rope(4, &backend);
    std::vector<float> x_values(3 * 4);
    std::vector<float> r_values(3 * 4);
    for (size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = 0.53f * static_cast<float>(i) - 2.2f;
        r_values[i] = 0.31f * static_cast<float>(i) - 0.7f;
    }
    Tensor x(Shape({1, 3, 4}), &backend, x_values);
    (void)rope.forward(x);
    Tensor relevance_out(Shape({1, 3, 4}), &backend, r_values);
    Tensor relevance_in = rope.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t p = 0; p < 6; ++p) {  // 3 positions * 2 pairs
        const float in_sum = relevance_in.data()[2 * p] + relevance_in.data()[2 * p + 1];
        const float out_sum = relevance_out.data()[2 * p] + relevance_out.data()[2 * p + 1];
        EXPECT_NEAR(in_sum, out_sum, 1e-3f) << "pair " << p;
    }
}

// At position 0 the rotation is the identity, so the epsilon rule degenerates to
// R_in == R_out elementwise (weight cos = 1 carries everything, weight sin = 0 carries
// nothing).
TEST_F(RoPEModuleTest, PropagateRelevanceAtPositionZeroIsIdentity) {
    RoPEModule rope(4, &backend);
    Tensor x(Shape({1, 1, 4}), &backend, {1.0f, -2.0f, 3.0f, 0.5f});
    (void)rope.forward(x);
    Tensor relevance_out(Shape({1, 1, 4}), &backend, {1.0f, 2.0f, -0.5f, 0.25f});
    Tensor relevance_in = rope.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_NEAR(relevance_in.data()[0], 1.0f, 1e-4f);
    EXPECT_NEAR(relevance_in.data()[1], 2.0f, 1e-4f);
    EXPECT_NEAR(relevance_in.data()[2], -0.5f, 1e-4f);
    EXPECT_NEAR(relevance_in.data()[3], 0.25f, 1e-4f);
}

TEST_F(RoPEModuleTest, PropagateRelevanceBeforeForwardThrows) {
    RoPEModule rope(4, &backend);
    Tensor relevance_out(Shape({1, 2, 4}), &backend);
    EXPECT_THROW({ (void)rope.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(RoPEModuleTest, PropagateRelevanceRejectsShapeMismatch) {
    RoPEModule rope(4, &backend);
    Tensor x(Shape({1, 2, 4}), &backend);
    (void)rope.forward(x);
    Tensor relevance_out(Shape({1, 3, 4}), &backend);
    EXPECT_THROW({ (void)rope.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Module contract
// ---------------------------------------------------------------------------

TEST_F(RoPEModuleTest, HasNoLearnableParameters) {
    RoPEModule rope(4, &backend);
    EXPECT_TRUE(rope.parameters().empty());
}

TEST_F(RoPEModuleTest, OpTypeIsElementwise) {
    RoPEModule rope(4, &backend);
    EXPECT_EQ(rope.op_type(), OpType::Elementwise);
}

// ---------------------------------------------------------------------------
// Death tests (exactly 3 -- forward / backward / propagate_relevance)
// ---------------------------------------------------------------------------

using RoPEModuleDeathTest = RoPEModuleTest;

TEST_F(RoPEModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RoPEModule rope(2, &backend);
    Tensor x(Shape({1, 1, 2}), &backend, {1.0f, 2.0f});
    (void)rope.forward(x);

    Tensor relevance_out(Shape({1, 1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)rope.propagate_relevance(relevance_out, LRPRuleConfig{}); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
