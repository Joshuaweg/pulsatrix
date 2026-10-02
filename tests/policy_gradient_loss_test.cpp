#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"

namespace pulsatrix {
namespace {

// Float32 softmax/log arithmetic is not exactly representable, so the numeric assertions below
// are tolerance claims against hand-computed constants (unlike DQNLoss's, whose fixture is
// exactly representable in binary32). 1e-6 is far tighter than any plausible formula error and
// far looser than float32's own rounding on these magnitudes.
constexpr float kTol = 1e-6f;

class PolicyGradientLossTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The one hand-derived fixture every numeric assertion below refers back to.
    //
    //   logits = [[1.0, 2.0, 3.0],     actions = [[2.0],    returns = [[ 2.0],
    //             [1.0, 1.0, 1.0]]                [0.0]]               [-3.0]]
    //
    // Row 0 -- stable softmax, max_logit = 3:
    //   exp(1-3) = 0.1353352832366127
    //   exp(2-3) = 0.3678794411714423
    //   exp(3-3) = 1.0
    //   exp_sum  = 1.5032147244080550,  log(exp_sum) = 0.4076059644443806
    //   p[0,:]   = [0.0900305731703805, 0.2447284710547976, 0.6652409557748219]
    //   log_softmax[0,:] = logit - 3 - 0.4076059644443806
    //                    = [-2.4076059644443806, -1.4076059644443806, -0.4076059644443806]
    //   a_0 = 2, G_0 = 2.0
    //   loss_0 = -log_softmax[0,2] * 2.0 = 0.4076059644443806 * 2.0 = 0.8152119288887612
    //
    // Row 1 -- all logits equal, so the distribution is exactly uniform:
    //   p[1,:] = [1/3, 1/3, 1/3],  log_softmax[1,:] = -log(3) = -1.0986122886681098
    //   a_1 = 0, G_1 = -3.0
    //   loss_1 = -(-1.0986122886681098) * (-3.0) = -3.2958368660043294
    //
    //   loss = (0.8152119288887612 + (-3.2958368660043294)) / 2 = -1.2403124685577841
    //
    // The loss is *negative*: a negative return makes that step's contribution negative, which
    // is correct for a policy-gradient surrogate and carries no information about policy
    // quality. That sign is the point of putting a negative return in the fixture at all.
    //
    // Gradient, grad[b,k] = G_b * (p[b,k] - 1{k == a_b}) / N,  N = 2:
    //   row 0 (G = 2, a = 2, scale 2/2 = 1):
    //     grad[0,0] = 1.0 * 0.0900305731703805               =  0.0900305731703805
    //     grad[0,1] = 1.0 * 0.2447284710547976               =  0.2447284710547976
    //     grad[0,2] = 1.0 * (0.6652409557748219 - 1.0)       = -0.3347590442251781
    //   row 1 (G = -3, a = 0, scale -3/2 = -1.5):
    //     grad[1,0] = -1.5 * (1/3 - 1.0) = -1.5 * (-2/3)     =  1.0
    //     grad[1,1] = -1.5 * (1/3)                           = -0.5
    //     grad[1,2] = -1.5 * (1/3)                           = -0.5
    //
    // Note every single entry is non-zero -- including columns 0 and 1 of row 0 and columns 1
    // and 2 of row 1, none of which is the action that row took. That density is the defining
    // structural difference from DQNLoss's masked gradient and is asserted on directly below.
    Tensor logits() { return Tensor(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f}); }
    Tensor actions() { return Tensor(Shape({2, 1}), &backend, {2.0f, 0.0f}); }
    Tensor returns() { return Tensor(Shape({2, 1}), &backend, {2.0f, -3.0f}); }
};

TEST_F(PolicyGradientLossTest, ForwardMatchesTheHandDerivedLoss) {
    PolicyGradientLoss loss(&backend);

    EXPECT_NEAR(loss.forward(logits(), actions(), returns()), -1.2403124685577841f, kTol);
}

TEST_F(PolicyGradientLossTest, BackwardMatchesTheHandDerivedGradientTensorElementwise) {
    PolicyGradientLoss loss(&backend);
    (void)loss.forward(logits(), actions(), returns());

    const Tensor grad = loss.backward();

    ASSERT_EQ(grad.rank(), 2);
    ASSERT_EQ(grad.shape().dim(0), 2);
    ASSERT_EQ(grad.shape().dim(1), 3);

    const float expected[6] = {0.0900305731703805f,  0.2447284710547976f, -0.3347590442251781f,
                               1.0f,                 -0.5f,               -0.5f};
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad.data()[i], expected[i], kTol) << "flat index " << i;
    }
}

// Spelled out separately from the elementwise check above so a failure says *which* property
// broke. This is the one detail most likely to be implemented wrong by analogy to DQNLoss,
// whose non-selected columns are exactly 0.0f: here they must not be, because softmax is
// normalized and probability mass pushed onto the taken action has to come off every other
// action. A masked implementation would pass a "selected column is right" test and fail here.
TEST_F(PolicyGradientLossTest, BackwardIsDenseAcrossEveryNonSelectedActionColumn) {
    PolicyGradientLoss loss(&backend);
    (void)loss.forward(logits(), actions(), returns());

    const Tensor grad = loss.backward();

    // Row 0 took action 2, so columns 0 and 1 are the non-selected ones -- and both carry the
    // real, positive `G * p / N` push-down, not zero.
    EXPECT_NE(grad.data()[0], 0.0f) << "non-selected column was masked to zero, DQN-style";
    EXPECT_NE(grad.data()[1], 0.0f) << "non-selected column was masked to zero, DQN-style";
    EXPECT_NEAR(grad.data()[0], 0.0900305731703805f, kTol);
    EXPECT_NEAR(grad.data()[1], 0.2447284710547976f, kTol);
    // Row 1 took action 0, so columns 1 and 2 are the non-selected ones.
    EXPECT_NE(grad.data()[4], 0.0f) << "non-selected column was masked to zero, DQN-style";
    EXPECT_NE(grad.data()[5], 0.0f) << "non-selected column was masked to zero, DQN-style";
    EXPECT_NEAR(grad.data()[4], -0.5f, kTol);
    EXPECT_NEAR(grad.data()[5], -0.5f, kTol);
}

// The structural invariant behind the density: a softmax row's gradient sums to
// G_b * (sum_k p[b,k] - 1) / N = 0, because the probabilities sum to 1. A masked (DQN-style)
// gradient would fail this on every row with a non-zero return.
TEST_F(PolicyGradientLossTest, EachGradientRowSumsToZero) {
    PolicyGradientLoss loss(&backend);
    (void)loss.forward(logits(), actions(), returns());

    const Tensor grad = loss.backward();

    for (int64_t b = 0; b < 2; ++b) {
        float row_sum = 0.0f;
        for (int64_t k = 0; k < 3; ++k) {
            row_sum += grad.data()[b * 3 + k];
        }
        EXPECT_NEAR(row_sum, 0.0f, kTol) << "row " << b;
    }
}

// Non-vacuity for the density test, and a direct check of the loss's dependence structure: the
// loss *does* depend on a non-taken action's logit (through the normalizer), unlike DQNLoss,
// whose loss is entirely independent of the non-selected columns.
TEST_F(PolicyGradientLossTest, ChangingANonSelectedLogitChangesTheLoss) {
    PolicyGradientLoss baseline_loss(&backend);
    const float baseline = baseline_loss.forward(logits(), actions(), returns());

    // Row 0 took action 2; raise column 0's logit, which is not the taken action.
    Tensor perturbed(Shape({2, 3}), &backend, {5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    PolicyGradientLoss other(&backend);

    EXPECT_GT(std::abs(other.forward(perturbed, actions(), returns()) - baseline), 0.1f);
}

// A zero return zeroes that step's entire gradient row -- the "this step taught us nothing"
// case, and the only way a row of this dense gradient is legitimately all zeros.
TEST_F(PolicyGradientLossTest, ZeroReturnsGiveZeroLossAndZeroGradient) {
    PolicyGradientLoss loss(&backend);
    Tensor zero_returns(Shape({2, 1}), &backend, {0.0f, 0.0f});

    EXPECT_FLOAT_EQ(loss.forward(logits(), actions(), zero_returns), 0.0f);

    const Tensor grad = loss.backward();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_EQ(grad.data()[i], 0.0f) << "flat index " << i;
    }
}

// Sign semantics, stated as behaviour rather than arithmetic: a positive return makes the taken
// action's gradient negative (gradient *descent* therefore raises that logit -- reinforcement),
// and a negative return flips it (descent lowers the logit -- punishment).
TEST_F(PolicyGradientLossTest, ReturnSignDecidesWhetherTheTakenActionIsReinforcedOrPunished) {
    Tensor uniform(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});
    Tensor action(Shape({1, 1}), &backend, {1.0f});

    PolicyGradientLoss positive(&backend);
    Tensor good(Shape({1, 1}), &backend, {2.0f});
    (void)positive.forward(uniform, action, good);
    const Tensor positive_grad = positive.backward();
    EXPECT_LT(positive_grad.data()[1], 0.0f) << "a positive return must push the taken logit up";
    EXPECT_GT(positive_grad.data()[0], 0.0f) << "and push every other logit down";

    PolicyGradientLoss negative(&backend);
    Tensor bad(Shape({1, 1}), &backend, {-2.0f});
    (void)negative.forward(uniform, action, bad);
    const Tensor negative_grad = negative.backward();
    EXPECT_GT(negative_grad.data()[1], 0.0f) << "a negative return must push the taken logit down";
    EXPECT_LT(negative_grad.data()[0], 0.0f) << "and pull every other logit up";
}

// Numerical stability: the subtract-the-row-max softmax must survive logits that would overflow
// exp() outright (exp(800) is +inf in both float and double) and logits far enough apart that a
// naive log(p) would log a zero. Both rows here are pathological for an unstabilized
// implementation and ordinary for a stabilized one.
TEST_F(PolicyGradientLossTest, IsNumericallyStableOnExtremeLogits) {
    PolicyGradientLoss loss(&backend);
    Tensor extreme(Shape({2, 3}), &backend, {800.0f, 801.0f, 802.0f, 0.0f, -200.0f, -400.0f});
    Tensor acts(Shape({2, 1}), &backend, {2.0f, 1.0f});
    Tensor rets(Shape({2, 1}), &backend, {1.0f, 1.0f});

    const float value = loss.forward(extreme, acts, rets);

    ASSERT_TRUE(std::isfinite(value)) << "unstabilized softmax overflowed";
    // Row 0 is the fixture's [1,2,3] shifted by 799 -- softmax is shift-invariant, so its
    // contribution is exactly the fixture's 0.4076059644443806. Row 1's taken action has
    // log-probability ~= -200 (its logit minus a log-sum-exp of ~0), so loss_1 ~= 200.
    EXPECT_NEAR(value, (0.4076059644443806f + 200.0f) / 2.0f, 1e-3f);

    const Tensor grad = loss.backward();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(grad.data()[i])) << "flat index " << i;
    }
}

TEST_F(PolicyGradientLossTest, BackwardThrowsBeforeForward) {
    PolicyGradientLoss loss(&backend);

    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(PolicyGradientLossTest, AcceptsAnActionEncodingWithinTheIntegerTolerance) {
    PolicyGradientLoss loss(&backend);
    // A policy's float round-trip may leave 2 as 2.00005; that must still decode to action 2.
    Tensor nearly_integer(Shape({2, 1}), &backend, {2.00005f, 0.0f});

    EXPECT_NEAR(loss.forward(logits(), nearly_integer, returns()), -1.2403124685577841f, kTol);
}

TEST_F(PolicyGradientLossTest, ForwardThrowsOnAFractionalActionEncoding) {
    PolicyGradientLoss loss(&backend);
    // An un-argmaxed probability, say -- silently rounding it would train the wrong column.
    Tensor fractional(Shape({2, 1}), &backend, {1.5f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(logits(), fractional, returns()); }, std::invalid_argument);
}

TEST_F(PolicyGradientLossTest, ForwardThrowsOnAnOutOfRangeActionIndex) {
    PolicyGradientLoss loss(&backend);
    Tensor too_large(Shape({2, 1}), &backend, {3.0f, 0.0f});  // action_dim is 3, so 3 is past the end
    Tensor negative(Shape({2, 1}), &backend, {-1.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(logits(), too_large, returns()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(logits(), negative, returns()); }, std::invalid_argument);
}

// A rejected forward() must not leave a half-populated cache that a later backward() would
// silently consume.
TEST_F(PolicyGradientLossTest, ARejectedForwardLeavesBackwardStillUnprimed) {
    PolicyGradientLoss loss(&backend);
    Tensor too_large(Shape({2, 1}), &backend, {3.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(logits(), too_large, returns()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(PolicyGradientLossTest, ForwardThrowsOnMismatchedBatchSizes) {
    PolicyGradientLoss loss(&backend);
    Tensor short_actions(Shape({1, 1}), &backend, {0.0f});
    Tensor short_returns(Shape({1, 1}), &backend, {0.0f});

    EXPECT_THROW({ (void)loss.forward(logits(), short_actions, returns()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(logits(), actions(), short_returns); }, std::invalid_argument);
}

TEST_F(PolicyGradientLossTest, ForwardThrowsOnWrongRank) {
    PolicyGradientLoss loss(&backend);
    Tensor flat_logits(Shape({6}), &backend, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Tensor flat_actions(Shape({2}), &backend, {2.0f, 0.0f});

    // Same numel as the valid tensors, wrong rank -- the case a bare "numel matches" check
    // would wave through.
    EXPECT_THROW({ (void)loss.forward(flat_logits, actions(), returns()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(logits(), flat_actions, returns()); }, std::invalid_argument);
}

TEST_F(PolicyGradientLossTest, ForwardThrowsOnAReturnsTensorWiderThanOneColumn) {
    PolicyGradientLoss loss(&backend);
    Tensor wide(Shape({2, 2}), &backend, {1.0f, 0.0f, 1.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(logits(), actions(), wide); }, std::invalid_argument);
}

// The single-action-dimension edge: action_dim == 1 is degenerate but well-formed. The only
// available action has probability exactly 1, so its log-probability is 0, the loss is 0, and
// the gradient is identically 0 -- a policy with no choice cannot be improved.
TEST_F(PolicyGradientLossTest, SingleActionDimensionHasZeroLossAndZeroGradient) {
    PolicyGradientLoss loss(&backend);
    Tensor one_logit(Shape({2, 1}), &backend, {3.0f, -7.0f});
    Tensor a(Shape({2, 1}), &backend, {0.0f, 0.0f});

    EXPECT_NEAR(loss.forward(one_logit, a, returns()), 0.0f, kTol);
    const Tensor grad = loss.backward();
    ASSERT_EQ(grad.numel(), 2);
    EXPECT_NEAR(grad.data()[0], 0.0f, kTol);
    EXPECT_NEAR(grad.data()[1], 0.0f, kTol);
}

}  // namespace
}  // namespace pulsatrix
