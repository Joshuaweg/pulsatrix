#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gae.hpp"

namespace pulsatrix {
namespace {

// Every hand-derived quantity below is exactly representable in binary floating point (halves
// and quarters throughout), deliberately: the done-boundary cut and the lambda == 0 collapse
// are claims of *exactness*, and a test that could only assert "close" would not distinguish a
// hard cut from a very heavy discount.
class GAETest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The shared 4-step rollout, with a done boundary at step 1 (mid-rollout, not at the end).
    Tensor rewards() { return Tensor(Shape({4, 1}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}); }
    Tensor dones() { return Tensor(Shape({4, 1}), &backend, {0.0f, 1.0f, 0.0f, 0.0f}); }
    Tensor values() { return Tensor(Shape({4, 1}), &backend, {0.5f, 1.0f, 2.0f, 3.0f}); }
    static constexpr float kBootstrap = 4.0f;
};

// Hand derivation -- gamma = 0.5, lambda = 0.5, bootstrap_value = 4.0:
//   rewards = [1, 2, 3, 4]   dones = [0, 1, 0, 0]   values = [0.5, 1, 2, 3]
//   V_next  = [values[1], values[2], values[3], bootstrap] = [1, 2, 3, 4]
//
//   delta[t] = r[t] + gamma*(1-done[t])*V_next[t] - V[t]
//     delta[0] = 1 + 0.5*1*1 - 0.5 = 1.0
//     delta[1] = 2 + 0.5*0*2 - 1.0 = 1.0      <- done: the bootstrap term is cut entirely
//     delta[2] = 3 + 0.5*1*3 - 2.0 = 2.5
//     delta[3] = 4 + 0.5*1*4 - 3.0 = 3.0      <- last step: V_next is the caller's bootstrap
//
//   A[t] = delta[t] + gamma*lambda*(1-done[t])*A[t+1], gamma*lambda = 0.25, A[4] := 0
//     A[3] = 3.0  + 0.25*1*0     = 3.0
//     A[2] = 2.5  + 0.25*1*3.0   = 3.25
//     A[1] = 1.0  + 0.25*0*3.25  = 1.0        <- done: the trace is cut, A[2] does not leak back
//     A[0] = 1.0  + 0.25*1*1.0   = 1.25
//
//   returns[t] = A[t] + V[t] = [1.75, 2.0, 5.25, 6.0]
TEST_F(GAETest, MatchesTheHandDerivedAdvantagesAndCriticTargets) {
    const GAEResult result = ComputeGAE(rewards(), dones(), values(), kBootstrap, 0.5f, 0.5f, &backend);

    ASSERT_EQ(result.advantages.rank(), 2);
    EXPECT_EQ(result.advantages.shape().dim(0), 4);
    EXPECT_EQ(result.advantages.shape().dim(1), 1);
    ASSERT_EQ(result.returns.rank(), 2);
    EXPECT_EQ(result.returns.shape().dim(0), 4);
    EXPECT_EQ(result.returns.shape().dim(1), 1);

    EXPECT_FLOAT_EQ(result.advantages.data()[0], 1.25f);
    EXPECT_FLOAT_EQ(result.advantages.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(result.advantages.data()[2], 3.25f);
    EXPECT_FLOAT_EQ(result.advantages.data()[3], 3.0f);

    EXPECT_FLOAT_EQ(result.returns.data()[0], 1.75f);
    EXPECT_FLOAT_EQ(result.returns.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(result.returns.data()[2], 5.25f);
    EXPECT_FLOAT_EQ(result.returns.data()[3], 6.0f);
}

// The GAE identity, asserted structurally rather than against the hand numbers again: the
// critic's target is the advantage plus the critic's *own* current estimate -- not the raw
// Monte-Carlo return-to-go RolloutBuffer::compute_returns() would produce (which, for this
// rollout at gamma = 0.5, would be [2.0, 2.0, 5.0, 4.0] -- different in three of four rows).
TEST_F(GAETest, ReturnsAreExactlyAdvantagesPlusValues) {
    const Tensor v = values();
    const GAEResult result = ComputeGAE(rewards(), dones(), v, kBootstrap, 0.5f, 0.5f, &backend);

    for (int64_t t = 0; t < 4; ++t) {
        EXPECT_EQ(result.returns.data()[t], result.advantages.data()[t] + v.data()[t]) << "at t = " << t;
    }
}

// lambda == 0 collapses the recursion to the one-step TD residual itself: A[t] == delta[t],
// exactly, for every t. Asserted with EXPECT_EQ, not EXPECT_FLOAT_EQ -- the trace term is
// multiplied by a literal zero, so any leakage at all is a bug in the recursion, not rounding.
//   delta = [1.0, 1.0, 2.5, 3.0] (derived above, independent of lambda)
TEST_F(GAETest, LambdaZeroReducesAdvantagesToTheOneStepTDResidual) {
    const GAEResult result = ComputeGAE(rewards(), dones(), values(), kBootstrap, 0.5f, 0.0f, &backend);

    EXPECT_EQ(result.advantages.data()[0], 1.0f);
    EXPECT_EQ(result.advantages.data()[1], 1.0f);
    EXPECT_EQ(result.advantages.data()[2], 2.5f);
    EXPECT_EQ(result.advantages.data()[3], 3.0f);

    // returns[t] = delta[t] + V[t] follows.
    EXPECT_EQ(result.returns.data()[0], 1.5f);
    EXPECT_EQ(result.returns.data()[1], 2.0f);
    EXPECT_EQ(result.returns.data()[2], 4.5f);
    EXPECT_EQ(result.returns.data()[3], 6.0f);
}

// lambda == 1 accumulates every discounted residual to the end of the episode segment, which
// telescopes to the discounted-reward sum over that segment (plus the segment's own bootstrap)
// minus V(s_t). Hand-derived both ways, and the two must agree:
//
//   Recursion (gamma*lambda = 0.5):
//     A[3] = 3.0 + 0.5*1*0   = 3.0
//     A[2] = 2.5 + 0.5*1*3.0 = 4.0
//     A[1] = 1.0 + 0.5*0*4.0 = 1.0    <- done at t=1 closes the segment
//     A[0] = 1.0 + 0.5*1*1.0 = 1.5
//
//   Telescoped (segment [0,1] ends terminal, so no bootstrap; segment [2,3] runs off the end
//   of the rollout, so it bootstraps from the caller's V = 4.0):
//     A[0] = r0 + gamma*r1 - V0            = 1 + 0.5*2 - 0.5         = 1.5   OK
//     A[1] = r1 - V1                       = 2 - 1                   = 1.0   OK
//     A[2] = r2 + gamma*r3 + gamma^2*B - V2= 3 + 0.5*4 + 0.25*4 - 2  = 4.0   OK
//     A[3] = r3 + gamma*B - V3             = 4 + 0.5*4 - 3           = 3.0   OK
TEST_F(GAETest, LambdaOneMatchesTheTelescopedMonteCarloAdvantage) {
    const GAEResult result = ComputeGAE(rewards(), dones(), values(), kBootstrap, 0.5f, 1.0f, &backend);

    EXPECT_FLOAT_EQ(result.advantages.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(result.advantages.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(result.advantages.data()[2], 4.0f);
    EXPECT_FLOAT_EQ(result.advantages.data()[3], 3.0f);
}

// A terminal step must cut the bootstrap *exactly*, not merely discount it heavily -- a
// terminal state has no successor whose value could be backed up. Deliberately huge successor
// and bootstrap values, so any leakage at all is unmissable, and EXPECT_EQ rather than
// EXPECT_FLOAT_EQ. Same assertion shape as DQNTargetTest.DoneRowTargetEqualsItsRewardExactly.
TEST_F(GAETest, TerminalStepCutsBothTheBootstrapAndTheTraceExactly) {
    Tensor r(Shape({2, 1}), &backend, {0.375f, 0.0f});
    Tensor d(Shape({2, 1}), &backend, {1.0f, 0.0f});
    Tensor v(Shape({2, 1}), &backend, {0.0f, 1.0e6f});

    const GAEResult result = ComputeGAE(r, d, v, 1.0e6f, 0.99f, 0.95f, &backend);

    // Step 0 is terminal: V_next (= 1.0e6) is cut, and A[1] -- a large number -- must not
    // propagate backwards through the trace either. A[0] is therefore exactly r0 - V0.
    EXPECT_EQ(result.advantages.data()[0], 0.375f);
}

// The last stored step's successor value can only come from the caller (RolloutBuffer stores no
// next_observation), so the bootstrap argument must actually reach exactly that one step and no
// other. Two runs differing only in the bootstrap: with lambda = 0 the trace cannot carry the
// difference anywhere, so only the final row may move.
TEST_F(GAETest, BootstrapValueAffectsOnlyTheFinalStepsDelta) {
    const GAEResult low = ComputeGAE(rewards(), dones(), values(), 0.0f, 0.5f, 0.0f, &backend);
    const GAEResult high = ComputeGAE(rewards(), dones(), values(), 4.0f, 0.5f, 0.0f, &backend);

    EXPECT_EQ(low.advantages.data()[0], high.advantages.data()[0]);
    EXPECT_EQ(low.advantages.data()[1], high.advantages.data()[1]);
    EXPECT_EQ(low.advantages.data()[2], high.advantages.data()[2]);
    // delta[3] = 4 + 0.5*bootstrap - 3, so 1.0 against 3.0.
    EXPECT_FLOAT_EQ(low.advantages.data()[3], 1.0f);
    EXPECT_FLOAT_EQ(high.advantages.data()[3], 3.0f);
}

// A single-step rollout has no `values[t+1]` at all: the bootstrap is the only successor value
// in play. Degenerate but entirely legal, and the loop's `t + 1 < batch_size` branch is never
// otherwise exercised on its false side alone.
TEST_F(GAETest, SingleStepRolloutUsesOnlyTheBootstrapValue) {
    Tensor r(Shape({1, 1}), &backend, {1.0f});
    Tensor d(Shape({1, 1}), &backend, {0.0f});
    Tensor v(Shape({1, 1}), &backend, {2.0f});

    const GAEResult result = ComputeGAE(r, d, v, 8.0f, 0.5f, 0.5f, &backend);

    EXPECT_FLOAT_EQ(result.advantages.data()[0], 1.0f + 0.5f * 8.0f - 2.0f);  // 3.0
    EXPECT_FLOAT_EQ(result.returns.data()[0], 5.0f);
}

TEST_F(GAETest, RejectsMalformedShapesAndOutOfRangeDiscounts) {
    Tensor r(Shape({3, 1}), &backend, {1.0f, 1.0f, 1.0f});
    Tensor d(Shape({3, 1}), &backend, {0.0f, 0.0f, 0.0f});
    Tensor v(Shape({3, 1}), &backend, {0.0f, 0.0f, 0.0f});

    Tensor rank1(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW((void)ComputeGAE(rank1, d, v, 0.0f, 0.9f, 0.9f, &backend), std::invalid_argument);

    Tensor wide(Shape({3, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW((void)ComputeGAE(wide, d, v, 0.0f, 0.9f, 0.9f, &backend), std::invalid_argument);

    Tensor short_column(Shape({2, 1}), &backend, {0.0f, 0.0f});
    EXPECT_THROW((void)ComputeGAE(r, short_column, v, 0.0f, 0.9f, 0.9f, &backend), std::invalid_argument);
    EXPECT_THROW((void)ComputeGAE(r, d, short_column, 0.0f, 0.9f, 0.9f, &backend), std::invalid_argument);

    EXPECT_THROW((void)ComputeGAE(r, d, v, 0.0f, -0.1f, 0.9f, &backend), std::invalid_argument);
    EXPECT_THROW((void)ComputeGAE(r, d, v, 0.0f, 1.1f, 0.9f, &backend), std::invalid_argument);
    EXPECT_THROW((void)ComputeGAE(r, d, v, 0.0f, 0.9f, -0.1f, &backend), std::invalid_argument);
    EXPECT_THROW((void)ComputeGAE(r, d, v, 0.0f, 0.9f, 1.1f, &backend), std::invalid_argument);

    // Both endpoints of both ranges are legal: gamma == 1 is undiscounted, lambda == 1 is
    // Monte-Carlo, lambda == 0 is one-step TD, gamma == 0 is myopic.
    EXPECT_NO_THROW((void)ComputeGAE(r, d, v, 0.0f, 0.0f, 0.0f, &backend));
    EXPECT_NO_THROW((void)ComputeGAE(r, d, v, 0.0f, 1.0f, 1.0f, &backend));
}

}  // namespace
}  // namespace pulsatrix
