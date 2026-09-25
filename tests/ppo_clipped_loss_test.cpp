#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"

namespace pulsatrix {
namespace {

constexpr float kTol = 1e-5f;

class PPOClippedLossTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // An old log-probability chosen so that the ratio comes out at exactly `ratio`, given that
    // the new policy assigns `new_prob` to the taken action:
    //   ratio = exp(log new_prob - old_log_prob)  =>  old_log_prob = log(new_prob / ratio).
    // This is how every row below is placed in a chosen ratio region by construction rather
    // than by trial and error.
    static float old_log_prob_for(float new_prob, float ratio) { return std::log(new_prob / ratio); }
};

// =========================================================================================
// THE load-bearing test: all three ratio regions x both advantage signs, hand-derived.
// =========================================================================================
//
// Setup, six rows, action_dim = 2, every row's logits are {0, 0} so the new policy is uniform:
//   p[b, 0] = p[b, 1] = 0.5 exactly, and the taken action is column 0 in every row.
// clip_epsilon = 0.2, so the trust region is [1 - eps, 1 + eps] = [0.8, 1.2].
//
//   row | ratio | A    | region        | mask | min(r*A, clip(r)*A)          | loss_b
//   ----+-------+------+---------------+------+------------------------------+-------
//    0  |  0.5  | +2.0 | below 1-eps   |  1   | min(1.0, 0.8*2 = 1.6) = 1.0  | -1.0
//    1  |  0.5  | -2.0 | below 1-eps   |  0   | min(-1.0, -1.6)      = -1.6  | +1.6
//    2  |  1.0  | +2.0 | inside        |  1   | min(2.0, 2.0)        =  2.0  | -2.0
//    3  |  1.0  | -2.0 | inside        |  1   | min(-2.0, -2.0)      = -2.0  | +2.0
//    4  |  1.5  | +2.0 | above 1+eps   |  0   | min(3.0, 1.2*2 = 2.4)=  2.4  | -2.4
//    5  |  1.5  | -2.0 | above 1+eps   |  1   | min(-3.0, -2.4)      = -3.0  | +3.0
//
//   loss = (-1.0 + 1.6 - 2.0 + 2.0 - 2.4 + 3.0) / 6 = 1.2 / 6 = 0.2
//
// Mask rule: zero iff (A >= 0 and ratio > 1+eps) -- row 4 -- or (A < 0 and ratio < 1-eps) --
// row 1. Those are exactly the two "already past the trust region in the direction that would
// increase the objective further" cases. Note rows 0 and 5 are *also* outside the trust region,
// but in the direction that hurts the objective, and so keep their full unclipped gradient:
// PPO does want to pull those steps back.
//
// Gradient, with N = 6, p = 0.5 everywhere, taken action = column 0:
//   grad[b,k] = -mask[b] * A[b] * ratio[b] * (1{k==0} - 0.5) / 6
//             = -mask * A * ratio * (+0.5) / 6   for k = 0
//             = -mask * A * ratio * (-0.5) / 6   for k = 1
//
//   row 0: -(1)(+2.0)(0.5)(0.5)/6 = -0.5/6  = -0.08333333 | +0.08333333
//   row 1:  mask 0                =  0      EXACTLY       |  0 EXACTLY
//   row 2: -(1)(+2.0)(1.0)(0.5)/6 = -1.0/6  = -0.16666667 | +0.16666667
//   row 3: -(1)(-2.0)(1.0)(0.5)/6 = +1.0/6  = +0.16666667 | -0.16666667
//   row 4:  mask 0                =  0      EXACTLY       |  0 EXACTLY
//   row 5: -(1)(-2.0)(1.5)(0.5)/6 = +1.5/6  = +0.25       | -0.25
TEST_F(PPOClippedLossTest, SixRegionGradientMatchesTheHandDerivedMaskAndValues) {
    constexpr float kEps = 0.2f;
    constexpr float kProb = 0.5f;  // uniform over two actions, from logits {0, 0}

    Tensor logits(Shape({6, 2}), &backend,
                  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor actions(Shape({6, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor old_log_probs(Shape({6, 1}), &backend,
                         {old_log_prob_for(kProb, 0.5f), old_log_prob_for(kProb, 0.5f),
                          old_log_prob_for(kProb, 1.0f), old_log_prob_for(kProb, 1.0f),
                          old_log_prob_for(kProb, 1.5f), old_log_prob_for(kProb, 1.5f)});
    Tensor advantages(Shape({6, 1}), &backend, {2.0f, -2.0f, 2.0f, -2.0f, 2.0f, -2.0f});

    PPOClippedLoss loss(&backend);
    const float value = loss.forward(logits, actions, old_log_probs, advantages, kEps);

    // The loss value is a direct consequence of the six per-row `min` selections tabulated
    // above -- it pins down which branch of the min was taken in every region.
    EXPECT_NEAR(value, 0.2f, kTol);

    const Tensor grad = loss.backward();
    ASSERT_EQ(grad.rank(), 2);
    ASSERT_EQ(grad.shape().dim(0), 6);
    ASSERT_EQ(grad.shape().dim(1), 2);

    // --- Region: ratio below 1-eps -------------------------------------------------------
    // Row 0, A > 0: NOT clipped out (the ratio fell below the region, which already reduces
    // the objective -- PPO leaves that gradient alone so the step can be pulled back up).
    EXPECT_NEAR(grad.data()[0], -0.5f / 6.0f, kTol);
    EXPECT_NEAR(grad.data()[1], 0.5f / 6.0f, kTol);

    // Row 1, A < 0: clipped out. The whole row must be EXACTLY zero, not merely small -- the
    // mask multiplies the row weight by a literal 0.0f, so this is a structural zero. If the
    // mask rule were ever weakened to "clip the value but keep the gradient", this row would
    // read -(1)(-2.0)(0.5)(0.5)/6 = +0.0833..., which EXPECT_EQ against 0 rejects outright.
    EXPECT_EQ(grad.data()[2], 0.0f);
    EXPECT_EQ(grad.data()[3], 0.0f);

    // --- Region: ratio inside the trust region -------------------------------------------
    // Row 2, A > 0.
    EXPECT_NEAR(grad.data()[4], -1.0f / 6.0f, kTol);
    EXPECT_NEAR(grad.data()[5], 1.0f / 6.0f, kTol);
    // Row 3, A < 0 -- same magnitude, opposite sign: a negative advantage pushes probability
    // *off* the taken action.
    EXPECT_NEAR(grad.data()[6], 1.0f / 6.0f, kTol);
    EXPECT_NEAR(grad.data()[7], -1.0f / 6.0f, kTol);

    // --- Region: ratio above 1+eps -------------------------------------------------------
    // Row 4, A > 0: clipped out. EXACTLY zero -- this is the canonical PPO case (the policy
    // has already increased the probability of a good action past the trust region, and the
    // objective refuses to pay for going further). Unmasked it would read -1.5/6 = -0.25.
    EXPECT_EQ(grad.data()[8], 0.0f);
    EXPECT_EQ(grad.data()[9], 0.0f);

    // Row 5, A < 0: NOT clipped out -- the ratio ran above the region for a *bad* action,
    // which hurts the objective, so the full gradient stands and pulls it back.
    EXPECT_NEAR(grad.data()[10], 0.25f, kTol);
    EXPECT_NEAR(grad.data()[11], -0.25f, kTol);
}

// The two masked rows above are the only ones whose gradient is zero; a "mask everything
// outside the trust region" bug would also zero rows 0 and 5 and still pass an all-zero check
// on rows 1 and 4. This asserts the complement explicitly: the two same-region rows with the
// opposite advantage sign are non-zero.
TEST_F(PPOClippedLossTest, OnlyTheTwoObjectiveIncreasingClippedRowsAreMasked) {
    constexpr float kProb = 0.5f;
    Tensor logits(Shape({4, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor actions(Shape({4, 1}), &backend, {0.0f, 0.0f, 0.0f, 0.0f});
    Tensor old_log_probs(Shape({4, 1}), &backend,
                         {old_log_prob_for(kProb, 0.5f), old_log_prob_for(kProb, 0.5f),
                          old_log_prob_for(kProb, 1.5f), old_log_prob_for(kProb, 1.5f)});
    Tensor advantages(Shape({4, 1}), &backend, {2.0f, -2.0f, 2.0f, -2.0f});

    PPOClippedLoss loss(&backend);
    (void)loss.forward(logits, actions, old_log_probs, advantages, 0.2f);
    const Tensor grad = loss.backward();

    EXPECT_NE(grad.data()[0], 0.0f);  // ratio < 1-eps, A > 0 -> kept
    EXPECT_EQ(grad.data()[2], 0.0f);  // ratio < 1-eps, A < 0 -> masked
    EXPECT_EQ(grad.data()[4], 0.0f);  // ratio > 1+eps, A > 0 -> masked
    EXPECT_NE(grad.data()[6], 0.0f);  // ratio > 1+eps, A < 0 -> kept
}

// A zero advantage carries no signal in either direction; it falls on the `A >= 0` side of the
// mask rule, but the gradient is zero anyway because the advantage itself multiplies it. Worth
// pinning: the `>= 0` vs `> 0` boundary in the mask predicate is arbitrary at exactly zero
// precisely because nothing downstream can observe the difference.
TEST_F(PPOClippedLossTest, ZeroAdvantageProducesZeroGradientInEveryRegion) {
    constexpr float kProb = 0.5f;
    Tensor logits(Shape({3, 2}), &backend, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    Tensor actions(Shape({3, 1}), &backend, {0.0f, 1.0f, 0.0f});
    Tensor old_log_probs(Shape({3, 1}), &backend,
                         {old_log_prob_for(kProb, 0.5f), old_log_prob_for(kProb, 1.0f),
                          old_log_prob_for(kProb, 1.5f)});
    Tensor advantages(Shape({3, 1}), &backend, {0.0f, 0.0f, 0.0f});

    PPOClippedLoss loss(&backend);
    EXPECT_NEAR(loss.forward(logits, actions, old_log_probs, advantages, 0.2f), 0.0f, kTol);
    const Tensor grad = loss.backward();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_EQ(grad.data()[i], 0.0f) << "at element " << i;
    }
}

// Cross-check at the trivial starting point: first epoch, first minibatch, before any update
// has moved the policy away from the one that collected the data. There every ratio is exactly
// 1, every row is inside the trust region, and the gradient reduces to
//   -A * 1 * (1{k==a} - p) / N  ==  A * (p - 1{k==a}) / N,
// which is *identically* PolicyGradientLoss::backward()'s gradient with `advantages` playing
// the role of `returns`. Asserted element-wise against the real class, not against a
// re-derivation of its formula.
//
// The loss *values* are deliberately not compared: PPO's surrogate at ratio == 1 evaluates to
// mean(-A), a constant in the logits, while PolicyGradientLoss evaluates to
// mean(-log p_a * G). Both are surrogates whose value carries no meaning; the gradient is the
// object the two share, and the object this cross-check is about.
TEST_F(PPOClippedLossTest, AtRatioOneTheGradientEqualsPolicyGradientLossWithAdvantagesAsReturns) {
    Tensor logits(Shape({2, 3}), &backend, {1.0f, 2.0f, 0.0f, 0.5f, -0.5f, 0.25f});
    Tensor actions(Shape({2, 1}), &backend, {1.0f, 2.0f});
    Tensor advantages(Shape({2, 1}), &backend, {1.5f, -0.75f});

    // The data-collecting policy *is* the current policy: its log-probabilities are this very
    // network's, so old_log_prob == new_log_prob and ratio == exp(0) == 1 in both rows.
    std::vector<float> old_lp(2);
    for (int64_t b = 0; b < 2; ++b) {
        const float* row = logits.data() + b * 3;
        float max_logit = std::max(row[0], std::max(row[1], row[2]));
        float exp_sum = 0.0f;
        for (int64_t a = 0; a < 3; ++a) {
            exp_sum += std::exp(row[a] - max_logit);
        }
        const int64_t index = static_cast<int64_t>(actions.data()[b]);
        old_lp[static_cast<size_t>(b)] = row[index] - max_logit - std::log(exp_sum);
    }
    Tensor old_log_probs(Shape({2, 1}), &backend, old_lp);

    PPOClippedLoss ppo(&backend);
    (void)ppo.forward(logits, actions, old_log_probs, advantages, 0.2f);
    const Tensor ppo_grad = ppo.backward();

    PolicyGradientLoss pg(&backend);
    (void)pg.forward(logits, actions, advantages);
    const Tensor pg_grad = pg.backward();

    ASSERT_EQ(ppo_grad.numel(), pg_grad.numel());
    for (int64_t i = 0; i < ppo_grad.numel(); ++i) {
        EXPECT_NEAR(ppo_grad.data()[i], pg_grad.data()[i], kTol) << "at element " << i;
    }

    // And the surrogate value at ratio == 1 really is mean(-A), the other half of the claim.
    EXPECT_NEAR(ppo.forward(logits, actions, old_log_probs, advantages, 0.2f), -(1.5f - 0.75f) / 2.0f, kTol);
}

// The gradient is dense across every action column, not sparse to the taken action -- softmax
// is a normalized distribution, so raising one action's probability necessarily lowers the
// others'. The same structural property PolicyGradientLoss documents, re-asserted here because
// the obvious wrong analogy (DQNLoss's masked, taken-action-only gradient) is one line away.
// The row's columns must also sum to zero: the gradient lives in the tangent space of the
// simplex.
TEST_F(PPOClippedLossTest, GradientIsDenseAcrossActionsAndSumsToZeroPerRow) {
    Tensor logits(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.0f});
    Tensor actions(Shape({1, 1}), &backend, {1.0f});
    // log(e^1 + e^2 + e^0) = log(11.10734) = 2.40760, so log p_1 = 2 - 2.40760 = -0.40760.
    // Matching old_log_prob puts the ratio at 1, safely inside the trust region -- otherwise
    // this row could be masked and its "dense" gradient would be trivially all-zero.
    Tensor old_log_probs(Shape({1, 1}), &backend, {-0.40760f});
    Tensor advantages(Shape({1, 1}), &backend, {1.0f});

    PPOClippedLoss loss(&backend);
    (void)loss.forward(logits, actions, old_log_probs, advantages, 0.2f);
    const Tensor grad = loss.backward();

    EXPECT_NE(grad.data()[0], 0.0f);
    EXPECT_NE(grad.data()[1], 0.0f);
    EXPECT_NE(grad.data()[2], 0.0f);
    EXPECT_NEAR(grad.data()[0] + grad.data()[1] + grad.data()[2], 0.0f, kTol);
}

TEST_F(PPOClippedLossTest, BackwardBeforeForwardThrows) {
    PPOClippedLoss loss(&backend);
    EXPECT_THROW((void)loss.backward(), std::logic_error);
}

TEST_F(PPOClippedLossTest, RejectsMalformedInputsAndOutOfRangeClipEpsilon) {
    PPOClippedLoss loss(&backend);
    Tensor logits(Shape({2, 3}), &backend, {1.0f, 2.0f, 0.0f, 0.5f, -0.5f, 0.25f});
    Tensor actions(Shape({2, 1}), &backend, {1.0f, 2.0f});
    Tensor old_log_probs(Shape({2, 1}), &backend, {-1.0f, -1.0f});
    Tensor advantages(Shape({2, 1}), &backend, {1.0f, 1.0f});

    Tensor rank1(Shape({3}), &backend, {1.0f, 2.0f, 0.0f});
    EXPECT_THROW((void)loss.forward(rank1, actions, old_log_probs, advantages, 0.2f), std::invalid_argument);

    Tensor mismatched(Shape({3, 1}), &backend, {0.0f, 0.0f, 0.0f});
    EXPECT_THROW((void)loss.forward(logits, mismatched, old_log_probs, advantages, 0.2f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(logits, actions, mismatched, advantages, 0.2f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(logits, actions, old_log_probs, mismatched, 0.2f), std::invalid_argument);

    Tensor fractional(Shape({2, 1}), &backend, {0.5f, 1.0f});
    EXPECT_THROW((void)loss.forward(logits, fractional, old_log_probs, advantages, 0.2f), std::invalid_argument);

    Tensor out_of_range(Shape({2, 1}), &backend, {3.0f, 1.0f});
    EXPECT_THROW((void)loss.forward(logits, out_of_range, old_log_probs, advantages, 0.2f), std::invalid_argument);
    Tensor negative_action(Shape({2, 1}), &backend, {-1.0f, 1.0f});
    EXPECT_THROW((void)loss.forward(logits, negative_action, old_log_probs, advantages, 0.2f),
                 std::invalid_argument);

    // clip_epsilon must be a real trust region: 0 would clamp the ratio to the single point 1
    // and kill every gradient; >= 1 would put the lower bound at or below 0, which no
    // probability ratio can reach.
    EXPECT_THROW((void)loss.forward(logits, actions, old_log_probs, advantages, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(logits, actions, old_log_probs, advantages, -0.1f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(logits, actions, old_log_probs, advantages, 1.0f), std::invalid_argument);
    EXPECT_NO_THROW((void)loss.forward(logits, actions, old_log_probs, advantages, 0.999f));
}

using PPOClippedLossDeathTest = PPOClippedLossTest;

// forward() reads all four tensors in raw host loops over Tensor::data() -- undefined behavior
// on a CUDA-backed Tensor, so each is PULSATRIX_ASSERT-guarded (mission_host_loop_guards.md). No
// real GPU needed: this reuses LinearModuleDeathTest's mislabeled-Tensor pattern.
//
// One death test, not four -- byte-for-byte PolicyGradientLossDeathTest's own count decision
// and reasoning, which this class's forward() structurally mirrors. The mission fixes the count
// at one per raw-host-loop *entry point*, and the four guards here are adjacent lines on a
// single entry path covering one guarded-argument role: a caller-supplied host tensor read by
// the same loops. `actions`, `old_log_probs` and `advantages` are all (N, 1) columns validated
// identically, so extra cases would re-cover a path rather than cover a new one.
TEST_F(PPOClippedLossDeathTest, ForwardAbortsOnNonCpuNewLogits) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    PPOClippedLoss loss(&backend);
    Tensor cuda_logits(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f}, DeviceType::Cuda);
    Tensor actions(Shape({2, 1}), &backend, {0.0f, 1.0f});
    Tensor old_log_probs(Shape({2, 1}), &backend, {-1.0f, -1.0f});
    Tensor advantages(Shape({2, 1}), &backend, {1.0f, 1.0f});
    EXPECT_DEATH({ (void)loss.forward(cuda_logits, actions, old_log_probs, advantages, 0.2f); },
                 "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
