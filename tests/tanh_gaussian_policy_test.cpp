#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tanh_gaussian_policy.hpp"

namespace pulsatrix {
namespace {

class TanhGaussianPolicyTest : public ::testing::Test {
protected:
    CPUBackend backend;
    TanhGaussianPolicy policy{&backend};
};

// ---------------------------------------------------------------------------------------
// Forward: the reparameterization identity, hand-computed end to end.
// ---------------------------------------------------------------------------------------

// The mission's required independent numeric proof of the *forward* formula (not its
// gradient). Every number below is computed by hand from the definitions, not read back out of
// the implementation:
//   log_std = ln(2)          -> std = exp(ln 2)          = 2
//   u       = mean + std*eps = 0.5 + 2*0.25              = 1.0   (exactly)
//   action  = tanh(1)                                    = 0.76159415595576485
//   1 - a^2                                              = 0.41997434161402614
//   log_prob = -0.5*eps^2 - log_std - 0.5*log(2pi) - log(1 - a^2 + 1e-6)
//            = -0.03125 - 0.69314718055994531 - 0.91893853320467274 + 0.86755927987104350
//            = -0.77577643389357455
// The four terms are asserted individually as well as in sum, so a sign error in any one of
// them cannot hide inside a coincidentally-correct total.
TEST_F(TanhGaussianPolicyTest, ForwardMatchesHandComputedReparameterizationIdentity) {
    Tensor mean(Shape({1, 1}), &backend, {0.5f});
    Tensor log_std(Shape({1, 1}), &backend, {0.69314718f});  // ln(2)
    Tensor epsilon(Shape({1, 1}), &backend, {0.25f});

    TanhGaussianSample sample = policy.forward(mean, log_std, epsilon);

    ASSERT_EQ(sample.action.numel(), 1);
    ASSERT_EQ(sample.log_prob.numel(), 1);
    // u == 1.0 exactly, so the action is exactly tanh(1).
    EXPECT_NEAR(sample.action.data()[0], 0.76159415595576485f, 1e-6f);

    const float quadratic = -0.03125f;
    const float minus_log_std = -0.69314718f;
    const float normalizer = -0.91893853f;
    const float squash_correction = 0.86755928f;  // -log(1 - tanh(1)^2 + 1e-6)
    EXPECT_NEAR(sample.log_prob.data()[0], quadratic + minus_log_std + normalizer + squash_correction, 1e-5f);
    EXPECT_NEAR(sample.log_prob.data()[0], -0.77577643f, 1e-5f);
}

// log_prob is the *joint* log-density: a diagonal Gaussian's dimensions are independent, so it
// is the sum of the per-dimension log-densities, emitted as one (N, 1) column. Proven here by
// running the same three (mean, log_std, epsilon) triples as three separate width-1 policies
// and checking the widths-1 log_probs add up to the width-3 one -- an independent route to the
// same number that does not restate the implementation's own loop.
TEST_F(TanhGaussianPolicyTest, ForwardLogProbIsSummedOverActionDim) {
    const std::vector<float> means{0.4f, -1.2f, 0.9f};
    const std::vector<float> log_stds{0.3f, -0.6f, 1.1f};
    const std::vector<float> epsilons{1.1f, -0.8f, 0.35f};

    Tensor mean(Shape({1, 3}), &backend, means);
    Tensor log_std(Shape({1, 3}), &backend, log_stds);
    Tensor epsilon(Shape({1, 3}), &backend, epsilons);
    TanhGaussianSample wide = policy.forward(mean, log_std, epsilon);

    ASSERT_EQ(wide.log_prob.shape().rank(), 2);
    EXPECT_EQ(wide.log_prob.shape().dim(0), 1);
    EXPECT_EQ(wide.log_prob.shape().dim(1), 1) << "log_prob is per-sample, summed over action_dim";

    float summed = 0.0f;
    for (size_t d = 0; d < means.size(); ++d) {
        TanhGaussianPolicy narrow(&backend);
        Tensor m(Shape({1, 1}), &backend, {means[d]});
        Tensor ls(Shape({1, 1}), &backend, {log_stds[d]});
        Tensor e(Shape({1, 1}), &backend, {epsilons[d]});
        TanhGaussianSample one = narrow.forward(m, ls, e);
        summed += one.log_prob.data()[0];
        // ...and each dimension's action is unaffected by the others, as a diagonal
        // distribution requires.
        EXPECT_NEAR(wide.action.data()[static_cast<int64_t>(d)], one.action.data()[0], 1e-6f) << "dim " << d;
    }
    EXPECT_NEAR(wide.log_prob.data()[0], summed, 1e-5f);
}

TEST_F(TanhGaussianPolicyTest, ForwardEmitsActionShapedLikeMeanAndOneLogProbPerRow) {
    Tensor mean(Shape({4, 3}), &backend);
    Tensor log_std(Shape({4, 3}), &backend);
    Tensor epsilon(Shape({4, 3}), &backend);

    TanhGaussianSample sample = policy.forward(mean, log_std, epsilon);

    EXPECT_EQ(sample.action.shape().rank(), 2);
    EXPECT_EQ(sample.action.shape().dim(0), 4);
    EXPECT_EQ(sample.action.shape().dim(1), 3);
    EXPECT_EQ(sample.log_prob.shape().dim(0), 4);
    EXPECT_EQ(sample.log_prob.shape().dim(1), 1);
}

TEST_F(TanhGaussianPolicyTest, ForwardWithZeroEpsilonReturnsTanhOfMeanExactly) {
    // The deterministic/"mean action" path SAC uses at evaluation time: epsilon = 0 collapses
    // u onto mean regardless of log_std, so the action is tanh(mean) alone.
    Tensor mean(Shape({1, 3}), &backend, {0.0f, 1.0f, -2.0f});
    Tensor log_std(Shape({1, 3}), &backend, {3.0f, -4.0f, 0.5f});
    Tensor epsilon(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});

    TanhGaussianSample sample = policy.forward(mean, log_std, epsilon);

    EXPECT_FLOAT_EQ(sample.action.data()[0], 0.0f);
    EXPECT_NEAR(sample.action.data()[1], std::tanh(1.0f), 1e-6f);
    EXPECT_NEAR(sample.action.data()[2], std::tanh(-2.0f), 1e-6f);
}

// ---------------------------------------------------------------------------------------
// Forward: the action's bound.
// ---------------------------------------------------------------------------------------

// tanh is a strict bijection onto the *open* interval (-1, 1), so no finite u can produce
// exactly +-1. Swept across a deliberately harsh grid -- including |u| ~ 8.5, where the action
// is within 1.2e-7 of the boundary -- because a naive "clamp to [-1, 1]" implementation, or a
// log-prob formula that happened to blow up near saturation, would both pass a mild fixture.
// The grid is built from explicit u values (then split into mean/std*epsilon) rather than an
// unconstrained mean x log_std cross product, so the *pre-squash* magnitude being tested is
// the thing actually controlled.
TEST_F(TanhGaussianPolicyTest, ActionIsAlwaysStrictlyInsideTheOpenUnitInterval) {
    const std::vector<float> targets{-8.5f, -6.0f, -3.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 3.0f, 6.0f, 8.5f};
    const std::vector<float> log_stds{-3.0f, 0.0f, 1.5f};

    bool saw_near_boundary = false;
    for (float target_u : targets) {
        for (float log_std_value : log_stds) {
            const float std_value = std::exp(log_std_value);
            // Split the target u across mean and std*epsilon so both contribute.
            const float epsilon_value = 0.5f * target_u / std_value;
            const float mean_value = 0.5f * target_u;

            TanhGaussianPolicy local(&backend);
            Tensor mean(Shape({1, 1}), &backend, {mean_value});
            Tensor log_std(Shape({1, 1}), &backend, {log_std_value});
            Tensor epsilon(Shape({1, 1}), &backend, {epsilon_value});
            TanhGaussianSample sample = local.forward(mean, log_std, epsilon);

            const float action = sample.action.data()[0];
            EXPECT_GT(action, -1.0f) << "u = " << target_u << ", log_std = " << log_std_value;
            EXPECT_LT(action, 1.0f) << "u = " << target_u << ", log_std = " << log_std_value;
            EXPECT_TRUE(std::isfinite(sample.log_prob.data()[0]))
                << "u = " << target_u << ", log_std = " << log_std_value;
            if (std::fabs(action) > 0.999f) {
                saw_near_boundary = true;
            }
        }
    }
    // The sweep genuinely reached the hard part of the range, so the strict-bound assertions
    // above are not vacuously satisfied by a fixture that never left the linear region.
    EXPECT_TRUE(saw_near_boundary);
}

// The honest float32 caveat to the test above, pinned as behavior rather than left to be
// rediscovered: `tanh` is mathematically never exactly +-1, but float32 has no room left to
// represent the difference once |u| exceeds roughly 9.5, so std::tanh returns exactly 1.0f.
// That is a representation limit, not a formula error -- and it is exactly the case
// kLogProbStabilizer exists for: log(1 - 1 + 1e-6) is finite, where log(0) would be -inf and
// would poison every downstream loss. Asserted as an exact value so a change to the stabilizer
// cannot slip through silently.
TEST_F(TanhGaussianPolicyTest, FullySaturatedActionStillYieldsAFiniteStabilizedLogProb) {
    Tensor mean(Shape({1, 2}), &backend, {40.0f, -40.0f});
    Tensor log_std(Shape({1, 2}), &backend, {0.0f, 0.0f});
    Tensor epsilon(Shape({1, 2}), &backend, {0.0f, 0.0f});

    TanhGaussianSample sample = policy.forward(mean, log_std, epsilon);

    ASSERT_FLOAT_EQ(sample.action.data()[0], 1.0f) << "float32 saturation is the premise of this test";
    ASSERT_FLOAT_EQ(sample.action.data()[1], -1.0f);
    // Per dimension: -0 - 0 - 0.5*log(2pi) - log(1e-6) = -0.91893853 + 13.81551056.
    const float per_dim = -0.91893853f + 13.81551056f;
    EXPECT_TRUE(std::isfinite(sample.log_prob.data()[0]));
    EXPECT_NEAR(sample.log_prob.data()[0], 2.0f * per_dim, 1e-3f);
}

// ---------------------------------------------------------------------------------------
// Forward: external-boundary validation.
// ---------------------------------------------------------------------------------------

TEST_F(TanhGaussianPolicyTest, ForwardThrowsOnMismatchedLogStdShape) {
    Tensor mean(Shape({2, 3}), &backend);
    Tensor log_std(Shape({2, 4}), &backend);
    Tensor epsilon(Shape({2, 3}), &backend);

    EXPECT_THROW({ (void)policy.forward(mean, log_std, epsilon); }, std::invalid_argument);
}

TEST_F(TanhGaussianPolicyTest, ForwardThrowsOnMismatchedEpsilonShape) {
    Tensor mean(Shape({2, 3}), &backend);
    Tensor log_std(Shape({2, 3}), &backend);
    Tensor epsilon(Shape({3, 3}), &backend);

    EXPECT_THROW({ (void)policy.forward(mean, log_std, epsilon); }, std::invalid_argument);
}

// Rank-2 is a real requirement, not a formality: log_prob is summed over the last axis and
// emitted one row per sample, which is undefined for a rank-1 block.
TEST_F(TanhGaussianPolicyTest, ForwardThrowsOnNonRank2Mean) {
    Tensor mean(Shape({3}), &backend);
    Tensor log_std(Shape({3}), &backend);
    Tensor epsilon(Shape({3}), &backend);

    EXPECT_THROW({ (void)policy.forward(mean, log_std, epsilon); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------------------
// Backward: validation and the two degenerate seeds.
// ---------------------------------------------------------------------------------------

TEST_F(TanhGaussianPolicyTest, BackwardThrowsBeforeForward) {
    Tensor grad_action(Shape({1, 2}), &backend, {1.0f, 1.0f});
    Tensor grad_log_prob(Shape({1, 2}), &backend, {1.0f, 1.0f});

    EXPECT_THROW({ (void)policy.backward(grad_action, grad_log_prob); }, std::logic_error);
}

TEST_F(TanhGaussianPolicyTest, BackwardThrowsOnMismatchedGradActionShape) {
    Tensor mean(Shape({1, 3}), &backend);
    Tensor log_std(Shape({1, 3}), &backend);
    Tensor epsilon(Shape({1, 3}), &backend);
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({1, 4}), &backend);
    Tensor grad_log_prob(Shape({1, 3}), &backend);
    EXPECT_THROW({ (void)policy.backward(grad_action, grad_log_prob); }, std::invalid_argument);
}

// grad_log_prob is required at the *action* shape (N, action_dim), not log_prob's own (N, 1) --
// log_prob is a sum over the row, so every element of the row receives the same incoming
// scalar, and asking for the broadcast block keeps both arguments one shape. Passing the
// un-broadcast (N, 1) column is therefore a rejected caller error, pinned here so the contract
// is unambiguous.
TEST_F(TanhGaussianPolicyTest, BackwardThrowsOnUnbroadcastGradLogProb) {
    Tensor mean(Shape({2, 3}), &backend);
    Tensor log_std(Shape({2, 3}), &backend);
    Tensor epsilon(Shape({2, 3}), &backend);
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({2, 3}), &backend);
    Tensor grad_log_prob(Shape({2, 1}), &backend);
    EXPECT_THROW({ (void)policy.backward(grad_action, grad_log_prob); }, std::invalid_argument);
}

// With no gradient arriving through log_prob, the whole thing collapses to the plain tanh chain
// rule -- the simplest slice of the two-gradient combination, isolated so a bug in the
// log-prob branch cannot be what makes the combined check pass.
TEST_F(TanhGaussianPolicyTest, BackwardWithZeroGradLogProbIsThePlainTanhChainRule) {
    Tensor mean(Shape({1, 2}), &backend, {0.3f, -0.7f});
    Tensor log_std(Shape({1, 2}), &backend, {0.0f, 0.0f});  // std = 1
    Tensor epsilon(Shape({1, 2}), &backend, {0.5f, -1.5f});
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({1, 2}), &backend, {1.0f, -2.0f});
    Tensor grad_log_prob(Shape({1, 2}), &backend, {0.0f, 0.0f});
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    const float a0 = std::tanh(0.3f + 0.5f);
    const float a1 = std::tanh(-0.7f - 1.5f);
    EXPECT_NEAR(grads.grad_mean.data()[0], 1.0f * (1.0f - a0 * a0), 1e-6f);
    EXPECT_NEAR(grads.grad_mean.data()[1], -2.0f * (1.0f - a1 * a1), 1e-6f);
    // std == 1 here, so d(u)/d(log_std) is just epsilon.
    EXPECT_NEAR(grads.grad_log_std.data()[0], 1.0f * (1.0f - a0 * a0) * 0.5f, 1e-6f);
    EXPECT_NEAR(grads.grad_log_std.data()[1], -2.0f * (1.0f - a1 * a1) * -1.5f, 1e-6f);
}

// The converse isolation: with no gradient arriving through the action, grad_log_std still
// picks up log_prob's explicit -log_std term (the -grad_log_prob summand), which is the one
// piece of this backward that bypasses u entirely and so is invisible to the check above.
TEST_F(TanhGaussianPolicyTest, BackwardWithZeroGradActionStillCarriesTheMinusLogStdTerm) {
    // epsilon = 0 makes u = mean = 0, so action = 0, 1 - a^2 = 1, and d(log_prob)/d(u) =
    // 2*0*1/(1+c) = 0: the *only* surviving path to log_std is the explicit -log_std term.
    Tensor mean(Shape({1, 1}), &backend, {0.0f});
    Tensor log_std(Shape({1, 1}), &backend, {0.4f});
    Tensor epsilon(Shape({1, 1}), &backend, {0.0f});
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({1, 1}), &backend, {0.0f});
    Tensor grad_log_prob(Shape({1, 1}), &backend, {3.0f});
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    EXPECT_NEAR(grads.grad_mean.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(grads.grad_log_std.data()[0], -3.0f, 1e-6f) << "d(-log_std)/d(log_std) * grad_log_prob";
}

// One fully hand-derived element of the *combined* two-gradient case, computed independently
// from the formulas rather than by running the code:
//   log_std = 0 -> std = 1; epsilon = 1 -> u = 0.5 + 1 = 1.5; a = tanh(1.5) = 0.90514825
//   1 - a^2 = 0.18070664
//   d(log_prob)/d(u) = 2*a*(1-a^2)/(1-a^2+1e-6) = 1.81028645...
//   grad_u = 2.0*0.18070664 + 0.5*1.81028646 = 0.36141328 + 0.90514323 = 1.26655651
//   grad_mean    = grad_u                                      = 1.26655651
//   grad_log_std = grad_u*std*epsilon - grad_log_prob = 1.26655651 - 0.5 = 0.76655651
TEST_F(TanhGaussianPolicyTest, BackwardCombinesBothIncomingGradientsPerHandDerivation) {
    Tensor mean(Shape({1, 1}), &backend, {0.5f});
    Tensor log_std(Shape({1, 1}), &backend, {0.0f});
    Tensor epsilon(Shape({1, 1}), &backend, {1.0f});
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({1, 1}), &backend, {2.0f});
    Tensor grad_log_prob(Shape({1, 1}), &backend, {0.5f});
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    EXPECT_NEAR(grads.grad_mean.data()[0], 1.26655651f, 1e-5f);
    EXPECT_NEAR(grads.grad_log_std.data()[0], 0.76655651f, 1e-5f);
}

TEST_F(TanhGaussianPolicyTest, BackwardEmitsGradientsShapedLikeMean) {
    Tensor mean(Shape({3, 2}), &backend);
    Tensor log_std(Shape({3, 2}), &backend);
    Tensor epsilon(Shape({3, 2}), &backend);
    (void)policy.forward(mean, log_std, epsilon);

    Tensor grad_action(Shape({3, 2}), &backend);
    Tensor grad_log_prob(Shape({3, 2}), &backend);
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    EXPECT_EQ(grads.grad_mean.shape().dim(0), 3);
    EXPECT_EQ(grads.grad_mean.shape().dim(1), 2);
    EXPECT_EQ(grads.grad_log_std.shape().dim(0), 3);
    EXPECT_EQ(grads.grad_log_std.shape().dim(1), 2);
}

// ---------------------------------------------------------------------------------------
// Central finite-difference gradient check -- this mission's load-bearing correctness proof.
//
// The gradient shape here is novel for this codebase: two incoming gradients (one through the
// action, one through the log-probability) combined into one gradient per learned tensor, with
// one of the log_std terms bypassing the shared intermediate u entirely. A hand-derived
// formula of that shape is exactly what finite differences exist to catch, so both mean *and*
// log_std are swept, perturbing each element and re-running the *entire* forward (tanh, log-
// prob and all) rather than any linearized stand-in.
//
// The fixture is deliberately non-uniform (N != action_dim, log_std spanning both signs with
// std from ~0.43 to ~2.72, sign-alternating epsilon, sign-alternating grad_action, and two
// different-signed grad_log_prob rows) rather than small-and-uniform: RWKVModule's mission
// found the hard way that a uniform small-scale fixture can make such a check pass *vacuously*
// against the tolerance floor. Tolerance is relative (1e-3 + 2e-2*|numeric|) for the same
// established reason. Every element's numeric gradient is additionally asserted well clear of
// the floor.
//
// grad_log_prob is per-row-constant by construction, which is not a convenience: log_prob is a
// sum over the row, so backward()'s (N, action_dim) grad_log_prob argument *is* the row scalar
// broadcast, and the scalar loss below applies it once per row accordingly.
// ---------------------------------------------------------------------------------------
namespace {

constexpr int64_t kN = 2;
constexpr int64_t kActionDim = 3;

const std::vector<float>& fd_mean_values() {
    static const std::vector<float> v{0.80f, -0.60f, 1.20f, 0.50f, -0.90f, 1.10f};
    return v;
}
const std::vector<float>& fd_log_std_values() {
    static const std::vector<float> v{0.35f, -0.85f, 1.00f, 0.15f, -0.45f, 0.70f};
    return v;
}
const std::vector<float>& fd_epsilon_values() {
    static const std::vector<float> v{1.30f, -0.70f, 0.40f, -1.60f, 0.95f, -0.25f};
    return v;
}
const std::vector<float>& fd_grad_action_values() {
    static const std::vector<float> v{1.00f, -0.50f, 0.30f, 0.80f, -0.20f, 0.90f};
    return v;
}
// One incoming log-prob gradient per row (SAC's alpha/N is a per-sample constant).
const std::vector<float>& fd_grad_log_prob_per_row() {
    static const std::vector<float> v{0.70f, -0.40f};
    return v;
}
// The same per-row values broadcast across action_dim -- what backward() is handed.
std::vector<float> fd_grad_log_prob_broadcast() {
    std::vector<float> out(static_cast<size_t>(kN * kActionDim));
    for (int64_t n = 0; n < kN; ++n) {
        for (int64_t d = 0; d < kActionDim; ++d) {
            out[static_cast<size_t>(n * kActionDim + d)] = fd_grad_log_prob_per_row()[static_cast<size_t>(n)];
        }
    }
    return out;
}

// L = sum_i action_i * grad_action_i + sum_n log_prob_n * grad_log_prob_n, so dL/d(action) and
// dL/d(log_prob) are exactly the seeds handed to backward().
double scalar_loss(CPUBackend& backend, const std::vector<float>& mean_values,
                   const std::vector<float>& log_std_values) {
    TanhGaussianPolicy local(&backend);
    Tensor mean(Shape({kN, kActionDim}), &backend, mean_values);
    Tensor log_std(Shape({kN, kActionDim}), &backend, log_std_values);
    Tensor epsilon(Shape({kN, kActionDim}), &backend, fd_epsilon_values());
    TanhGaussianSample sample = local.forward(mean, log_std, epsilon);

    double total = 0.0;
    for (int64_t i = 0; i < sample.action.numel(); ++i) {
        total += static_cast<double>(sample.action.data()[i]) *
                 static_cast<double>(fd_grad_action_values()[static_cast<size_t>(i)]);
    }
    for (int64_t n = 0; n < kN; ++n) {
        total += static_cast<double>(sample.log_prob.data()[n]) *
                 static_cast<double>(fd_grad_log_prob_per_row()[static_cast<size_t>(n)]);
    }
    return total;
}

float relative_tolerance(float numeric) { return 1e-3f + 2e-2f * std::fabs(numeric); }

}  // namespace

TEST_F(TanhGaussianPolicyTest, BackwardGradMeanMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor mean(Shape({kN, kActionDim}), &backend, fd_mean_values());
    Tensor log_std(Shape({kN, kActionDim}), &backend, fd_log_std_values());
    Tensor epsilon(Shape({kN, kActionDim}), &backend, fd_epsilon_values());
    Tensor grad_action(Shape({kN, kActionDim}), &backend, fd_grad_action_values());
    Tensor grad_log_prob(Shape({kN, kActionDim}), &backend, fd_grad_log_prob_broadcast());

    (void)policy.forward(mean, log_std, epsilon);
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    ASSERT_EQ(grads.grad_mean.numel(), kN * kActionDim);
    for (size_t idx = 0; idx < fd_mean_values().size(); ++idx) {
        std::vector<float> plus = fd_mean_values();
        std::vector<float> minus = fd_mean_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric = static_cast<float>(
            (scalar_loss(backend, plus, fd_log_std_values()) - scalar_loss(backend, minus, fd_log_std_values())) /
            (2.0 * static_cast<double>(h)));
        EXPECT_NEAR(grads.grad_mean.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "mean index " << idx;
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "mean index " << idx << " is too small to be a non-vacuous check";
    }
}

TEST_F(TanhGaussianPolicyTest, BackwardGradLogStdMatchesFiniteDifference) {
    const float h = 1e-3f;
    Tensor mean(Shape({kN, kActionDim}), &backend, fd_mean_values());
    Tensor log_std(Shape({kN, kActionDim}), &backend, fd_log_std_values());
    Tensor epsilon(Shape({kN, kActionDim}), &backend, fd_epsilon_values());
    Tensor grad_action(Shape({kN, kActionDim}), &backend, fd_grad_action_values());
    Tensor grad_log_prob(Shape({kN, kActionDim}), &backend, fd_grad_log_prob_broadcast());

    (void)policy.forward(mean, log_std, epsilon);
    TanhGaussianGrad grads = policy.backward(grad_action, grad_log_prob);

    ASSERT_EQ(grads.grad_log_std.numel(), kN * kActionDim);
    for (size_t idx = 0; idx < fd_log_std_values().size(); ++idx) {
        std::vector<float> plus = fd_log_std_values();
        std::vector<float> minus = fd_log_std_values();
        plus[idx] += h;
        minus[idx] -= h;
        const float numeric = static_cast<float>(
            (scalar_loss(backend, fd_mean_values(), plus) - scalar_loss(backend, fd_mean_values(), minus)) /
            (2.0 * static_cast<double>(h)));
        EXPECT_NEAR(grads.grad_log_std.data()[static_cast<int64_t>(idx)], numeric, relative_tolerance(numeric))
            << "log_std index " << idx;
        EXPECT_GT(std::fabs(numeric), 1e-2f) << "log_std index " << idx << " is too small to be a non-vacuous check";
    }
}

using TanhGaussianPolicyDeathTest = TanhGaussianPolicyTest;

// Exactly one death test, per the mission's Requirements section: forward() dereferences
// Tensor::data() directly in a raw host loop (exp/tanh/log have no backend primitive), which is
// undefined behavior on a CUDA-backed Tensor, and its three PULSATRIX_ASSERTs are adjacent lines on
// one entry path covering a single guarded-argument role -- one test covers that role (see
// mission_host_loop_guards.md; LinearModuleDeathTest is the mislabeled-Tensor pattern reused
// here, so no real GPU is needed).
//
// backward() is NOT independently guarded and gets no death test: it only reads
// last_action_/last_std_/last_epsilon_, which only forward() ever populates -- after rejecting
// a non-Cpu tensor. No reachable call sequence gets a non-Cpu tensor into that cache, so a
// guard there would be untestable dead code. Identical to Reparameterize::backward()'s
// precedent.
TEST_F(TanhGaussianPolicyDeathTest, ForwardAbortsOnNonCpuMean) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor mean(Shape({1, 2}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    Tensor log_std(Shape({1, 2}), &backend, {0.0f, 0.0f});
    Tensor epsilon(Shape({1, 2}), &backend, {0.0f, 0.0f});
    EXPECT_DEATH({ (void)policy.forward(mean, log_std, epsilon); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
