/** @file continuous_cartpole_env_test.cpp
 *  @brief ContinuousCartPoleEnv physics, action-range validation, termination, determinism.
 *
 *  Two load-bearing tests:
 *
 *  1. HandTracedHalfForceStepMatchesIndependentReference -- the CartPole equations were
 *     evaluated independently of this C++ implementation (double precision, the constants and
 *     integration order fixed in the mission plan) at action = 0.5 and the resulting state is
 *     pinned below as literal constants. This is the correctness proof for the *continuous*
 *     force mapping, which no existing test covers.
 *  2. ActionPlusOneReproducesDiscreteRightPush / ActionMinusOneReproducesDiscreteLeftPush --
 *     a direct, bit-exact trajectory comparison against CartPoleEnv, the environment whose
 *     physics is already verified. These tie this class's physics to a trusted implementation
 *     rather than only to a re-derivation, and they fail immediately if any constant or
 *     integration step here drifts from the discrete original.
 *
 *  Every other physics test in this file is a consistency check.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "pulsatrix/cartpole_env.hpp"
#include "pulsatrix/continuous_cartpole_env.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/environment.hpp"

namespace pulsatrix {
namespace {

constexpr float kTol = 1e-5f;

class ContinuousCartPoleEnvTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor state(float x, float x_dot, float theta, float theta_dot) {
        return Tensor(Shape({1, 4}), &backend, {x, x_dot, theta, theta_dot});
    }

    Tensor action(float fraction) { return Tensor(Shape({1, 1}), &backend, {fraction}); }
};

// ---------------------------------------------------------------------------------------
// Construction / interface description
// ---------------------------------------------------------------------------------------

TEST_F(ContinuousCartPoleEnvTest, ConstructionThrowsOnZeroMaxSteps) {
    EXPECT_THROW({ ContinuousCartPoleEnv env(&backend, 0); }, std::invalid_argument);
}

TEST_F(ContinuousCartPoleEnvTest, ConstructionThrowsOnNegativeMaxSteps) {
    EXPECT_THROW({ ContinuousCartPoleEnv env(&backend, -5); }, std::invalid_argument);
}

// The whole reason this class exists: a continuous action space, reported as such through the
// unchanged Environment interface.
TEST_F(ContinuousCartPoleEnvTest, DescribesAContinuousActionSpace) {
    ContinuousCartPoleEnv env(&backend);
    EXPECT_EQ(env.observation_dim(), 4);
    EXPECT_EQ(env.action_dim(), 1);
    EXPECT_FALSE(env.is_discrete());
    EXPECT_EQ(env.max_steps(), 200);
}

TEST_F(ContinuousCartPoleEnvTest, UsableThroughTheEnvironmentBasePointer) {
    ContinuousCartPoleEnv concrete(&backend, 5);
    Environment& env = concrete;

    Tensor observation = env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_EQ(observation.rank(), 2);
    EXPECT_EQ(observation.shape().dim(0), 1);
    EXPECT_EQ(observation.shape().dim(1), env.observation_dim());

    const StepResult result = env.step(action(0.25f));
    EXPECT_FLOAT_EQ(result.reward, 1.0f);
    EXPECT_FALSE(result.done);
}

// ---------------------------------------------------------------------------------------
// The hand-traced correctness proof (continuous force mapping)
// ---------------------------------------------------------------------------------------

/**
 * Independently computed reference, from the exact formulas (gravity 9.8, masspole 0.1,
 * total_mass 1.1, length 0.5, polemass_length 0.05, force_mag 10.0, tau 0.02), NOT from this
 * implementation:
 *
 *   initial state : x=0.1, x_dot=-0.2, theta=0.05, theta_dot=0.3
 *   action        = 0.5  ->  force = 0.5 * 10.0 = 5.0
 *   costheta      = cos(0.05)                  =  0.9987502603949663
 *   sintheta      = sin(0.05)                  =  0.04997916927067833
 *   temp          = (5.0 + 0.05*0.3^2*sin)/1.1 =  4.545659005692471
 *   thetaacc      = (9.8*sin - cos*temp) / (0.5*(4/3 - 0.1*cos^2/1.1))
 *                                              = -6.518614147978367
 *   xacc          = temp - 0.05*thetaacc*cos/1.1
 *                                              =  4.841589350133731
 *   explicit Euler, positions from PRE-update velocities:
 *   x         = 0.1  + 0.02*(-0.2)             =  0.096
 *   x_dot     = -0.2 + 0.02*4.841589350133731  = -0.1031682129973254
 *   theta     = 0.05 + 0.02*0.3                =  0.056
 *   theta_dot = 0.3  + 0.02*(-6.518614147978)  =  0.16962771704043264
 *
 * Note this is the *same initial state* cartpole_env_test.cpp's own hand-traced test uses, so
 * the two references are directly comparable: at action=1.0 the force is 10.0 and the results
 * coincide with that test's pinned values (asserted separately, below).
 */
TEST_F(ContinuousCartPoleEnvTest, HandTracedHalfForceStepMatchesIndependentReference) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.1f, -0.2f, 0.05f, 0.3f));

    const StepResult result = env.step(action(0.5f));

    EXPECT_NEAR(result.observation.data()[0], 0.096f, kTol);
    EXPECT_NEAR(result.observation.data()[1], -0.1031682129973254f, kTol);
    EXPECT_NEAR(result.observation.data()[2], 0.056f, kTol);
    EXPECT_NEAR(result.observation.data()[3], 0.16962771704043264f, kTol);
    EXPECT_FLOAT_EQ(result.reward, 1.0f);
    EXPECT_FALSE(result.done);
    EXPECT_EQ(env.step_count(), 1);
}

/**
 * Second independent reference, at the value the discrete environment cannot express at all:
 * action = 0.0 means force = 0.0, so the pole falls under gravity alone.
 *
 *   temp      = (0.0 + 0.05*0.3^2*sin(0.05))/1.1 =  0.00020446023792550225
 *   thetaacc  =  0.7879791281169457
 *   xacc      = -0.035568010643556226
 *   x         =  0.096
 *   x_dot     = -0.20071136021287114
 *   theta     =  0.056
 *   theta_dot =  0.3157595825623389
 *
 * thetaacc is *positive* here (the pole, already tilted to +0.05 rad, accelerates further
 * from vertical) whereas both discrete actions produce a large-magnitude thetaacc dominated
 * by the cart force. A zero-force step is the qualitatively new behavior a continuous action
 * space buys, so it gets its own pinned reference.
 */
TEST_F(ContinuousCartPoleEnvTest, HandTracedZeroForceStepMatchesIndependentReference) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.1f, -0.2f, 0.05f, 0.3f));

    const StepResult result = env.step(action(0.0f));

    EXPECT_NEAR(result.observation.data()[0], 0.096f, kTol);
    EXPECT_NEAR(result.observation.data()[1], -0.20071136021287114f, kTol);
    EXPECT_NEAR(result.observation.data()[2], 0.056f, kTol);
    EXPECT_NEAR(result.observation.data()[3], 0.3157595825623389f, kTol);
}

// ---------------------------------------------------------------------------------------
// Equivalence with the already-verified discrete environment at the two extremes
// ---------------------------------------------------------------------------------------

// Runs both environments from the same initial state for `steps` steps and asserts the
// observations, rewards and done flags agree bit-exactly at every step. Both classes compute
// in double and emit float, so if the constants and integration order match, the emitted
// floats are identical -- EXPECT_FLOAT_EQ, not a tolerance, is the right assertion.
void ExpectTrajectoriesIdentical(CPUBackend* backend, float continuous_action, float discrete_index,
                                 const Tensor& initial_state, int steps) {
    ContinuousCartPoleEnv continuous(backend, 1000);
    CartPoleEnv discrete(backend, 1000);

    const Tensor continuous_start = continuous.reset(initial_state);
    const Tensor discrete_start = discrete.reset(initial_state);
    for (int64_t i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(continuous_start.data()[i], discrete_start.data()[i]) << "initial component " << i;
    }

    Tensor continuous_a(Shape({1, 1}), backend, {continuous_action});
    Tensor discrete_a(Shape({1, 1}), backend, {discrete_index});

    for (int step = 0; step < steps; ++step) {
        const StepResult c = continuous.step(continuous_a);
        const StepResult d = discrete.step(discrete_a);
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(c.observation.data()[i], d.observation.data()[i])
                << "step " << step << ", component " << i;
        }
        EXPECT_FLOAT_EQ(c.reward, d.reward) << "step " << step;
        EXPECT_EQ(c.done, d.done) << "step " << step;
    }
}

// THE cross-check: action=+1.0 is a full-strength rightward push, which is exactly what
// CartPoleEnv's discrete action 1 does.
TEST_F(ContinuousCartPoleEnvTest, ActionPlusOneReproducesDiscreteRightPush) {
    ExpectTrajectoriesIdentical(&backend, 1.0f, 1.0f, state(0.1f, -0.2f, 0.05f, 0.3f), 12);
    ExpectTrajectoriesIdentical(&backend, 1.0f, 1.0f, state(0.0f, 0.0f, 0.0f, 0.0f), 12);
}

// ... and action=-1.0 is a full-strength leftward push: CartPoleEnv's discrete action 0.
TEST_F(ContinuousCartPoleEnvTest, ActionMinusOneReproducesDiscreteLeftPush) {
    ExpectTrajectoriesIdentical(&backend, -1.0f, 0.0f, state(0.1f, -0.2f, 0.05f, 0.3f), 12);
    ExpectTrajectoriesIdentical(&backend, -1.0f, 0.0f, state(-0.3f, 0.15f, -0.04f, -0.2f), 12);
}

// A direct, value-level restatement of the same claim against cartpole_env_test.cpp's own
// pinned hand-traced constants: at action=1.0 this class must land on the literal numbers
// that test proved for the discrete environment. If ExpectTrajectoriesIdentical ever passed
// because *both* classes were wrong in the same way, this would still catch it.
TEST_F(ContinuousCartPoleEnvTest, ActionPlusOneMatchesTheDiscreteHandTracedReference) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.1f, -0.2f, 0.05f, 0.3f));

    const StepResult result = env.step(action(1.0f));

    EXPECT_NEAR(result.observation.data()[0], 0.096f, kTol);
    EXPECT_NEAR(result.observation.data()[1], -0.005625065781779709f, kTol);
    EXPECT_NEAR(result.observation.data()[2], 0.056f, kTol);
    EXPECT_NEAR(result.observation.data()[3], 0.02349585151852651f, kTol);
}

// ---------------------------------------------------------------------------------------
// The continuous action space's own behavior: force scales with the action
// ---------------------------------------------------------------------------------------

// Explicit (not semi-implicit) Euler is observable in exactly one place: position after the
// first step depends only on the PRE-update velocity. From rest, any nonzero push accelerates
// the cart but must leave x exactly 0 for this first step.
TEST_F(ContinuousCartPoleEnvTest, UsesExplicitEulerOrderNotSemiImplicit) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    const StepResult result = env.step(action(0.5f));

    EXPECT_FLOAT_EQ(result.observation.data()[0], 0.0f);  // x unchanged
    EXPECT_FLOAT_EQ(result.observation.data()[2], 0.0f);  // theta unchanged
    EXPECT_GT(result.observation.data()[1], 0.0f);        // x_dot pushed right
    EXPECT_LT(result.observation.data()[3], 0.0f);        // pole tips left relative to cart
}

TEST_F(ContinuousCartPoleEnvTest, OppositeActionsAreMirrorImagesFromRest) {
    ContinuousCartPoleEnv right_env(&backend);
    (void)right_env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const StepResult right = right_env.step(action(0.5f));

    ContinuousCartPoleEnv left_env(&backend);
    (void)left_env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const StepResult left = left_env.step(action(-0.5f));

    EXPECT_NEAR(right.observation.data()[1], -left.observation.data()[1], kTol);
    EXPECT_NEAR(right.observation.data()[3], -left.observation.data()[3], kTol);
    EXPECT_GT(right.observation.data()[1], 0.0f);
    EXPECT_LT(left.observation.data()[1], 0.0f);
}

// From rest (theta = theta_dot = 0), every term in the acceleration chain is linear in force:
// temp = force/total_mass, thetaacc = -temp / (length*(4/3 - masspole/total_mass)), and xacc
// is a linear combination of the two. So x_dot after one step must scale exactly with the
// action. Linearity is the property that distinguishes a genuinely scaled force from a
// thresholded one that merely happens to agree at +/-1.
TEST_F(ContinuousCartPoleEnvTest, ResultingVelocityIsLinearInTheAction) {
    ContinuousCartPoleEnv full(&backend);
    (void)full.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const float full_x_dot = full.step(action(1.0f)).observation.data()[1];

    ContinuousCartPoleEnv half(&backend);
    (void)half.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const float half_x_dot = half.step(action(0.5f)).observation.data()[1];

    ContinuousCartPoleEnv quarter(&backend);
    (void)quarter.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const float quarter_x_dot = quarter.step(action(0.25f)).observation.data()[1];

    EXPECT_NEAR(half_x_dot, full_x_dot * 0.5f, kTol);
    EXPECT_NEAR(quarter_x_dot, full_x_dot * 0.25f, kTol);
}

// The degenerate action the discrete environment cannot express: zero force leaves the cart's
// velocity essentially untouched (only the pole's angular-momentum coupling perturbs it).
TEST_F(ContinuousCartPoleEnvTest, ZeroActionAppliesNoCartForce) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    const StepResult result = env.step(action(0.0f));

    // With theta = theta_dot = 0 and force = 0, every acceleration term is exactly zero, so
    // the state must be completely unchanged.
    EXPECT_FLOAT_EQ(result.observation.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(result.observation.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(result.observation.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(result.observation.data()[3], 0.0f);
}

// ---------------------------------------------------------------------------------------
// Termination conditions -- each tested independently of the others
// ---------------------------------------------------------------------------------------

TEST_F(ContinuousCartPoleEnvTest, TerminatesOnPositionBoundRegardlessOfAction) {
    for (float a : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
        ContinuousCartPoleEnv env(&backend);
        // |x| = 2.45 > 2.4 already, and x_dot = 0 keeps it there through the position update.
        (void)env.reset(state(2.45f, 0.0f, 0.0f, 0.0f));
        const StepResult result = env.step(action(a));
        EXPECT_TRUE(result.done) << "action " << a;
        EXPECT_FLOAT_EQ(result.reward, 1.0f) << "action " << a;
    }
}

TEST_F(ContinuousCartPoleEnvTest, TerminatesOnNegativePositionBound) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(-2.45f, 0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(env.step(action(0.0f)).done);
}

TEST_F(ContinuousCartPoleEnvTest, TerminatesOnAngleBoundRegardlessOfAction) {
    for (float a : {-1.0f, 0.0f, 1.0f}) {
        ContinuousCartPoleEnv env(&backend);
        // |theta| = 0.25 > 0.20943951, theta_dot = 0 keeps it past the threshold.
        (void)env.reset(state(0.0f, 0.0f, 0.25f, 0.0f));
        const StepResult result = env.step(action(a));
        EXPECT_TRUE(result.done) << "action " << a;
    }
}

TEST_F(ContinuousCartPoleEnvTest, TerminatesOnNegativeAngleBound) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, -0.25f, 0.0f));
    EXPECT_TRUE(env.step(action(0.0f)).done);
}

TEST_F(ContinuousCartPoleEnvTest, DoesNotTerminateJustInsideBothThresholds) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(2.39f, 0.0f, 0.2f, 0.0f));
    EXPECT_FALSE(env.step(action(0.0f)).done);
}

TEST_F(ContinuousCartPoleEnvTest, TerminatesAfterExactlyMaxSteps) {
    ContinuousCartPoleEnv env(&backend, 3);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    EXPECT_FALSE(env.step(action(0.0f)).done);
    EXPECT_FALSE(env.step(action(0.0f)).done);
    const StepResult third = env.step(action(0.0f));
    EXPECT_TRUE(third.done);
    EXPECT_EQ(env.step_count(), 3);
}

TEST_F(ContinuousCartPoleEnvTest, ResetClearsTheStepCounter) {
    ContinuousCartPoleEnv env(&backend, 3);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    (void)env.step(action(0.0f));
    (void)env.step(action(0.0f));
    EXPECT_EQ(env.step_count(), 2);

    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_EQ(env.step_count(), 0);
    EXPECT_FALSE(env.step(action(0.0f)).done);
}

// ---------------------------------------------------------------------------------------
// reset() determinism
// ---------------------------------------------------------------------------------------

TEST_F(ContinuousCartPoleEnvTest, ResetIsDeterministicForAGivenSeed) {
    ContinuousCartPoleEnv a(&backend, 200, 7);
    ContinuousCartPoleEnv b(&backend, 200, 7);

    for (int episode = 0; episode < 3; ++episode) {
        const Tensor first = a.reset();
        const Tensor second = b.reset();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(first.data()[i], second.data()[i]) << "episode " << episode << ", component " << i;
        }
    }
}

TEST_F(ContinuousCartPoleEnvTest, DifferentSeedsGiveDifferentInitialStates) {
    ContinuousCartPoleEnv a(&backend, 200, 7);
    ContinuousCartPoleEnv b(&backend, 200, 12345);

    const Tensor first = a.reset();
    const Tensor second = b.reset();

    bool any_different = false;
    for (int64_t i = 0; i < 4; ++i) {
        if (first.data()[i] != second.data()[i]) {
            any_different = true;
        }
    }
    EXPECT_TRUE(any_different);
}

// The LCG is the same generator CartPoleEnv uses, seeded and advanced identically, so a given
// seed yields the same reset sequence in both classes. Pinning that means a Phase 4
// continuous-control experiment can be compared against a Phase 2/3 discrete one from a
// genuinely identical starting distribution.
TEST_F(ContinuousCartPoleEnvTest, ResetSequenceMatchesTheDiscreteEnvironmentForTheSameSeed) {
    ContinuousCartPoleEnv continuous(&backend, 200, 7);
    CartPoleEnv discrete(&backend, 200, 7);

    for (int episode = 0; episode < 3; ++episode) {
        const Tensor c = continuous.reset();
        const Tensor d = discrete.reset();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(c.data()[i], d.data()[i]) << "episode " << episode << ", component " << i;
        }
    }
}

TEST_F(ContinuousCartPoleEnvTest, RandomResetStaysInsideTheConventionalPerturbationRange) {
    ContinuousCartPoleEnv env(&backend);
    for (int episode = 0; episode < 20; ++episode) {
        const Tensor observation = env.reset();
        ASSERT_EQ(observation.shape().dim(1), 4);
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_LE(std::abs(observation.data()[i]), 0.05f + kTol) << "component " << i;
        }
    }
}

TEST_F(ContinuousCartPoleEnvTest, ExplicitResetOverridesTheStateExactly) {
    ContinuousCartPoleEnv env(&backend);
    const Tensor observation = env.reset(state(1.5f, -0.25f, 0.1f, 0.75f));
    EXPECT_FLOAT_EQ(observation.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(observation.data()[1], -0.25f);
    EXPECT_FLOAT_EQ(observation.data()[2], 0.1f);
    EXPECT_FLOAT_EQ(observation.data()[3], 0.75f);
}

// ---------------------------------------------------------------------------------------
// Validation (external boundary -> throws, not asserts)
// ---------------------------------------------------------------------------------------

TEST_F(ContinuousCartPoleEnvTest, StepThrowsIfCalledBeforeReset) {
    ContinuousCartPoleEnv env(&backend);
    EXPECT_THROW({ (void)env.step(action(0.0f)); }, std::invalid_argument);
}

TEST_F(ContinuousCartPoleEnvTest, ResetThrowsOnWrongInitialStateShape) {
    ContinuousCartPoleEnv env(&backend);
    Tensor wrong_width(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});
    EXPECT_THROW({ (void)env.reset(wrong_width); }, std::invalid_argument);

    Tensor wrong_rank(Shape({4}), &backend, {0.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_THROW({ (void)env.reset(wrong_rank); }, std::invalid_argument);

    Tensor wrong_batch(Shape({2, 4}), &backend);
    EXPECT_THROW({ (void)env.reset(wrong_batch); }, std::invalid_argument);
}

TEST_F(ContinuousCartPoleEnvTest, StepThrowsOnWrongActionShape) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    Tensor wrong_rank(Shape({1}), &backend, {0.5f});
    EXPECT_THROW({ (void)env.step(wrong_rank); }, std::invalid_argument);

    Tensor two_actions(Shape({1, 2}), &backend, {0.5f, 0.5f});
    EXPECT_THROW({ (void)env.step(two_actions); }, std::invalid_argument);

    Tensor batched(Shape({2, 1}), &backend, {0.5f, 0.5f});
    EXPECT_THROW({ (void)env.step(batched); }, std::invalid_argument);
}

TEST_F(ContinuousCartPoleEnvTest, StepAcceptsTheWholeInteriorOfTheActionRange) {
    ContinuousCartPoleEnv env(&backend, 1000);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    for (float a : {-1.0f, -0.999f, -0.5f, -1e-7f, 0.0f, 1e-7f, 0.5f, 0.999f, 1.0f}) {
        EXPECT_NO_THROW({ (void)env.step(action(a)); }) << "action " << a;
    }
}

TEST_F(ContinuousCartPoleEnvTest, StepThrowsOnActionBeyondTheTolerance) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_THROW({ (void)env.step(action(1.0001f)); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(-1.0001f)); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(2.0f)); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(-5.0f)); }, std::invalid_argument);
}

// Every comparison against NaN is false, so a range check phrased as "throw if outside the
// band" would wave NaN through and permanently poison the state. Phrased as "throw unless
// inside," it is rejected.
TEST_F(ContinuousCartPoleEnvTest, StepThrowsOnNonFiniteAction) {
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_THROW({ (void)env.step(action(std::nanf(""))); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(INFINITY)); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(-INFINITY)); }, std::invalid_argument);
}

// A tanh-squashed policy's output can round to a hair beyond 1.0. That must be accepted AND
// clamped: the resulting force has to be exactly force_mag, not force_mag * 1.00001. Bit-exact
// equality with the action=1.0 trajectory is the only assertion that actually proves clamping
// happened -- a tolerance would pass either way.
TEST_F(ContinuousCartPoleEnvTest, ActionJustOutsideTheRangeIsAcceptedAndClampedExactly) {
    ContinuousCartPoleEnv probe(&backend);
    (void)probe.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    EXPECT_NO_THROW({ (void)probe.step(action(1.00001f)); });

    ContinuousCartPoleEnv tolerated(&backend);
    (void)tolerated.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    const StepResult from_tolerated = tolerated.step(action(1.00001f));

    ContinuousCartPoleEnv exact(&backend);
    (void)exact.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    const StepResult from_exact = exact.step(action(1.0f));

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(from_tolerated.observation.data()[i], from_exact.observation.data()[i])
            << "component " << i;
    }
}

TEST_F(ContinuousCartPoleEnvTest, ActionJustBelowTheRangeIsAcceptedAndClampedExactly) {
    ContinuousCartPoleEnv probe(&backend);
    (void)probe.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    EXPECT_NO_THROW({ (void)probe.step(action(-1.00001f)); });

    ContinuousCartPoleEnv tolerated(&backend);
    (void)tolerated.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    const StepResult from_tolerated = tolerated.step(action(-1.00001f));

    ContinuousCartPoleEnv exact(&backend);
    (void)exact.reset(state(0.1f, -0.2f, 0.05f, 0.3f));
    const StepResult from_exact = exact.step(action(-1.0f));

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(from_tolerated.observation.data()[i], from_exact.observation.data()[i])
            << "component " << i;
    }
}

using ContinuousCartPoleEnvDeathTest = ContinuousCartPoleEnvTest;

// step() runs the physics against Tensor::data() directly -- undefined behavior on a
// CUDA-backed Tensor. One death test, matching CartPoleEnv's precedent: one guarded argument
// role.
TEST_F(ContinuousCartPoleEnvDeathTest, StepAbortsOnNonCpuAction) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    ContinuousCartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    Tensor cuda_action(Shape({1, 1}), &backend, {0.5f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)env.step(cuda_action); }, "PULSATRIX_ASSERT failed");
}

// ---------------------------------------------------------------------------------------
// Full episode loop through the base interface
// ---------------------------------------------------------------------------------------

/**
 * @brief Test-only continuous agent: emits a deterministic, observation-independent force
 *        fraction sweeping [-1, 1] from its own small LCG. Not an algorithm and not
 *        production code -- it exists to prove the continuous environment supports a full
 *        episode loop through Environment& before SAC (Mission 1/2) exists.
 */
class SweepingContinuousAgent {
public:
    SweepingContinuousAgent(DeviceBackend* backend, uint32_t seed) : backend_(backend), lcg_state_(seed) {}

    [[nodiscard]] Tensor act() {
        lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
        const float unit = static_cast<float>((lcg_state_ >> 8) & 0xFFFFu) / 65535.0f;  // [0, 1]
        return Tensor(Shape({1, 1}), backend_, {unit * 2.0f - 1.0f});                   // [-1, 1]
    }

private:
    DeviceBackend* backend_;
    uint32_t lcg_state_;
};

TEST_F(ContinuousCartPoleEnvTest, RandomContinuousAgentRunsAFullEpisodeToTermination) {
    ContinuousCartPoleEnv concrete(&backend, 200, 42);
    Environment& env = concrete;
    SweepingContinuousAgent agent(&backend, 1234);

    Tensor observation = env.reset();
    ASSERT_EQ(observation.shape().dim(0), 1);
    ASSERT_EQ(observation.shape().dim(1), 4);

    int64_t steps = 0;
    float total_reward = 0.0f;
    bool done = false;
    while (!done) {
        const StepResult result = env.step(agent.act());
        EXPECT_FLOAT_EQ(result.reward, 1.0f);
        EXPECT_EQ(result.observation.shape().dim(1), 4);

        total_reward += result.reward;
        observation = result.observation;
        done = result.done;
        ++steps;
        ASSERT_LE(steps, concrete.max_steps()) << "episode failed to terminate within max_steps";
    }

    EXPECT_GT(steps, 0);
    // Reward is exactly 1.0 per step, so the return is exactly the episode length.
    EXPECT_FLOAT_EQ(total_reward, static_cast<float>(steps));
    EXPECT_EQ(concrete.step_count(), steps);
}

}  // namespace
}  // namespace pulsatrix
