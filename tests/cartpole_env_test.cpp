/** @file cartpole_env_test.cpp
 *  @brief CartPoleEnv physics, termination, validation -- plus the Environment+Agent loop
 *         proof (a test-local RandomDiscreteAgent running a full episode end to end).
 *
 *  The load-bearing test here is HandTracedSingleStepMatchesIndependentReference: the exact
 *  CartPole equations were evaluated independently of this C++ implementation (double
 *  precision, the constants and integration order fixed in the mission plan) and the
 *  resulting state is pinned below as literal constants. Every other physics test in this
 *  file is a consistency check; that one is the correctness proof.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "pulsatrix/agent.hpp"
#include "pulsatrix/cartpole_env.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/environment.hpp"

namespace pulsatrix {
namespace {

constexpr float kTol = 1e-5f;

class CartPoleEnvTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor state(float x, float x_dot, float theta, float theta_dot) {
        return Tensor(Shape({1, 4}), &backend, {x, x_dot, theta, theta_dot});
    }

    Tensor action(float index) { return Tensor(Shape({1, 1}), &backend, {index}); }
};

// ---------------------------------------------------------------------------------------
// Construction / interface description
// ---------------------------------------------------------------------------------------

TEST_F(CartPoleEnvTest, ConstructionThrowsOnZeroMaxSteps) {
    EXPECT_THROW({ CartPoleEnv env(&backend, 0); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, ConstructionThrowsOnNegativeMaxSteps) {
    EXPECT_THROW({ CartPoleEnv env(&backend, -5); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, DescribesItsSpaces) {
    CartPoleEnv env(&backend);
    EXPECT_EQ(env.observation_dim(), 4);
    EXPECT_EQ(env.action_dim(), 2);
    EXPECT_TRUE(env.is_discrete());
    EXPECT_EQ(env.max_steps(), 200);
}

TEST_F(CartPoleEnvTest, UsableThroughTheEnvironmentBasePointer) {
    CartPoleEnv concrete(&backend, 5);
    Environment& env = concrete;

    Tensor observation = env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_EQ(observation.rank(), 2);
    EXPECT_EQ(observation.shape().dim(0), 1);
    EXPECT_EQ(observation.shape().dim(1), env.observation_dim());

    const StepResult result = env.step(action(1.0f));
    EXPECT_FLOAT_EQ(result.reward, 1.0f);
    EXPECT_FALSE(result.done);
}

// ---------------------------------------------------------------------------------------
// The hand-traced correctness proof
// ---------------------------------------------------------------------------------------

/**
 * Independently computed reference, from the exact formulas (gravity 9.8, masspole 0.1,
 * total_mass 1.1, length 0.5, polemass_length 0.05, force_mag 10.0, tau 0.02), NOT from
 * this implementation:
 *
 *   initial state : x=0.1, x_dot=-0.2, theta=0.05, theta_dot=0.3, action=1 (force=+10)
 *   costheta      = cos(0.05)              =  0.9987502603949663
 *   sintheta      = sin(0.05)              =  0.04997916927067833
 *   temp          = (10 + 0.05*0.3^2*sin)/1.1
 *                                          =  9.091113551147014
 *   thetaacc      = (9.8*sin - cos*temp) / (0.5*(4/3 - 0.1*cos^2/1.1))
 *                                          = -13.825207424073673
 *   xacc          = temp - 0.05*thetaacc*cos/1.1
 *                                          =  9.718746710911015
 *   explicit Euler, positions from PRE-update velocities:
 *   x        = 0.1  + 0.02*(-0.2)          =  0.096
 *   x_dot    = -0.2 + 0.02*9.718746710911  = -0.005625065781779709
 *   theta    = 0.05 + 0.02*0.3             =  0.056
 *   theta_dot= 0.3  + 0.02*(-13.8252074241)=  0.02349585151852651
 */
TEST_F(CartPoleEnvTest, HandTracedSingleStepMatchesIndependentReference) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.1f, -0.2f, 0.05f, 0.3f));

    const StepResult result = env.step(action(1.0f));

    EXPECT_NEAR(result.observation.data()[0], 0.096f, kTol);
    EXPECT_NEAR(result.observation.data()[1], -0.005625065781779709f, kTol);
    EXPECT_NEAR(result.observation.data()[2], 0.056f, kTol);
    EXPECT_NEAR(result.observation.data()[3], 0.02349585151852651f, kTol);
    EXPECT_FLOAT_EQ(result.reward, 1.0f);
    EXPECT_FALSE(result.done);
    EXPECT_EQ(env.step_count(), 1);
}

// Explicit (not semi-implicit) Euler is observable in exactly one place: position after the
// first step depends only on the PRE-update velocity. From x=0, x_dot=0, a rightward push
// accelerates the cart but must leave x exactly 0 for this first step. Semi-implicit Euler
// would have moved it. This pins the integration order the mission fixed.
TEST_F(CartPoleEnvTest, UsesExplicitEulerOrderNotSemiImplicit) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    const StepResult result = env.step(action(1.0f));

    EXPECT_FLOAT_EQ(result.observation.data()[0], 0.0f);  // x unchanged
    EXPECT_FLOAT_EQ(result.observation.data()[2], 0.0f);  // theta unchanged
    EXPECT_GT(result.observation.data()[1], 0.0f);        // x_dot pushed right
    EXPECT_LT(result.observation.data()[3], 0.0f);        // pole tips left relative to cart
}

TEST_F(CartPoleEnvTest, LeftAndRightPushesAreMirrorImagesFromRest) {
    CartPoleEnv right_env(&backend);
    (void)right_env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const StepResult right = right_env.step(action(1.0f));

    CartPoleEnv left_env(&backend);
    (void)left_env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    const StepResult left = left_env.step(action(0.0f));

    EXPECT_NEAR(right.observation.data()[1], -left.observation.data()[1], kTol);
    EXPECT_NEAR(right.observation.data()[3], -left.observation.data()[3], kTol);
    EXPECT_GT(right.observation.data()[1], 0.0f);
    EXPECT_LT(left.observation.data()[1], 0.0f);
}

// ---------------------------------------------------------------------------------------
// Termination conditions -- each tested independently of the others
// ---------------------------------------------------------------------------------------

TEST_F(CartPoleEnvTest, TerminatesOnPositionBoundRegardlessOfAction) {
    for (float a : {0.0f, 1.0f}) {
        CartPoleEnv env(&backend);
        // |x| = 2.45 > 2.4 already, and x_dot = 0 keeps it there through the position update.
        (void)env.reset(state(2.45f, 0.0f, 0.0f, 0.0f));
        const StepResult result = env.step(action(a));
        EXPECT_TRUE(result.done) << "action " << a;
        EXPECT_FLOAT_EQ(result.reward, 1.0f);
    }
}

TEST_F(CartPoleEnvTest, TerminatesOnNegativePositionBound) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(-2.45f, 0.0f, 0.0f, 0.0f));
    EXPECT_TRUE(env.step(action(1.0f)).done);
}

TEST_F(CartPoleEnvTest, TerminatesOnAngleBoundRegardlessOfAction) {
    for (float a : {0.0f, 1.0f}) {
        CartPoleEnv env(&backend);
        // |theta| = 0.25 > 0.20943951, theta_dot = 0 keeps it past the threshold.
        (void)env.reset(state(0.0f, 0.0f, 0.25f, 0.0f));
        const StepResult result = env.step(action(a));
        EXPECT_TRUE(result.done) << "action " << a;
    }
}

TEST_F(CartPoleEnvTest, TerminatesOnNegativeAngleBound) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, -0.25f, 0.0f));
    EXPECT_TRUE(env.step(action(1.0f)).done);
}

TEST_F(CartPoleEnvTest, DoesNotTerminateJustInsideBothThresholds) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(2.39f, 0.0f, 0.2f, 0.0f));
    EXPECT_FALSE(env.step(action(1.0f)).done);
}

TEST_F(CartPoleEnvTest, TerminatesAfterExactlyMaxSteps) {
    CartPoleEnv env(&backend, 3);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    EXPECT_FALSE(env.step(action(1.0f)).done);
    EXPECT_FALSE(env.step(action(0.0f)).done);
    const StepResult third = env.step(action(1.0f));
    EXPECT_TRUE(third.done);
    EXPECT_EQ(env.step_count(), 3);
}

TEST_F(CartPoleEnvTest, ResetClearsTheStepCounter) {
    CartPoleEnv env(&backend, 3);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    (void)env.step(action(1.0f));
    (void)env.step(action(1.0f));
    EXPECT_EQ(env.step_count(), 2);

    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_EQ(env.step_count(), 0);
    EXPECT_FALSE(env.step(action(1.0f)).done);
}

// ---------------------------------------------------------------------------------------
// reset() determinism
// ---------------------------------------------------------------------------------------

TEST_F(CartPoleEnvTest, ResetIsDeterministicForAGivenSeed) {
    CartPoleEnv a(&backend, 200, 7);
    CartPoleEnv b(&backend, 200, 7);

    for (int episode = 0; episode < 3; ++episode) {
        const Tensor first = a.reset();
        const Tensor second = b.reset();
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(first.data()[i], second.data()[i]) << "episode " << episode << ", component " << i;
        }
    }
}

TEST_F(CartPoleEnvTest, DifferentSeedsGiveDifferentInitialStates) {
    CartPoleEnv a(&backend, 200, 7);
    CartPoleEnv b(&backend, 200, 12345);

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

TEST_F(CartPoleEnvTest, RandomResetStaysInsideTheConventionalPerturbationRange) {
    CartPoleEnv env(&backend);
    for (int episode = 0; episode < 20; ++episode) {
        const Tensor observation = env.reset();
        ASSERT_EQ(observation.shape().dim(1), 4);
        for (int64_t i = 0; i < 4; ++i) {
            EXPECT_LE(std::abs(observation.data()[i]), 0.05f + kTol) << "component " << i;
        }
    }
}

TEST_F(CartPoleEnvTest, ExplicitResetOverridesTheStateExactly) {
    CartPoleEnv env(&backend);
    const Tensor observation = env.reset(state(1.5f, -0.25f, 0.1f, 0.75f));
    EXPECT_FLOAT_EQ(observation.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(observation.data()[1], -0.25f);
    EXPECT_FLOAT_EQ(observation.data()[2], 0.1f);
    EXPECT_FLOAT_EQ(observation.data()[3], 0.75f);
}

// ---------------------------------------------------------------------------------------
// Validation (external boundary -> throws, not asserts)
// ---------------------------------------------------------------------------------------

TEST_F(CartPoleEnvTest, StepThrowsIfCalledBeforeReset) {
    CartPoleEnv env(&backend);
    EXPECT_THROW({ (void)env.step(action(1.0f)); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, ResetThrowsOnWrongInitialStateShape) {
    CartPoleEnv env(&backend);
    Tensor wrong_width(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});
    EXPECT_THROW({ (void)env.reset(wrong_width); }, std::invalid_argument);

    Tensor wrong_rank(Shape({4}), &backend, {0.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_THROW({ (void)env.reset(wrong_rank); }, std::invalid_argument);

    Tensor wrong_batch(Shape({2, 4}), &backend);
    EXPECT_THROW({ (void)env.reset(wrong_batch); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, StepThrowsOnWrongActionShape) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));

    Tensor wrong_rank(Shape({1}), &backend, {1.0f});
    EXPECT_THROW({ (void)env.step(wrong_rank); }, std::invalid_argument);

    Tensor two_actions(Shape({1, 2}), &backend, {1.0f, 0.0f});
    EXPECT_THROW({ (void)env.step(two_actions); }, std::invalid_argument);

    Tensor batched(Shape({2, 1}), &backend, {1.0f, 0.0f});
    EXPECT_THROW({ (void)env.step(batched); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, StepThrowsOnOutOfRangeActionIndex) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_THROW({ (void)env.step(action(2.0f)); }, std::invalid_argument);
    EXPECT_THROW({ (void)env.step(action(-1.0f)); }, std::invalid_argument);
}

TEST_F(CartPoleEnvTest, StepThrowsOnNonIntegerActionEncoding) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_THROW({ (void)env.step(action(0.5f)); }, std::invalid_argument);
}

// A float round-trip of a whole-number index must still be accepted -- the 1e-4 tolerance
// exists for exactly this, and a check that only accepted bit-exact 0.0f/1.0f would be a
// trap for any policy that computes its action index in floating point.
TEST_F(CartPoleEnvTest, StepAcceptsNearIntegerActionEncoding) {
    CartPoleEnv env(&backend);
    (void)env.reset(state(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_NO_THROW({ (void)env.step(action(1.00001f)); });
}

// ---------------------------------------------------------------------------------------
// Environment + Agent loop
// ---------------------------------------------------------------------------------------

/**
 * @brief Test-only Agent: picks a uniformly random discrete action from its own small
 *        deterministic LCG. Not an algorithm and not production code -- it exists solely to
 *        prove the Environment+Agent pair supports a full episode loop before any real
 *        policy exists (Phase 2-4 build those).
 */
class RandomDiscreteAgent : public Agent {
public:
    RandomDiscreteAgent(DeviceBackend* backend, int64_t action_dim, uint32_t seed)
        : backend_(backend), action_dim_(action_dim), lcg_state_(seed) {}

    [[nodiscard]] Tensor act(const Tensor& observation) override {
        (void)observation;  // A random policy is, by construction, observation-independent.
        lcg_state_ = lcg_state_ * 1664525u + 1013904223u;
        const int64_t index = static_cast<int64_t>((lcg_state_ >> 16) % static_cast<uint32_t>(action_dim_));
        return Tensor(Shape({1, 1}), backend_, {static_cast<float>(index)});
    }

private:
    DeviceBackend* backend_;
    int64_t action_dim_;
    uint32_t lcg_state_;
};

TEST_F(CartPoleEnvTest, RandomAgentRunsAFullEpisodeToTermination) {
    CartPoleEnv env(&backend, 200, 42);
    RandomDiscreteAgent agent(&backend, env.action_dim(), 1234);

    Tensor observation = env.reset();
    ASSERT_EQ(observation.shape().dim(0), 1);
    ASSERT_EQ(observation.shape().dim(1), 4);

    int64_t steps = 0;
    float total_reward = 0.0f;
    bool done = false;
    while (!done) {
        const Tensor chosen = agent.act(observation);
        ASSERT_EQ(chosen.shape().dim(0), 1);
        ASSERT_EQ(chosen.shape().dim(1), 1);

        const StepResult result = env.step(chosen);
        EXPECT_FLOAT_EQ(result.reward, 1.0f);
        EXPECT_EQ(result.observation.shape().dim(0), 1);
        EXPECT_EQ(result.observation.shape().dim(1), 4);

        total_reward += result.reward;
        observation = result.observation;
        done = result.done;
        ++steps;
        ASSERT_LE(steps, env.max_steps()) << "episode failed to terminate within max_steps";
    }

    EXPECT_GT(steps, 0);
    EXPECT_LE(steps, env.max_steps());
    // Reward is exactly 1.0 per step, so the return is exactly the episode length.
    EXPECT_FLOAT_EQ(total_reward, static_cast<float>(steps));
    EXPECT_EQ(env.step_count(), steps);
}

// The whole point of the Environment/Agent split: a loop written against the base classes
// alone compiles and runs, with no knowledge of CartPole or of the agent's concrete type.
TEST_F(CartPoleEnvTest, EpisodeLoopWorksThroughBaseReferencesOnly) {
    CartPoleEnv concrete_env(&backend, 25, 3);
    RandomDiscreteAgent concrete_agent(&backend, concrete_env.action_dim(), 99);
    Environment& env = concrete_env;
    Agent& agent = concrete_agent;

    Tensor observation = env.reset();
    int64_t steps = 0;
    bool done = false;
    while (!done && steps < 25) {
        const StepResult result = env.step(agent.act(observation));
        observation = result.observation;
        done = result.done;
        ++steps;
    }
    EXPECT_TRUE(done);
}

}  // namespace
}  // namespace pulsatrix
