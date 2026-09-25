/** @file hypergrid_env_test.cpp
 *  @brief HyperGridEnv construction validation, reward formula, and step/episode mechanics.
 *
 *  The load-bearing test here is RewardMatchesHandDerivedValuesAtNamedCells: the reward
 *  formula (mission_hypergrid_env.md's Design section) is evaluated independently of this
 *  implementation at three named cells (exact corner, near-corner-band, interior) for
 *  ndim=2, side_length=8, and pinned below as literal constants.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/environment.hpp"
#include "pulsatrix/hypergrid_env.hpp"

namespace pulsatrix {
namespace {

class HyperGridEnvTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor state2(float x0, float x1) { return Tensor(Shape({1, 2}), &backend, {x0, x1}); }

    Tensor action(float index) { return Tensor(Shape({1, 1}), &backend, {index}); }
};

// ---------------------------------------------------------------------------------------
// Construction validation
// ---------------------------------------------------------------------------------------

TEST_F(HyperGridEnvTest, ConstructionThrowsOnZeroNdim) {
    EXPECT_THROW({ HyperGridEnv env(&backend, 0); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, ConstructionThrowsOnSideLengthBelowFour) {
    EXPECT_THROW({ HyperGridEnv env(&backend, 2, 3); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, ConstructionThrowsOnNegativeRewardConstant) {
    EXPECT_THROW({ HyperGridEnv env(&backend, 2, 8, -0.1f); }, std::invalid_argument);
    EXPECT_THROW({ HyperGridEnv env(&backend, 2, 8, 0.1f, -0.5f); }, std::invalid_argument);
    EXPECT_THROW({ HyperGridEnv env(&backend, 2, 8, 0.1f, 0.5f, -2.0f); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, ConstructionThrowsOnZeroMaxSteps) {
    EXPECT_THROW({ HyperGridEnv env(&backend, 2, 8, 0.1f, 0.5f, 2.0f, 0); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, DescribesItsSpaces) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_EQ(env.observation_dim(), 2);
    EXPECT_EQ(env.action_dim(), 3);  // 2 increment actions + stop
    EXPECT_TRUE(env.is_discrete());
    EXPECT_EQ(env.ndim(), 2);
    EXPECT_EQ(env.side_length(), 8);
    EXPECT_EQ(env.max_steps(), 64);
}

// ---------------------------------------------------------------------------------------
// reset()
// ---------------------------------------------------------------------------------------

TEST_F(HyperGridEnvTest, ResetReturnsOrigin) {
    HyperGridEnv env(&backend, 2, 8);
    Tensor obs = env.reset();
    EXPECT_EQ(obs.rank(), 2);
    EXPECT_EQ(obs.shape().dim(0), 1);
    EXPECT_EQ(obs.shape().dim(1), 2);
    EXPECT_FLOAT_EQ(obs.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(obs.data()[1], 0.0f);
}

TEST_F(HyperGridEnvTest, ResetWithInitialStateRoundTrips) {
    HyperGridEnv env(&backend, 2, 8);
    Tensor obs = env.reset(state2(3.0f, 5.0f));
    EXPECT_FLOAT_EQ(obs.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(obs.data()[1], 5.0f);
}

TEST_F(HyperGridEnvTest, ResetWithInitialStateThrowsOnWrongShape) {
    HyperGridEnv env(&backend, 2, 8);
    Tensor wrong = Tensor(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});
    EXPECT_THROW({ (void)env.reset(wrong); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, ResetWithInitialStateThrowsOnOutOfRangeCoordinate) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_THROW({ (void)env.reset(state2(8.0f, 0.0f)); }, std::invalid_argument);   // side_length is 8, max index 7
    EXPECT_THROW({ (void)env.reset(state2(0.0f, -1.0f)); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, ResetWithInitialStateThrowsOnNonIntegerCoordinate) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_THROW({ (void)env.reset(state2(2.5f, 0.0f)); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------------------
// reward() -- the hand-derived correctness proof
// ---------------------------------------------------------------------------------------

/**
 * Independently derived, from the formula in mission_hypergrid_env.md's Design section
 * (u_i = x_i/(H-1); near_corner iff every u_i <= 0.25 or >= 0.75; exact_corner iff every
 * x_i in {0, H-1}), for ndim=2, side_length=8 (H-1=7), r0=0.1, r1=0.5, r2=2.0:
 *
 * - (0, 0): exact corner (u=(0,0), both <= 0.25) -> R = 0.1 + 0.5 + 2.0 = 2.6
 * - (1, 1): near-corner band, not exact corner (u=(1/7,1/7)~=0.1429, both <= 0.25;
 *           x_i != 0 and != 7) -> R = 0.1 + 0.5 = 0.6
 * - (4, 4): interior (u=(4/7,4/7)~=0.5714, neither <=0.25 nor >=0.75) -> R = 0.1
 */
TEST_F(HyperGridEnvTest, RewardMatchesHandDerivedValuesAtNamedCells) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_NEAR(env.reward(state2(0.0f, 0.0f)), 2.6f, 1e-5f);
    EXPECT_NEAR(env.reward(state2(1.0f, 1.0f)), 0.6f, 1e-5f);
    EXPECT_NEAR(env.reward(state2(4.0f, 4.0f)), 0.1f, 1e-5f);
}

TEST_F(HyperGridEnvTest, RewardRequiresEveryDimensionInBandForNearCornerBonus) {
    // (1, 4): dim 0 is in the near-corner band, dim 1 is interior -> near_corner is false
    // (requires *every* dimension in-band), so R = r0 only.
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_NEAR(env.reward(state2(1.0f, 4.0f)), 0.1f, 1e-5f);
}

TEST_F(HyperGridEnvTest, RewardOfArbitraryStateDoesNotRequireReset) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_NEAR(env.reward(state2(7.0f, 7.0f)), 2.6f, 1e-5f);
}

TEST_F(HyperGridEnvTest, RewardThrowsOnInvalidState) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_THROW({ (void)env.reward(state2(8.0f, 0.0f)); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------------------
// step()
// ---------------------------------------------------------------------------------------

TEST_F(HyperGridEnvTest, StepThrowsIfNotReset) {
    HyperGridEnv env(&backend, 2, 8);
    EXPECT_THROW({ (void)env.step(action(0.0f)); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, LegalIncrementMovesOneCoordinateNoReward) {
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset();
    StepResult result = env.step(action(0.0f));  // increment dim 0
    EXPECT_FLOAT_EQ(result.observation.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(result.observation.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(result.reward, 0.0f);
    EXPECT_FALSE(result.done);
}

TEST_F(HyperGridEnvTest, IllegalOffGridIncrementThrows) {
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset(state2(7.0f, 0.0f));  // dim 0 already at side_length-1
    EXPECT_THROW({ (void)env.step(action(0.0f)); }, std::invalid_argument);
}

TEST_F(HyperGridEnvTest, StopActionTerminatesWithRewardOfCurrentState) {
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset(state2(1.0f, 1.0f));
    StepResult result = env.step(action(2.0f));  // stop (action_dim() - 1 == ndim())
    EXPECT_FLOAT_EQ(result.observation.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(result.observation.data()[1], 1.0f);
    EXPECT_NEAR(result.reward, 0.6f, 1e-5f);  // hand-derived above
    EXPECT_TRUE(result.done);
}

TEST_F(HyperGridEnvTest, StepAfterDoneThrowsLogicError) {
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset();
    static_cast<void>(env.step(action(2.0f)));  // stop immediately -> done
    EXPECT_THROW({ (void)env.step(action(0.0f)); }, std::logic_error);
}

TEST_F(HyperGridEnvTest, MaxStepsCapForcesTerminationWithCurrentStateReward) {
    HyperGridEnv env(&backend, 2, 8, 0.1f, 0.5f, 2.0f, /*max_steps=*/3);
    (void)env.reset();
    static_cast<void>(env.step(action(0.0f)));  // (1, 0)
    static_cast<void>(env.step(action(0.0f)));  // (2, 0)
    StepResult result = env.step(action(0.0f));  // (3, 0), 3rd step -> cap reached
    EXPECT_TRUE(result.done);
    EXPECT_NEAR(result.reward, env.reward(state2(3.0f, 0.0f)), 1e-5f);
}

TEST_F(HyperGridEnvTest, ResetAfterDoneAllowsANewEpisode) {
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset();
    static_cast<void>(env.step(action(2.0f)));  // stop
    Tensor obs = env.reset();
    EXPECT_FLOAT_EQ(obs.data()[0], 0.0f);
    StepResult result = env.step(action(0.0f));
    EXPECT_FALSE(result.done);
}

TEST_F(HyperGridEnvTest, HandTracedEpisodeToACornerViaStop) {
    // From origin: increment dim 0 seven times, dim 1 seven times, then stop at (7, 7),
    // an exact corner. Reward is 0 at every intermediate step and 2.6 only at stop.
    HyperGridEnv env(&backend, 2, 8);
    (void)env.reset();
    for (int i = 0; i < 7; ++i) {
        StepResult r = env.step(action(0.0f));
        EXPECT_FLOAT_EQ(r.reward, 0.0f);
        EXPECT_FALSE(r.done);
    }
    for (int i = 0; i < 7; ++i) {
        StepResult r = env.step(action(1.0f));
        EXPECT_FLOAT_EQ(r.reward, 0.0f);
        EXPECT_FALSE(r.done);
    }
    StepResult final_result = env.step(action(2.0f));  // stop
    EXPECT_FLOAT_EQ(final_result.observation.data()[0], 7.0f);
    EXPECT_FLOAT_EQ(final_result.observation.data()[1], 7.0f);
    EXPECT_NEAR(final_result.reward, 2.6f, 1e-5f);
    EXPECT_TRUE(final_result.done);
}

TEST_F(HyperGridEnvTest, UsableThroughTheEnvironmentBasePointer) {
    HyperGridEnv concrete(&backend, 2, 8);
    Environment& env = concrete;
    Tensor obs = env.reset();
    EXPECT_EQ(obs.shape().dim(1), env.observation_dim());
    StepResult result = env.step(action(0.0f));
    EXPECT_FALSE(result.done);
}

}  // namespace
}  // namespace pulsatrix
