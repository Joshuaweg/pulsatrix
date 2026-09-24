#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/rollout_buffer.hpp"

namespace exai {
namespace {

constexpr int64_t kObsDim = 2;
constexpr int64_t kActDim = 1;

class RolloutBufferTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Step `i` is made recognizable in every field at once, so a row of a returned batch can
    // be checked for internal consistency (observation, action and log_prob all from the same
    // step) rather than merely for "some plausible value turned up".
    void add_marked(RolloutBuffer& buffer, int i, float reward = 1.0f, bool done = false) {
        const float f = static_cast<float>(i);
        Tensor observation(Shape({1, kObsDim}), &backend, {10.0f + f, -(10.0f + f)});
        Tensor action(Shape({1, kActDim}), &backend, {100.0f + f});
        buffer.add(observation, action, reward, -0.5f - f, done);
    }

    // Adds one step per (reward, done) pair, marked as above.
    void add_episode_steps(RolloutBuffer& buffer, const std::vector<float>& rewards,
                           const std::vector<bool>& dones) {
        for (size_t i = 0; i < rewards.size(); ++i) {
            add_marked(buffer, static_cast<int>(i), rewards[i], dones[i]);
        }
    }
};

TEST_F(RolloutBufferTest, ConstructorThrowsOnNonPositiveMaxLength) {
    EXPECT_THROW({ RolloutBuffer buffer(0, kObsDim, kActDim, &backend); }, std::invalid_argument);
    EXPECT_THROW({ RolloutBuffer buffer(-3, kObsDim, kActDim, &backend); }, std::invalid_argument);
}

TEST_F(RolloutBufferTest, ConstructorThrowsOnNonPositiveObservationDim) {
    EXPECT_THROW({ RolloutBuffer buffer(4, 0, kActDim, &backend); }, std::invalid_argument);
}

TEST_F(RolloutBufferTest, ConstructorThrowsOnNonPositiveActionDim) {
    EXPECT_THROW({ RolloutBuffer buffer(4, kObsDim, -1, &backend); }, std::invalid_argument);
}

TEST_F(RolloutBufferTest, StartsEmptyWithTheRequestedLength) {
    RolloutBuffer buffer(7, kObsDim, kActDim, &backend);

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.max_length(), 7);
    EXPECT_EQ(buffer.observation_dim(), kObsDim);
    EXPECT_EQ(buffer.action_dim(), kActDim);
}

TEST_F(RolloutBufferTest, SizeGrowsOneStepPerAdd) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);

    for (int i = 0; i < 4; ++i) {
        add_marked(buffer, i);
        EXPECT_EQ(buffer.size(), i + 1);
    }
}

// The mission's hand-derived single-episode case: rewards [1, 1, 1], gamma = 0.9, the last
// step terminal. G_2 = 1, G_1 = 1 + 0.9*1 = 1.9, G_0 = 1 + 0.9 + 0.81 = 2.71. Asserted
// against those literal values, not against "the returns decrease" -- a monotonicity check
// passes for any number of wrong discount arithmetics.
TEST_F(RolloutBufferTest, DiscountedReturnMatchesHandDerivedSingleEpisodeValues) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, true});

    const RolloutBatch batch = buffer.compute_returns(0.9f);

    ASSERT_EQ(batch.returns.numel(), 3);
    EXPECT_FLOAT_EQ(batch.returns.data()[0], 2.71f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 1.9f);
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 1.0f);
}

// gamma == 1 is legal and means undiscounted return-to-go: G_t is just the remaining reward
// sum, 3/2/1 for the same three unit rewards.
TEST_F(RolloutBufferTest, UndiscountedReturnIsThePlainRemainingRewardSum) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, true});

    const RolloutBatch batch = buffer.compute_returns(1.0f);

    EXPECT_FLOAT_EQ(batch.returns.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 1.0f);
}

// The mission's second hand-derived case, and the one genuinely easy-to-get-wrong detail in
// this mission's math: two episodes inside one rollout, rewards [1, 1(done), 1, 1, 1(done)],
// gamma = 0.9.
//
//   Episode 1 (steps 0..1): G_1 = 1,             G_0 = 1 + 0.9*1        = 1.9
//   Episode 2 (steps 2..4): G_4 = 1, G_3 = 1.9,  G_2 = 1 + 0.9 + 0.81   = 2.71
//
// Without the reset at step 1's done flag, a naive single reverse pass would instead produce
// G_1 = 1 + 0.9*2.71 = 3.439 and G_0 = 1 + 0.9*3.439 = 4.0951 -- episode 2's return leaking
// backwards across a terminal state into episode 1. Both of those wrong values are still
// monotone decreasing, which is exactly why this asserts literals.
TEST_F(RolloutBufferTest, ReturnAccumulatorResetsAtEveryEpisodeBoundary) {
    RolloutBuffer buffer(5, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {false, true, false, false, true});

    const RolloutBatch batch = buffer.compute_returns(0.9f);

    ASSERT_EQ(batch.returns.numel(), 5);
    // Episode 1, computed independently of everything after step 1.
    EXPECT_FLOAT_EQ(batch.returns.data()[0], 1.9f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 1.0f);
    // Episode 2, identical to the standalone three-step episode above.
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 2.71f);
    EXPECT_FLOAT_EQ(batch.returns.data()[3], 1.9f);
    EXPECT_FLOAT_EQ(batch.returns.data()[4], 1.0f);
    // Spelled out separately so a failure names the actual defect rather than just a number
    // mismatch: these are the values a missing reset would produce.
    EXPECT_NE(batch.returns.data()[0], 4.0951f) << "episode 2's return leaked back into step 0";
    EXPECT_NE(batch.returns.data()[1], 3.439f) << "episode 2's return leaked back into step 1";
}

// A rollout that ends mid-episode (no terminal flag on the last stored step) is the normal
// fixed-length-rollout case, not an error: the return-to-go is simply truncated at the
// rollout boundary. Rewards [1, 1, 1] with no done at all, gamma = 0.9 -- same values as the
// single-episode case, confirming the terminal flag on the *last* step changes nothing.
TEST_F(RolloutBufferTest, RolloutEndingMidEpisodeTruncatesTheReturnWithoutError) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, false});

    const RolloutBatch batch = buffer.compute_returns(0.9f);

    EXPECT_FLOAT_EQ(batch.returns.data()[0], 2.71f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 1.9f);
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 1.0f);
}

// Distinct rewards, so a "returns happen to be right because every reward is 1" coincidence
// cannot hide a mis-indexed reward lookup. Rewards [1, 2, 3], gamma = 0.5, single episode:
//   G_2 = 3, G_1 = 2 + 0.5*3 = 3.5, G_0 = 1 + 0.5*3.5 = 2.75.
TEST_F(RolloutBufferTest, DiscountedReturnUsesEachStepsOwnReward) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 2.0f, 3.0f}, {false, false, true});

    const RolloutBatch batch = buffer.compute_returns(0.5f);

    EXPECT_FLOAT_EQ(batch.returns.data()[0], 2.75f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 3.5f);
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 3.0f);
}

TEST_F(RolloutBufferTest, ComputeReturnsThrowsOnGammaOutsideTheUnitInterval) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, true});

    EXPECT_THROW({ (void)buffer.compute_returns(0.0f); }, std::invalid_argument);
    EXPECT_THROW({ (void)buffer.compute_returns(1.5f); }, std::invalid_argument);
    EXPECT_THROW({ (void)buffer.compute_returns(-0.1f); }, std::invalid_argument);
    EXPECT_NO_THROW({ (void)buffer.compute_returns(1.0f); }) << "gamma == 1 is undiscounted, not invalid";
    EXPECT_NO_THROW({ (void)buffer.compute_returns(0.99f); });
}

TEST_F(RolloutBufferTest, ComputeReturnsReturnsExactlyTheExpectedRowShapes) {
    RolloutBuffer buffer(8, kObsDim, kActDim, &backend);
    for (int i = 0; i < 5; ++i) {
        add_marked(buffer, i);
    }

    const RolloutBatch batch = buffer.compute_returns(0.9f);

    // Leading dimension is size(), not max_length() -- a partially-filled rollout must not
    // hand back its never-written trailing slots.
    ASSERT_EQ(batch.observations.rank(), 2);
    EXPECT_EQ(batch.observations.shape().dim(0), 5);
    EXPECT_EQ(batch.observations.shape().dim(1), kObsDim);
    ASSERT_EQ(batch.actions.rank(), 2);
    EXPECT_EQ(batch.actions.shape().dim(0), 5);
    EXPECT_EQ(batch.actions.shape().dim(1), kActDim);
    ASSERT_EQ(batch.returns.rank(), 2);
    EXPECT_EQ(batch.returns.shape().dim(0), 5);
    EXPECT_EQ(batch.returns.shape().dim(1), 1);
    ASSERT_EQ(batch.log_probs.rank(), 2);
    EXPECT_EQ(batch.log_probs.shape().dim(0), 5);
    EXPECT_EQ(batch.log_probs.shape().dim(1), 1);
}

// Content, not just shape: row t of every returned tensor must be step t's own data, in
// stored order (a rollout is ordered, unlike a sampled replay batch).
TEST_F(RolloutBufferTest, BatchRowsCarryEachStepsOwnObservationActionAndLogProb) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    for (int i = 0; i < 4; ++i) {
        add_marked(buffer, i);
    }

    const RolloutBatch batch = buffer.compute_returns(0.9f);

    for (int64_t t = 0; t < 4; ++t) {
        const float f = static_cast<float>(t);
        EXPECT_FLOAT_EQ(batch.observations.data()[t * kObsDim], 10.0f + f) << "row " << t;
        EXPECT_FLOAT_EQ(batch.observations.data()[t * kObsDim + 1], -(10.0f + f)) << "row " << t;
        EXPECT_FLOAT_EQ(batch.actions.data()[t * kActDim], 100.0f + f) << "row " << t;
        EXPECT_FLOAT_EQ(batch.log_probs.data()[t], -0.5f - f) << "row " << t;
    }
}

// compute_returns() is a const pure reduction, not a consuming read: the explicit clear() is
// what empties the buffer.
TEST_F(RolloutBufferTest, ComputeReturnsDoesNotConsumeTheRollout) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, true});

    const RolloutBatch first = buffer.compute_returns(0.9f);
    const RolloutBatch second = buffer.compute_returns(0.9f);

    EXPECT_EQ(buffer.size(), 3);
    for (int64_t t = 0; t < 3; ++t) {
        EXPECT_FLOAT_EQ(first.returns.data()[t], second.returns.data()[t]) << "row " << t;
    }
}

// ---------------------------------------------------------------------------------------
// rewards() / dones() -- the two raw-storage accessors PPO's ComputeGAE() needs (Phase 3
// Mission 4). Not new logic, just new visibility into state add() already stored, so these
// tests assert exactly that: what went in comes back out, in order, at the right shape.
// ---------------------------------------------------------------------------------------

TEST_F(RolloutBufferTest, RewardsAndDonesReturnExactlyWhatWasAddedInStoredOrder) {
    RolloutBuffer buffer(6, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, -2.5f, 3.25f, 0.0f, 7.5f}, {false, true, false, false, true});

    const Tensor rewards = buffer.rewards();
    const Tensor dones = buffer.dones();

    ASSERT_EQ(rewards.numel(), 5);
    ASSERT_EQ(dones.numel(), 5);
    EXPECT_FLOAT_EQ(rewards.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(rewards.data()[1], -2.5f);
    EXPECT_FLOAT_EQ(rewards.data()[2], 3.25f);
    EXPECT_FLOAT_EQ(rewards.data()[3], 0.0f);
    EXPECT_FLOAT_EQ(rewards.data()[4], 7.5f);

    // 0.0f/1.0f floats, the encoding ReplayBatch/ComputeDQNTarget/ComputeGAE all share -- and
    // ComputeGAE multiplies by (1 - dones[t]) rather than thresholding, so anything other than
    // exactly 0 or 1 would silently scale its bootstrap.
    EXPECT_FLOAT_EQ(dones.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(dones.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(dones.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(dones.data()[3], 0.0f);
    EXPECT_FLOAT_EQ(dones.data()[4], 1.0f);
}

// Leading dimension is size(), not max_length(): a partially-filled rollout must not hand back
// its never-written trailing slots, exactly as compute_returns() must not. Shape is (N, 1),
// which is precisely what ComputeGAE's require_column() demands of both arguments.
TEST_F(RolloutBufferTest, RewardsAndDonesAreColumnShapedAtTheCurrentSize) {
    RolloutBuffer buffer(8, kObsDim, kActDim, &backend);
    for (int i = 0; i < 3; ++i) {
        add_marked(buffer, i);
    }

    const Tensor rewards = buffer.rewards();
    const Tensor dones = buffer.dones();

    ASSERT_EQ(rewards.rank(), 2);
    EXPECT_EQ(rewards.shape().dim(0), 3);
    EXPECT_EQ(rewards.shape().dim(1), 1);
    ASSERT_EQ(dones.rank(), 2);
    EXPECT_EQ(dones.shape().dim(0), 3);
    EXPECT_EQ(dones.shape().dim(1), 1);

    // Same leading dimension as the batch, so row t of each belongs to the same stored step --
    // the invariant ComputeGAE relies on when it pairs rewards[t] with values[t].
    const RolloutBatch batch = buffer.compute_returns(0.9f);
    EXPECT_EQ(rewards.shape().dim(0), batch.returns.shape().dim(0));
    EXPECT_EQ(dones.shape().dim(0), batch.observations.shape().dim(0));
}

// Like compute_returns(), these are const pure reads: they neither consume the rollout nor
// leak the buffer's own storage (a caller writing into the returned tensor must not corrupt
// the next epoch's read of the same rollout, which PPO performs repeatedly).
TEST_F(RolloutBufferTest, RewardsAndDonesDoNotConsumeOrAliasTheRollout) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 2.0f, 3.0f}, {false, false, true});

    Tensor first = buffer.rewards();
    first.data()[0] = -99.0f;

    EXPECT_EQ(buffer.size(), 3) << "a read must not consume the rollout";
    const Tensor second = buffer.rewards();
    EXPECT_FLOAT_EQ(second.data()[0], 1.0f) << "the returned tensor aliased the buffer's storage";
    EXPECT_FLOAT_EQ(second.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(second.data()[2], 3.0f);
}

// After clear() and a refill, the accessors must report the *second* rollout only -- the same
// "stale rows beyond size_ are unreachable" property ClearMakesTheBufferGenuinelyReusable pins
// for compute_returns(), now for the two raw accessors PPO reads on every rollout.
TEST_F(RolloutBufferTest, RewardsAndDonesReflectOnlyTheCurrentRolloutAfterClear) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {true, true, true});

    buffer.clear();
    EXPECT_EQ(buffer.rewards().numel(), 0) << "a cleared buffer has no steps to report";
    EXPECT_EQ(buffer.dones().numel(), 0);
    EXPECT_EQ(buffer.rewards().shape().dim(1), 1) << "an empty read is still (0, 1), not degenerate";

    add_episode_steps(buffer, {5.0f, 6.0f}, {false, true});
    const Tensor rewards = buffer.rewards();
    const Tensor dones = buffer.dones();
    ASSERT_EQ(rewards.numel(), 2);
    EXPECT_FLOAT_EQ(rewards.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(rewards.data()[1], 6.0f);
    EXPECT_FLOAT_EQ(dones.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(dones.data()[1], 1.0f);
}

TEST_F(RolloutBufferTest, AddPastMaxLengthThrowsLogicError) {
    RolloutBuffer buffer(2, kObsDim, kActDim, &backend);
    add_marked(buffer, 0);
    add_marked(buffer, 1);

    ASSERT_EQ(buffer.size(), 2);
    // std::logic_error specifically, *not* std::invalid_argument: a full rollout is a
    // usage-protocol violation (the caller skipped compute_returns()+clear()), not a
    // malformed argument, and the two must stay separately catchable.
    EXPECT_THROW({ add_marked(buffer, 2); }, std::logic_error);
    EXPECT_EQ(buffer.size(), 2) << "a rejected add must not advance the buffer";
}

TEST_F(RolloutBufferTest, AddThrowsOnMismatchedObservationShape) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor bad(Shape({1, kObsDim + 1}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);

    EXPECT_THROW({ buffer.add(bad, action, 1.0f, 0.0f, false); }, std::invalid_argument);
    EXPECT_EQ(buffer.size(), 0) << "a rejected add must not advance the buffer";
}

TEST_F(RolloutBufferTest, AddThrowsOnMismatchedActionShape) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend);
    Tensor bad(Shape({1, kActDim + 2}), &backend);

    EXPECT_THROW({ buffer.add(observation, bad, 1.0f, 0.0f, false); }, std::invalid_argument);
}

TEST_F(RolloutBufferTest, AddThrowsOnWrongRank) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor flat(Shape({kObsDim}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);

    // Same numel as a valid (1, observation_dim) row, wrong rank -- the case a bare "numel
    // matches" check would wave through.
    EXPECT_THROW({ buffer.add(flat, action, 1.0f, 0.0f, false); }, std::invalid_argument);
}

TEST_F(RolloutBufferTest, AddThrowsOnMultiRowObservation) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor batched(Shape({2, kObsDim}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);

    EXPECT_THROW({ buffer.add(batched, action, 1.0f, 0.0f, false); }, std::invalid_argument);
}

// The two rejection kinds are genuinely different types, asserted in one place so the
// distinction cannot rot into "both are logic_error subclasses, close enough".
TEST_F(RolloutBufferTest, ShapeMismatchAndOverCapacityThrowDistinctExceptionTypes) {
    RolloutBuffer buffer(1, kObsDim, kActDim, &backend);
    Tensor bad(Shape({1, kObsDim + 1}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);

    EXPECT_THROW({ buffer.add(bad, action, 1.0f, 0.0f, false); }, std::invalid_argument);

    add_marked(buffer, 0);
    bool caught_invalid_argument = false;
    bool caught_logic_error = false;
    try {
        add_marked(buffer, 1);
    } catch (const std::invalid_argument&) {
        caught_invalid_argument = true;
    } catch (const std::logic_error&) {
        caught_logic_error = true;
    }
    EXPECT_FALSE(caught_invalid_argument);
    EXPECT_TRUE(caught_logic_error);
}

// The full rollout cycle, which is the class's actual usage protocol: fill, reduce, clear,
// and refill to max_length again without throwing. size() == 0 alone would not prove the
// buffer is genuinely reusable rather than internally still full.
TEST_F(RolloutBufferTest, ClearMakesTheBufferGenuinelyReusable) {
    RolloutBuffer buffer(3, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f, 1.0f}, {false, false, true});
    (void)buffer.compute_returns(0.9f);

    buffer.clear();
    ASSERT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.max_length(), 3) << "clear() must not change the configured length";

    for (int i = 0; i < 3; ++i) {
        EXPECT_NO_THROW({ add_marked(buffer, i, 2.0f, i == 2); }) << "add #" << i << " after clear()";
    }
    EXPECT_EQ(buffer.size(), 3);
    // And the second rollout's returns come from the *second* rollout's rewards (2.0 each),
    // with no trace of the first: stale storage rows beyond size_ are genuinely unreachable.
    const RolloutBatch batch = buffer.compute_returns(1.0f);
    EXPECT_FLOAT_EQ(batch.returns.data()[0], 6.0f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 4.0f);
    EXPECT_FLOAT_EQ(batch.returns.data()[2], 2.0f);
}

TEST_F(RolloutBufferTest, ClearOnAnAlreadyEmptyBufferIsANoOp) {
    RolloutBuffer buffer(2, kObsDim, kActDim, &backend);

    buffer.clear();
    buffer.clear();

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_NO_THROW({ add_marked(buffer, 0); });
}

// A partially-filled rollout is the normal case when an episode ends early, so reducing one
// must work rather than requiring the buffer to be full first.
TEST_F(RolloutBufferTest, ComputeReturnsWorksOnAPartiallyFilledRollout) {
    RolloutBuffer buffer(10, kObsDim, kActDim, &backend);
    add_episode_steps(buffer, {1.0f, 1.0f}, {false, true});

    const RolloutBatch batch = buffer.compute_returns(0.5f);

    ASSERT_EQ(batch.returns.numel(), 2);
    EXPECT_FLOAT_EQ(batch.returns.data()[0], 1.5f);
    EXPECT_FLOAT_EQ(batch.returns.data()[1], 1.0f);
}

// An empty rollout reduces to an empty batch rather than throwing: "nothing was collected
// this iteration" is a state a training loop can reach (an environment that terminates
// immediately), and (0, *) tensors let the caller's own loop handle it with no special case.
TEST_F(RolloutBufferTest, ComputeReturnsOnAnEmptyBufferYieldsAnEmptyBatch) {
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);

    RolloutBatch batch = buffer.compute_returns(0.9f);

    EXPECT_EQ(batch.returns.numel(), 0);
    EXPECT_EQ(batch.observations.shape().dim(0), 0);
    EXPECT_EQ(batch.observations.shape().dim(1), kObsDim);
    EXPECT_EQ(batch.actions.shape().dim(0), 0);
    EXPECT_EQ(batch.log_probs.shape().dim(0), 0);
}

using RolloutBufferDeathTest = RolloutBufferTest;

// add() copies rows out of Tensor::data() in a raw host loop -- undefined behavior on a
// CUDA-backed Tensor, so both tensor arguments are EXAI_ASSERT-guarded
// (mission_host_loop_guards.md). No real GPU needed: see LinearModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses. One death test per distinct guarded argument
// role, as in ReplayBufferDeathTest: an observation-shaped tensor and an action-shaped one.
TEST_F(RolloutBufferDeathTest, AddAbortsOnNonCpuObservation) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    Tensor action(Shape({1, kActDim}), &backend, {0.0f});
    EXPECT_DEATH({ buffer.add(observation, action, 1.0f, 0.0f, false); }, "EXAI_ASSERT failed");
}

TEST_F(RolloutBufferDeathTest, AddAbortsOnNonCpuAction) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RolloutBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend, {1.0f, 2.0f});
    Tensor action(Shape({1, kActDim}), &backend, {0.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ buffer.add(observation, action, 1.0f, 0.0f, false); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
