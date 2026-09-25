#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/replay_buffer.hpp"

namespace pulsatrix {
namespace {

constexpr int64_t kObsDim = 2;
constexpr int64_t kActDim = 1;

class ReplayBufferTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Transition `i` is made recognizable in *every* field at once, so a row of a sampled
    // batch can be checked for internal consistency (all five fields from the same
    // transition), not just for "some plausible observation value turned up".
    void add_marked(ReplayBuffer& buffer, int i) {
        const float f = static_cast<float>(i);
        Tensor observation(Shape({1, kObsDim}), &backend, {10.0f + f, -(10.0f + f)});
        Tensor action(Shape({1, kActDim}), &backend, {100.0f + f});
        Tensor next_observation(Shape({1, kObsDim}), &backend, {20.0f + f, -(20.0f + f)});
        buffer.add(observation, action, 1000.0f + f, next_observation, i % 2 == 1);
    }

    // Recovers transition index `i` from row `row` of a batch produced from add_marked()
    // transitions, and asserts every other field of that row agrees with it.
    int decode_and_check_row(const ReplayBatch& batch, int64_t row) {
        const float observation_marker = batch.observations.data()[row * kObsDim];
        const int i = static_cast<int>(observation_marker - 10.0f);
        const float f = static_cast<float>(i);
        EXPECT_FLOAT_EQ(batch.observations.data()[row * kObsDim + 1], -(10.0f + f)) << "row " << row;
        EXPECT_FLOAT_EQ(batch.actions.data()[row * kActDim], 100.0f + f) << "row " << row;
        EXPECT_FLOAT_EQ(batch.rewards.data()[row], 1000.0f + f) << "row " << row;
        EXPECT_FLOAT_EQ(batch.next_observations.data()[row * kObsDim], 20.0f + f) << "row " << row;
        EXPECT_FLOAT_EQ(batch.next_observations.data()[row * kObsDim + 1], -(20.0f + f)) << "row " << row;
        EXPECT_FLOAT_EQ(batch.dones.data()[row], i % 2 == 1 ? 1.0f : 0.0f) << "row " << row;
        return i;
    }

    // The set of transition indices the buffer's live content actually consists of, obtained
    // by repeated full-size sampling. Sampling is *with replacement*, so one full-size batch
    // is not guaranteed to touch every stored slot; repeated draws from a fixed-seed LCG are
    // deterministic, so this is exhaustive in practice without being probabilistic in
    // outcome.
    std::set<int> observed_content(ReplayBuffer& buffer, int draws = 50) {
        std::set<int> seen;
        for (int d = 0; d < draws; ++d) {
            ReplayBatch batch = buffer.sample(buffer.size());
            for (int64_t row = 0; row < buffer.size(); ++row) {
                seen.insert(decode_and_check_row(batch, row));
            }
        }
        return seen;
    }
};

TEST_F(ReplayBufferTest, ConstructorThrowsOnNonPositiveCapacity) {
    EXPECT_THROW({ ReplayBuffer buffer(0, kObsDim, kActDim, &backend); }, std::invalid_argument);
    EXPECT_THROW({ ReplayBuffer buffer(-3, kObsDim, kActDim, &backend); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, ConstructorThrowsOnNonPositiveObservationDim) {
    EXPECT_THROW({ ReplayBuffer buffer(4, 0, kActDim, &backend); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, ConstructorThrowsOnNonPositiveActionDim) {
    EXPECT_THROW({ ReplayBuffer buffer(4, kObsDim, -1, &backend); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, StartsEmptyWithTheRequestedCapacity) {
    ReplayBuffer buffer(7, kObsDim, kActDim, &backend);

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 7);
    EXPECT_EQ(buffer.observation_dim(), kObsDim);
    EXPECT_EQ(buffer.action_dim(), kActDim);
}

TEST_F(ReplayBufferTest, SizeGrowsToCapacityThenStops) {
    ReplayBuffer buffer(3, kObsDim, kActDim, &backend);

    for (int i = 0; i < 3; ++i) {
        add_marked(buffer, i);
        EXPECT_EQ(buffer.size(), i + 1);
    }
    for (int i = 3; i < 8; ++i) {
        add_marked(buffer, i);
        EXPECT_EQ(buffer.size(), 3) << "after add #" << i;
        EXPECT_EQ(buffer.capacity(), 3);
    }
}

// The load-bearing circular-buffer proof: not "3 things are still in there" but "*these*
// three things, and specifically not the two that were overwritten".
TEST_F(ReplayBufferTest, CircularOverwriteRetainsExactlyTheNewestTransitionsByContent) {
    ReplayBuffer buffer(3, kObsDim, kActDim, &backend);
    for (int i = 0; i < 5; ++i) {
        add_marked(buffer, i);
    }

    ASSERT_EQ(buffer.size(), 3);
    const std::set<int> content = observed_content(buffer);

    EXPECT_EQ(content, (std::set<int>{2, 3, 4}));
    // Spelled out separately so a failure says *which* half of the claim broke: the two
    // oldest transitions are genuinely gone, and all three survivors are genuinely present.
    EXPECT_EQ(content.count(0), 0u) << "transition 0 should have been overwritten";
    EXPECT_EQ(content.count(1), 0u) << "transition 1 should have been overwritten";
    EXPECT_EQ(content.count(2), 1u);
    EXPECT_EQ(content.count(3), 1u);
    EXPECT_EQ(content.count(4), 1u);
}

// A partially-filled buffer must never hand back one of its zero-filled, never-written
// slots -- which is what sampling over [0, capacity()) instead of [0, size()) would do.
TEST_F(ReplayBufferTest, SamplingAPartiallyFilledBufferNeverReturnsUnwrittenSlots) {
    ReplayBuffer buffer(5, kObsDim, kActDim, &backend);
    add_marked(buffer, 0);
    add_marked(buffer, 1);

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_LT(buffer.size(), buffer.capacity());
    const std::set<int> content = observed_content(buffer);

    EXPECT_EQ(content, (std::set<int>{0, 1}));
    // A never-written slot decodes to marker -10 (observation 0.0f), which would show up as
    // index -10 in `content`; the set equality above already excludes it, and this makes the
    // intent explicit.
    EXPECT_EQ(content.count(-10), 0u) << "a zero-filled unwritten slot was sampled";
}

TEST_F(ReplayBufferTest, SampleReturnsExactlyTheExpectedRowShapes) {
    ReplayBuffer buffer(10, kObsDim, kActDim, &backend);
    for (int i = 0; i < 10; ++i) {
        add_marked(buffer, i);
    }

    ReplayBatch batch = buffer.sample(4);

    ASSERT_EQ(batch.observations.rank(), 2);
    EXPECT_EQ(batch.observations.shape().dim(0), 4);
    EXPECT_EQ(batch.observations.shape().dim(1), kObsDim);
    ASSERT_EQ(batch.actions.rank(), 2);
    EXPECT_EQ(batch.actions.shape().dim(0), 4);
    EXPECT_EQ(batch.actions.shape().dim(1), kActDim);
    ASSERT_EQ(batch.rewards.rank(), 2);
    EXPECT_EQ(batch.rewards.shape().dim(0), 4);
    EXPECT_EQ(batch.rewards.shape().dim(1), 1);
    ASSERT_EQ(batch.next_observations.rank(), 2);
    EXPECT_EQ(batch.next_observations.shape().dim(0), 4);
    EXPECT_EQ(batch.next_observations.shape().dim(1), kObsDim);
    ASSERT_EQ(batch.dones.rank(), 2);
    EXPECT_EQ(batch.dones.shape().dim(0), 4);
    EXPECT_EQ(batch.dones.shape().dim(1), 1);
}

TEST_F(ReplayBufferTest, SameSeedAndSameContentSampleIdentically) {
    ReplayBuffer a(16, kObsDim, kActDim, &backend, 42);
    ReplayBuffer b(16, kObsDim, kActDim, &backend, 42);
    for (int i = 0; i < 16; ++i) {
        add_marked(a, i);
        add_marked(b, i);
    }

    ReplayBatch batch_a = a.sample(8);
    ReplayBatch batch_b = b.sample(8);

    ASSERT_EQ(batch_a.observations.numel(), batch_b.observations.numel());
    for (int64_t i = 0; i < batch_a.observations.numel(); ++i) {
        EXPECT_FLOAT_EQ(batch_a.observations.data()[i], batch_b.observations.data()[i]) << "index " << i;
    }
    for (int64_t i = 0; i < batch_a.rewards.numel(); ++i) {
        EXPECT_FLOAT_EQ(batch_a.rewards.data()[i], batch_b.rewards.data()[i]) << "index " << i;
    }
}

// Non-vacuity for the test above: the sampled indices are genuinely seed-driven, not a
// constant sequence that would make any two buffers agree.
TEST_F(ReplayBufferTest, DifferentSeedsSampleDifferently) {
    ReplayBuffer a(16, kObsDim, kActDim, &backend, 42);
    ReplayBuffer b(16, kObsDim, kActDim, &backend, 4242);
    for (int i = 0; i < 16; ++i) {
        add_marked(a, i);
        add_marked(b, i);
    }

    ReplayBatch batch_a = a.sample(8);
    ReplayBatch batch_b = b.sample(8);

    bool any_difference = false;
    for (int64_t i = 0; i < batch_a.observations.numel(); ++i) {
        if (batch_a.observations.data()[i] != batch_b.observations.data()[i]) {
            any_difference = true;
        }
    }
    EXPECT_TRUE(any_difference) << "two different seeds produced an identical batch";
}

// With replacement, not without: a full-size batch may repeat a transition. A
// sample-without-replacement implementation could never produce this, so the check has
// teeth.
TEST_F(ReplayBufferTest, SamplingIsWithReplacement) {
    ReplayBuffer buffer(2, kObsDim, kActDim, &backend);
    add_marked(buffer, 0);
    add_marked(buffer, 1);

    bool saw_repeat = false;
    for (int d = 0; d < 20 && !saw_repeat; ++d) {
        ReplayBatch batch = buffer.sample(2);
        if (batch.observations.data()[0] == batch.observations.data()[kObsDim]) {
            saw_repeat = true;
        }
    }
    EXPECT_TRUE(saw_repeat) << "20 full-size draws from a 2-element buffer never repeated a transition";
}

TEST_F(ReplayBufferTest, SampleThrowsWhenBatchSizeExceedsCurrentSize) {
    ReplayBuffer buffer(10, kObsDim, kActDim, &backend);
    add_marked(buffer, 0);
    add_marked(buffer, 1);

    // 3 > size() == 2, even though 3 <= capacity() == 10: the unwritten slots are not
    // transitions and must not be silently handed back.
    EXPECT_THROW({ (void)buffer.sample(3); }, std::invalid_argument);
    EXPECT_NO_THROW({ (void)buffer.sample(2); });
}

// Its own case rather than an instance of the general exceeds-size rule: "sample a buffer
// nothing has been added to yet" is the boundary a training loop actually hits first.
TEST_F(ReplayBufferTest, SampleThrowsOnAnEmptyBuffer) {
    ReplayBuffer buffer(10, kObsDim, kActDim, &backend);

    ASSERT_EQ(buffer.size(), 0);
    EXPECT_THROW({ (void)buffer.sample(1); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, SampleThrowsOnNonPositiveBatchSize) {
    ReplayBuffer buffer(10, kObsDim, kActDim, &backend);
    add_marked(buffer, 0);

    EXPECT_THROW({ (void)buffer.sample(0); }, std::invalid_argument);
    EXPECT_THROW({ (void)buffer.sample(-2); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, AddThrowsOnMismatchedObservationShape) {
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor bad(Shape({1, kObsDim + 1}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);
    Tensor next_observation(Shape({1, kObsDim}), &backend);

    EXPECT_THROW({ buffer.add(bad, action, 1.0f, next_observation, false); }, std::invalid_argument);
    EXPECT_EQ(buffer.size(), 0) << "a rejected add must not advance the buffer";
}

TEST_F(ReplayBufferTest, AddThrowsOnMismatchedActionShape) {
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend);
    Tensor bad(Shape({1, kActDim + 2}), &backend);
    Tensor next_observation(Shape({1, kObsDim}), &backend);

    EXPECT_THROW({ buffer.add(observation, bad, 1.0f, next_observation, false); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, AddThrowsOnMismatchedNextObservationShape) {
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);
    Tensor bad(Shape({2, kObsDim}), &backend);

    EXPECT_THROW({ buffer.add(observation, action, 1.0f, bad, false); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, AddThrowsOnWrongRank) {
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor flat(Shape({kObsDim}), &backend);
    Tensor action(Shape({1, kActDim}), &backend);
    Tensor next_observation(Shape({1, kObsDim}), &backend);

    // Same numel as a valid (1, observation_dim) row, wrong rank -- the case a bare
    // "numel matches" check would wave through.
    EXPECT_THROW({ buffer.add(flat, action, 1.0f, next_observation, false); }, std::invalid_argument);
}

TEST_F(ReplayBufferTest, DoneIsStoredAsAFloatFlag) {
    ReplayBuffer buffer(1, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend, {1.0f, 2.0f});
    Tensor action(Shape({1, kActDim}), &backend, {0.0f});
    Tensor next_observation(Shape({1, kObsDim}), &backend, {3.0f, 4.0f});

    buffer.add(observation, action, 5.0f, next_observation, true);
    EXPECT_FLOAT_EQ(buffer.sample(1).dones.data()[0], 1.0f);

    buffer.add(observation, action, 5.0f, next_observation, false);
    EXPECT_FLOAT_EQ(buffer.sample(1).dones.data()[0], 0.0f);
}

using ReplayBufferDeathTest = ReplayBufferTest;

// add() copies rows out of Tensor::data() in a raw host loop -- undefined behavior on a
// CUDA-backed Tensor, so all three tensor arguments are PULSATRIX_ASSERT-guarded
// (mission_host_loop_guards.md). No real GPU needed: see LinearModuleDeathTest for the
// mislabeled-Tensor testing pattern this reuses.
//
// Two death tests, not three and not one. MSELoss's precedent is one death test per
// *distinct* guarded argument role (prediction, target), not one per line of guard code;
// add() has only two distinct roles -- an observation-shaped tensor and an action-shaped
// tensor. `next_observation` is structurally identical to `observation` (same width, same
// validation path, adjacent guard line), so a third case would re-exercise a path already
// covered rather than cover a new one. The mission's own requirement of exactly 2 and
// MSELoss's precedent agree on that reading.
TEST_F(ReplayBufferDeathTest, AddAbortsOnNonCpuObservation) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend, {1.0f, 2.0f}, DeviceType::Cuda);
    Tensor action(Shape({1, kActDim}), &backend, {0.0f});
    Tensor next_observation(Shape({1, kObsDim}), &backend, {3.0f, 4.0f});
    EXPECT_DEATH({ buffer.add(observation, action, 1.0f, next_observation, false); }, "PULSATRIX_ASSERT failed");
}

TEST_F(ReplayBufferDeathTest, AddAbortsOnNonCpuAction) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    ReplayBuffer buffer(4, kObsDim, kActDim, &backend);
    Tensor observation(Shape({1, kObsDim}), &backend, {1.0f, 2.0f});
    Tensor action(Shape({1, kActDim}), &backend, {0.0f}, DeviceType::Cuda);
    Tensor next_observation(Shape({1, kObsDim}), &backend, {3.0f, 4.0f});
    EXPECT_DEATH({ buffer.add(observation, action, 1.0f, next_observation, false); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
