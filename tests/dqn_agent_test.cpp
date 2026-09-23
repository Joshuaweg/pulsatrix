#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/dqn_agent.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"
#include "exai/sequential_module.hpp"

namespace exai {
namespace {

constexpr int64_t kObsDim = 2;
constexpr int64_t kActionDim = 3;
constexpr int64_t kGreedyAction = 1;

class DQNAgentTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // A hand-set Q-network whose output is known exactly and has an unambiguous maximum:
    // the weight matrix is left at LinearModule's zero initialization, so for *any* input
    //     q = x @ 0 + bias = [1.0, 5.0, 2.0]
    // and argmax q = 1, by a clear margin. No margin-of-error, no tie, no dependence on the
    // observation -- so "the agent acted greedily" is a decidable claim on every single draw.
    DQNAgentTest() : q_network(kObsDim, kActionDim, &backend) { q_network.set_bias({1.0f, 5.0f, 2.0f}); }

    LinearModule q_network;

    Tensor observation() { return Tensor(Shape({1, kObsDim}), &backend, {0.3f, -0.7f}); }

    // The action indices an agent produces over `draws` successive act() calls.
    std::vector<int64_t> action_sequence(DQNAgent& agent, int draws) {
        std::vector<int64_t> actions;
        actions.reserve(static_cast<size_t>(draws));
        for (int d = 0; d < draws; ++d) {
            const Tensor action = agent.act(observation());
            EXPECT_EQ(action.rank(), 2);
            EXPECT_EQ(action.shape().dim(0), 1);
            EXPECT_EQ(action.shape().dim(1), 1);
            actions.push_back(static_cast<int64_t>(action.data()[0]));
        }
        return actions;
    }
};

TEST_F(DQNAgentTest, ExposesItsConstructionParameters) {
    DQNAgent agent(&q_network, kActionDim, 0.25f, &backend);

    EXPECT_EQ(agent.action_dim(), kActionDim);
    EXPECT_FLOAT_EQ(agent.epsilon(), 0.25f);
}

TEST_F(DQNAgentTest, ActReturnsTheActionAsAOneByOneFloatTensor) {
    DQNAgent agent(&q_network, kActionDim, 0.0f, &backend);

    const Tensor action = agent.act(observation());

    // The exact encoding CartPoleEnv::step() accepts -- a (1,1) float-encoded index.
    ASSERT_EQ(action.rank(), 2);
    EXPECT_EQ(action.shape().dim(0), 1);
    EXPECT_EQ(action.shape().dim(1), 1);
    EXPECT_FLOAT_EQ(action.data()[0], static_cast<float>(kGreedyAction));
}

// Boundary 1: epsilon == 0 must be *always* greedy, not merely usually.
TEST_F(DQNAgentTest, EpsilonZeroAlwaysTakesTheGreedyAction) {
    DQNAgent agent(&q_network, kActionDim, 0.0f, &backend);

    const std::vector<int64_t> actions = action_sequence(agent, 500);

    for (size_t d = 0; d < actions.size(); ++d) {
        ASSERT_EQ(actions[d], kGreedyAction) << "draw " << d << " was not the greedy action";
    }
}

// Boundary 2: epsilon == 1 must be *always* exploring. The claim is deliberately not "never
// emits the greedy index" -- a uniform draw over [0, action_dim) includes the greedy index, so
// that assertion would be false for any correct implementation. The checkable, distinguishing
// claim is a conjunction:
//   (a) the sequence is genuinely not the constant greedy sequence epsilon=0 produces;
//   (b) every action in the space is reached, so the draw is uniform over the whole range and
//       not, say, "greedy or greedy+1";
//   (c) the greedy index *does* occur by chance -- which is what makes (a) non-trivial: the
//       test is not passing merely because the greedy action happens to be unreachable.
// Together these separate a correct epsilon=1 agent from an epsilon=0 one, from an off-by-one
// range bug, and from an "explore means anything but greedy" bug.
TEST_F(DQNAgentTest, EpsilonOneAlwaysExploresAndNeverActsGreedilyByPolicy) {
    DQNAgent exploring(&q_network, kActionDim, 1.0f, &backend, 7);
    DQNAgent greedy(&q_network, kActionDim, 0.0f, &backend, 7);

    const std::vector<int64_t> explored = action_sequence(exploring, 600);
    const std::vector<int64_t> greedy_actions = action_sequence(greedy, 600);

    // (a)
    EXPECT_NE(explored, greedy_actions) << "an epsilon=1 agent reproduced the epsilon=0 action sequence exactly";
    int non_greedy = 0;
    int greedy_by_chance = 0;
    std::set<int64_t> distinct;
    for (int64_t a : explored) {
        ASSERT_GE(a, 0);
        ASSERT_LT(a, kActionDim);
        distinct.insert(a);
        if (a == kGreedyAction) {
            ++greedy_by_chance;
        } else {
            ++non_greedy;
        }
    }
    // (b)
    EXPECT_EQ(distinct, (std::set<int64_t>{0, 1, 2})) << "the random draw did not cover the whole action space";
    EXPECT_GT(non_greedy, 0);
    // (c) -- the non-vacuity clause. Roughly 1/3 of 600 draws should land on the greedy index.
    EXPECT_GT(greedy_by_chance, 0) << "the greedy action was never drawn, so (a) proves nothing about exploration";
    EXPECT_GT(greedy_by_chance, 100) << "the uniform draw is badly skewed away from the greedy index";
    EXPECT_LT(greedy_by_chance, 300) << "the uniform draw is badly skewed toward the greedy index";
}

// The quantitative middle of the two boundaries: epsilon actually scales the exploration rate,
// rather than merely switching it on somewhere between 0 and 1. With epsilon = 0.5 and 3
// actions, a draw is non-greedy with probability 0.5 * (2/3) = 1/3.
TEST_F(DQNAgentTest, IntermediateEpsilonExploresAtRoughlyTheRequestedRate) {
    DQNAgent agent(&q_network, kActionDim, 0.5f, &backend, 11);

    const std::vector<int64_t> actions = action_sequence(agent, 1200);
    int non_greedy = 0;
    for (int64_t a : actions) {
        if (a != kGreedyAction) {
            ++non_greedy;
        }
    }

    const double rate = static_cast<double>(non_greedy) / static_cast<double>(actions.size());
    // Expected 1/3; a generous band, since the point is to exclude 0 (never explores) and
    // 2/3 (explores at epsilon = 1), not to test the LCG's uniformity.
    EXPECT_GT(rate, 0.22) << "epsilon = 0.5 explored far too rarely";
    EXPECT_LT(rate, 0.45) << "epsilon = 0.5 explored far too often";
}

TEST_F(DQNAgentTest, SameSeedGivesTheIdenticalActionSequence) {
    DQNAgent a(&q_network, kActionDim, 0.5f, &backend, 1234);
    DQNAgent b(&q_network, kActionDim, 0.5f, &backend, 1234);

    EXPECT_EQ(action_sequence(a, 100), action_sequence(b, 100));
}

// Non-vacuity for the determinism test: the stream is genuinely seed-driven, not a constant
// sequence that would make any two agents agree.
TEST_F(DQNAgentTest, DifferentSeedsGiveDifferentActionSequences) {
    DQNAgent a(&q_network, kActionDim, 0.5f, &backend, 1234);
    DQNAgent b(&q_network, kActionDim, 0.5f, &backend, 99);

    EXPECT_NE(action_sequence(a, 100), action_sequence(b, 100));
}

TEST_F(DQNAgentTest, ActGreedyIgnoresEpsilonEntirely) {
    DQNAgent agent(&q_network, kActionDim, 1.0f, &backend);

    for (int d = 0; d < 200; ++d) {
        const Tensor action = agent.act_greedy(observation());
        ASSERT_FLOAT_EQ(action.data()[0], static_cast<float>(kGreedyAction)) << "draw " << d;
    }
}

// act_greedy() consumes no randomness, so interleaving an evaluation rollout with training
// cannot perturb the training run's own action sequence.
TEST_F(DQNAgentTest, ActGreedyDoesNotAdvanceTheExplorationStream) {
    DQNAgent undisturbed(&q_network, kActionDim, 0.5f, &backend, 2024);
    DQNAgent interleaved(&q_network, kActionDim, 0.5f, &backend, 2024);

    std::vector<int64_t> expected;
    std::vector<int64_t> actual;
    for (int d = 0; d < 60; ++d) {
        expected.push_back(static_cast<int64_t>(undisturbed.act(observation()).data()[0]));
        (void)interleaved.act_greedy(observation());
        actual.push_back(static_cast<int64_t>(interleaved.act(observation()).data()[0]));
        (void)interleaved.act_greedy(observation());
    }

    EXPECT_EQ(actual, expected);
}

TEST_F(DQNAgentTest, SetEpsilonChangesTheBehaviourItAdvertises) {
    DQNAgent agent(&q_network, kActionDim, 1.0f, &backend, 5);
    ASSERT_NE(action_sequence(agent, 200), std::vector<int64_t>(200, kGreedyAction));

    agent.set_epsilon(0.0f);

    EXPECT_FLOAT_EQ(agent.epsilon(), 0.0f);
    EXPECT_EQ(action_sequence(agent, 200), std::vector<int64_t>(200, kGreedyAction));
}

TEST_F(DQNAgentTest, SetEpsilonThrowsOutsideTheUnitInterval) {
    DQNAgent agent(&q_network, kActionDim, 0.5f, &backend);

    EXPECT_THROW({ agent.set_epsilon(-0.01f); }, std::invalid_argument);
    EXPECT_THROW({ agent.set_epsilon(1.01f); }, std::invalid_argument);
    EXPECT_FLOAT_EQ(agent.epsilon(), 0.5f) << "a rejected set_epsilon must not have taken effect";
    EXPECT_NO_THROW({ agent.set_epsilon(0.0f); });
    EXPECT_NO_THROW({ agent.set_epsilon(1.0f); });
}

TEST_F(DQNAgentTest, ConstructorThrowsOnInvalidArguments) {
    EXPECT_THROW({ DQNAgent agent(nullptr, kActionDim, 0.5f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ DQNAgent agent(&q_network, 0, 0.5f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ DQNAgent agent(&q_network, -2, 0.5f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ DQNAgent agent(&q_network, kActionDim, -0.1f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ DQNAgent agent(&q_network, kActionDim, 1.1f, &backend); }, std::invalid_argument);
}

TEST_F(DQNAgentTest, ThrowsWhenTheNetworkOutputWidthDisagreesWithActionDim) {
    // action_dim and q_network are independent constructor arguments a caller can mismatch;
    // an unchecked argmax would then read past the end of the network's output row.
    DQNAgent agent(&q_network, kActionDim + 1, 0.0f, &backend);

    EXPECT_THROW({ (void)agent.act(observation()); }, std::invalid_argument);
    EXPECT_THROW({ (void)agent.act_greedy(observation()); }, std::invalid_argument);
}

// The agent knows nothing about its network's internals, so a multi-layer container works
// unchanged -- this is the shape Mission 1's training loop will actually build.
TEST_F(DQNAgentTest, WorksWithAMultiLayerSequentialQNetwork) {
    LinearModule first(kObsDim, 4, &backend);
    ReluModule activation(&backend);
    LinearModule second(4, kActionDim, &backend);
    first.set_bias({1.0f, 1.0f, 1.0f, 1.0f});          // post-ReLU output is [1,1,1,1]
    second.set_weight({0.0f, 1.0f, 0.0f,               // -> q = [0, 4, 0] + [0, 0, 0.5]
                       0.0f, 1.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 1.0f, 0.0f});
    second.set_bias({0.0f, 0.0f, 0.5f});
    SequentialModule network({&first, &activation, &second});
    DQNAgent agent(&network, kActionDim, 0.0f, &backend);

    EXPECT_FLOAT_EQ(agent.act_greedy(observation()).data()[0], 1.0f);
}

using DQNAgentDeathTest = DQNAgentTest;

// act() argmaxes over the network output in a raw host loop over Tensor::data(); that output
// inherits its device from the observation, so a CUDA-backed observation is silent UB
// (mission_host_loop_guards.md). One death test, per the mission's one-per-entry-point rule:
// act() has exactly one caller-supplied Tensor argument, and act_greedy() reaches the same
// guarded argument role through the same helper.
TEST_F(DQNAgentDeathTest, ActAbortsOnNonCpuObservation) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    DQNAgent agent(&q_network, kActionDim, 0.0f, &backend);
    Tensor cuda_observation(Shape({1, kObsDim}), &backend, {0.3f, -0.7f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)agent.act(cuda_observation); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
