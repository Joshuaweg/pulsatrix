#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pulsatrix/categorical_policy_agent.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

constexpr int64_t kObsDim = 2;
constexpr int64_t kActionDim = 3;

// The fixture's policy network leaves LinearModule's zero weight initialization alone and sets
// the bias, so for *any* observation the logits are exactly the bias:
//
//     logits = x @ 0 + bias = [1.0, 2.0, 3.0]
//
// Stable softmax (max_logit = 3):
//     exp(1-3) = 0.1353352832366127
//     exp(2-3) = 0.3678794411714423
//     exp(3-3) = 1.0
//     exp_sum  = 1.5032147244080550,  log(exp_sum) = 0.4076059644443806
//     p        = [0.0900305731703805, 0.2447284710547976, 0.6652409557748219]
//     cum p    = [0.0900305731703805, 0.3347590442251781, 1.0]
//     log p    = [-2.4076059644443806, -1.4076059644443806, -0.4076059644443806]
//
// These are the same numbers PolicyGradientLoss's fixture row 0 uses, deliberately: the two
// classes compute the same stabilized softmax and a shared hand-derivation makes a divergence
// between them visible.
constexpr float kProb0 = 0.0900305731703805f;
constexpr float kProb1 = 0.2447284710547976f;
constexpr float kProb2 = 0.6652409557748219f;
constexpr float kLogProb1 = -1.4076059644443806f;

// Tight enough that no plausible formula error survives, loose enough for float32 rounding.
constexpr float kTol = 1e-6f;

class CategoricalPolicyAgentTest : public ::testing::Test {
protected:
    CPUBackend backend;

    CategoricalPolicyAgentTest() : policy_network(kObsDim, kActionDim, &backend) {
        policy_network.set_bias({1.0f, 2.0f, 3.0f});
    }

    LinearModule policy_network;

    Tensor observation() { return Tensor(Shape({1, kObsDim}), &backend, {0.3f, -0.7f}); }

    // The action indices an agent produces over `draws` successive act() calls.
    std::vector<int64_t> action_sequence(CategoricalPolicyAgent& agent, int draws) {
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

    // The LCG this codebase uses everywhere (Numerical Recipes constants), reimplemented here
    // *independently of the agent* so the hand-traced tests below check the implementation
    // against an external computation rather than against itself.
    static float first_uniform_draw(uint32_t seed) {
        const uint32_t state = seed * 1664525u + 1013904223u;
        const uint32_t draw = (state >> 8) & 0xFFFFu;
        return static_cast<float>(draw) / 65536.0f;
    }
};

TEST_F(CategoricalPolicyAgentTest, ExposesItsConstructionParameters) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend);

    EXPECT_EQ(agent.action_dim(), kActionDim);
}

TEST_F(CategoricalPolicyAgentTest, ActReturnsTheActionAsAOneByOneFloatTensor) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend);

    const Tensor action = agent.act(observation());

    // The exact encoding CartPoleEnv::step() accepts -- a (1,1) float-encoded index.
    ASSERT_EQ(action.rank(), 2);
    EXPECT_EQ(action.shape().dim(0), 1);
    EXPECT_EQ(action.shape().dim(1), 1);
    EXPECT_GE(action.data()[0], 0.0f);
    EXPECT_LT(action.data()[0], static_cast<float>(kActionDim));
    EXPECT_FLOAT_EQ(action.data()[0], std::round(action.data()[0]));
}

// The primary sampling-correctness proof: three seeds, three hand-traced first draws, three
// different buckets of the inverse CDF. Each `u` below is computed from the LCG's own integer
// recurrence, not read off the agent, and each expected action is the smallest index whose
// cumulative probability reaches that `u`:
//
//   seed  6: state = 6*1664525 + 1013904223 mod 2^32, (state >> 8) & 0xFFFF = 1879
//            u = 1879/65536  = 0.0286712646484375 <= 0.0900305732 -> action 0
//   seed  7: draw = 8381,  u = 8381/65536  = 0.1278839111328125, above cum[0] = 0.0900305732
//            and at or below cum[1] = 0.3347590442               -> action 1
//   seed 42: draw = 39345, u = 39345/65536 = 0.6003570556640625, above cum[1]  -> action 2
//
// All three buckets matter: "always the last index" (a scan that never breaks), "always the
// first index" (a scan comparing the wrong way round), and an off-by-one in the cumulative
// comparison are each caught by at least one of these three cases and by no single one of them.
TEST_F(CategoricalPolicyAgentTest, SampledActionMatchesTheHandTracedInverseCdfDraw) {
    struct Case {
        uint32_t seed;
        float expected_u;
        int64_t expected_action;
    };
    const Case cases[3] = {{6, 1879.0f / 65536.0f, 0}, {7, 8381.0f / 65536.0f, 1}, {42, 39345.0f / 65536.0f, 2}};

    for (const Case& c : cases) {
        // The draw the agent will consume, recomputed independently from the LCG recurrence.
        ASSERT_FLOAT_EQ(first_uniform_draw(c.seed), c.expected_u) << "seed " << c.seed;

        // The inverse CDF applied to that draw, computed here from the hand-derived
        // probabilities rather than by calling anything on the agent.
        const float cumulative[3] = {kProb0, kProb0 + kProb1, kProb0 + kProb1 + kProb2};
        int64_t independently_sampled = kActionDim - 1;
        for (int64_t a = 0; a < kActionDim; ++a) {
            if (cumulative[a] >= c.expected_u) {
                independently_sampled = a;
                break;
            }
        }
        ASSERT_EQ(independently_sampled, c.expected_action)
            << "the test's own hand-derivation is inconsistent for seed " << c.seed;

        CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend, c.seed);
        const Tensor action = agent.act(observation());

        EXPECT_FLOAT_EQ(action.data()[0], static_cast<float>(c.expected_action)) << "seed " << c.seed;
    }
}

// Supplementary to the hand-traced draw above, not a substitute for it: a single traced sample
// cannot see a systematically biased sampler, and a frequency check cannot prove any particular
// draw is right. 6000 samples of a deliberately lopsided distribution (0.090 / 0.245 / 0.665)
// separates it from uniform (0.333 each) by a wide margin at this tolerance.
TEST_F(CategoricalPolicyAgentTest, EmpiricalFrequenciesApproximateTheTrueProbabilities) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend, 2024);
    constexpr int kSamples = 6000;

    const std::vector<int64_t> actions = action_sequence(agent, kSamples);

    int counts[3] = {0, 0, 0};
    for (int64_t a : actions) {
        ASSERT_GE(a, 0);
        ASSERT_LT(a, kActionDim);
        ++counts[a];
    }
    const float expected[3] = {kProb0, kProb1, kProb2};
    for (int64_t a = 0; a < kActionDim; ++a) {
        const float frequency = static_cast<float>(counts[a]) / static_cast<float>(kSamples);
        EXPECT_NEAR(frequency, expected[a], 0.02f) << "action " << a << " frequency " << frequency;
    }
}

// log_prob() must be the log-probability of the action actually sampled. The expected value is
// recomputed here from the network's known logits by an independent stabilized softmax -- no
// internal method of the agent is consulted.
TEST_F(CategoricalPolicyAgentTest, LogProbMatchesAnIndependentlyRecomputedSoftmax) {
    // The cached logits, known exactly because the network's weights are zero.
    const float logits[3] = {1.0f, 2.0f, 3.0f};
    const float max_logit = 3.0f;
    float exp_sum = 0.0f;
    for (int a = 0; a < 3; ++a) {
        exp_sum += std::exp(logits[a] - max_logit);
    }
    float expected_log_softmax[3];
    for (int a = 0; a < 3; ++a) {
        expected_log_softmax[a] = logits[a] - max_logit - std::log(exp_sum);
    }
    // Cross-check the independent computation against the hand-derived constant.
    ASSERT_NEAR(expected_log_softmax[1], kLogProb1, kTol);

    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend, 2024);
    for (int d = 0; d < 200; ++d) {
        const int64_t sampled = static_cast<int64_t>(agent.act(observation()).data()[0]);
        ASSERT_GE(sampled, 0);
        ASSERT_LT(sampled, kActionDim);
        EXPECT_NEAR(agent.log_prob(), expected_log_softmax[sampled], kTol) << "draw " << d;
        // The reported log-probability is a genuine log-probability of the right magnitude.
        EXPECT_NEAR(std::exp(agent.log_prob()), std::exp(expected_log_softmax[sampled]), kTol);
    }
}

// The hand-traced seed 7 case again, now read through log_prob(): that draw samples action 1,
// whose log-probability is log(0.2447284710547976) = -1.4076059644443806.
TEST_F(CategoricalPolicyAgentTest, LogProbAfterTheHandTracedDrawIsTheHandDerivedValue) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend, 7);

    ASSERT_FLOAT_EQ(agent.act(observation()).data()[0], 1.0f);

    EXPECT_NEAR(agent.log_prob(), kLogProb1, kTol);
    EXPECT_NEAR(std::exp(agent.log_prob()), kProb1, kTol);
}

TEST_F(CategoricalPolicyAgentTest, LogProbThrowsBeforeAnyAct) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend);

    // Not 0.0f (a probability of 1) -- a silently plausible lie a training loop would
    // happily backpropagate.
    EXPECT_THROW({ (void)agent.log_prob(); }, std::logic_error);
}

TEST_F(CategoricalPolicyAgentTest, ActGreedyTakesTheArgmaxLogitEveryTime) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend);

    for (int d = 0; d < 200; ++d) {
        const Tensor action = agent.act_greedy(observation());
        ASSERT_EQ(action.rank(), 2);
        ASSERT_FLOAT_EQ(action.data()[0], 2.0f) << "draw " << d;  // argmax [1, 2, 3] = 2
    }
}

// act_greedy() consumes no randomness, so interleaving an evaluation rollout with training
// cannot perturb the training run's own sampled action sequence.
TEST_F(CategoricalPolicyAgentTest, ActGreedyDoesNotAdvanceTheSamplingStream) {
    CategoricalPolicyAgent undisturbed(&policy_network, kActionDim, &backend, 2024);
    CategoricalPolicyAgent interleaved(&policy_network, kActionDim, &backend, 2024);

    std::vector<int64_t> expected;
    std::vector<int64_t> actual;
    for (int d = 0; d < 60; ++d) {
        expected.push_back(static_cast<int64_t>(undisturbed.act(observation()).data()[0]));
        (void)interleaved.act_greedy(observation());
        actual.push_back(static_cast<int64_t>(interleaved.act(observation()).data()[0]));
        (void)interleaved.act_greedy(observation());
    }

    EXPECT_EQ(actual, expected);
    // Non-vacuity: the sequence is genuinely a sampled one, not a constant that any
    // stream-perturbation would leave unchanged anyway.
    EXPECT_NE(expected, std::vector<int64_t>(expected.size(), expected[0]));
}

// act_greedy() samples nothing, so it must leave log_prob() reporting the last *sampled*
// action's probability -- otherwise an evaluation rollout run between act() and the training
// loop's log_prob() read would silently corrupt the stored log-probability.
TEST_F(CategoricalPolicyAgentTest, ActGreedyDoesNotUpdateLogProb) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend, 7);

    ASSERT_FLOAT_EQ(agent.act(observation()).data()[0], 1.0f);
    const float after_act = agent.log_prob();

    for (int d = 0; d < 10; ++d) {
        (void)agent.act_greedy(observation());
        EXPECT_FLOAT_EQ(agent.log_prob(), after_act) << "act_greedy call " << d;
    }
    EXPECT_NEAR(after_act, kLogProb1, kTol);
}

// The other half of the same guarantee: act_greedy() before any act() must not prime
// log_prob() either.
TEST_F(CategoricalPolicyAgentTest, ActGreedyAloneLeavesLogProbUnprimed) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim, &backend);

    (void)agent.act_greedy(observation());

    EXPECT_THROW({ (void)agent.log_prob(); }, std::logic_error);
}

TEST_F(CategoricalPolicyAgentTest, SameSeedGivesTheIdenticalActionSequence) {
    CategoricalPolicyAgent a(&policy_network, kActionDim, &backend, 1234);
    CategoricalPolicyAgent b(&policy_network, kActionDim, &backend, 1234);

    EXPECT_EQ(action_sequence(a, 100), action_sequence(b, 100));
}

// Non-vacuity for the determinism test: the stream is genuinely seed-driven, not a constant
// sequence that would make any two agents agree.
TEST_F(CategoricalPolicyAgentTest, DifferentSeedsGiveDifferentActionSequences) {
    CategoricalPolicyAgent a(&policy_network, kActionDim, &backend, 1234);
    CategoricalPolicyAgent b(&policy_network, kActionDim, &backend, 99);

    EXPECT_NE(action_sequence(a, 100), action_sequence(b, 100));
}

// A near-deterministic distribution: one logit far above the rest leaves essentially no
// probability anywhere else, so the sampler must return that action on every draw. This is the
// boundary an unstabilized softmax (exp of a large logit) or a mis-ordered scan would break.
TEST_F(CategoricalPolicyAgentTest, AnOverwhelminglyPeakedDistributionAlwaysSamplesItsPeak) {
    LinearModule peaked(kObsDim, kActionDim, &backend);
    peaked.set_bias({0.0f, 60.0f, 0.0f});  // p ~= [8.8e-27, 1 - 1.8e-26, 8.8e-27]
    CategoricalPolicyAgent agent(&peaked, kActionDim, &backend, 3);

    const std::vector<int64_t> actions = action_sequence(agent, 500);

    EXPECT_EQ(actions, std::vector<int64_t>(500, 1));
    EXPECT_NEAR(agent.log_prob(), 0.0f, 1e-6f) << "log-probability of a near-certain action must be ~0";
}

// The uniform boundary: equal logits must give equal sampling frequencies -- the case a
// max-subtraction bug that collapsed the distribution would break loudly.
TEST_F(CategoricalPolicyAgentTest, EqualLogitsSampleUniformly) {
    LinearModule flat(kObsDim, kActionDim, &backend);
    flat.set_bias({4.0f, 4.0f, 4.0f});
    CategoricalPolicyAgent agent(&flat, kActionDim, &backend, 11);
    constexpr int kSamples = 6000;

    const std::vector<int64_t> actions = action_sequence(agent, kSamples);

    int counts[3] = {0, 0, 0};
    for (int64_t a : actions) {
        ++counts[a];
    }
    for (int64_t a = 0; a < kActionDim; ++a) {
        const float frequency = static_cast<float>(counts[a]) / static_cast<float>(kSamples);
        EXPECT_NEAR(frequency, 1.0f / 3.0f, 0.02f) << "action " << a;
    }
    EXPECT_NEAR(agent.log_prob(), -std::log(3.0f), kTol);
}

// The single-action-dimension edge: degenerate but well-formed. The only action has probability
// exactly 1, so log_prob() is 0 and the draw cannot land anywhere else.
TEST_F(CategoricalPolicyAgentTest, SingleActionDimensionAlwaysSamplesActionZero) {
    LinearModule single(kObsDim, 1, &backend);
    single.set_bias({-7.0f});
    CategoricalPolicyAgent agent(&single, 1, &backend);

    for (int d = 0; d < 100; ++d) {
        ASSERT_FLOAT_EQ(agent.act(observation()).data()[0], 0.0f) << "draw " << d;
        ASSERT_FLOAT_EQ(agent.log_prob(), 0.0f) << "draw " << d;
    }
    EXPECT_FLOAT_EQ(agent.act_greedy(observation()).data()[0], 0.0f);
}

TEST_F(CategoricalPolicyAgentTest, ConstructorThrowsOnInvalidArguments) {
    EXPECT_THROW({ CategoricalPolicyAgent agent(nullptr, kActionDim, &backend); }, std::invalid_argument);
    EXPECT_THROW({ CategoricalPolicyAgent agent(&policy_network, 0, &backend); }, std::invalid_argument);
    EXPECT_THROW({ CategoricalPolicyAgent agent(&policy_network, -2, &backend); }, std::invalid_argument);
}

TEST_F(CategoricalPolicyAgentTest, ThrowsWhenTheNetworkOutputWidthDisagreesWithActionDim) {
    // action_dim and policy_network are independent constructor arguments a caller can
    // mismatch; an unchecked softmax scan would then read past the end of the output row.
    CategoricalPolicyAgent agent(&policy_network, kActionDim + 1, &backend);

    EXPECT_THROW({ (void)agent.act(observation()); }, std::invalid_argument);
    EXPECT_THROW({ (void)agent.act_greedy(observation()); }, std::invalid_argument);
}

// A rejected act() must not leave log_prob() primed with a value from a forward pass that
// never produced an action.
TEST_F(CategoricalPolicyAgentTest, ARejectedActLeavesLogProbUnprimed) {
    CategoricalPolicyAgent agent(&policy_network, kActionDim + 1, &backend);

    EXPECT_THROW({ (void)agent.act(observation()); }, std::invalid_argument);
    EXPECT_THROW({ (void)agent.log_prob(); }, std::logic_error);
}

// The agent knows nothing about its network's internals, so a multi-layer container works
// unchanged -- this is the shape Mission 1's REINFORCE training loop will actually build.
TEST_F(CategoricalPolicyAgentTest, WorksWithAMultiLayerSequentialPolicyNetwork) {
    LinearModule first(kObsDim, 4, &backend);
    ReluModule activation(&backend);
    LinearModule second(4, kActionDim, &backend);
    first.set_bias({1.0f, 1.0f, 1.0f, 1.0f});  // post-ReLU output is [1,1,1,1]
    second.set_weight({0.0f, 1.0f, 0.0f,       // -> logits = [0, 4, 0] + [0, 0, 0.5]
                       0.0f, 1.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 1.0f, 0.0f});
    second.set_bias({0.0f, 0.0f, 0.5f});
    SequentialModule network({&first, &activation, &second});
    CategoricalPolicyAgent agent(&network, kActionDim, &backend);

    EXPECT_FLOAT_EQ(agent.act_greedy(observation()).data()[0], 1.0f);
    const int64_t sampled = static_cast<int64_t>(agent.act(observation()).data()[0]);
    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, kActionDim);
    EXPECT_LT(agent.log_prob(), 0.0f);
}

}  // namespace
}  // namespace pulsatrix
