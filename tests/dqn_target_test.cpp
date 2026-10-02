#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dqn_target.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

class DQNTargetTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Hand derivation:
//   next_q_target = [[1.0, 5.0, 3.0],   rewards = [[0.5],   dones = [[0.0],   gamma = 0.9
//                    [7.0, 2.0, 4.0]]              [1.0]]            [1.0]]
//   row 0: max = 5.0, not done -> 0.5 + 0.9 * 1 * 5.0 = 0.5 + 4.5 = 5.0
//   row 1: done      -> 1.0 + 0.9 * 0 * 7.0 = 1.0
TEST_F(DQNTargetTest, VanillaTargetMatchesTheHandDerivedValues) {
    Tensor next_q_target(Shape({2, 3}), &backend, {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 4.0f});
    Tensor rewards(Shape({2, 1}), &backend, {0.5f, 1.0f});
    Tensor dones(Shape({2, 1}), &backend, {0.0f, 1.0f});

    const Tensor targets = ComputeDQNTarget(next_q_target, rewards, dones, 0.9f, &backend);

    ASSERT_EQ(targets.rank(), 2);
    EXPECT_EQ(targets.shape().dim(0), 2);
    EXPECT_EQ(targets.shape().dim(1), 1);
    EXPECT_FLOAT_EQ(targets.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(targets.data()[1], 1.0f);
}

// The done flag must zero the bootstrapped term *exactly*, not merely shrink it -- a terminal
// state has no successor whose value could be backed up. Asserted as bitwise equality against
// the reward, with a deliberately huge next-state Q-value that any leakage would show up in.
TEST_F(DQNTargetTest, DoneRowTargetEqualsItsRewardExactly) {
    Tensor next_q_target(Shape({1, 3}), &backend, {1.0e6f, 5.0e5f, 9.0e5f});
    Tensor rewards(Shape({1, 1}), &backend, {0.375f});
    Tensor dones(Shape({1, 1}), &backend, {1.0f});

    const Tensor targets = ComputeDQNTarget(next_q_target, rewards, dones, 0.99f, &backend);

    // EXPECT_EQ, not EXPECT_FLOAT_EQ: "equals its reward", full stop. Even a 1-ULP leak from
    // a 1.0e6 next-state value would be a real bug in the termination handling.
    EXPECT_EQ(targets.data()[0], 0.375f);
}

// The mirror of the test above, so a "dones is always treated as 1" bug cannot pass both.
TEST_F(DQNTargetTest, NotDoneRowBootstrapsFromTheMaxNextQValue) {
    Tensor next_q_target(Shape({1, 3}), &backend, {1.0f, 5.0f, 3.0f});
    Tensor rewards(Shape({1, 1}), &backend, {0.375f});
    Tensor dones(Shape({1, 1}), &backend, {0.0f});

    const Tensor targets = ComputeDQNTarget(next_q_target, rewards, dones, 0.5f, &backend);

    EXPECT_FLOAT_EQ(targets.data()[0], 0.375f + 0.5f * 5.0f);  // 2.875
}

TEST_F(DQNTargetTest, VanillaTargetUsesTheMaximumNotTheFirstOrLastColumn) {
    // Max sits in the middle column in row 0 and the last in row 1, so neither a
    // "always column 0" nor an "always last column" bug survives.
    Tensor next_q_target(Shape({2, 3}), &backend, {-1.0f, 4.0f, 0.0f, -5.0f, -6.0f, 2.0f});
    Tensor rewards(Shape({2, 1}), &backend, {0.0f, 0.0f});
    Tensor dones(Shape({2, 1}), &backend, {0.0f, 0.0f});

    const Tensor targets = ComputeDQNTarget(next_q_target, rewards, dones, 1.0f, &backend);

    EXPECT_FLOAT_EQ(targets.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(targets.data()[1], 2.0f);
}

// ---------------------------------------------------------------------------------------
// Double DQN
// ---------------------------------------------------------------------------------------

// The load-bearing divergence case (van Hasselt et al. 2016). Row 0 is constructed so the two
// networks *disagree* about which next action is best; row 1 so they agree.
//
//   next_q_online = [[10.0, 1.0],    argmax row 0 = 0   argmax row 1 = 0
//                    [ 3.0, 1.0]]
//   next_q_target = [[ 2.0, 8.0],    argmax row 0 = 1   argmax row 1 = 0
//                    [ 9.0, 4.0]]
//   rewards = [[0.25], [0.25]],  dones = [[0.0], [0.0]],  gamma = 0.5
//
//   Vanilla (target's own argmax, then target's value):
//     row 0: 0.25 + 0.5 * 8.0 = 4.25      row 1: 0.25 + 0.5 * 9.0 = 4.75
//   Double (online's argmax, then target's value at that action):
//     row 0: 0.25 + 0.5 * 2.0 = 1.25      row 1: 0.25 + 0.5 * 9.0 = 4.75
//
// Row 0's 4.25 vs 1.25 is the whole point: vanilla latched onto the target network's own
// optimistic 8.0, Double DQN evaluated the action the *online* network actually preferred.
class DoubleDQNDivergenceTest : public DQNTargetTest {
protected:
    Tensor next_q_online() { return Tensor(Shape({2, 2}), &backend, {10.0f, 1.0f, 3.0f, 1.0f}); }
    Tensor next_q_target() { return Tensor(Shape({2, 2}), &backend, {2.0f, 8.0f, 9.0f, 4.0f}); }
    Tensor rewards() { return Tensor(Shape({2, 1}), &backend, {0.25f, 0.25f}); }
    Tensor dones() { return Tensor(Shape({2, 1}), &backend, {0.0f, 0.0f}); }
};

TEST_F(DoubleDQNDivergenceTest, TheTwoTargetFunctionsProvablyDivergeWhenTheNetworksDisagree) {
    const Tensor vanilla = ComputeDQNTarget(next_q_target(), rewards(), dones(), 0.5f, &backend);
    const Tensor doubled =
        ComputeDoubleDQNTarget(next_q_online(), next_q_target(), rewards(), dones(), 0.5f, &backend);

    // Both hand-derived literals, then the divergence itself stated as its own claim.
    EXPECT_FLOAT_EQ(vanilla.data()[0], 4.25f);
    EXPECT_FLOAT_EQ(doubled.data()[0], 1.25f);
    EXPECT_NE(vanilla.data()[0], doubled.data()[0])
        << "vanilla and Double DQN produced the same target on a row where the online and "
           "target networks disagree about the best next action";

    // Row 1: the networks agree, so the two formulations must coincide -- otherwise the
    // divergence above could be explained by Double DQN simply computing something unrelated.
    EXPECT_FLOAT_EQ(vanilla.data()[1], 4.75f);
    EXPECT_FLOAT_EQ(doubled.data()[1], 4.75f);
}

// Double DQN must evaluate with the *target* network, never with the online one. If it
// mistakenly evaluated with next_q_online, row 0's target would be 0.25 + 0.5*10.0 = 5.25.
TEST_F(DoubleDQNDivergenceTest, DoubleTargetEvaluatesWithTheTargetNetworkNotTheOnlineOne) {
    const Tensor doubled =
        ComputeDoubleDQNTarget(next_q_online(), next_q_target(), rewards(), dones(), 0.5f, &backend);

    EXPECT_NE(doubled.data()[0], 5.25f) << "Double DQN evaluated the selected action with the online network";
    EXPECT_FLOAT_EQ(doubled.data()[0], 1.25f);
}

TEST_F(DoubleDQNDivergenceTest, SwappingTheTwoNetworkArgumentsChangesTheResult) {
    // Argument-order non-vacuity: selection and evaluation are genuinely read from different
    // tensors, so swapping them cannot produce the same answer.
    const Tensor normal =
        ComputeDoubleDQNTarget(next_q_online(), next_q_target(), rewards(), dones(), 0.5f, &backend);
    const Tensor swapped =
        ComputeDoubleDQNTarget(next_q_target(), next_q_online(), rewards(), dones(), 0.5f, &backend);

    // Swapped: select with [2.0, 8.0] -> action 1, evaluate in [10.0, 1.0] -> 1.0.
    EXPECT_FLOAT_EQ(swapped.data()[0], 0.25f + 0.5f * 1.0f);  // 0.75
    EXPECT_NE(normal.data()[0], swapped.data()[0]);
}

TEST_F(DoubleDQNDivergenceTest, DoubleTargetAlsoZeroesTheBootstrapOnDone) {
    Tensor all_done(Shape({2, 1}), &backend, {1.0f, 1.0f});

    const Tensor doubled =
        ComputeDoubleDQNTarget(next_q_online(), next_q_target(), rewards(), all_done, 0.5f, &backend);

    EXPECT_EQ(doubled.data()[0], 0.25f);
    EXPECT_EQ(doubled.data()[1], 0.25f);
}

// ---------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------

TEST_F(DQNTargetTest, TargetFunctionsThrowOnMalformedShapes) {
    Tensor next_q(Shape({2, 3}), &backend, {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 4.0f});
    Tensor rewards(Shape({2, 1}), &backend, {0.0f, 0.0f});
    Tensor dones(Shape({2, 1}), &backend, {0.0f, 0.0f});
    Tensor wrong_rows(Shape({3, 1}), &backend, {0.0f, 0.0f, 0.0f});
    Tensor flat(Shape({6}), &backend, {1.0f, 5.0f, 3.0f, 7.0f, 2.0f, 4.0f});

    EXPECT_THROW({ (void)ComputeDQNTarget(flat, rewards, dones, 0.9f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)ComputeDQNTarget(next_q, wrong_rows, dones, 0.9f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)ComputeDQNTarget(next_q, rewards, wrong_rows, 0.9f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)ComputeDoubleDQNTarget(next_q, next_q, wrong_rows, dones, 0.9f, &backend); },
                 std::invalid_argument);
}

TEST_F(DQNTargetTest, DoubleTargetThrowsWhenTheTwoNetworkOutputsHaveDifferentShapes) {
    Tensor online(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor target(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor rewards(Shape({2, 1}), &backend, {0.0f, 0.0f});
    Tensor dones(Shape({2, 1}), &backend, {0.0f, 0.0f});

    EXPECT_THROW({ (void)ComputeDoubleDQNTarget(online, target, rewards, dones, 0.9f, &backend); },
                 std::invalid_argument);
}

TEST_F(DQNTargetTest, TargetFunctionsThrowOnAGammaOutsideTheUnitInterval) {
    Tensor next_q(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor rewards(Shape({1, 1}), &backend, {0.0f});
    Tensor dones(Shape({1, 1}), &backend, {0.0f});

    EXPECT_THROW({ (void)ComputeDQNTarget(next_q, rewards, dones, 1.5f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)ComputeDQNTarget(next_q, rewards, dones, -0.1f, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)ComputeDoubleDQNTarget(next_q, next_q, rewards, dones, 1.5f, &backend); },
                 std::invalid_argument);
    // Both endpoints are legitimate: gamma = 0 is a myopic agent, gamma = 1 an undiscounted one.
    EXPECT_NO_THROW({ (void)ComputeDQNTarget(next_q, rewards, dones, 0.0f, &backend); });
    EXPECT_NO_THROW({ (void)ComputeDQNTarget(next_q, rewards, dones, 1.0f, &backend); });
}

// ---------------------------------------------------------------------------------------
// SyncTargetNetwork
// ---------------------------------------------------------------------------------------

class SyncTargetNetworkTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Two structurally identical two-layer networks. `source` gets recognizable weights;
    // `destination` keeps LinearModule's zero initialization, so "did the sync happen" and
    // "did it happen correctly" are the same question.
    SyncTargetNetworkTest()
        : source_a(2, 3, &backend),
          source_b(3, 2, &backend),
          destination_a(2, 3, &backend),
          destination_b(3, 2, &backend),
          source({&source_a, &source_b}),
          destination({&destination_a, &destination_b}) {
        source_a.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        source_a.set_bias({7.0f, 8.0f, 9.0f});
        source_b.set_weight({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
        source_b.set_bias({16.0f, 17.0f});
    }

    LinearModule source_a;
    LinearModule source_b;
    LinearModule destination_a;
    LinearModule destination_b;
    SequentialModule source;
    SequentialModule destination;
};

TEST_F(SyncTargetNetworkTest, CopiesEveryParameterValueFromSourceToDestination) {
    // Precondition: they genuinely differ beforehand, so the post-sync equality has content.
    ASSERT_NE(source_a.weight().data()[0], destination_a.weight().data()[0]);

    SyncTargetNetwork(source, destination);

    std::vector<ParamRef> source_params = source.parameters();
    std::vector<ParamRef> destination_params = destination.parameters();
    ASSERT_EQ(source_params.size(), 4u);  // two LinearModules, weight + bias each
    ASSERT_EQ(destination_params.size(), source_params.size());
    for (size_t p = 0; p < source_params.size(); ++p) {
        ASSERT_EQ(source_params[p].value->numel(), destination_params[p].value->numel()) << "parameter " << p;
        for (int64_t e = 0; e < source_params[p].value->numel(); ++e) {
            EXPECT_EQ(destination_params[p].value->data()[e], source_params[p].value->data()[e])
                << "parameter " << p << ", element " << e;
        }
    }
}

// The load-bearing independence property: the sync must be a real value copy, not an alias.
// The source is mutated *through its own buffer* afterwards (not via set_weight(), which
// replaces the Tensor object wholesale and so would mask a genuine aliasing bug).
TEST_F(SyncTargetNetworkTest, MutatingSourceAfterSyncLeavesDestinationUnaffected) {
    SyncTargetNetwork(source, destination);
    ASSERT_EQ(destination_a.weight().data()[0], 1.0f);
    ASSERT_EQ(destination_b.bias().data()[1], 17.0f);

    std::vector<ParamRef> source_params = source.parameters();
    for (ParamRef& param : source_params) {
        for (int64_t e = 0; e < param.value->numel(); ++e) {
            param.value->data()[e] = -999.0f;
        }
    }

    // Every synced value in the destination is exactly what it was, untouched.
    EXPECT_EQ(destination_a.weight().data()[0], 1.0f);
    EXPECT_EQ(destination_a.weight().data()[5], 6.0f);
    EXPECT_EQ(destination_a.bias().data()[2], 9.0f);
    EXPECT_EQ(destination_b.weight().data()[0], 10.0f);
    EXPECT_EQ(destination_b.bias().data()[1], 17.0f);
    // And the source really did change, so the check above is not vacuous.
    EXPECT_EQ(source_a.weight().data()[0], -999.0f);
}

// The converse direction: the destination is the network an optimizer would later update, so
// writing into it must not reach back into the source either.
TEST_F(SyncTargetNetworkTest, MutatingDestinationAfterSyncLeavesSourceUnaffected) {
    SyncTargetNetwork(source, destination);

    std::vector<ParamRef> destination_params = destination.parameters();
    destination_params[0].value->data()[0] = 42.0f;

    EXPECT_EQ(source_a.weight().data()[0], 1.0f);
    EXPECT_EQ(destination_a.weight().data()[0], 42.0f);
}

// The copy is element-wise into the destination's *existing* buffers. Replacing its Tensor
// objects would dangle every ParamRef an optimizer is already holding, so the buffer address
// must survive the sync.
TEST_F(SyncTargetNetworkTest, SyncPreservesTheDestinationsParameterStorageIdentity) {
    std::vector<ParamRef> before = destination.parameters();
    const Tensor* value_before = before[0].value;
    const float* buffer_before = before[0].value->data();

    SyncTargetNetwork(source, destination);

    std::vector<ParamRef> after = destination.parameters();
    EXPECT_EQ(after[0].value, value_before);
    EXPECT_EQ(after[0].value->data(), buffer_before) << "sync replaced the destination's parameter buffer";
}

TEST_F(SyncTargetNetworkTest, SyncDoesNotTouchGradients) {
    std::vector<ParamRef> source_params = source.parameters();
    std::vector<ParamRef> destination_params = destination.parameters();
    source_params[0].grad->data()[0] = 5.0f;
    destination_params[0].grad->data()[0] = 3.0f;

    SyncTargetNetwork(source, destination);

    EXPECT_EQ(destination_params[0].grad->data()[0], 3.0f) << "sync is a pure value copy, unrelated to gradients";
    EXPECT_EQ(source_params[0].grad->data()[0], 5.0f);
}

TEST_F(SyncTargetNetworkTest, SyncedDestinationProducesTheSameForwardOutputAsTheSource) {
    SyncTargetNetwork(source, destination);
    Tensor input(Shape({1, 2}), &backend, {0.5f, -1.5f});

    const Tensor from_source = source.forward(input);
    const Tensor from_destination = destination.forward(input);

    ASSERT_EQ(from_source.numel(), from_destination.numel());
    for (int64_t i = 0; i < from_source.numel(); ++i) {
        EXPECT_FLOAT_EQ(from_destination.data()[i], from_source.data()[i]) << "index " << i;
    }
}

TEST_F(SyncTargetNetworkTest, ThrowsOnMismatchedParameterCounts) {
    LinearModule lonely(2, 3, &backend);

    // 4 parameters (two layers) vs 2 (one layer).
    EXPECT_THROW({ SyncTargetNetwork(source, lonely); }, std::invalid_argument);
    EXPECT_THROW({ SyncTargetNetwork(lonely, source); }, std::invalid_argument);
}

TEST_F(SyncTargetNetworkTest, ThrowsOnMismatchedParameterShapes) {
    LinearModule wrong_a(2, 3, &backend);
    LinearModule wrong_b(3, 5, &backend);  // same parameter *count*, different shapes
    SequentialModule wrong({&wrong_a, &wrong_b});

    EXPECT_THROW({ SyncTargetNetwork(source, wrong); }, std::invalid_argument);
}

// A rejected sync must not leave the destination half-copied.
TEST_F(SyncTargetNetworkTest, AMismatchedShapeIsDetectedBeforeAnyPartialCopyOfThatParameter) {
    LinearModule wrong_a(2, 3, &backend);
    LinearModule wrong_b(3, 5, &backend);
    SequentialModule wrong({&wrong_a, &wrong_b});

    EXPECT_THROW({ SyncTargetNetwork(source, wrong); }, std::invalid_argument);
    // Parameter index 2 (wrong_b's weight) is where the shapes first disagree; it must be
    // untouched, even though indices 0 and 1 matched and were legitimately copied.
    EXPECT_EQ(wrong_b.weight().data()[0], 0.0f);
    EXPECT_EQ(wrong_b.bias().data()[0], 0.0f);
}

TEST_F(SyncTargetNetworkTest, SyncingAParameterlessModulePairIsALegalNoOp) {
    // Not every Module has parameters; an empty parameter list is a legitimate answer (see
    // Module::parameters()'s own note), not an architecture mismatch to reject.
    ReluModule a(&backend);
    ReluModule b(&backend);
    EXPECT_NO_THROW({ SyncTargetNetwork(a, b); });
}

}  // namespace
}  // namespace pulsatrix
