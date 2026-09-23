#include <gtest/gtest.h>

#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/dqn_loss.hpp"

namespace exai {
namespace {

class DQNLossTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The one hand-derived fixture every numeric assertion below refers back to.
    //
    //   q_values = [[1.0, 2.0, 3.0],      actions = [[2.0],     targets = [[1.0],
    //               [4.0, 5.0, 6.0]]                 [0.0]]                [6.5]]
    //
    //   selected[0] = q_values[0, 2] = 3.0   diff[0] = 3.0 - 1.0 =  2.0   diff^2 = 4.00
    //   selected[1] = q_values[1, 0] = 4.0   diff[1] = 4.0 - 6.5 = -2.5   diff^2 = 6.25
    //   loss = (4.00 + 6.25) / 2 = 5.125
    //
    //   scale = 2/N = 2/2 = 1.0
    //   grad = [[0.0, 0.0,  2.0],    <- only column 2 (the taken action) is non-zero
    //           [-2.5, 0.0, 0.0]]    <- only column 0 (the taken action) is non-zero
    //
    // Every one of these values is exactly representable in binary32, so the assertions are
    // exact-equality claims, not tolerance claims.
    Tensor q_values() { return Tensor(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); }
    Tensor actions() { return Tensor(Shape({2, 1}), &backend, {2.0f, 0.0f}); }
    Tensor targets() { return Tensor(Shape({2, 1}), &backend, {1.0f, 6.5f}); }
};

TEST_F(DQNLossTest, ForwardMatchesTheHandDerivedLoss) {
    DQNLoss loss(&backend);

    EXPECT_FLOAT_EQ(loss.forward(q_values(), actions(), targets()), 5.125f);
}

TEST_F(DQNLossTest, BackwardMatchesTheHandDerivedGradientTensorElementwise) {
    DQNLoss loss(&backend);
    (void)loss.forward(q_values(), actions(), targets());

    const Tensor grad = loss.backward();

    ASSERT_EQ(grad.rank(), 2);
    ASSERT_EQ(grad.shape().dim(0), 2);
    ASSERT_EQ(grad.shape().dim(1), 3);

    const float expected[6] = {0.0f, 0.0f, 2.0f, -2.5f, 0.0f, 0.0f};
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(grad.data()[i], expected[i]) << "flat index " << i;
    }
}

// Spelled out separately from the elementwise check above so a failure says *which* property
// broke: the masking is the defining property of the DQN semi-gradient update, and "zero"
// here must mean exactly 0.0f, not merely small.
TEST_F(DQNLossTest, BackwardIsExactlyZeroOnEveryNonSelectedColumn) {
    DQNLoss loss(&backend);
    (void)loss.forward(q_values(), actions(), targets());

    const Tensor grad = loss.backward();

    // Row 0 took action 2; row 1 took action 0.
    EXPECT_EQ(grad.data()[0], 0.0f);
    EXPECT_EQ(grad.data()[1], 0.0f);
    EXPECT_NE(grad.data()[2], 0.0f);
    EXPECT_NE(grad.data()[3], 0.0f);
    EXPECT_EQ(grad.data()[4], 0.0f);
    EXPECT_EQ(grad.data()[5], 0.0f);
}

// Non-vacuity for the masking test: a Q-value in a non-selected column has no influence on
// the loss at all, which is the behavioural statement the zero gradient encodes.
TEST_F(DQNLossTest, ChangingANonSelectedQValueDoesNotChangeTheLoss) {
    DQNLoss loss(&backend);
    const float baseline = loss.forward(q_values(), actions(), targets());

    // Perturb q_values[0, 0] (row 0 took action 2, so column 0 is unselected) hugely.
    Tensor perturbed(Shape({2, 3}), &backend, {1000.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    DQNLoss other(&backend);

    EXPECT_FLOAT_EQ(other.forward(perturbed, actions(), targets()), baseline);
}

TEST_F(DQNLossTest, ZeroTDErrorGivesZeroLossAndZeroGradient) {
    DQNLoss loss(&backend);
    Tensor exact_targets(Shape({2, 1}), &backend, {3.0f, 4.0f});  // exactly the selected Q-values

    EXPECT_FLOAT_EQ(loss.forward(q_values(), actions(), exact_targets), 0.0f);

    const Tensor grad = loss.backward();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_EQ(grad.data()[i], 0.0f) << "flat index " << i;
    }
}

TEST_F(DQNLossTest, BackwardThrowsBeforeForward) {
    DQNLoss loss(&backend);

    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(DQNLossTest, AcceptsAnActionEncodingWithinTheIntegerTolerance) {
    DQNLoss loss(&backend);
    // A policy's float round-trip may leave 2 as 2.00005; that must still decode to action 2.
    Tensor nearly_integer(Shape({2, 1}), &backend, {2.00005f, 0.0f});

    EXPECT_FLOAT_EQ(loss.forward(q_values(), nearly_integer, targets()), 5.125f);
}

TEST_F(DQNLossTest, ForwardThrowsOnAFractionalActionEncoding) {
    DQNLoss loss(&backend);
    // An un-argmaxed probability, say -- silently rounding it would train the wrong column.
    Tensor fractional(Shape({2, 1}), &backend, {1.5f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(q_values(), fractional, targets()); }, std::invalid_argument);
}

TEST_F(DQNLossTest, ForwardThrowsOnAnOutOfRangeActionIndex) {
    DQNLoss loss(&backend);
    Tensor too_large(Shape({2, 1}), &backend, {3.0f, 0.0f});   // action_dim is 3, so 3 is past the end
    Tensor negative(Shape({2, 1}), &backend, {-1.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(q_values(), too_large, targets()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(q_values(), negative, targets()); }, std::invalid_argument);
}

// A rejected forward() must not leave a half-populated cache that a later backward() would
// silently consume.
TEST_F(DQNLossTest, ARejectedForwardLeavesBackwardStillUnprimed) {
    DQNLoss loss(&backend);
    Tensor too_large(Shape({2, 1}), &backend, {3.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(q_values(), too_large, targets()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

TEST_F(DQNLossTest, ForwardThrowsOnMismatchedBatchSizes) {
    DQNLoss loss(&backend);
    Tensor short_actions(Shape({1, 1}), &backend, {0.0f});
    Tensor short_targets(Shape({1, 1}), &backend, {0.0f});

    EXPECT_THROW({ (void)loss.forward(q_values(), short_actions, targets()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(q_values(), actions(), short_targets); }, std::invalid_argument);
}

TEST_F(DQNLossTest, ForwardThrowsOnWrongRank) {
    DQNLoss loss(&backend);
    Tensor flat_q(Shape({6}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor flat_actions(Shape({2}), &backend, {2.0f, 0.0f});

    // Same numel as the valid tensors, wrong rank -- the case a bare "numel matches" check
    // would wave through.
    EXPECT_THROW({ (void)loss.forward(flat_q, actions(), targets()); }, std::invalid_argument);
    EXPECT_THROW({ (void)loss.forward(q_values(), flat_actions, targets()); }, std::invalid_argument);
}

TEST_F(DQNLossTest, ForwardThrowsOnAnActionsTensorWiderThanOneColumn) {
    DQNLoss loss(&backend);
    Tensor wide(Shape({2, 2}), &backend, {2.0f, 0.0f, 0.0f, 0.0f});

    EXPECT_THROW({ (void)loss.forward(q_values(), wide, targets()); }, std::invalid_argument);
}

// The single-action-dimension edge: action_dim == 1 is degenerate but well-formed, and the
// gradient is then a plain MSE gradient with no masking left to do.
TEST_F(DQNLossTest, SingleActionDimensionReducesToPlainMSE) {
    DQNLoss loss(&backend);
    Tensor q(Shape({2, 1}), &backend, {3.0f, 4.0f});
    Tensor a(Shape({2, 1}), &backend, {0.0f, 0.0f});
    Tensor t(Shape({2, 1}), &backend, {1.0f, 6.5f});

    // Identical selected values and targets as the main fixture -> identical loss.
    EXPECT_FLOAT_EQ(loss.forward(q, a, t), 5.125f);
    const Tensor grad = loss.backward();
    ASSERT_EQ(grad.numel(), 2);
    EXPECT_FLOAT_EQ(grad.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(grad.data()[1], -2.5f);
}

using DQNLossDeathTest = DQNLossTest;

// forward() reads all three tensors in one raw host loop over Tensor::data() -- undefined
// behavior on a CUDA-backed Tensor, so each is EXAI_ASSERT-guarded
// (mission_host_loop_guards.md). No real GPU needed: this reuses LinearModuleDeathTest's
// mislabeled-Tensor pattern.
//
// One death test, not three. The mission's Requirements section fixes the count at one per
// raw-host-loop *entry point*, and within forward() the three guards are adjacent lines on a
// single entry path covering one guarded-argument role: a caller-supplied host tensor read by
// the same loop. ReplayBuffer's own precedent -- two tests for three guards, because
// `next_observation` re-exercised `observation`'s path -- points the same way; `actions` and
// `targets` are both (N, 1) columns validated identically, so extra cases would re-cover a
// path rather than cover a new one.
TEST_F(DQNLossDeathTest, ForwardAbortsOnNonCpuQValues) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    DQNLoss loss(&backend);
    Tensor cuda_q(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)loss.forward(cuda_q, actions(), targets()); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
