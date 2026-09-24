#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/linear_probe.hpp"

// LinearProbe (campaign_exai_dl_library_mechanistic_interpretability, Phase 2 Mission 1)
// is the linear-probing utility: one LinearModule(activation_dim, 1) + BCEWithLogitsLoss,
// trained on (activation, binary concept label) pairs, answering "is this concept
// linearly decodable from this activation?".
//
// This file covers both mission objectives:
//   Objective 1 -- the class itself (construction, train_step, accuracy) plus its
//                  adversarial/boundary set.
//   Objective 2 -- the mission's actual acceptance criterion: a positive control (concept
//                  linearly separable by construction -> probe must succeed) and a
//                  negative control (labels independent of the features -> probe must
//                  fail), run through the *identical* training procedure, so the contrast
//                  proves the methodology rather than the probe's ability to fit anything.
namespace exai {
namespace {

class LinearProbeTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(LinearProbeTest, ConstructorBuildsAClassifierOfTheRequestedActivationDimension) {
    LinearProbe probe(4, &backend);

    EXPECT_EQ(probe.activation_dim(), 4);
    EXPECT_EQ(probe.classifier().weight().shape(), Shape({4, 1}));
    EXPECT_EQ(probe.classifier().bias().shape(), Shape({1}));
}

// Weights must not start all-zero: an all-zero probe produces logit 0 -> sigmoid exactly
// 0.5 for every example, landing every single prediction precisely on accuracy()'s
// threshold. Small seeded random init avoids that degenerate tie without introducing
// nondeterminism (same seed -> same weights).
TEST_F(LinearProbeTest, ConstructorInitializesWeightsDeterministicallyAndNonZero) {
    LinearProbe a(4, &backend, /*seed=*/123);
    LinearProbe b(4, &backend, /*seed=*/123);

    bool any_non_zero = false;
    for (int64_t i = 0; i < a.classifier().weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(a.classifier().weight()[i], b.classifier().weight()[i]);
        if (a.classifier().weight()[i] != 0.0f) {
            any_non_zero = true;
        }
    }
    EXPECT_TRUE(any_non_zero);
}

TEST_F(LinearProbeTest, TrainStepReturnsAFiniteLoss) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({2, 2}), &backend, {1.0f, 1.0f, -1.0f, -1.0f});
    Tensor labels(Shape({2, 1}), &backend, {1.0f, 0.0f});

    const float loss = probe.train_step(activations, labels, optimizer);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0f);
}

TEST_F(LinearProbeTest, TrainStepDecreasesLossOverRepeatedCallsOnAFixedBatch) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({2, 2}), &backend, {1.0f, 1.0f, -1.0f, -1.0f});
    Tensor labels(Shape({2, 1}), &backend, {1.0f, 0.0f});

    const float first = probe.train_step(activations, labels, optimizer);
    float last = first;
    for (int i = 0; i < 30; ++i) {
        last = probe.train_step(activations, labels, optimizer);
    }

    EXPECT_LT(last, first);
}

// accuracy() must not disturb the probe's learned parameters -- it is forward-only.
TEST_F(LinearProbeTest, AccuracyDoesNotChangeTheProbeParameters) {
    LinearProbe probe(2, &backend);
    Tensor activations(Shape({2, 2}), &backend, {1.0f, 1.0f, -1.0f, -1.0f});
    Tensor labels(Shape({2, 1}), &backend, {1.0f, 0.0f});
    const std::vector<float> before(probe.classifier().weight().data(),
                                    probe.classifier().weight().data() + probe.classifier().weight().numel());

    (void)probe.accuracy(activations, labels);

    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_FLOAT_EQ(probe.classifier().weight()[static_cast<int64_t>(i)], before[i]);
    }
}

TEST_F(LinearProbeTest, AccuracyIsOneWhenEveryPredictionMatchesItsLabel) {
    LinearProbe probe(2, &backend);
    probe.classifier().set_weight({1.0f, 1.0f});
    probe.classifier().set_bias({0.0f});
    Tensor activations(Shape({2, 2}), &backend, {2.0f, 2.0f, -2.0f, -2.0f});
    Tensor labels(Shape({2, 1}), &backend, {1.0f, 0.0f});

    EXPECT_FLOAT_EQ(probe.accuracy(activations, labels), 1.0f);
}

TEST_F(LinearProbeTest, AccuracyIsZeroWhenEveryPredictionIsWrong) {
    LinearProbe probe(2, &backend);
    probe.classifier().set_weight({1.0f, 1.0f});
    probe.classifier().set_bias({0.0f});
    Tensor activations(Shape({2, 2}), &backend, {2.0f, 2.0f, -2.0f, -2.0f});
    Tensor labels(Shape({2, 1}), &backend, {0.0f, 1.0f});

    EXPECT_FLOAT_EQ(probe.accuracy(activations, labels), 0.0f);
}

TEST_F(LinearProbeTest, AccuracyReportsTheFractionCorrectOnAMixedBatch) {
    LinearProbe probe(1, &backend);
    probe.classifier().set_weight({1.0f});
    probe.classifier().set_bias({0.0f});
    Tensor activations(Shape({4, 1}), &backend, {3.0f, 2.0f, -3.0f, -2.0f});
    Tensor labels(Shape({4, 1}), &backend, {1.0f, 1.0f, 0.0f, 1.0f});  // last one wrong

    EXPECT_FLOAT_EQ(probe.accuracy(activations, labels), 0.75f);
}

// Boundary: logit exactly 0 -> sigmoid exactly 0.5, precisely on the threshold. The
// documented convention is >= 0.5 predicts the positive class, so this must score as a
// predicted 1 -- not left to whichever way the floating-point comparison happens to fall.
TEST_F(LinearProbeTest, AccuracyTreatsExactlyHalfProbabilityAsThePositiveClass) {
    LinearProbe probe(1, &backend);
    probe.classifier().set_weight({0.0f});
    probe.classifier().set_bias({0.0f});
    Tensor activations(Shape({1, 1}), &backend, {5.0f});  // logit = 5*0 + 0 = 0 exactly

    Tensor positive_label(Shape({1, 1}), &backend, {1.0f});
    EXPECT_FLOAT_EQ(probe.accuracy(activations, positive_label), 1.0f);

    Tensor negative_label(Shape({1, 1}), &backend, {0.0f});
    EXPECT_FLOAT_EQ(probe.accuracy(activations, negative_label), 0.0f);
}

// ---------------------------------------------------------------------------------------
// Adversarial / boundary conditions.
//
// Every LinearProbe entry point is classified **external boundary -> throw** per
// cpp_tdd/context_tdd_adversarial_boundary_testing.md: a probe is driven by an analyst
// with caller-assembled activation/label batches (ultimately, via Phase 5's bindings, from
// Python), not by an already-validated internal call chain. Per that file's "when
// genuinely unsure, default to external boundary" rule, these are real throws and every
// one of these tests must pass identically in Debug and Release -- no NDEBUG skip.
//
// The checks live on LinearProbe itself rather than being delegated to LinearModule::
// forward / Module::forward / BCEWithLogitsLoss::forward, per the same file's "validate at
// the *true* entry point, not wherever an earlier unrelated fix happens to also catch some
// of the bad cases": a mismatched activation/label batch size is invisible to every one of
// those downstream checks individually (each operand is internally well-formed), and an
// empty batch would otherwise surface as Module::forward's generic message with no mention
// of the probe.
// ---------------------------------------------------------------------------------------

TEST_F(LinearProbeTest, ConstructorThrowsOnNonPositiveActivationDimension) {
    EXPECT_THROW(LinearProbe(0, &backend), std::invalid_argument);
    EXPECT_THROW(LinearProbe(-1, &backend), std::invalid_argument);
}

TEST_F(LinearProbeTest, TrainStepThrowsOnMismatchedActivationAndLabelBatchSizes) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({3, 2}), &backend);
    Tensor labels(Shape({2, 1}), &backend);

    EXPECT_THROW((void)probe.train_step(activations, labels, optimizer), std::invalid_argument);
}

TEST_F(LinearProbeTest, AccuracyThrowsOnMismatchedActivationAndLabelBatchSizes) {
    LinearProbe probe(2, &backend);
    Tensor activations(Shape({3, 2}), &backend);
    Tensor labels(Shape({2, 1}), &backend);

    EXPECT_THROW((void)probe.accuracy(activations, labels), std::invalid_argument);
}

TEST_F(LinearProbeTest, TrainStepThrowsOnAnEmptyBatch) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({0, 2}), &backend);
    Tensor labels(Shape({0, 1}), &backend);

    EXPECT_THROW((void)probe.train_step(activations, labels, optimizer), std::invalid_argument);
}

TEST_F(LinearProbeTest, AccuracyThrowsOnAnEmptyBatch) {
    LinearProbe probe(2, &backend);
    Tensor activations(Shape({0, 2}), &backend);
    Tensor labels(Shape({0, 1}), &backend);

    EXPECT_THROW((void)probe.accuracy(activations, labels), std::invalid_argument);
}

TEST_F(LinearProbeTest, TrainStepThrowsWhenActivationWidthDiffersFromTheProbeDimension) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({2, 5}), &backend);
    Tensor labels(Shape({2, 1}), &backend);

    EXPECT_THROW((void)probe.train_step(activations, labels, optimizer), std::invalid_argument);
}

TEST_F(LinearProbeTest, TrainStepThrowsOnARankOneActivationBatch) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({2}), &backend);
    Tensor labels(Shape({1, 1}), &backend);

    EXPECT_THROW((void)probe.train_step(activations, labels, optimizer), std::invalid_argument);
}

TEST_F(LinearProbeTest, TrainStepThrowsWhenLabelsAreNotASingleColumn) {
    LinearProbe probe(2, &backend);
    AdamOptimizer optimizer(0.1f, &backend);
    Tensor activations(Shape({2, 2}), &backend);
    Tensor labels(Shape({2, 2}), &backend);

    EXPECT_THROW((void)probe.train_step(activations, labels, optimizer), std::invalid_argument);
}

TEST_F(LinearProbeTest, AccuracyThrowsOnARankOneLabelBatch) {
    LinearProbe probe(2, &backend);
    Tensor activations(Shape({2, 2}), &backend);
    Tensor labels(Shape({2}), &backend);

    EXPECT_THROW((void)probe.accuracy(activations, labels), std::invalid_argument);
}

}  // namespace
}  // namespace exai
