#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_probe.hpp"

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
namespace pulsatrix {
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

// =======================================================================================
// Objective 2 -- positive and negative control.
//
// The mission's actual acceptance criterion. A probe that reaches high accuracy proves
// nothing on its own: a classifier with enough freedom relative to the sample can fit
// arbitrary labels, so "the probe succeeded" and "the concept is linearly represented" are
// only the same statement if the same procedure is shown to *fail* when the concept
// demonstrably isn't there. Hence two tests over the same feature distribution, the same
// dimensionality, the same probe seed, the same optimizer, the same learning rate and the
// same epoch count -- differing in exactly one thing: whether the labels are a linear
// function of the features or independent of them.
//
// Both report accuracy on a **held-out** split drawn from the same generator but never
// trained on (the mission's exit gate asks for this to be decided explicitly rather than
// left implicit). Held-out is the stronger choice for the negative control in particular:
// on the training split a linear model could in principle memorize some of the random
// labels, so a low training accuracy would be partly an artifact of the model's capacity
// being small relative to the sample, whereas held-out accuracy on independent labels is
// chance for *any* model, which is exactly the property being asserted.
// =======================================================================================

// Deterministic generator, hand-rolled rather than std::normal_distribution / std::
// uniform_real_distribution -- both have implementation-defined mappings from the engine's
// output, so identical seeds would not give identical data across standard libraries, and
// the accuracy thresholds below are asserted on specific data. This is the same
// "hand-roll it at the call site" convention MnistConvNet's weight init already follows;
// it deliberately does not add an RNG or a randn() to Tensor.
class TestRng {
public:
    explicit TestRng(uint64_t seed) : state_(seed) {}

    /** @brief Uniform in [0, 1). */
    float uniform() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t bits = static_cast<uint32_t>(state_ >> 32);
        return static_cast<float>(bits) / 4294967296.0f;
    }

    /** @brief Standard normal, via Box-Muller. */
    float gaussian() {
        const float u1 = std::fmax(uniform(), 1e-7f);
        const float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }

private:
    uint64_t state_;
};

constexpr int64_t kDim = 8;
constexpr int64_t kTrainSize = 256;
constexpr int64_t kTestSize = 256;
constexpr int kEpochs = 300;
constexpr float kLearningRate = 0.05f;
constexpr unsigned kProbeSeed = 7;

struct Dataset {
    std::vector<float> features;  // kDim floats per example, row-major
    std::vector<float> labels;    // one float per example, 0 or 1
    int64_t size = 0;
};

// Gaussian features. `separable == true` derives each label from a fixed direction w
// (label = 1 iff w.x > 0), making the concept linearly decodable *by construction*;
// `separable == false` draws each label by an independent coin flip, so no function of the
// features -- linear or otherwise -- predicts it better than chance. Everything else about
// the two datasets is identical, including the feature draw order.
Dataset make_dataset(TestRng& rng, const std::vector<float>& w, int64_t n, bool separable) {
    Dataset data;
    data.size = n;
    data.features.reserve(static_cast<size_t>(n * kDim));
    data.labels.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        float dot = 0.0f;
        for (int64_t j = 0; j < kDim; ++j) {
            const float x = rng.gaussian();
            data.features.push_back(x);
            dot += w[static_cast<size_t>(j)] * x;
        }
        const float coin = rng.uniform();
        data.labels.push_back(separable ? (dot > 0.0f ? 1.0f : 0.0f) : (coin < 0.5f ? 1.0f : 0.0f));
    }
    return data;
}

// The single shared training procedure. Both controls call exactly this -- it takes no
// knobs, precisely so neither test can quietly be given an advantage the other didn't get.
float train_probe_and_score_heldout(DeviceBackend* backend, const Dataset& train, const Dataset& heldout) {
    LinearProbe probe(kDim, backend, kProbeSeed);
    AdamOptimizer optimizer(kLearningRate, backend);

    const Tensor train_x(Shape({train.size, kDim}), backend, train.features);
    const Tensor train_y(Shape({train.size, 1}), backend, train.labels);
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        (void)probe.train_step(train_x, train_y, optimizer);
    }

    const Tensor heldout_x(Shape({heldout.size, kDim}), backend, heldout.features);
    const Tensor heldout_y(Shape({heldout.size, 1}), backend, heldout.labels);
    return probe.accuracy(heldout_x, heldout_y);
}

std::vector<float> make_concept_direction(TestRng& rng) {
    std::vector<float> w(static_cast<size_t>(kDim));
    for (float& value : w) {
        value = rng.gaussian();
    }
    return w;
}

class LinearProbeControlTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(LinearProbeControlTest, PositiveControlRecoversAConceptThatIsLinearlySeparableByConstruction) {
    TestRng rng(20260923u);
    const std::vector<float> w = make_concept_direction(rng);
    const Dataset train = make_dataset(rng, w, kTrainSize, /*separable=*/true);
    const Dataset heldout = make_dataset(rng, w, kTestSize, /*separable=*/true);

    const float accuracy = train_probe_and_score_heldout(&backend, train, heldout);

    std::cout << "[LinearProbe] positive control, held-out accuracy: " << accuracy << std::endl;
    EXPECT_GE(accuracy, 0.95f) << "held-out accuracy on a by-construction linearly separable concept";
}

TEST_F(LinearProbeControlTest, NegativeControlFailsWhenLabelsAreIndependentOfTheFeatures) {
    TestRng rng(20260923u);
    const std::vector<float> w = make_concept_direction(rng);
    const Dataset train = make_dataset(rng, w, kTrainSize, /*separable=*/false);
    const Dataset heldout = make_dataset(rng, w, kTestSize, /*separable=*/false);

    const float accuracy = train_probe_and_score_heldout(&backend, train, heldout);

    std::cout << "[LinearProbe] negative control, held-out accuracy: " << accuracy << std::endl;
    EXPECT_LT(accuracy, 0.65f) << "held-out accuracy on labels carrying no information about the features";
}

}  // namespace
}  // namespace pulsatrix
