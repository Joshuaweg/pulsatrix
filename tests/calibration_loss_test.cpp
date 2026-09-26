/** @file calibration_loss_test.cpp
 *  @brief CalibrationLoss (Brier score) correctness -- known-calibrated, known-overconfident,
 *         perfect, and maximally-uncertain hand-derived cases.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class CalibrationLossTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor probs_row(std::initializer_list<float> values) { return Tensor(Shape({1, 3}), &backend, values); }

    Tensor target(float class_index) { return Tensor(Shape({1, 1}), &backend, {class_index}); }
};

// Known-calibrated: confidently and correctly predicts class 0.
// BS = (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14.
TEST_F(CalibrationLossTest, KnownCalibratedCaseMatchesHandDerivedValue) {
    CalibrationLoss loss(&backend);
    const float value = loss.forward(probs_row({0.7f, 0.2f, 0.1f}), target(0.0f));
    EXPECT_NEAR(value, 0.14f, 1e-5f);
}

// Known-overconfident: confidently predicts class 0, but the true class is 1.
// BS = (0.95-0)^2 + (0.03-1)^2 + (0.02-0)^2 = 0.9025 + 0.9409 + 0.0004 = 1.8438.
TEST_F(CalibrationLossTest, KnownOverconfidentCaseMatchesHandDerivedValue) {
    CalibrationLoss loss(&backend);
    const float value = loss.forward(probs_row({0.95f, 0.03f, 0.02f}), target(1.0f));
    EXPECT_NEAR(value, 1.8438f, 1e-4f);
}

// Perfect prediction: BS = 0 exactly.
TEST_F(CalibrationLossTest, PerfectPredictionGivesZero) {
    CalibrationLoss loss(&backend);
    const float value = loss.forward(probs_row({1.0f, 0.0f, 0.0f}), target(0.0f));
    EXPECT_NEAR(value, 0.0f, 1e-6f);
}

// Maximally uncertain (uniform) prediction over 3 classes, any target:
// BS = (1/3-1)^2 + 2*(1/3-0)^2 = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667.
TEST_F(CalibrationLossTest, UniformPredictionMatchesHandDerivedValue) {
    CalibrationLoss loss(&backend);
    const float value = loss.forward(probs_row({1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}), target(0.0f));
    EXPECT_NEAR(value, 0.6667f, 1e-3f);
}

TEST_F(CalibrationLossTest, BackwardMatchesHandDerivedGradient) {
    // probs=[0.7,0.2,0.1], target=0, N=1: grad = 2*(p-y)/1 = [2*(0.7-1), 2*(0.2-0), 2*(0.1-0)]
    //     = [-0.6, 0.4, 0.2].
    CalibrationLoss loss(&backend);
    (void)loss.forward(probs_row({0.7f, 0.2f, 0.1f}), target(0.0f));
    Tensor grad = loss.backward();
    ASSERT_EQ(grad.rank(), 2);
    EXPECT_NEAR(grad.data()[0], -0.6f, 1e-5f);
    EXPECT_NEAR(grad.data()[1], 0.4f, 1e-5f);
    EXPECT_NEAR(grad.data()[2], 0.2f, 1e-5f);
}

TEST_F(CalibrationLossTest, BackwardAveragesOverBatch) {
    // Two examples, batched: row0 probs=[1,0,0] target=0 (perfect, zero grad contribution);
    // row1 probs=[0.7,0.2,0.1] target=0 (grad [-0.6,0.4,0.2] before batch averaging).
    // Batch mean divides by N=2: row1's grad becomes [-0.3, 0.2, 0.1].
    CalibrationLoss loss(&backend);
    Tensor probs(Shape({2, 3}), &backend, {1.0f, 0.0f, 0.0f, 0.7f, 0.2f, 0.1f});
    Tensor targets(Shape({2, 1}), &backend, {0.0f, 0.0f});
    (void)loss.forward(probs, targets);
    Tensor grad = loss.backward();
    EXPECT_NEAR(grad.data()[0], 0.0f, 1e-6f);
    EXPECT_NEAR(grad.data()[1], 0.0f, 1e-6f);
    EXPECT_NEAR(grad.data()[2], 0.0f, 1e-6f);
    EXPECT_NEAR(grad.data()[3], -0.3f, 1e-5f);
    EXPECT_NEAR(grad.data()[4], 0.2f, 1e-5f);
    EXPECT_NEAR(grad.data()[5], 0.1f, 1e-5f);
}

TEST_F(CalibrationLossTest, ForwardThrowsOnWrongProbsRank) {
    CalibrationLoss loss(&backend);
    Tensor bad_probs(Shape({3}), &backend, {0.7f, 0.2f, 0.1f});
    EXPECT_THROW({ (void)loss.forward(bad_probs, target(0.0f)); }, std::invalid_argument);
}

TEST_F(CalibrationLossTest, ForwardThrowsOnBatchSizeMismatch) {
    CalibrationLoss loss(&backend);
    Tensor mismatched_target(Shape({2, 1}), &backend, {0.0f, 1.0f});
    EXPECT_THROW({ (void)loss.forward(probs_row({0.7f, 0.2f, 0.1f}), mismatched_target); }, std::invalid_argument);
}

TEST_F(CalibrationLossTest, ForwardThrowsOnNonIntegerTargetClass) {
    CalibrationLoss loss(&backend);
    EXPECT_THROW({ (void)loss.forward(probs_row({0.7f, 0.2f, 0.1f}), target(1.5f)); }, std::invalid_argument);
}

TEST_F(CalibrationLossTest, ForwardThrowsOnOutOfRangeTargetClass) {
    CalibrationLoss loss(&backend);
    EXPECT_THROW({ (void)loss.forward(probs_row({0.7f, 0.2f, 0.1f}), target(3.0f)); }, std::invalid_argument);
}

TEST_F(CalibrationLossTest, BackwardThrowsBeforeForward) {
    CalibrationLoss loss(&backend);
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
