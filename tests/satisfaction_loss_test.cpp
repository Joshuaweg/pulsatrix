#include "pulsatrix/satisfaction_loss.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class SatisfactionLossTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

using SatisfactionLossDeathTest = SatisfactionLossTest;

// ---------------------------------------------------------------------------
// Forward correctness
// ---------------------------------------------------------------------------

TEST_F(SatisfactionLossTest, ForwardComputesOneMinusArithmeticMeanForPEqualsOne) {
    SatisfactionLoss loss(&backend, 1.0f);
    Tensor truth_values(Shape({4}), &backend, {0.2f, 0.4f, 0.6f, 0.8f});
    const float value = loss.forward(truth_values);
    EXPECT_NEAR(value, 1.0f - 0.5f, 1e-5f);
}

TEST_F(SatisfactionLossTest, DefaultPMatchesAggregatorDefault) {
    SatisfactionLoss loss(&backend);  // default p == 2.0, RMS
    Tensor truth_values(Shape({2}), &backend, {0.6f, 0.8f});
    const float value = loss.forward(truth_values);
    const float rms = std::sqrt((0.36f + 0.64f) / 2.0f);
    EXPECT_NEAR(value, 1.0f - rms, 1e-5f);
}

TEST_F(SatisfactionLossTest, PerfectSatisfactionAcrossAllGroundingsGivesZeroLoss) {
    SatisfactionLoss loss(&backend, 2.0f);
    Tensor truth_values(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_NEAR(loss.forward(truth_values), 0.0f, 1e-5f);
}

TEST_F(SatisfactionLossTest, ZeroSatisfactionAcrossAllGroundingsGivesLossOfOne) {
    SatisfactionLoss loss(&backend, 2.0f);
    Tensor truth_values(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});
    EXPECT_NEAR(loss.forward(truth_values), 1.0f, 1e-5f);
}

TEST_F(SatisfactionLossTest, SingleGroundingIsHandledCorrectly) {
    SatisfactionLoss loss(&backend, 2.0f);
    Tensor truth_values(Shape({1}), &backend, {0.7f});
    EXPECT_NEAR(loss.forward(truth_values), 1.0f - 0.7f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Backward -- composed through AggregatorModule::backward(), verified against central
// finite differences on forward() directly (an integration-level check per this project's
// TDD discipline for a new seam, not just a unit-level formula restatement).
// ---------------------------------------------------------------------------

TEST_F(SatisfactionLossTest, BackwardMatchesCentralFiniteDifferences) {
    SatisfactionLoss loss(&backend, 2.0f);
    std::vector<float> x_values{0.3f, 0.6f, 0.9f, 0.2f};
    Tensor truth_values(Shape({4}), &backend, x_values);
    (void)loss.forward(truth_values);
    Tensor grad = loss.backward();

    auto loss_of = [&](const std::vector<float>& xv) {
        SatisfactionLoss probe(&backend, 2.0f);
        Tensor xp(Shape({4}), &backend, xv);
        return probe.forward(xp);
    };

    const float h = 1e-3f;
    for (size_t i = 0; i < x_values.size(); ++i) {
        std::vector<float> xp = x_values;
        std::vector<float> xm = x_values;
        xp[i] += h;
        xm[i] -= h;
        const float numeric = (loss_of(xp) - loss_of(xm)) / (2.0f * h);
        EXPECT_NEAR(grad.data()[static_cast<int64_t>(i)], numeric, 5e-3f) << "x[" << i << "]";
    }
}

TEST_F(SatisfactionLossTest, BackwardIsUniformNegativeOneOverNForPEqualsOne) {
    // For p == 1, agg_1 == mean, so d(loss)/dx_i == -1/n uniformly -- hand-derivable
    // without finite differences, the mission's own "closed form" gate.
    SatisfactionLoss loss(&backend, 1.0f);
    Tensor truth_values(Shape({4}), &backend, {0.1f, 0.9f, 0.5f, 0.5f});
    (void)loss.forward(truth_values);
    Tensor grad = loss.backward();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(grad.data()[i], -0.25f, 1e-5f) << "x[" << i << "]";
    }
}

TEST_F(SatisfactionLossTest, BackwardBeforeForwardThrows) {
    SatisfactionLoss loss(&backend);
    EXPECT_THROW({ (void)loss.backward(); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

TEST_F(SatisfactionLossTest, ConstructorRejectsPEqualsZeroDelegatedFromAggregator) {
    EXPECT_THROW({ SatisfactionLoss loss(&backend, 0.0f); }, std::invalid_argument);
}

TEST_F(SatisfactionLossTest, ForwardRejectsRankTwoTruthValues) {
    SatisfactionLoss loss(&backend);
    Tensor truth_values(Shape({2, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});
    EXPECT_THROW({ (void)loss.forward(truth_values); }, std::invalid_argument);
}

TEST_F(SatisfactionLossTest, ForwardRejectsRankZeroTruthValues) {
    SatisfactionLoss loss(&backend);
    Tensor truth_values(Shape({}), &backend, {0.5f});
    EXPECT_THROW({ (void)loss.forward(truth_values); }, std::invalid_argument);
}

// Empty truth_values -- inherited from AggregatorModule's own Module::forward() NVI
// precondition, not re-implemented here.
TEST_F(SatisfactionLossTest, ForwardRejectsEmptyTruthValues) {
    SatisfactionLoss loss(&backend);
    Tensor truth_values(Shape({0}), &backend);
    EXPECT_THROW({ (void)loss.forward(truth_values); }, std::invalid_argument);
}

TEST_F(SatisfactionLossDeathTest, ForwardAbortsOnNonCpuTruthValues) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    SatisfactionLoss loss(&backend);
    Tensor truth_values(Shape({2}), &backend, {0.1f, 0.2f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)loss.forward(truth_values); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
