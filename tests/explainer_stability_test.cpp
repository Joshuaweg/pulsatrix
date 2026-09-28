#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_stability.hpp"

// Promotes stability_test.cpp's test-only ComputeVariance/AllValuesExactlyEqual pattern
// into a real production API (plan's Phase A "blocking dependency" prerequisite -- see
// plans/okay-we-have-now-buzzing-moth.md). The ExplanationScoreCard's stability tile calls
// this for stochastic explainers (LIME/KernelSHAP); deterministic methods should show
// "0.0 (deterministic)" via is_deterministic rather than a numerically-noisy near-zero
// variance.
namespace pulsatrix {
namespace {

class ExplainerStabilityTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(ExplainerStabilityTest, IdenticalRepeatedRunsAreExactlyDeterministic) {
    std::vector<Attribution> runs;
    for (int i = 0; i < 5; ++i) {
        runs.push_back(Attribution{"saliency", Tensor(Shape({3}), &backend, {0.1f, 0.2f, 0.3f}), {}});
    }

    StabilityResult result = ComputeAttributionStability(runs);

    EXPECT_TRUE(result.is_deterministic);
    EXPECT_FLOAT_EQ(result.mean_variance, 0.0f);
}

TEST_F(ExplainerStabilityTest, VaryingRepeatedRunsReportPositiveVarianceAndNotDeterministic) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {1.0f, 5.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {2.0f, 5.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {3.0f, 5.0f}), {}});

    StabilityResult result = ComputeAttributionStability(runs);

    EXPECT_FALSE(result.is_deterministic);
    EXPECT_GT(result.mean_variance, 0.0f);
}

TEST_F(ExplainerStabilityTest, MeanVarianceAveragesAcrossAllElements) {
    // Element 0 varies (values 0,2,4 -> variance ~2.667); element 1 is constant (variance 0).
    // Mean across elements should be (2.667 + 0) / 2.
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {0.0f, 9.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {2.0f, 9.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {4.0f, 9.0f}), {}});

    StabilityResult result = ComputeAttributionStability(runs);

    EXPECT_NEAR(result.mean_variance, 8.0f / 3.0f / 2.0f, 1e-4f);
}

TEST_F(ExplainerStabilityTest, ThrowsOnEmptyRuns) {
    std::vector<Attribution> runs;
    EXPECT_THROW(ComputeAttributionStability(runs), std::invalid_argument);
}

TEST_F(ExplainerStabilityTest, ThrowsOnInconsistentShapesAcrossRuns) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {1.0f, 2.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({3}), &backend, {1.0f, 2.0f, 3.0f}), {}});

    EXPECT_THROW(ComputeAttributionStability(runs), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
