#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_conservation.hpp"

// Promotes lrp_conservation_test.cpp's test-only {sum_in, sum_out} pattern into a real
// production API (plan's Phase A "blocking dependency" prerequisite -- see
// plans/okay-we-have-now-buzzing-moth.md). The ExplanationScoreCard's conservation-delta
// tile calls this; it must not reimplement the math itself.
namespace pulsatrix {
namespace {

class ConservationResultTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(ConservationResultTest, SumsBothTensorsIndependently) {
    Tensor relevance_in(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor relevance_out(Shape({2}), &backend, {2.5f, 3.5f});

    ConservationResult result = ComputeConservation(relevance_in, relevance_out);

    EXPECT_FLOAT_EQ(result.relevance_in_sum, 6.0f);
    EXPECT_FLOAT_EQ(result.relevance_out_sum, 6.0f);
}

TEST_F(ConservationResultTest, DeltaIsAbsoluteDifferenceOfSums) {
    Tensor relevance_in(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor relevance_out(Shape({2}), &backend, {4.0f, 1.0f});

    ConservationResult result = ComputeConservation(relevance_in, relevance_out);

    EXPECT_FLOAT_EQ(result.delta(), 3.0f);
}

TEST_F(ConservationResultTest, DeltaIsNonNegativeRegardlessOfSign) {
    Tensor relevance_in(Shape({1}), &backend, {10.0f});
    Tensor relevance_out(Shape({1}), &backend, {1.0f});

    ConservationResult result = ComputeConservation(relevance_in, relevance_out);

    EXPECT_FLOAT_EQ(result.delta(), 9.0f);
}

TEST_F(ConservationResultTest, PerfectConservationHasZeroDelta) {
    Tensor relevance_in(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor relevance_out(Shape({1}), &backend, {6.0f});

    ConservationResult result = ComputeConservation(relevance_in, relevance_out);

    EXPECT_NEAR(result.delta(), 0.0f, 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
