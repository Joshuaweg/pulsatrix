/** @file sinusoidal_timestep_embedding_test.cpp
 *  @brief Unit tests for the fixed sinusoidal diffusion-timestep encoding.
 *
 *  No finite-difference check here either: the embedding is a non-learned, deterministic
 *  function of an integer with no backward pass (mission_diffusion_module.md). Correctness is
 *  established by direct comparison against the published formula evaluated independently.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "exai/cpu_backend.hpp"
#include "exai/sinusoidal_timestep_embedding.hpp"

namespace exai {
namespace {

class SinusoidalTimestepEmbeddingTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(SinusoidalTimestepEmbeddingTest, ShapeIsOneByEmbeddingDim) {
    Tensor emb = SinusoidalTimestepEmbedding(7, 8, &backend);

    EXPECT_EQ(emb.shape().rank(), 2);
    EXPECT_EQ(emb.shape().dim(0), 1);
    EXPECT_EQ(emb.shape().dim(1), 8);
    EXPECT_EQ(emb.numel(), 8);
}

TEST_F(SinusoidalTimestepEmbeddingTest, MatchesHandComputedValues) {
    // embedding_dim = 4 -> two pairs. Pair 0: exponent 0, so frequency = t / 1 = 5.
    // Pair 1: exponent 2/4 = 0.5, so frequency = 5 / sqrt(10000) = 5 / 100 = 0.05.
    Tensor emb = SinusoidalTimestepEmbedding(5, 4, &backend);

    ASSERT_EQ(emb.numel(), 4);
    EXPECT_NEAR(emb.data()[0], std::sin(5.0f), 1e-5f);
    EXPECT_NEAR(emb.data()[1], std::cos(5.0f), 1e-5f);
    EXPECT_NEAR(emb.data()[2], std::sin(0.05f), 1e-5f);
    EXPECT_NEAR(emb.data()[3], std::cos(0.05f), 1e-5f);
}

TEST_F(SinusoidalTimestepEmbeddingTest, TimestepZeroGivesAlternatingZerosAndOnes) {
    // sin(0) = 0, cos(0) = 1 for every pair, regardless of frequency -- the one case whose
    // exact value is known without evaluating any transcendental at a nontrivial argument.
    Tensor emb = SinusoidalTimestepEmbedding(0, 6, &backend);

    ASSERT_EQ(emb.numel(), 6);
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(emb.data()[2 * i], 0.0f, 1e-6f) << "sin slot of pair " << i;
        EXPECT_NEAR(emb.data()[2 * i + 1], 1.0f, 1e-6f) << "cos slot of pair " << i;
    }
}

TEST_F(SinusoidalTimestepEmbeddingTest, EveryPairSatisfiesTheTrigIdentity) {
    // sin^2 + cos^2 == 1 per pair, for any t and any frequency. Structural: it fails
    // immediately if the two slots of a pair were ever computed at different frequencies
    // (e.g. an off-by-one in the exponent index).
    Tensor emb = SinusoidalTimestepEmbedding(137, 16, &backend);

    for (int64_t i = 0; i < 8; ++i) {
        const float s = emb.data()[2 * i];
        const float c = emb.data()[2 * i + 1];
        EXPECT_NEAR(s * s + c * c, 1.0f, 1e-5f) << "pair " << i;
    }
}

TEST_F(SinusoidalTimestepEmbeddingTest, HigherPairsHaveLongerWavelengths) {
    // The geometric frequency schedule means the last pair's angle is base times smaller
    // than the first's -- at a large t the highest pair is still near its t = 0 value while
    // the lowest has wrapped many times. Pin the high pair's angle is tiny.
    Tensor emb = SinusoidalTimestepEmbedding(100, 8, &backend);

    // Pair 3: exponent 6/8 = 0.75, frequency = 100 / 10000^0.75 = 100 / 1000 = 0.1.
    EXPECT_NEAR(emb.data()[6], std::sin(0.1f), 1e-5f);
    EXPECT_NEAR(emb.data()[7], std::cos(0.1f), 1e-5f);
    // Pair 0: frequency = 100 exactly, which has wrapped ~16 times.
    EXPECT_NEAR(emb.data()[0], std::sin(100.0f), 1e-4f);
}

TEST_F(SinusoidalTimestepEmbeddingTest, DistinctTimestepsProduceDistinctEmbeddings) {
    Tensor a = SinusoidalTimestepEmbedding(3, 8, &backend);
    Tensor b = SinusoidalTimestepEmbedding(4, 8, &backend);

    float max_difference = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        max_difference = std::fmax(max_difference, std::fabs(a.data()[i] - b.data()[i]));
    }
    EXPECT_GT(max_difference, 1e-2f);
}

TEST_F(SinusoidalTimestepEmbeddingTest, IsDeterministicAcrossCalls) {
    Tensor first = SinusoidalTimestepEmbedding(42, 8, &backend);
    Tensor second = SinusoidalTimestepEmbedding(42, 8, &backend);

    for (int64_t i = 0; i < first.numel(); ++i) {
        EXPECT_FLOAT_EQ(first.data()[i], second.data()[i]) << "index " << i;
    }
}

TEST_F(SinusoidalTimestepEmbeddingTest, RespectsACustomBase) {
    // base = 100, dim = 4 -> pair 1 exponent 0.5, frequency = 10 / sqrt(100) = 1.0.
    Tensor emb = SinusoidalTimestepEmbedding(10, 4, &backend, 100.0f);

    EXPECT_NEAR(emb.data()[2], std::sin(1.0f), 1e-5f);
    EXPECT_NEAR(emb.data()[3], std::cos(1.0f), 1e-5f);
}

TEST_F(SinusoidalTimestepEmbeddingTest, ThrowsOnNonPositiveEmbeddingDim) {
    EXPECT_THROW({ (void)SinusoidalTimestepEmbedding(1, 0, &backend); }, std::invalid_argument);
    EXPECT_THROW({ (void)SinusoidalTimestepEmbedding(1, -4, &backend); }, std::invalid_argument);
}

TEST_F(SinusoidalTimestepEmbeddingTest, ThrowsOnOddEmbeddingDim) {
    EXPECT_THROW({ (void)SinusoidalTimestepEmbedding(1, 5, &backend); }, std::invalid_argument);
}

}  // namespace
}  // namespace exai
