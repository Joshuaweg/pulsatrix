// TDA-2: pairwise distances via GEMM, with a certified float32 error bound.
#include "pulsatrix/tda/distance_matrix.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class DistanceMatrixTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // n points with coordinates offset + scale * N(0, 1), from a fixed seed.
    Tensor Gaussian(int64_t n, int64_t d, uint32_t seed, float scale = 1.0f, float offset = 0.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(n * d));
        for (float& x : v) x = offset + scale * normal(rng);
        return Tensor(Shape({n, d}), &backend, v);
    }

    // The exact distances, by long double sums of float differences.
    static std::vector<long double> Reference(const Tensor& points) {
        const int64_t n = points.shape().dim(0);
        const int64_t d = points.numel() / n;
        const std::vector<float> x = points.to_host_vector();
        std::vector<long double> out;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                long double s = 0;
                for (int64_t k = 0; k < d; ++k) {
                    const long double diff = static_cast<long double>(x[static_cast<std::size_t>(i * d + k)]) -
                                             x[static_cast<std::size_t>(j * d + k)];
                    s += diff * diff;
                }
                out.push_back(std::sqrt(s));
            }
        }
        return out;
    }

    // Every stored distance is within the matrix's bound of the reference.
    static void ExpectWithinBound(const DistanceMatrix& m, const std::vector<long double>& exact) {
        ASSERT_EQ(m.values.size(), exact.size());
        long double worst = 0;
        for (std::size_t k = 0; k < exact.size(); ++k) {
            worst = std::max(worst, std::fabs(static_cast<long double>(m.values[k]) - exact[k]));
        }
        EXPECT_LE(static_cast<double>(worst), m.error_bound) << "the bound doesn't cover the error";
    }
};

TEST_F(DistanceMatrixTest, ATriangleGivesItsSidesInPdistOrder) {
    const Tensor points(Shape({3, 2}), &backend, {0, 0, 3, 0, 0, 4});
    for (DistanceMethod method : {DistanceMethod::Gemm, DistanceMethod::Exact}) {
        const DistanceMatrix m = PairwiseDistances(points, {method});
        ASSERT_EQ(m.size, 3);
        ASSERT_EQ(m.values.size(), 3u);
        EXPECT_NEAR(m.values[0], 3.0f, m.error_bound);  // (0, 1)
        EXPECT_NEAR(m.values[1], 4.0f, m.error_bound);  // (0, 2)
        EXPECT_NEAR(m.values[2], 5.0f, m.error_bound);  // (1, 2)
        EXPECT_FLOAT_EQ(m.max_distance, 5.0f);
        EXPECT_LT(m.error_bound, 1e-5);
    }
}

TEST_F(DistanceMatrixTest, IndexAndAtFollowPdistOrderInEitherDirection) {
    const DistanceMatrix m = PairwiseDistances(Gaussian(5, 3, 1));
    std::size_t k = 0;
    for (int64_t i = 0; i < 5; ++i) {
        for (int64_t j = i + 1; j < 5; ++j, ++k) {
            EXPECT_EQ(m.index(i, j), k);
            EXPECT_EQ(m.index(j, i), k);
            EXPECT_EQ(m.at(j, i), m.values[k]);
        }
        EXPECT_EQ(m.at(i, i), 0.0f);
    }
    EXPECT_THROW((void)m.index(2, 2), std::out_of_range);
    EXPECT_THROW((void)m.at(0, 5), std::out_of_range);
    EXPECT_THROW((void)m.at(-1, 0), std::out_of_range);
}

TEST_F(DistanceMatrixTest, GemmAndExactAgreeWithinTheBoundOnRandomPoints) {
    const Tensor points = Gaussian(150, 64, 2);
    const std::vector<long double> exact = Reference(points);
    const DistanceMatrix gemm = PairwiseDistances(points);
    const DistanceMatrix direct = PairwiseDistances(points, {DistanceMethod::Exact});
    ExpectWithinBound(gemm, exact);
    ExpectWithinBound(direct, exact);
    // Well-separated points: the GEMM's bound is small next to the distances (about 11 here).
    EXPECT_LT(gemm.error_bound, 1e-3 * gemm.max_distance);
    // Exact's bound is the float32 rounding of the stored distances, half a unit in the last place.
    EXPECT_LE(direct.error_bound, 0x1p-24 * direct.max_distance * 1.01);
    EXPECT_LE(direct.error_bound, gemm.error_bound);
}

TEST_F(DistanceMatrixTest, ALargeSharedOffsetCancelsButStaysWithinTheBound) {
    // Points spread by 0.01 per coordinate, 1000 from the origin: |x|^2 is about 1e8 times
    // |x - y|^2, so the GEMM's float32 cancellation leaves few correct digits. The bound says so,
    // and covers it.
    const Tensor points = Gaussian(60, 32, 3, /*scale=*/0.01f, /*offset=*/1000.0f / std::sqrt(32.0f));
    const std::vector<long double> exact = Reference(points);
    const DistanceMatrix raw = PairwiseDistances(points);
    ExpectWithinBound(raw, exact);
    EXPECT_GT(raw.error_bound, 0.1 * raw.max_distance) << "the bound should admit the cancellation";

    // Centering removes the offset before the GEMM: Euclidean distances are unchanged, and the
    // bound shrinks with the norms.
    DistanceOptions centered;
    centered.center = true;
    const DistanceMatrix c = PairwiseDistances(points, centered);
    ExpectWithinBound(c, exact);
    EXPECT_LT(c.error_bound, 1e-3 * c.max_distance);
    EXPECT_LT(c.error_bound * 100, raw.error_bound);
}

TEST_F(DistanceMatrixTest, IdenticalPointsGetZeroWithinTheBound) {
    std::vector<float> v = {0.3f, -1.7f, 2.25f, 0.3f, -1.7f, 2.25f, 1.0f, 1.0f, 1.0f};
    const Tensor points(Shape({3, 3}), &backend, v);
    for (DistanceMethod method : {DistanceMethod::Gemm, DistanceMethod::Exact}) {
        const DistanceMatrix m = PairwiseDistances(points, {method});
        EXPECT_GE(m.values[0], 0.0f);
        EXPECT_LE(m.values[0], m.error_bound);
    }
    EXPECT_EQ(PairwiseDistances(points, {DistanceMethod::Exact}).values[0], 0.0f);
}

TEST_F(DistanceMatrixTest, BlockSizeDoesNotChangeTheResultBeyondTheBound) {
    const Tensor points = Gaussian(37, 20, 4);
    const DistanceMatrix whole = PairwiseDistances(points);
    for (int64_t rows : {1, 5, 36, 37, 100}) {
        DistanceOptions o;
        o.block_rows = rows;
        const DistanceMatrix m = PairwiseDistances(points, o);
        ASSERT_EQ(m.values.size(), whole.values.size());
        for (std::size_t k = 0; k < m.values.size(); ++k) {
            EXPECT_NEAR(m.values[k], whole.values[k], m.error_bound + whole.error_bound) << "rows " << rows;
        }
    }
}

TEST_F(DistanceMatrixTest, NormalizeGivesTheChordOfTheAngle) {
    const Tensor points = Gaussian(40, 16, 5, 1.0f, 0.5f);
    DistanceOptions o;
    o.normalize = true;
    const DistanceMatrix m = PairwiseDistances(points, o);
    const std::vector<float> x = points.to_host_vector();
    std::size_t k = 0;
    for (int64_t i = 0; i < 40; ++i) {
        for (int64_t j = i + 1; j < 40; ++j, ++k) {
            double dot = 0, ni = 0, nj = 0;
            for (int64_t a = 0; a < 16; ++a) {
                const double xi = x[static_cast<std::size_t>(i * 16 + a)];
                const double xj = x[static_cast<std::size_t>(j * 16 + a)];
                dot += xi * xj;
                ni += xi * xi;
                nj += xj * xj;
            }
            const double chord = std::sqrt(std::max(0.0, 2 - 2 * dot / std::sqrt(ni * nj)));
            EXPECT_NEAR(m.values[k], chord, m.error_bound + 1e-12);
        }
    }
    EXPECT_LE(m.max_distance, 2.0f + static_cast<float>(m.error_bound));
}

TEST_F(DistanceMatrixTest, ACloudsDistancesMatchItsTensors) {
    PointCloud cloud({PointPooling::PerPosition});
    cloud.add(Gaussian(2, 12, 6));  // treated as (batch 2, 12) -> rank 2: one position each
    cloud.add(Gaussian(3, 12, 7));
    const DistanceMatrix from_cloud = PairwiseDistances(cloud, &backend);
    const DistanceMatrix from_tensor = PairwiseDistances(cloud.ToTensor(&backend));
    EXPECT_EQ(from_cloud.size, 5);
    EXPECT_EQ(from_cloud.values, from_tensor.values);
    EXPECT_EQ(from_cloud.error_bound, from_tensor.error_bound);
}

TEST_F(DistanceMatrixTest, BadInputsThrow) {
    EXPECT_THROW((void)PairwiseDistances(Tensor(Shape({4}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)PairwiseDistances(Gaussian(1, 3, 8)), std::invalid_argument);
    EXPECT_THROW((void)PairwiseDistances(Tensor(Shape({2, 2}), &backend, {1, std::numeric_limits<float>::quiet_NaN(), 0, 0})),
                 std::invalid_argument);
    DistanceOptions negative;
    negative.block_rows = -1;
    EXPECT_THROW((void)PairwiseDistances(Gaussian(3, 3, 9), negative), std::invalid_argument);
    DistanceOptions normalize;
    normalize.normalize = true;
    EXPECT_THROW((void)PairwiseDistances(Tensor(Shape({2, 2}), &backend, {0, 0, 1, 1}), normalize),
                 std::invalid_argument);
    EXPECT_THROW((void)PairwiseDistances(PointCloud(), &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
