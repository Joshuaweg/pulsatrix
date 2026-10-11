// INT-1: nearest neighbors, mutual nearest neighbors, mutual-kNN alignment and hubness, checked
// against PyTorch and Huh et al.'s mutual_knn (tools/generate_neighbors_reference_values.py).
#include "pulsatrix/nearest_neighbors.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class NeighborsTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The reference script's inputs: offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i²).
    Tensor Make(double seed, int64_t n, int64_t p, double offset = 0.0) {
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t a = 0; a < p; ++a) {
                const double di = static_cast<double>(i);
                const double da = static_cast<double>(a);
                v.push_back(static_cast<float>(offset + std::sin(seed + 0.37 * di + 1.13 * da + 0.05 * di * da + 0.71 * di * di)));
            }
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    Tensor Gaussian(int64_t n, int64_t p, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(n * p));
        for (float& x : v) {
            x = normal(rng);
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    Tensor Points(int64_t n, int64_t p, std::vector<float> v) { return Tensor(Shape({n, p}), &backend, std::move(v)); }
};

TEST_F(NeighborsTest, MatchPyTorchForEachMetric) {
    const Tensor x = Make(0.0, 12, 5, /*offset=*/0.8);
    NeighborOptions o;
    o.k = 3;
    EXPECT_EQ(NearestNeighbors(x, o).indices,
              (std::vector<int64_t>{7, 4, 10, 5, 3, 4, 6, 11, 8, 5, 1, 4, 7, 1, 0, 3, 1, 4,
                                    11, 2, 3, 10, 4, 0, 9, 0, 10, 8, 0, 10, 7, 0, 4, 6, 2, 3}));
    o.center = true;
    EXPECT_EQ(NearestNeighbors(x, o).indices,
              (std::vector<int64_t>{7, 4, 10, 5, 3, 4, 6, 11, 9, 5, 1, 4, 7, 1, 5, 3, 1, 4,
                                    11, 2, 3, 10, 4, 0, 9, 0, 10, 8, 0, 2, 7, 4, 0, 6, 2, 3}));
    o.metric = NeighborMetric::Euclidean;
    for (bool center : {false, true}) {  // centering doesn't change distances
        o.center = center;
        const Neighbors n = NearestNeighbors(x, o);
        EXPECT_EQ(n.indices, (std::vector<int64_t>{7, 4, 10, 5, 3, 4, 6, 11, 9, 5, 1, 4, 7, 1, 5, 3, 1, 4,
                                                   11, 2, 3, 10, 4, 0, 9, 0, 10, 8, 0, 10, 7, 4, 0, 6, 2, 3}));
        const double nearest[] = {0.80510997014127528, 0.50444710941153881, 1.0544169032081041, 0.43211481323378681,
                                  0.65396660590047562, 0.43211481323378681, 0.57644350312443149, 0.40223823472035036,
                                  0.092123216605531258, 0.092123216605531258, 0.40223823472035036, 0.57644350312443149};
        for (int64_t q = 0; q < 12; ++q) {
            EXPECT_NEAR(n.score(q, 0), nearest[q], 1e-12);
        }
    }
}

TEST_F(NeighborsTest, MutualKnnAlignmentMatchesHuhEtAl) {
    const Tensor a = Make(0.5, 40, 16);
    const Tensor b = Make(2.5, 40, 24);
    NeighborOptions o;
    const std::pair<int64_t, double> expected[] = {
        {1, 0.45000000000000001}, {5, 0.56500000000000006}, {10, 0.50249999999999995}};
    for (const auto& [k, value] : expected) {
        o.k = k;
        EXPECT_NEAR(MutualKnnAlignment(a, b, o), value, 1e-12) << "k = " << k;
    }
}

TEST_F(NeighborsTest, ScoresAreSortedAndNeverTheQueryItself) {
    const Tensor x = Gaussian(30, 8, 1);
    NeighborOptions o;
    o.k = 29;
    for (NeighborMetric m : {NeighborMetric::Cosine, NeighborMetric::Euclidean}) {
        o.metric = m;
        const Neighbors n = NearestNeighbors(x, o);
        ASSERT_EQ(n.rows, 30);
        ASSERT_EQ(n.k, 29);
        for (int64_t q = 0; q < 30; ++q) {
            for (int64_t r = 0; r < 29; ++r) {
                EXPECT_NE(n.index(q, r), q);
                if (r > 0) {
                    if (m == NeighborMetric::Cosine) {
                        EXPECT_GE(n.score(q, r - 1), n.score(q, r));
                    } else {
                        EXPECT_LE(n.score(q, r - 1), n.score(q, r));
                    }
                }
            }
        }
    }
}

TEST_F(NeighborsTest, TiesKeepTheLowerIndexFirst) {
    // Points 1, 2 and 3 are all at distance 1 from point 0.
    const Tensor x = Points(4, 2, {0, 0, 1, 0, 0, 1, -1, 0});
    NeighborOptions o;
    o.k = 3;
    o.metric = NeighborMetric::Euclidean;
    const Neighbors n = NearestNeighbors(x, o);
    EXPECT_EQ(n.index(0, 0), 1);
    EXPECT_EQ(n.index(0, 1), 2);
    EXPECT_EQ(n.index(0, 2), 3);
    EXPECT_DOUBLE_EQ(n.score(0, 2), 1.0);
}

TEST_F(NeighborsTest, CenteringChangesCosineNeighborsButNotDistances) {
    // A common offset of 3 dominates every vector, so uncentered cosines are all near 1 and the
    // neighbors follow the offset's interaction with each row; centering removes it.
    const Tensor x = Make(0.0, 12, 5, /*offset=*/3.0);
    NeighborOptions o;
    o.k = 3;
    const Neighbors raw = NearestNeighbors(x, o);
    o.center = true;
    const Neighbors centered = NearestNeighbors(x, o);
    EXPECT_NE(raw.indices, centered.indices);
    EXPECT_GT(raw.score(0, 2), 0.9);
    EXPECT_LT(centered.score(0, 2), raw.score(0, 2));

    o.metric = NeighborMetric::Euclidean;
    const Neighbors d_centered = NearestNeighbors(x, o);
    o.center = false;
    const Neighbors d_raw = NearestNeighbors(x, o);
    EXPECT_EQ(d_raw.indices, d_centered.indices);
    for (std::size_t i = 0; i < d_raw.scores.size(); ++i) {
        EXPECT_NEAR(d_raw.scores[i], d_centered.scores[i], 1e-12);
    }
}

TEST_F(NeighborsTest, QueriesAgainstAnotherSet) {
    const Tensor points = Points(3, 2, {0, 0, 10, 0, 0, 10});
    const Tensor queries = Points(2, 2, {9, 1, 1, 8});
    NeighborOptions o;
    o.k = 1;
    o.metric = NeighborMetric::Euclidean;
    const Neighbors n = NearestNeighbors(queries, points, o);
    EXPECT_EQ(n.indices, (std::vector<int64_t>{1, 2}));
    EXPECT_NEAR(n.score(0, 0), std::sqrt(2.0), 1e-12);
    o.k = 3;  // every point may be a neighbor: none is the query itself
    EXPECT_EQ(NearestNeighbors(queries, points, o).index(0, 2), 2);
}

TEST_F(NeighborsTest, AZeroVectorHasCosineZero) {
    const Tensor x = Points(3, 2, {0, 0, 1, 0, -1, 0});
    NeighborOptions o;
    o.k = 2;
    const Neighbors n = NearestNeighbors(x, o);
    EXPECT_EQ(n.index(0, 0), 1);  // both at 0: the lower index first
    EXPECT_DOUBLE_EQ(n.score(0, 0), 0.0);
    EXPECT_EQ(n.index(1, 0), 0);  // 0 beats -1
    EXPECT_DOUBLE_EQ(n.score(1, 1), -1.0);
}

TEST_F(NeighborsTest, FindsMutualNearestNeighborsAcrossTwoSets) {
    // a[0] and b[1] choose each other; a[1]'s nearest is b[1] too, but b[1] prefers a[0].
    const Tensor a = Points(2, 2, {0, 0, 3, 0});
    const Tensor b = Points(3, 2, {-5, 0, 1, 0, 9, 0});
    NeighborOptions o;
    o.k = 1;
    o.metric = NeighborMetric::Euclidean;
    EXPECT_EQ(MutualNearestNeighbors(a, b, o), (std::vector<std::pair<int64_t, int64_t>>{{0, 1}}));
    o.k = 2;
    EXPECT_EQ(MutualNearestNeighbors(a, b, o),
              (std::vector<std::pair<int64_t, int64_t>>{{0, 0}, {0, 1}, {1, 1}, {1, 2}}));
}

TEST_F(NeighborsTest, AlignmentIsOneForTheSameGeometryAndChanceForUnrelatedOnes) {
    const Tensor a = Gaussian(200, 16, 2);
    std::vector<float> scaled = a.to_host_vector();
    for (float& v : scaled) {
        v *= 3.0f;
    }
    NeighborOptions o;
    o.k = 10;
    EXPECT_DOUBLE_EQ(MutualKnnAlignment(a, Tensor(a.shape(), &backend, scaled), o), 1.0);
    // Unrelated: about k / (n - 1) = 0.05 by chance.
    const double chance = MutualKnnAlignment(a, Gaussian(200, 32, 3), o);
    EXPECT_NEAR(chance, 10.0 / 199.0, 0.02);
}

TEST_F(NeighborsTest, HubnessGrowsWithDimension) {
    // Radovanović et al.: with the same number of points, k-occurrence skews as the dimension
    // grows, so a few points become everyone's neighbor.
    NeighborOptions o;
    o.k = 5;
    o.metric = NeighborMetric::Euclidean;
    const Hubness low = MeasureHubness(NearestNeighbors(Gaussian(300, 3, 4), o), 300);
    const Hubness high = MeasureHubness(NearestNeighbors(Gaussian(300, 300, 5), o), 300);
    EXPECT_LT(low.skewness, 1.0);
    EXPECT_GT(high.skewness, low.skewness + 1.0);
    EXPECT_GT(high.never_a_neighbor, low.never_a_neighbor);
    EXPECT_GT(high.max_occurrence, low.max_occurrence);

    int64_t total = 0;
    for (int64_t c : high.occurrences) {
        total += c;
    }
    EXPECT_EQ(total, 300 * 5);  // every list has k entries
}

TEST_F(NeighborsTest, HubnessOfAPlantedHub) {
    Neighbors n;
    n.rows = 4;
    n.k = 1;
    n.indices = {3, 3, 3, 0};
    const Hubness h = MeasureHubness(n, 4);
    EXPECT_EQ(h.occurrences, (std::vector<int64_t>{1, 0, 0, 3}));
    EXPECT_EQ(h.max_occurrence, 3);
    EXPECT_DOUBLE_EQ(h.max_fraction, 0.75);
    EXPECT_DOUBLE_EQ(h.never_a_neighbor, 0.5);
    EXPECT_GT(h.skewness, 0.0);
    EXPECT_THROW((void)MeasureHubness(n, 3), std::invalid_argument);
    EXPECT_THROW((void)MeasureHubness(n, 0), std::invalid_argument);
}

TEST_F(NeighborsTest, RefusesBadInputs) {
    const Tensor x = Gaussian(5, 3, 6);
    NeighborOptions o;
    o.k = 5;  // only 4 other points
    EXPECT_THROW((void)NearestNeighbors(x, o), std::invalid_argument);
    o.k = 0;
    EXPECT_THROW((void)NearestNeighbors(x, o), std::invalid_argument);
    o.k = 2;
    EXPECT_THROW((void)NearestNeighbors(Tensor(Shape({5}), &backend, std::vector<float>(5, 1.0f)), o),
                 std::invalid_argument);
    std::vector<float> bad = x.to_host_vector();
    bad[4] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)NearestNeighbors(Tensor(x.shape(), &backend, bad), o), std::invalid_argument);
    EXPECT_THROW((void)NearestNeighbors(x, Gaussian(5, 4, 7), o), std::invalid_argument);
    EXPECT_THROW((void)MutualNearestNeighbors(x, Gaussian(5, 4, 7), o), std::invalid_argument);
    EXPECT_THROW((void)MutualKnnAlignment(x, Gaussian(6, 3, 8), o), std::invalid_argument);
    const Neighbors n = NearestNeighbors(x, o);
    EXPECT_THROW((void)n.index(5, 0), std::out_of_range);
    EXPECT_THROW((void)n.score(0, 2), std::out_of_range);
}

}  // namespace
}  // namespace pulsatrix
