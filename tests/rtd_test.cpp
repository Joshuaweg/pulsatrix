// TDA-11: Representation Topology Divergence (Barannikov et al., ICML 2022), checked against the
// paper's Theorem 2.1, and CompareRepresentations().
#include "pulsatrix/tda/rtd.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"

namespace pulsatrix {
namespace {

constexpr double kPi = 3.14159265358979323846;

class RtdTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Gaussian blobs: point i is centers[i % k] plus sigma N(0, I), in 2 dimensions.
    static std::vector<float> Blobs(const std::vector<std::array<float, 2>>& centers, int64_t n, float sigma, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal;
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            const auto& c = centers[static_cast<std::size_t>(i) % centers.size()];
            v.push_back(c[0] + sigma * normal(rng));
            v.push_back(c[1] + sigma * normal(rng));
        }
        return v;
    }

    static std::vector<std::pair<double, double>> Finite(const PersistenceDiagram& d) {
        std::vector<std::pair<double, double>> out;
        for (const PersistencePair& p : d.pairs) {
            if (!p.essential()) out.emplace_back(p.birth, p.death);
        }
        std::sort(out.begin(), out.end());
        return out;
    }
};

TEST_F(RtdTest, IdenticalRepresentationsHaveAnEmptyCrossBarcode) {
    const std::vector<float> v = Blobs({{0, 0}, {5, 0}}, 40, 1.0f, 1);
    const DistanceMatrix w = PairwiseDistances(Tensor(Shape({40, 2}), &backend, v), {DistanceMethod::Exact});
    EXPECT_TRUE(CrossBarcode(w, w, 1).pairs.empty());
    EXPECT_TRUE(CrossBarcode(w, w, 2).pairs.empty());
}

TEST_F(RtdTest, AgainstAllZeroDistancesTheCrossBarcodeIsTheBarcodeOneDimensionDown) {
    // Theorem 2.1: if every distance in the second representation is 0, R-Cross-Barcode_{k+1}
    // equals the first representation's ordinary barcode in dimension k.
    std::vector<float> circle;
    for (int k = 0; k < 16; ++k) {
        const double a = 2 * kPi * k / 16;
        circle.push_back(static_cast<float>(std::cos(a)));
        circle.push_back(static_cast<float>(std::sin(a)));
    }
    for (const std::vector<float>& v : {circle, Blobs({{0, 0}, {4, 1}}, 18, 0.5f, 2)}) {
        const int64_t n = static_cast<int64_t>(v.size() / 2);
        const DistanceMatrix w = PairwiseDistances(Tensor(Shape({n, 2}), &backend, v), {DistanceMethod::Exact});
        DistanceMatrix zero = w;
        std::fill(zero.values.begin(), zero.values.end(), 0.0f);
        zero.max_distance = 0;
        RipsOptions o;
        o.max_dimension = 1;
        o.threshold = w.max_distance;
        const RipsResult plain = ComputeRipsPersistence(w, o);
        EXPECT_EQ(Finite(CrossBarcode(w, zero, 1)), Finite(plain.diagrams[0])) << n;
        EXPECT_EQ(Finite(CrossBarcode(w, zero, 2)), Finite(plain.diagrams[1])) << n;
    }
}

TEST_F(RtdTest, RotationAndScaleDontCount) {
    const int64_t n = 60;
    const std::vector<float> v = Blobs({{0, 0}, {6, 0}, {0, 6}}, n, 1.0f, 3);
    std::vector<float> turned;
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    for (int64_t i = 0; i < n; ++i) {
        const float x = v[static_cast<std::size_t>(2 * i)], y = v[static_cast<std::size_t>(2 * i + 1)];
        turned.push_back(3 * (c * x - s * y));
        turned.push_back(3 * (s * x + c * y));
    }
    const RtdResult r = Rtd(Tensor(Shape({n, 2}), &backend, v), Tensor(Shape({n, 2}), &backend, turned));
    EXPECT_EQ(r.trials, 1);  // 60 points < sample_size: all of them, once
    EXPECT_LT(r.rtd, 1e-3);
}

TEST_F(RtdTest, ClustersThatMergeElsewhereRaiseRtd) {
    // Three well-separated clusters, against the same points with a little jitter and against the
    // same points with the clusters pulled together into one blob.
    const int64_t n = 90;
    const std::vector<float> a = Blobs({{0, 0}, {10, 0}, {5, 9}}, n, 1.0f, 4);
    std::vector<float> jittered = a, merged = a;
    std::mt19937 rng(5);
    std::normal_distribution<float> normal;
    for (float& x : jittered) x += 0.05f * normal(rng);
    for (int64_t i = 0; i < n; ++i) {
        // Pull each point towards the middle: the clusters overlap.
        merged[static_cast<std::size_t>(2 * i)] = 5 + 0.15f * (a[static_cast<std::size_t>(2 * i)] - 5) + normal(rng);
        merged[static_cast<std::size_t>(2 * i + 1)] = 3 + 0.15f * (a[static_cast<std::size_t>(2 * i + 1)] - 3) + normal(rng);
    }
    const Tensor ta(Shape({n, 2}), &backend, a);
    const double near = Rtd(ta, Tensor(Shape({n, 2}), &backend, jittered)).rtd;
    const double far = Rtd(ta, Tensor(Shape({n, 2}), &backend, merged)).rtd;
    EXPECT_GT(far, 5 * near) << "near " << near << " far " << far;
}

TEST_F(RtdTest, BatchesAreSampledReproducibly) {
    const int64_t n = 300;
    const std::vector<float> a = Blobs({{0, 0}, {8, 0}}, n, 1.0f, 6), b = Blobs({{0, 0}, {3, 0}}, n, 1.0f, 7);
    RtdOptions o;
    o.sample_size = 80;
    o.trials = 3;
    o.seed = 9;
    const Tensor ta(Shape({n, 2}), &backend, a), tb(Shape({n, 2}), &backend, b);
    const RtdResult first = Rtd(ta, tb, o), again = Rtd(ta, tb, o);
    EXPECT_EQ(first.trials, 3);
    EXPECT_EQ(first.rtd, again.rtd);
    EXPECT_DOUBLE_EQ(first.rtd, (first.rtd_ab + first.rtd_ba) / 2);
    EXPECT_GT(first.rtd, 0.0);
    o.seed = 10;
    EXPECT_NE(Rtd(ta, tb, o).rtd, first.rtd);
}

TEST_F(RtdTest, CkaCanBeHighWhereRtdSeesTheClustersSplit) {
    // Two clusters far apart dominate CKA. In b, each cluster is split in two: CKA barely moves,
    // RTD does, and a jittered copy of a sets the scale for "barely".
    const int64_t n = 80;
    const std::vector<float> a = Blobs({{-20, 0}, {20, 0}}, n, 1.0f, 11);
    std::vector<float> split = a, jittered = a;
    std::mt19937 rng(12);
    std::normal_distribution<float> normal;
    for (int64_t i = 0; i < n; ++i) {
        // Odd-numbered points of each cluster move up by 4, even ones down by 4.
        split[static_cast<std::size_t>(2 * i + 1)] += ((i / 2) % 2 == 0) ? 4.0f : -4.0f;
    }
    for (float& x : jittered) x += 0.05f * normal(rng);
    const Tensor ta(Shape({n, 2}), &backend, a);
    const RepresentationComparison vs_split = CompareRepresentations(ta, Tensor(Shape({n, 2}), &backend, split));
    const RepresentationComparison vs_jitter = CompareRepresentations(ta, Tensor(Shape({n, 2}), &backend, jittered));
    EXPECT_GT(vs_split.cka, 0.9);
    EXPECT_GT(vs_jitter.cka, 0.99);
    // About 0.26 against 0.054 with these seeds: RTD rises almost fivefold while CKA stays above 0.9.
    EXPECT_GT(vs_split.rtd.rtd, 3 * vs_jitter.rtd.rtd)
        << "split " << vs_split.rtd.rtd << " jitter " << vs_jitter.rtd.rtd;
    EXPECT_GT(vs_jitter.mutual_knn, vs_split.mutual_knn);
}

TEST_F(RtdTest, BadInputsThrow) {
    const Tensor a(Shape({5, 2}), &backend, {0, 0, 1, 0, 0, 1, 1, 1, 2, 2});
    const Tensor b(Shape({4, 2}), &backend, {0, 0, 1, 0, 0, 1, 1, 1});
    EXPECT_THROW((void)Rtd(a, b), std::invalid_argument);
    EXPECT_THROW((void)Rtd(b, b, RtdOptions{3, 1, 0}), std::invalid_argument);
    EXPECT_THROW((void)Rtd(b, b, RtdOptions{100, 0, 0}), std::invalid_argument);
    const Tensor same(Shape({4, 2}), &backend, std::vector<float>(8, 1.0f));
    EXPECT_THROW((void)Rtd(same, b), std::invalid_argument);  // every distance 0
    const DistanceMatrix w = PairwiseDistances(a), v = PairwiseDistances(b);
    EXPECT_THROW((void)CrossBarcode(w, v, 1), std::invalid_argument);
    EXPECT_THROW((void)CrossBarcode(w, w, 0), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
