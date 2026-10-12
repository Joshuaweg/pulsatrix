// TDA-3: H0 persistence as a minimum spanning tree, checked against Kruskal's algorithm and
// hand-worked configurations.
#include "pulsatrix/tda/h0.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class H0Test : public ::testing::Test {
protected:
    CPUBackend backend;

    DistanceMatrix Distances(int64_t n, int64_t d, std::vector<float> v) {
        return PairwiseDistances(Tensor(Shape({n, d}), &backend, v), {DistanceMethod::Exact});
    }

    // Kruskal's algorithm, an independent way to the same tree lengths.
    static std::vector<float> KruskalLengths(const DistanceMatrix& m) {
        std::vector<std::tuple<float, int64_t, int64_t>> all;
        for (int64_t i = 0; i < m.size; ++i) {
            for (int64_t j = i + 1; j < m.size; ++j) all.emplace_back(m.at(i, j), i, j);
        }
        std::sort(all.begin(), all.end());
        std::vector<int64_t> root(static_cast<std::size_t>(m.size));
        std::iota(root.begin(), root.end(), int64_t{0});
        auto find = [&](int64_t x) {
            while (root[static_cast<std::size_t>(x)] != x) x = root[static_cast<std::size_t>(x)];
            return x;
        };
        std::vector<float> lengths;
        for (const auto& [length, i, j] : all) {
            const int64_t a = find(i), b = find(j);
            if (a == b) continue;
            root[static_cast<std::size_t>(a)] = b;
            lengths.push_back(length);
        }
        return lengths;
    }
};

TEST_F(H0Test, PointsOnALineMergeAtTheirGaps) {
    const H0Result h0 = ComputeH0(Distances(4, 1, {7, 0, 3, 1}));
    ASSERT_EQ(h0.edges.size(), 3u);
    EXPECT_FLOAT_EQ(h0.edges[0].length, 1);  // 0 and 1
    EXPECT_FLOAT_EQ(h0.edges[1].length, 2);  // 1 and 3
    EXPECT_FLOAT_EQ(h0.edges[2].length, 4);  // 3 and 7
    EXPECT_EQ(h0.edges[0].a, 1);  // points 1 (value 0) and 3 (value 1)
    EXPECT_EQ(h0.edges[0].b, 3);

    const PersistenceDiagram& d = h0.diagram;
    EXPECT_EQ(d.dimension, 0);
    ASSERT_EQ(d.pairs.size(), 4u);
    for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(d.pairs[static_cast<std::size_t>(k)].birth, 0.0);
        EXPECT_FALSE(d.pairs[static_cast<std::size_t>(k)].essential());
    }
    EXPECT_DOUBLE_EQ(d.pairs[2].death, 4.0);
    EXPECT_TRUE(d.pairs[3].essential());
    EXPECT_EQ(d.essential_count(), 1);
    EXPECT_EQ(d.pairs[3].persistence(), std::numeric_limits<double>::infinity());
}

TEST_F(H0Test, PrimMatchesKruskalOnRandomPoints) {
    std::mt19937 rng(21);
    std::normal_distribution<float> normal;
    for (int64_t n : {2, 3, 17, 120}) {
        std::vector<float> v(static_cast<std::size_t>(n * 5));
        for (float& x : v) x = normal(rng);
        const DistanceMatrix m = Distances(n, 5, v);
        const H0Result h0 = ComputeH0(m);
        const std::vector<float> expected = KruskalLengths(m);
        ASSERT_EQ(h0.edges.size(), expected.size()) << n;
        for (std::size_t k = 0; k < expected.size(); ++k) EXPECT_EQ(h0.edges[k].length, expected[k]) << n;
        // Every edge is a real pair at its stored distance.
        for (const MstEdge& e : h0.edges) {
            EXPECT_LT(e.a, e.b);
            EXPECT_EQ(e.length, m.at(e.a, e.b));
        }
        // And together they connect every point.
        const std::vector<int64_t> one = SingleLinkageClusters(h0.edges, n, std::numeric_limits<double>::infinity());
        EXPECT_TRUE(std::all_of(one.begin(), one.end(), [](int64_t l) { return l == 0; }));
    }
}

TEST_F(H0Test, DuplicatePointsKeepTheirZeroEdgeButNotAZeroPersistencePair) {
    const H0Result h0 = ComputeH0(Distances(3, 2, {1, 1, 1, 1, 4, 5}));
    ASSERT_EQ(h0.edges.size(), 2u);
    EXPECT_EQ(h0.edges[0].length, 0.0f);
    ASSERT_EQ(h0.diagram.pairs.size(), 2u);  // (0, 5) and the essential one
    EXPECT_DOUBLE_EQ(h0.diagram.pairs[0].death, 5.0);
}

TEST_F(H0Test, TiesBreakTheSameWayEveryTime) {
    // A unit square: every side is 1, so any three sides are a minimum tree.
    const DistanceMatrix m = Distances(4, 2, {0, 0, 1, 0, 1, 1, 0, 1});
    const H0Result first = ComputeH0(m);
    ASSERT_EQ(first.edges.size(), 3u);
    for (const MstEdge& e : first.edges) EXPECT_FLOAT_EQ(e.length, 1.0f);
    // Prim from point 0 takes 0-1, then 1-2 (the lower index of a tie), then 0-3 (the earlier
    // parent of a tie); edges are listed by (length, a, b).
    EXPECT_EQ(std::make_tuple(first.edges[0].a, first.edges[0].b), std::make_tuple(int64_t{0}, int64_t{1}));
    EXPECT_EQ(std::make_tuple(first.edges[1].a, first.edges[1].b), std::make_tuple(int64_t{0}, int64_t{3}));
    EXPECT_EQ(std::make_tuple(first.edges[2].a, first.edges[2].b), std::make_tuple(int64_t{1}, int64_t{2}));
    const H0Result again = ComputeH0(m);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(again.edges[k].a, first.edges[k].a);
        EXPECT_EQ(again.edges[k].b, first.edges[k].b);
    }
}

TEST_F(H0Test, TheDiagramCarriesTheDistancesErrorBound) {
    std::vector<float> v(40);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.7f * static_cast<float>(i)) + 100.0f;
    const DistanceMatrix m = PairwiseDistances(Tensor(Shape({10, 4}), &backend, v));
    ASSERT_GT(m.error_bound, 0.0);
    EXPECT_EQ(ComputeH0(m).diagram.error_bound, m.error_bound);
}

TEST_F(H0Test, CuttingTheTreeGivesTheClusters) {
    // Two groups of three, far apart: 0, 2, 4 near the origin and 1, 3, 5 near 100.
    const DistanceMatrix m = Distances(6, 1, {0, 100, 1, 101, 2, 103});
    const H0Result h0 = ComputeH0(m);
    EXPECT_FLOAT_EQ(h0.edges.back().length, 98.0f);  // the gap between the groups merges last
    EXPECT_EQ(SingleLinkageClusters(h0.edges, 6, 10.0), (std::vector<int64_t>{0, 1, 0, 1, 0, 1}));
    EXPECT_EQ(SingleLinkageClusters(h0.edges, 6, 1.5), (std::vector<int64_t>{0, 1, 0, 1, 0, 2}));
    EXPECT_EQ(SingleLinkageClusters(h0.edges, 6, -1.0), (std::vector<int64_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(SingleLinkageClusters(h0.edges, 6, 1000.0), (std::vector<int64_t>(6, 0)));
}

TEST_F(H0Test, BadInputsThrow) {
    DistanceMatrix one;
    one.size = 1;
    EXPECT_THROW((void)ComputeH0(one), std::invalid_argument);
    EXPECT_THROW((void)SingleLinkageClusters({{0, 3, 1.0f}}, 3, 2.0), std::invalid_argument);
    EXPECT_THROW((void)SingleLinkageClusters({}, -1, 2.0), std::invalid_argument);
    EXPECT_TRUE(SingleLinkageClusters({}, 0, 2.0).empty());
}

}  // namespace
}  // namespace pulsatrix
