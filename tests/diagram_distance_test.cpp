// TDA-5: bottleneck and Wasserstein distances between persistence diagrams, checked against brute
// force over every matching, the metric axioms, and the stability theorem on perturbed clouds.
#include "pulsatrix/tda/diagram_distance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"

namespace pulsatrix {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

PersistenceDiagram Diagram(std::vector<PersistencePair> pairs, int dimension = 1) {
    PersistenceDiagram d;
    d.dimension = dimension;
    d.pairs = std::move(pairs);
    return d;
}

// Every way to match finite points, each to a point of the other diagram or to the diagonal: the
// smallest largest move (q = infinity) or smallest sum of q-th powers.
double BruteForce(const std::vector<PersistencePair>& a, const std::vector<PersistencePair>& b, double q) {
    const std::size_t n1 = a.size(), n2 = b.size(), n = n1 + n2;
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    auto linf = [](const PersistencePair& x, const PersistencePair& y) {
        return std::max(std::abs(x.birth - y.birth), std::abs(x.death - y.death));
    };
    auto diag = [](const PersistencePair& x) { return (x.death - x.birth) / 2; };
    double best = kInf;
    do {
        // Row i (a's points, then b's diagonal copies) goes to column perm[i] (b's points, then
        // a's diagonal copies).
        double worst = 0, sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t j = perm[i];
            double c = 0;
            if (i < n1 && j < n2) c = linf(a[i], b[j]);
            if (i < n1 && j >= n2) c = diag(a[i]);
            if (i >= n1 && j < n2) c = diag(b[j]);
            worst = std::max(worst, c);
            sum += std::pow(c, std::isinf(q) ? 1.0 : q);
        }
        best = std::min(best, std::isinf(q) ? worst : std::pow(sum, 1.0 / q));
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

std::vector<PersistencePair> RandomPairs(std::mt19937& rng, int count) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<PersistencePair> pairs;
    for (int k = 0; k < count; ++k) {
        const double birth = u(rng);
        pairs.push_back({birth, birth + 0.01 + u(rng)});
    }
    return pairs;
}

TEST(DiagramDistanceTest, ASinglePointGoesToThePointOrToTheDiagonal) {
    const PersistenceDiagram a = Diagram({{0.0, 4.0}});
    const PersistenceDiagram near = Diagram({{0.5, 4.25}});
    const PersistenceDiagram empty = Diagram({});
    EXPECT_NEAR(BottleneckDistance(a, near).value, 0.5, 1e-12);  // max(0.5, 0.25)
    EXPECT_NEAR(BottleneckDistance(a, empty).value, 2.0, 1e-12);  // half its persistence
    EXPECT_NEAR(BottleneckDistance(empty, empty).value, 0.0, 1e-12);
    // Two short-lived points are cheaper to send to the diagonal than to each other.
    const PersistenceDiagram x = Diagram({{0.0, 0.2}}), y = Diagram({{5.0, 5.2}});
    EXPECT_NEAR(BottleneckDistance(x, y).value, 0.1, 1e-12);
    EXPECT_NEAR(WassersteinDistance(x, y, 1), 0.2, 1e-12);
    EXPECT_NEAR(WassersteinDistance(x, y, 2), std::sqrt(0.02), 1e-12);
}

TEST(DiagramDistanceTest, BottleneckAndWassersteinMatchBruteForce) {
    std::mt19937 rng(8);
    for (int trial = 0; trial < 40; ++trial) {
        const int n1 = static_cast<int>(rng() % 4), n2 = static_cast<int>(rng() % 4);
        const std::vector<PersistencePair> a = RandomPairs(rng, n1), b = RandomPairs(rng, n2);
        const PersistenceDiagram da = Diagram(a), db = Diagram(b);
        EXPECT_NEAR(BottleneckDistance(da, db).value, BruteForce(a, b, kInf), 1e-12) << trial;
        for (double q : {1.0, 2.0, 3.5}) {
            EXPECT_NEAR(WassersteinDistance(da, db, q), BruteForce(a, b, q), 1e-9) << trial << " q " << q;
        }
        EXPECT_NEAR(WassersteinDistance(da, db, kInf), BottleneckDistance(da, db).value, 1e-12);
    }
}

TEST(DiagramDistanceTest, TheyAreMetrics) {
    std::mt19937 rng(9);
    for (int trial = 0; trial < 20; ++trial) {
        const PersistenceDiagram a = Diagram(RandomPairs(rng, 12)), b = Diagram(RandomPairs(rng, 9)),
                                 c = Diagram(RandomPairs(rng, 15));
        const double ab = BottleneckDistance(a, b).value, bc = BottleneckDistance(b, c).value,
                     ac = BottleneckDistance(a, c).value;
        EXPECT_EQ(BottleneckDistance(a, a).value, 0.0);
        EXPECT_EQ(ab, BottleneckDistance(b, a).value);
        EXPECT_LE(ac, ab + bc + 1e-12);
        const double wab = WassersteinDistance(a, b), wbc = WassersteinDistance(b, c), wac = WassersteinDistance(a, c);
        EXPECT_NEAR(WassersteinDistance(a, a), 0.0, 1e-12);
        EXPECT_NEAR(wab, WassersteinDistance(b, a), 1e-9);
        EXPECT_LE(wac, wab + wbc + 1e-9);
        EXPECT_LE(ab, wab + 1e-12);  // bottleneck <= every Wasserstein
    }
}

TEST(DiagramDistanceTest, EssentialFeaturesMatchEachOtherByBirth) {
    const PersistenceDiagram a = Diagram({{0.0, 1.0}, {0.0, kInf}, {2.0, kInf}}, 0);
    const PersistenceDiagram b = Diagram({{0.0, 1.0}, {0.5, kInf}, {2.25, kInf}}, 0);
    EXPECT_DOUBLE_EQ(BottleneckDistance(a, b).value, 0.5);
    EXPECT_DOUBLE_EQ(WassersteinDistance(a, b, 1), 0.75);
    const PersistenceDiagram one = Diagram({{0.0, kInf}}, 0);
    EXPECT_EQ(BottleneckDistance(a, one).value, kInf);
    EXPECT_EQ(WassersteinDistance(a, one), kInf);
}

TEST(DiagramDistanceTest, TheBottleneckIsCertifiedByBothDiagramsBounds) {
    PersistenceDiagram a = Diagram({{0.0, 1.0}}), b = Diagram({{0.1, 1.0}});
    a.error_bound = 1e-6;
    b.error_bound = 3e-6;
    const Certified<double> d = BottleneckDistance(a, b);
    ASSERT_EQ(d.terms.size(), 2u);
    EXPECT_DOUBLE_EQ(d.bound(), 4e-6);
    EXPECT_NEAR(d.value, 0.1, 1e-15);
}

TEST(DiagramDistanceTest, StabilityHoldsForPerturbedPointClouds) {
    // Chazal, de Silva and Oudot: moving the distances by at most e moves every Rips diagram by at
    // most e in bottleneck distance. Checked on random clouds and perturbations, in H0 and H1.
    CPUBackend backend;
    std::mt19937 rng(10);
    std::normal_distribution<float> normal;
    int nontrivial = 0;
    for (int trial = 0; trial < 25; ++trial) {
        const int64_t n = 30;
        std::vector<float> x(static_cast<std::size_t>(n * 2)), y;
        for (float& v : x) v = normal(rng);
        const float scale = std::pow(10.0f, -static_cast<float>(trial % 4) - 1.0f);
        y = x;
        for (float& v : y) v += scale * normal(rng);
        const DistanceMatrix dx = PairwiseDistances(Tensor(Shape({n, 2}), &backend, x), {DistanceMethod::Exact});
        const DistanceMatrix dy = PairwiseDistances(Tensor(Shape({n, 2}), &backend, y), {DistanceMethod::Exact});
        double moved = 0;
        for (std::size_t k = 0; k < dx.values.size(); ++k) {
            moved = std::max(moved, std::abs(static_cast<double>(dx.values[k]) - dy.values[k]));
        }
        // A common threshold, above both enclosing radii, so no feature is cut short differently.
        RipsOptions o;
        o.threshold = std::max(dx.max_distance, dy.max_distance);
        const RipsResult rx = ComputeRipsPersistence(dx, o), ry = ComputeRipsPersistence(dy, o);
        for (int dim = 0; dim <= 1; ++dim) {
            const double d = BottleneckDistance(rx.diagrams[static_cast<std::size_t>(dim)],
                                                ry.diagrams[static_cast<std::size_t>(dim)]).value;
            EXPECT_LE(d, moved + 1e-6) << "trial " << trial << " dim " << dim;
            if (d > 0.5 * moved) ++nontrivial;
        }
    }
    EXPECT_GT(nontrivial, 10) << "the bound should be close to tight in some trials";
}

TEST(DiagramDistanceTest, LargerDiagramsAgreeBetweenTheTwoMethodsAtInfinity) {
    std::mt19937 rng(11);
    const PersistenceDiagram a = Diagram(RandomPairs(rng, 120)), b = Diagram(RandomPairs(rng, 100));
    const double bottleneck = BottleneckDistance(a, b).value;
    EXPECT_GT(bottleneck, 0.0);
    EXPECT_LE(bottleneck, WassersteinDistance(a, b, 1));
    EXPECT_LE(WassersteinDistance(a, b, 8), WassersteinDistance(a, b, 2) + 1e-12);
    EXPECT_GE(WassersteinDistance(a, b, 64), bottleneck - 1e-12);
}

TEST(DiagramDistanceTest, BadInputsThrow) {
    const PersistenceDiagram h0 = Diagram({}, 0), h1 = Diagram({}, 1);
    EXPECT_THROW((void)BottleneckDistance(h0, h1), std::invalid_argument);
    EXPECT_THROW((void)WassersteinDistance(h0, h1), std::invalid_argument);
    EXPECT_THROW((void)WassersteinDistance(h1, h1, 0.5), std::invalid_argument);
    EXPECT_THROW((void)WassersteinDistance(h1, h1, std::nan("")), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
