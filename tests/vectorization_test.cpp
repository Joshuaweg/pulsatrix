// TDA-6: Betti curves, persistence landscapes, persistence images and persistent entropy, checked
// on hand-worked diagrams, against their definitions, and for the landscapes' stability.
#include "pulsatrix/tda/vectorization.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace pulsatrix {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

PersistenceDiagram Diagram(std::vector<PersistencePair> pairs) {
    PersistenceDiagram d;
    d.dimension = 1;
    d.pairs = std::move(pairs);
    return d;
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

TEST(VectorizationTest, LinearGridIncludesBothEnds) {
    EXPECT_EQ(LinearGrid(0, 1, 5), (std::vector<double>{0, 0.25, 0.5, 0.75, 1}));
    EXPECT_EQ(LinearGrid(2, 3, 1), (std::vector<double>{2}));
    EXPECT_THROW((void)LinearGrid(0, 1, 0), std::invalid_argument);
    EXPECT_THROW((void)LinearGrid(0, kInf, 3), std::invalid_argument);
}

TEST(VectorizationTest, TheBettiCurveCountsWhatIsAliveIncludingEssentialFeatures) {
    const PersistenceDiagram d = Diagram({{0, 2}, {1, 3}, {1.5, kInf}});
    // At t: [0,2) and [1,3) and [1.5, inf).
    EXPECT_EQ(BettiCurve(d, {-1, 0, 1, 1.5, 2, 3, 100}), (std::vector<int64_t>{0, 1, 2, 3, 2, 1, 1}));
    EXPECT_EQ(BettiCurve(Diagram({}), {0, 1}), (std::vector<int64_t>{0, 0}));
    EXPECT_THROW((void)BettiCurve(d, {std::nan("")}), std::invalid_argument);
}

TEST(VectorizationTest, LandscapesAreTheLayeredTentFunctions) {
    // Tents of (0, 4) and (1, 3): peaks 2 at t = 2 and 1 at t = 2.
    const PersistenceDiagram d = Diagram({{0, 4}, {1, 3}, {0.5, kInf}});
    const Certified<PersistenceLandscape> l = ComputePersistenceLandscape(d, {0, 1, 1.5, 2, 3, 4, 5}, 3);
    ASSERT_EQ(l.value.layers.size(), 3u);
    EXPECT_EQ(l.value.layers[0], (std::vector<double>{0, 1, 1.5, 2, 1, 0, 0}));
    EXPECT_EQ(l.value.layers[1], (std::vector<double>{0, 0, 0.5, 1, 0, 0, 0}));
    EXPECT_EQ(l.value.layers[2], (std::vector<double>(7, 0.0)));  // the essential feature is left out
    EXPECT_THROW((void)ComputePersistenceLandscape(d, {0}, 0), std::invalid_argument);
}

TEST(VectorizationTest, LandscapesMoveNoMoreThanTheBottleneckDistance) {
    std::mt19937 rng(12);
    const std::vector<double> grid = LinearGrid(0, 2, 201);
    for (int trial = 0; trial < 30; ++trial) {
        const PersistenceDiagram a = Diagram(RandomPairs(rng, 10)), b = Diagram(RandomPairs(rng, 7));
        const double bottleneck = BottleneckDistance(a, b).value;
        const auto la = ComputePersistenceLandscape(a, grid, 4).value, lb = ComputePersistenceLandscape(b, grid, 4).value;
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t i = 0; i < grid.size(); ++i) {
                EXPECT_LE(std::abs(la.layers[k][i] - lb.layers[k][i]), bottleneck + 1e-12) << trial;
            }
        }
    }
}

TEST(VectorizationTest, LandscapesCarryTheDiagramsErrorBound) {
    PersistenceDiagram d = Diagram({{0, 1}});
    d.error_bound = 2e-6;
    const auto l = ComputePersistenceLandscape(d, {0.5});
    ASSERT_EQ(l.terms.size(), 1u);
    EXPECT_EQ(l.bound(), 2e-6);
}

TEST(VectorizationTest, APersistenceImagePutsEachPointsWeightAtItsBirthAndPersistence) {
    // One point at birth 1, persistence 2, on a 10 x 10 image of [0, 2] x [0, 4], sigma 0.05: all
    // its mass sits in the pixel holding (1, 2), weighted 2 / 4.
    PersistenceImageOptions o;
    o.width = 10;
    o.height = 10;
    o.birth_max = 2.0;
    o.persistence_max = 4.0;
    o.sigma = 0.05;
    const PersistenceImage image = ComputePersistenceImage(Diagram({{1.0, 3.0}}), o);
    ASSERT_EQ(image.values.size(), 100u);
    const double total = std::accumulate(image.values.begin(), image.values.end(), 0.0);
    EXPECT_NEAR(total, 0.5, 1e-9);
    // Birth 1 is the boundary of columns 4 and 5, persistence 2 that of rows 4 and 5, so four pixels
    // share it. A pixel is 0.2 = 4 sigma wide and 0.4 = 8 sigma tall: it holds the Gaussian's mass
    // within that distance on one side of the point, along each axis.
    const double across = 0.5 * std::erf(4.0 / std::sqrt(2.0)), up = 0.5 * std::erf(8.0 / std::sqrt(2.0));
    for (int64_t r : {4, 5}) {
        for (int64_t c : {4, 5}) {
            EXPECT_NEAR(image.values[static_cast<std::size_t>(r * 10 + c)], 0.5 * across * up, 1e-12);
        }
    }
}

TEST(VectorizationTest, PersistenceImagesAddUpOverPointsAndIgnoreTheDiagonal) {
    std::mt19937 rng(13);
    const std::vector<PersistencePair> a = RandomPairs(rng, 6), b = RandomPairs(rng, 4);
    std::vector<PersistencePair> both = a;
    both.insert(both.end(), b.begin(), b.end());
    both.push_back({0.7, 0.7});   // on the diagonal: weight 0
    both.push_back({0.2, kInf});  // essential: left out
    PersistenceImageOptions o;
    o.birth_max = 1.0;
    o.persistence_max = 1.01;
    o.sigma = 0.1;
    const PersistenceImage ia = ComputePersistenceImage(Diagram(a), o), ib = ComputePersistenceImage(Diagram(b), o),
                           iab = ComputePersistenceImage(Diagram(both), o);
    for (std::size_t k = 0; k < iab.values.size(); ++k) EXPECT_NEAR(iab.values[k], ia.values[k] + ib.values[k], 1e-12);
    EXPECT_DOUBLE_EQ(iab.sigma, 0.1);

    PersistenceImageOptions bad;
    bad.width = 0;
    EXPECT_THROW((void)ComputePersistenceImage(Diagram(a), bad), std::invalid_argument);
    // Defaults take the ranges from the diagram, and an empty diagram gives an empty image.
    const PersistenceImage empty = ComputePersistenceImage(Diagram({}));
    EXPECT_EQ(std::accumulate(empty.values.begin(), empty.values.end(), 0.0), 0.0);
}

TEST(VectorizationTest, PersistentEntropyIsTheEntropyOfTheLifetimes) {
    EXPECT_EQ(PersistentEntropy(Diagram({})), 0.0);
    EXPECT_DOUBLE_EQ(PersistentEntropy(Diagram({{0, 3}})), 0.0);
    EXPECT_NEAR(PersistentEntropy(Diagram({{0, 1}, {2, 3}, {5, 6}, {1, 2}})), std::log(4.0), 1e-12);
    // Persistences 1 and 3: -(1/4 log 1/4 + 3/4 log 3/4); scaling every point leaves it unchanged.
    const double expected = -(0.25 * std::log(0.25) + 0.75 * std::log(0.75));
    EXPECT_NEAR(PersistentEntropy(Diagram({{0, 1}, {0, 3}, {0, kInf}})), expected, 1e-12);
    EXPECT_NEAR(PersistentEntropy(Diagram({{0, 10}, {0, 30}})), expected, 1e-12);
}

}  // namespace
}  // namespace pulsatrix
