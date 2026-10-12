// TDA-7: farthest-point landmarks, with a certified covering radius and the subsampling bound it
// gives persistence diagrams.
#include "pulsatrix/tda/landmarks.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"

namespace pulsatrix {
namespace {

constexpr double kPi = 3.14159265358979323846;

class LandmarksTest : public ::testing::Test {
protected:
    CPUBackend backend;

    static double Exact(const std::vector<float>& x, int64_t d, int64_t i, int64_t j) {
        double s = 0;
        for (int64_t k = 0; k < d; ++k) {
            const double diff = static_cast<double>(x[static_cast<std::size_t>(i * d + k)]) - x[static_cast<std::size_t>(j * d + k)];
            s += diff * diff;
        }
        return std::sqrt(s);
    }

    // The exact covering radius of the given landmarks.
    static double ExactCoveringRadius(const std::vector<float>& x, int64_t n, int64_t d, const std::vector<int64_t>& l) {
        double radius = 0;
        for (int64_t i = 0; i < n; ++i) {
            double nearest = std::numeric_limits<double>::infinity();
            for (int64_t j : l) nearest = std::min(nearest, Exact(x, d, i, j));
            radius = std::max(radius, nearest);
        }
        return radius;
    }

    std::vector<float> NoisyCircle(int count, float noise, uint32_t seed, float offset = 0.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> jitter(-noise, noise);
        std::vector<float> v;
        for (int k = 0; k < count; ++k) {
            const double a = 2 * kPi * k / count;
            v.push_back(offset + static_cast<float>(std::cos(a)) + jitter(rng));
            v.push_back(offset + static_cast<float>(std::sin(a)) + jitter(rng));
        }
        return v;
    }
};

TEST_F(LandmarksTest, PointsOnALineArePickedFarthestFirst) {
    std::vector<float> line;
    for (int i = 0; i < 10; ++i) line.push_back(static_cast<float>(i));
    LandmarkOptions o;
    o.count = 4;
    const Landmarks l = FarthestPointLandmarks(Tensor(Shape({10, 1}), &backend, line), o);
    // From 0: 9 is farthest; then 4 and 5 are both 4 away (4 wins the tie); then 2 and 7 are 2
    // away from {0, 4, 9}... 2 is first.
    EXPECT_EQ(l.indices, (std::vector<int64_t>{0, 9, 4, 2}));
    ASSERT_EQ(l.insertion_radii.size(), 4u);
    EXPECT_TRUE(std::isinf(l.insertion_radii[0]));
    EXPECT_NEAR(l.insertion_radii[1], 9.0, 1e-5);
    EXPECT_NEAR(l.insertion_radii[2], 4.0, 1e-5);
    EXPECT_NEAR(l.insertion_radii[3], 2.0, 1e-5);
    EXPECT_NEAR(l.covering_radius.value, 2.0, 1e-5);  // 7 is 2 from 9, 6 is 2 from 4
    EXPECT_EQ(l.nearest[8], 1);                        // 8 is nearest 9, the second landmark
    EXPECT_NEAR(l.distance_to_nearest[8], 1.0, 1e-5);
    EXPECT_EQ(l.nearest[9], 1);
    EXPECT_EQ(l.distance_to_nearest[9], 0.0);
}

TEST_F(LandmarksTest, TheCertifiedRadiusCoversEveryPointAndIsWithinTwiceTheOptimum) {
    std::mt19937 rng(14);
    std::normal_distribution<float> normal;
    for (int trial = 0; trial < 5; ++trial) {
        const int64_t n = 11, d = 3;
        std::vector<float> x(static_cast<std::size_t>(n * d));
        for (float& v : x) v = normal(rng);
        LandmarkOptions o;
        o.count = 3;
        const Landmarks l = FarthestPointLandmarks(Tensor(Shape({n, d}), &backend, x), o);
        const double exact = ExactCoveringRadius(x, n, d, l.indices);
        EXPECT_LE(exact, l.covering_radius.value + l.covering_radius.bound());
        EXPECT_GE(exact, l.covering_radius.value - l.covering_radius.bound());
        // Gonzalez: greedy is within twice the best 3 centers, found here by brute force.
        double optimum = std::numeric_limits<double>::infinity();
        for (int64_t a = 0; a < n; ++a) {
            for (int64_t b = a + 1; b < n; ++b) {
                for (int64_t c = b + 1; c < n; ++c) optimum = std::min(optimum, ExactCoveringRadius(x, n, d, {a, b, c}));
            }
        }
        EXPECT_LE(exact, 2 * optimum + 1e-9) << trial;
    }
}

TEST_F(LandmarksTest, SelectionStopsAtTheRadiusOrWhenEveryPointIsALandmark) {
    const std::vector<float> circle = NoisyCircle(200, 0.0f, 1);
    LandmarkOptions o;
    o.radius = 0.2;
    const Landmarks l = FarthestPointLandmarks(Tensor(Shape({200, 2}), &backend, circle), o);
    EXPECT_LE(l.covering_radius.value, 0.2);
    EXPECT_GT(l.insertion_radii.back(), 0.2);  // the last one added was still needed
    EXPECT_LT(l.indices.size(), 30u);

    const std::vector<float> three = {0, 0, 1, 0, 1, 0};  // a repeated point
    LandmarkOptions all;
    all.count = 10;
    const Landmarks every = FarthestPointLandmarks(Tensor(Shape({3, 2}), &backend, three), all);
    EXPECT_EQ(every.indices, (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(every.covering_radius.value, 0.0);
}

TEST_F(LandmarksTest, ALandmarkDiagramIsWithinItsSubsamplingBoundOfTheFullOne) {
    const int64_t n = 300;
    const std::vector<float> circle = NoisyCircle(static_cast<int>(n), 0.05f, 2);
    const Tensor points(Shape({n, 2}), &backend, circle);
    LandmarkOptions o;
    o.count = 50;
    const Landmarks l = FarthestPointLandmarks(points, o);

    const RipsResult full = ComputeRipsPersistence(PairwiseDistances(points), {});
    const Tensor sub = LandmarkPoints(points, l);
    ASSERT_EQ(sub.shape().dim(0), 50);
    const RipsResult sampled = ComputeRipsPersistence(PairwiseDistances(sub), {});
    for (int dim = 0; dim <= 1; ++dim) {
        const PersistenceDiagram certified = WithSubsamplingBound(sampled.diagrams[static_cast<std::size_t>(dim)], l);
        EXPECT_GT(certified.error_bound, sampled.diagrams[static_cast<std::size_t>(dim)].error_bound);
        const Certified<double> d = BottleneckDistance(certified, full.diagrams[static_cast<std::size_t>(dim)]);
        EXPECT_LE(d.value, d.bound()) << "dim " << dim;
    }
    // The circle's loop survives, far above the bound.
    double longest = 0;
    for (const PersistencePair& p : sampled.diagrams[1].pairs) longest = std::max(longest, p.persistence());
    EXPECT_GT(longest, 4 * l.SubsamplingTerm().bound);
}

TEST_F(LandmarksTest, CenteringShrinksTheFloatErrorOfAnOffsetCloud) {
    const std::vector<float> far = NoisyCircle(100, 0.01f, 3, /*offset=*/1000.0f);
    const Tensor points(Shape({100, 2}), &backend, far);
    LandmarkOptions o;
    o.count = 20;
    const Landmarks raw = FarthestPointLandmarks(points, o);
    o.center = true;
    const Landmarks centered = FarthestPointLandmarks(points, o);
    EXPECT_LT(centered.covering_radius.bound() * 100, raw.covering_radius.bound());
    const double exact = ExactCoveringRadius(far, 100, 2, centered.indices);
    EXPECT_NEAR(exact, centered.covering_radius.value, centered.covering_radius.bound());
}

TEST_F(LandmarksTest, ACloudAndBadOptions) {
    PointCloud cloud;
    cloud.add(Tensor(Shape({4, 2}), &backend, {0, 0, 3, 0, 0, 4, 3, 4}));
    LandmarkOptions o;
    o.count = 2;
    const Landmarks l = FarthestPointLandmarks(cloud, &backend, o);
    EXPECT_EQ(l.indices, (std::vector<int64_t>{0, 3}));  // (3, 4) is 5 from the origin

    const Tensor points(Shape({4, 2}), &backend, {0, 0, 3, 0, 0, 4, 3, 4});
    EXPECT_THROW((void)FarthestPointLandmarks(points, {}), std::invalid_argument);  // no stopping rule
    LandmarkOptions bad;
    bad.count = -1;
    EXPECT_THROW((void)FarthestPointLandmarks(points, bad), std::invalid_argument);
    bad.count = 2;
    bad.first = 4;
    EXPECT_THROW((void)FarthestPointLandmarks(points, bad), std::invalid_argument);
    bad.first = 0;
    bad.radius = std::nan("");
    EXPECT_THROW((void)FarthestPointLandmarks(points, bad), std::invalid_argument);
    Landmarks wrong;
    wrong.indices = {7};
    EXPECT_THROW((void)LandmarkPoints(points, wrong), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
