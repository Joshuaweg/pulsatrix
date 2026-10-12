// TDA-8: significance tests for persistence diagrams: the universal null distribution (Bobrowski
// and Skraba 2023), bootstrap bands (Fasy et al. 2014), Gaussian baselines and rank p-values.
#include "pulsatrix/tda/significance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

class SignificanceTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Uniform points in the unit square, from raw mt19937 output.
    static std::vector<float> UniformSquare(int count, uint32_t seed) {
        std::mt19937 rng(seed);
        std::vector<float> v;
        for (int k = 0; k < 2 * count; ++k) v.push_back(static_cast<float>(rng()) / 4294967296.0f);
        return v;
    }

    // count points on a circle of radius r around (0.5, 0.5), with uniform jitter.
    static void AddCircle(std::vector<float>& v, int count, float r, float jitter, uint32_t seed) {
        std::mt19937 rng(seed);
        for (int k = 0; k < count; ++k) {
            const double a = 2 * kPi * k / count;
            const float jx = jitter * (static_cast<float>(rng()) / 4294967296.0f - 0.5f);
            const float jy = jitter * (static_cast<float>(rng()) / 4294967296.0f - 0.5f);
            v.push_back(0.5f + r * static_cast<float>(std::cos(a)) + jx);
            v.push_back(0.5f + r * static_cast<float>(std::sin(a)) + jy);
        }
    }

    PersistenceDiagram H1(const std::vector<float>& v) {
        const int64_t n = static_cast<int64_t>(v.size() / 2);
        return ComputeRipsPersistence(PairwiseDistances(Tensor(Shape({n, 2}), &backend, v)), {}).diagrams[1];
    }
};

TEST_F(SignificanceTest, TheUniversalNullFollowsItsFormula) {
    PersistenceDiagram d;
    d.dimension = 1;
    d.pairs = {{1.0, 2.0}, {1.0, 4.0}, {2.0, 3.0}, {0.5, kInf}, {0.0, 1.0}};  // the last two aren't tested
    const UniversalNullResult r = UniversalNullTest(d, 0.05);
    ASSERT_EQ(r.features.size(), 3u);
    const double l1 = std::log(std::log(2.0)), l2 = std::log(std::log(4.0)), l3 = std::log(std::log(1.5));
    const double mean = (l1 + l2 + l3) / 3;
    EXPECT_NEAR(r.mean_log_log, mean, 1e-15);
    EXPECT_EQ(r.features[1].index, 1);
    EXPECT_NEAR(r.features[1].ratio, 4.0, 1e-15);
    const double ell = l2 - 0.57721566490153286061 - mean;
    EXPECT_NEAR(r.features[1].ell, ell, 1e-14);
    EXPECT_NEAR(r.features[1].p_value, std::exp(-std::exp(ell)), 1e-14);
    EXPECT_DOUBLE_EQ(r.threshold, 0.05 / 3);
}

TEST_F(SignificanceTest, OnUniformNoiseThePValuesAreCalibrated) {
    // Pooling the loops of five uniform clouds, p-values should be roughly uniform: about a tenth
    // below 0.1 and half below 0.5 (the conjecture's tails converge slowly, so the bands are wide).
    std::vector<double> p_values;
    int significant = 0;
    for (uint32_t seed = 1; seed <= 5; ++seed) {
        const UniversalNullResult r = UniversalNullTest(H1(UniformSquare(500, seed)));
        for (const FeatureSignificance& f : r.features) {
            p_values.push_back(f.p_value);
            significant += f.significant ? 1 : 0;
        }
    }
    ASSERT_GT(p_values.size(), 300u);
    auto fraction_below = [&](double x) {
        return static_cast<double>(std::count_if(p_values.begin(), p_values.end(), [x](double p) { return p < x; })) /
               static_cast<double>(p_values.size());
    };
    EXPECT_NEAR(fraction_below(0.1), 0.1, 0.06);
    EXPECT_NEAR(fraction_below(0.5), 0.5, 0.12);
    EXPECT_LE(significant, 1) << "Bonferroni at 0.05 should flag almost nothing in noise";
}

TEST_F(SignificanceTest, ACircleInNoiseIsTheOneSignificantLoop) {
    // A ring with clutter around it but not inside it: uniform points inside the disk would fill
    // the loop in at once (death / birth about 3, not significant, as it shouldn't be).
    std::vector<float> v;
    const std::vector<float> clutter = UniformSquare(400, 7);
    for (std::size_t k = 0; k < clutter.size(); k += 2) {
        const float dx = clutter[k] - 0.5f, dy = clutter[k + 1] - 0.5f;
        if (std::sqrt(dx * dx + dy * dy) > 0.4f) v.insert(v.end(), {clutter[k], clutter[k + 1]});
    }
    AddCircle(v, 120, 0.35f, 0.02f, 8);
    const PersistenceDiagram d = H1(v);
    const UniversalNullResult r = UniversalNullTest(d);
    int significant = 0;
    double best_ratio = 0;
    for (const FeatureSignificance& f : r.features) {
        significant += f.significant ? 1 : 0;
        best_ratio = std::max(best_ratio, f.ratio);
    }
    double min_p = 1;
    for (const FeatureSignificance& f : r.features) min_p = std::min(min_p, f.p_value);
    EXPECT_EQ(significant, 1) << "best ratio " << best_ratio << " min p " << min_p << " threshold " << r.threshold
                              << " features " << r.features.size();
    for (const FeatureSignificance& f : r.features) {
        if (f.significant) EXPECT_EQ(f.ratio, best_ratio);
    }
}

TEST_F(SignificanceTest, TheBootstrapBandSeparatesTheCircleFromNoise) {
    std::vector<float> circle;
    AddCircle(circle, 80, 0.4f, 0.04f, 9);
    const DistanceMatrix dc = PairwiseDistances(Tensor(Shape({80, 2}), &backend, circle));
    BootstrapOptions o;
    o.resamples = 40;
    o.seed = 3;
    const BootstrapBandResult rc = BootstrapBand(dc, 1, o);
    ASSERT_EQ(rc.distances.size(), 40u);
    EXPECT_TRUE(std::is_sorted(rc.distances.begin(), rc.distances.end()));
    ASSERT_EQ(rc.significant.size(), 1u);
    EXPECT_DOUBLE_EQ(rc.diagram.pairs[static_cast<std::size_t>(rc.significant[0])].persistence(), MaxPersistence(rc.diagram));
    // The same seed gives the same band.
    EXPECT_EQ(BootstrapBand(dc, 1, o).band, rc.band);

    const std::vector<float> noise = UniformSquare(80, 10);
    const DistanceMatrix dn = PairwiseDistances(Tensor(Shape({80, 2}), &backend, noise));
    EXPECT_TRUE(BootstrapBand(dn, 1, o).significant.empty());
}

TEST_F(SignificanceTest, AGaussianBaselineMatchesTheMeanCovarianceAndRank) {
    // Points in a plane of R^3 (z = x + 2y), with a large mean: the baseline keeps both.
    std::mt19937 rng(11);
    std::vector<float> v;
    for (int i = 0; i < 200; ++i) {
        const float x = static_cast<float>(rng()) / 4294967296.0f, y = 3 * static_cast<float>(rng()) / 4294967296.0f;
        v.insert(v.end(), {x + 10, y, x + 2 * y});
    }
    const Tensor points(Shape({200, 3}), &backend, v);
    const Tensor g = GaussianBaseline(points, 20000, 5);
    ASSERT_EQ(g.shape().dim(0), 20000);
    ASSERT_EQ(g.shape().dim(1), 3);
    const std::vector<float> s = g.to_host_vector();
    auto moments = [](const std::vector<float>& x, int64_t n, std::vector<double>& mean, std::vector<double>& cov) {
        mean.assign(3, 0);
        cov.assign(9, 0);
        for (int64_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) mean[static_cast<std::size_t>(a)] += x[static_cast<std::size_t>(i * 3 + a)] / static_cast<double>(n);
        }
        for (int64_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    cov[static_cast<std::size_t>(a * 3 + b)] += (x[static_cast<std::size_t>(i * 3 + a)] - mean[static_cast<std::size_t>(a)]) *
                                                                (x[static_cast<std::size_t>(i * 3 + b)] - mean[static_cast<std::size_t>(b)]) /
                                                                static_cast<double>(n - 1);
                }
            }
        }
    };
    std::vector<double> m0, c0, m1, c1;
    moments(v, 200, m0, c0);
    moments(s, 20000, m1, c1);
    for (int a = 0; a < 3; ++a) EXPECT_NEAR(m1[static_cast<std::size_t>(a)], m0[static_cast<std::size_t>(a)], 0.05);
    for (int k = 0; k < 9; ++k) EXPECT_NEAR(c1[static_cast<std::size_t>(k)], c0[static_cast<std::size_t>(k)], 0.05 * (1 + std::abs(c0[static_cast<std::size_t>(k)])));
    // Every baseline point lies in the plane (z = (x - 10) + 2y), up to float rounding.
    for (int64_t i = 0; i < 20000; ++i) {
        const double x = s[static_cast<std::size_t>(i * 3)], y = s[static_cast<std::size_t>(i * 3 + 1)], z = s[static_cast<std::size_t>(i * 3 + 2)];
        ASSERT_NEAR(z, (x - 10) + 2 * y, 1e-3);
    }
}

TEST_F(SignificanceTest, ACircleBeatsItsGaussianBaselines) {
    std::vector<float> circle;
    AddCircle(circle, 100, 0.4f, 0.03f, 12);
    const Tensor points(Shape({100, 2}), &backend, circle);
    const double observed = MaxPersistence(H1(circle));
    std::vector<double> baselines;
    for (uint32_t seed = 0; seed < 19; ++seed) {
        baselines.push_back(MaxPersistence(H1(GaussianBaseline(points, 100, seed).to_host_vector())));
    }
    EXPECT_DOUBLE_EQ(BaselinePValue(observed, baselines), 1.0 / 20.0);  // the smallest possible
}

TEST_F(SignificanceTest, RankPValuesAndHelpers) {
    EXPECT_DOUBLE_EQ(BaselinePValue(5, {1, 2, 3}), 0.25);
    EXPECT_DOUBLE_EQ(BaselinePValue(2, {1, 2, 3}), 0.75);  // ties count against the observed
    EXPECT_DOUBLE_EQ(BaselinePValue(0, {1, 2, 3}), 1.0);
    EXPECT_THROW((void)BaselinePValue(1, {}), std::invalid_argument);
    EXPECT_THROW((void)BaselinePValue(std::nan(""), {1}), std::invalid_argument);

    PersistenceDiagram d;
    d.pairs = {{0, 1}, {0.5, 3}, {1, kInf}};
    EXPECT_DOUBLE_EQ(MaxPersistence(d), 2.5);
    EXPECT_EQ(MaxPersistence(PersistenceDiagram{}), 0.0);

    const Tensor points(Shape({3, 1}), &backend, {0, 1, 5});
    const DistanceMatrix m = PairwiseDistances(points, {DistanceMethod::Exact});
    const DistanceMatrix sub = SubDistanceMatrix(m, {2, 0, 2});
    EXPECT_EQ(sub.size, 3);
    EXPECT_EQ(sub.values, (std::vector<float>{5, 0, 5}));
    EXPECT_FLOAT_EQ(sub.max_distance, 5);
    EXPECT_THROW((void)SubDistanceMatrix(m, {3}), std::invalid_argument);
}

TEST_F(SignificanceTest, BadInputsThrow) {
    PersistenceDiagram h0;
    h0.pairs = {{0, 1}, {0, 2}};
    EXPECT_THROW((void)UniversalNullTest(h0), std::invalid_argument);
    PersistenceDiagram one;
    one.dimension = 1;
    one.pairs = {{1, 2}};
    EXPECT_THROW((void)UniversalNullTest(one), std::invalid_argument);
    one.pairs.push_back({1, 3});
    EXPECT_THROW((void)UniversalNullTest(one, 0.0), std::invalid_argument);
    EXPECT_THROW((void)UniversalNullTest(one, 1.0), std::invalid_argument);

    const DistanceMatrix m = PairwiseDistances(Tensor(Shape({4, 1}), &backend, {0, 1, 2, 3}));
    BootstrapOptions o;
    o.resamples = 0;
    EXPECT_THROW((void)BootstrapBand(m, 1, o), std::invalid_argument);
    o.resamples = 2;
    EXPECT_THROW((void)BootstrapBand(m, 2, o), std::invalid_argument);  // max_dimension is 1
    EXPECT_THROW((void)GaussianBaseline(Tensor(Shape({1, 2}), &backend, {1, 2}), 5), std::invalid_argument);
    EXPECT_THROW((void)GaussianBaseline(Tensor(Shape({2, 2}), &backend, {1, 2, 3, 4}), 0), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
