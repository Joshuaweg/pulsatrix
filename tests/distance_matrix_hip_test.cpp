// TDA-2 and TDA-7 on the GPU: the GEMMs run through hipBLAS, and the distances and landmark covering
// radii stay within their certified float32 error bounds, including when cancellation makes those
// bounds large.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/landmarks.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Gaussian(int64_t n, int64_t d, uint32_t seed, float scale, float offset) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(n * d));
    for (float& x : v) x = offset + scale * normal(rng);
    return v;
}

// The largest |stored - exact| over every pair, against long double sums.
double WorstError(const DistanceMatrix& m, const std::vector<float>& x, int64_t d) {
    long double worst = 0;
    std::size_t k = 0;
    for (int64_t i = 0; i < m.size; ++i) {
        for (int64_t j = i + 1; j < m.size; ++j, ++k) {
            long double s = 0;
            for (int64_t a = 0; a < d; ++a) {
                const long double diff = static_cast<long double>(x[static_cast<std::size_t>(i * d + a)]) -
                                         x[static_cast<std::size_t>(j * d + a)];
                s += diff * diff;
            }
            worst = std::max(worst, std::fabs(static_cast<long double>(m.values[k]) - std::sqrt(s)));
        }
    }
    return static_cast<double>(worst);
}

TEST(DistanceMatrixHip, HipblasDistancesStayWithinTheBound) {
    HIPBackend hip;
    const int64_t n = 400, d = 768;  // a language model's width
    const std::vector<float> x = Gaussian(n, d, 11, 1.0f, 0.0f);
    const Tensor points(Shape({n, d}), &hip, x, DeviceType::Hip);
    DistanceOptions o;
    o.block_rows = 128;  // several blocks, the last one short
    const DistanceMatrix m = PairwiseDistances(points, o);
    EXPECT_LE(WorstError(m, x, d), m.error_bound);
    EXPECT_LT(m.error_bound, 1e-3 * m.max_distance);

    CPUBackend cpu;
    const DistanceMatrix on_cpu = PairwiseDistances(Tensor(Shape({n, d}), &cpu, x), o);
    for (std::size_t k = 0; k < m.values.size(); ++k) {
        ASSERT_NEAR(m.values[k], on_cpu.values[k], m.error_bound + on_cpu.error_bound);
    }
}

TEST(DistanceMatrixHip, CancellationOnTheGpuIsCoveredByTheBound) {
    HIPBackend hip;
    const int64_t n = 200, d = 256;
    const std::vector<float> x = Gaussian(n, d, 12, 0.01f, 1000.0f / 16.0f);
    const Tensor points(Shape({n, d}), &hip, x, DeviceType::Hip);
    const DistanceMatrix raw = PairwiseDistances(points);
    EXPECT_LE(WorstError(raw, x, d), raw.error_bound);
    DistanceOptions centered;
    centered.center = true;
    const DistanceMatrix c = PairwiseDistances(points, centered);
    EXPECT_LE(WorstError(c, x, d), c.error_bound);
    EXPECT_LT(c.error_bound * 100, raw.error_bound);
}

TEST(DistanceMatrixHip, LandmarkCoveringRadiiAreCertifiedOnTheGpu) {
    HIPBackend hip;
    const int64_t n = 2000, d = 384;
    for (float offset : {0.0f, 50.0f}) {
        const std::vector<float> x = Gaussian(n, d, 13, 1.0f, offset);
        const Tensor points(Shape({n, d}), &hip, x, DeviceType::Hip);
        LandmarkOptions o;
        o.count = 64;
        const Landmarks l = FarthestPointLandmarks(points, o);
        ASSERT_EQ(l.indices.size(), 64u);
        double exact = 0;
        for (int64_t i = 0; i < n; ++i) {
            long double nearest = 1e300L;
            for (int64_t j : l.indices) {
                long double s = 0;
                for (int64_t a = 0; a < d; ++a) {
                    const long double diff = static_cast<long double>(x[static_cast<std::size_t>(i * d + a)]) -
                                             x[static_cast<std::size_t>(j * d + a)];
                    s += diff * diff;
                }
                nearest = std::min(nearest, std::sqrt(s));
            }
            exact = std::max(exact, static_cast<double>(nearest));
        }
        EXPECT_NEAR(exact, l.covering_radius.value, l.covering_radius.bound()) << "offset " << offset;
    }
}

}  // namespace
}  // namespace pulsatrix
