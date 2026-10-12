// TDA-2 on the GPU: the GEMM runs through hipBLAS, and its distances stay within the certified
// float32 error bound, including when cancellation makes that bound large.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/tda/distance_matrix.hpp"

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

}  // namespace
}  // namespace pulsatrix
