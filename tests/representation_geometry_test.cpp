// INT-4: anisotropy, effective rank and TwoNN intrinsic dimension, checked against numpy and the
// TwoNN code of scikit-dimension and DADApy (tools/generate_geometry_reference_values.py).
#include "pulsatrix/representation_geometry.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/matrix_decompositions.hpp"

namespace pulsatrix {
namespace {

class GeometryTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The reference script's inputs: offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i²).
    std::vector<float> MakeValues(double seed, int64_t n, int64_t p, double offset = 0.0) {
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t a = 0; a < p; ++a) {
                const double di = static_cast<double>(i);
                const double da = static_cast<double>(a);
                v.push_back(static_cast<float>(offset + std::sin(seed + 0.37 * di + 1.13 * da + 0.05 * di * da + 0.71 * di * di)));
            }
        }
        return v;
    }

    Tensor Make(double seed, int64_t n, int64_t p, double offset = 0.0) { return Tensor(Shape({n, p}), &backend, MakeValues(seed, n, p, offset)); }

    // Uniform in [-1, 1) from mt19937's raw output, the same on every platform.
    static float Uniform(std::mt19937& rng) { return static_cast<float>(rng()) / 4294967296.0f * 2.0f - 1.0f; }

    // n points spread uniformly over a dim-dimensional cube, placed in `width` dimensions by fixed
    // random linear map, so the width says nothing about the dimension.
    Tensor Embedded(int64_t n, int64_t dim, int64_t width, uint32_t seed) {
        std::mt19937 rng(seed);
        std::vector<float> map(static_cast<std::size_t>(dim * width));
        for (float& m : map) m = Uniform(rng);
        std::vector<float> v(static_cast<std::size_t>(n * width), 0.0f);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t d = 0; d < dim; ++d) {
                const float z = Uniform(rng);
                for (int64_t c = 0; c < width; ++c) v[static_cast<std::size_t>(i * width + c)] += z * map[static_cast<std::size_t>(d * width + c)];
            }
        }
        return Tensor(Shape({n, width}), &backend, v);
    }
};

// ---- reference values ----------------------------------------------------------------------------

TEST_F(GeometryTest, AnisotropyMatchesThePairByPairMean) {
    const Tensor x = Make(0.3, 30, 12, 0.8);
    EXPECT_NEAR(Anisotropy(x), 0.57362539796455625, 1e-12);
    EXPECT_NEAR(Anisotropy(x, /*center=*/true), -0.033510468647212244, 1e-12);
}

TEST_F(GeometryTest, EffectiveRankMatchesNumpy) {
    // Tall: the p x p Gram matrix.
    const Tensor tall = Make(0.3, 30, 12, 0.8);
    const EffectiveRank t = MeasureEffectiveRank(tall, /*center=*/false);
    EXPECT_NEAR(t.effective_rank, 8.0208198728587732, 1e-10);
    EXPECT_NEAR(t.participation_ratio, 2.6050240308259762, 1e-10);
    EXPECT_EQ(t.singular_values.size(), 12u);
    const EffectiveRank tc = MeasureEffectiveRank(tall);
    EXPECT_NEAR(tc.effective_rank, 8.3416568476073198, 1e-10);
    EXPECT_NEAR(tc.participation_ratio, 5.8914123870890762, 1e-10);

    // Wide: the n x n Gram matrix. Centered it has rank 9, and the tenth singular value is zero.
    const Tensor wide = Make(1.1, 10, 40);
    const EffectiveRank w = MeasureEffectiveRank(wide, /*center=*/false);
    EXPECT_NEAR(w.effective_rank, 8.1667575686547771, 1e-10);
    EXPECT_NEAR(w.participation_ratio, 5.5739187343778909, 1e-10);
    const EffectiveRank wc = MeasureEffectiveRank(wide);
    EXPECT_NEAR(wc.effective_rank, 7.5684934708067537, 1e-10);
    EXPECT_NEAR(wc.participation_ratio, 5.4303896256753266, 1e-10);
    ASSERT_EQ(wc.singular_values.size(), 10u);
    EXPECT_NEAR(wc.singular_values[0], 6.99280174, 1e-7);
    EXPECT_NEAR(wc.singular_values[8], 0.554624228, 1e-8);
    EXPECT_EQ(wc.singular_values[9], 0.0);
}

TEST_F(GeometryTest, TwoNNMatchesScikitDimensionAndDadapy) {
    const Tensor x = Make(0.7, 200, 3);
    const TwoNNResult r = TwoNNDimension(x);
    EXPECT_NEAR(r.dimension, 2.3883419110333435, 1e-10);
    EXPECT_NEAR(r.mle_dimension, 2.1987407818445646, 1e-10);
    EXPECT_EQ(r.points, 200);
    EXPECT_EQ(r.duplicates, 0);
    EXPECT_NEAR(TwoNNDimension(x, 0.25).dimension, 2.3077336550348995, 1e-10);
}

// ---- falsifiers ----------------------------------------------------------------------------------

// The reason the measure exists: a shared offset makes unrelated rows look alike by cosine, and
// centering takes it away again.
TEST_F(GeometryTest, AnisotropyRisesWithACommonDirectionAndCenteringRemovesIt) {
    const int64_t n = 300, p = 64;
    std::mt19937 rng(5);
    std::vector<float> spread(static_cast<std::size_t>(n * p)), shifted(spread.size());
    for (std::size_t k = 0; k < spread.size(); ++k) {
        spread[k] = Uniform(rng);
        // A shift of 6 along one axis: about 36 / (36 + 64 / 3) = 0.63 of each row's squared length.
        shifted[k] = spread[k] + (k % static_cast<std::size_t>(p) == 0 ? 6.0f : 0.0f);
    }
    const Tensor a(Shape({n, p}), &backend, spread), b(Shape({n, p}), &backend, shifted);
    EXPECT_LT(std::abs(Anisotropy(a)), 0.01);
    EXPECT_GT(Anisotropy(b), 0.5);
    EXPECT_LT(std::abs(Anisotropy(b, /*center=*/true)), 0.01);

    // Rows all pointing the same way, at different lengths.
    const Tensor same(Shape({3, 2}), &backend, std::vector<float>{1, 2, 2, 4, 0.5f, 1});
    EXPECT_NEAR(Anisotropy(same), 1.0, 1e-12);
    // Two opposite rows.
    EXPECT_NEAR(Anisotropy(Tensor(Shape({2, 2}), &backend, std::vector<float>{1, 1, -1, -1})), -1.0, 1e-12);
}

TEST_F(GeometryTest, EffectiveRankCountsEqualDirections) {
    // k orthogonal rows of equal length: every singular value is equal, so both counts are k.
    for (int64_t k : {2, 3, 7}) {
        std::vector<float> v(static_cast<std::size_t>(k * 10), 0.0f);
        for (int64_t i = 0; i < k; ++i) v[static_cast<std::size_t>(i * 10 + i)] = 2.0f;
        const EffectiveRank r = MeasureEffectiveRank(Tensor(Shape({k, 10}), &backend, v), /*center=*/false);
        EXPECT_NEAR(r.effective_rank, static_cast<double>(k), 1e-12) << k;
        EXPECT_NEAR(r.participation_ratio, static_cast<double>(k), 1e-12) << k;
    }
    // Two parallel rows: one direction.
    const EffectiveRank one = MeasureEffectiveRank(Tensor(Shape({2, 3}), &backend, std::vector<float>{1, 2, 2, -2, -4, -4}), false);
    EXPECT_NEAR(one.effective_rank, 1.0, 1e-12);
    EXPECT_EQ(one.singular_values[1], 0.0);
    // Points on a 3-dimensional subspace of a 40-wide space, many of them: rank 3 at most.
    const EffectiveRank sub = MeasureEffectiveRank(Embedded(500, 3, 40, 2));
    EXPECT_LE(sub.effective_rank, 3.0 + 1e-9);
    EXPECT_GT(sub.effective_rank, 2.0);
    EXPECT_EQ(sub.singular_values[3], 0.0);
}

// TwoNN recovers the dimension points vary in, not the width they're written in.
TEST_F(GeometryTest, TwoNNFindsTheDimensionNotTheWidth) {
    for (int64_t dim : {2, 5}) {
        const TwoNNResult r = TwoNNDimension(Embedded(1500, dim, 40, 7));
        EXPECT_NEAR(r.dimension, static_cast<double>(dim), 0.15 * static_cast<double>(dim)) << dim;
        EXPECT_NEAR(r.mle_dimension, static_cast<double>(dim), 0.15 * static_cast<double>(dim)) << dim;
    }
}

TEST_F(GeometryTest, TwoNNDropsRepeatedRows) {
    std::vector<float> v = MakeValues(0.7, 200, 3);
    const std::vector<float> first_rows(v.begin(), v.begin() + 3 * 5);
    v.insert(v.end(), first_rows.begin(), first_rows.end());  // rows 0-4 again
    v.insert(v.end(), v.begin() + 3 * 7, v.begin() + 3 * 8);  // row 7 again
    const TwoNNResult r = TwoNNDimension(Tensor(Shape({206, 3}), &backend, v));
    EXPECT_EQ(r.points, 200);
    EXPECT_EQ(r.duplicates, 6);
    EXPECT_NEAR(r.dimension, 2.3883419110333435, 1e-10);
}

TEST_F(GeometryTest, RejectsWhatItCannotMeasure) {
    const Tensor one_row = Make(0.0, 1, 4);
    EXPECT_THROW((void)Anisotropy(one_row), std::invalid_argument);
    EXPECT_THROW((void)MeasureEffectiveRank(one_row), std::invalid_argument);
    EXPECT_THROW((void)Anisotropy(Tensor(Shape({4}), &backend, std::vector<float>{1, 2, 3, 4})), std::invalid_argument);
    EXPECT_THROW((void)Anisotropy(Tensor(Shape({2, 2}), &backend, std::vector<float>{1, 2, 0, 0})), std::invalid_argument);
    EXPECT_THROW((void)Anisotropy(Tensor(Shape({2, 2}), &backend, std::vector<float>{1, 2, 1, 2}), /*center=*/true), std::invalid_argument);
    EXPECT_THROW((void)Anisotropy(Tensor(Shape({2, 2}), &backend, std::vector<float>{1, NAN, 1, 2})), std::invalid_argument);
    EXPECT_THROW((void)MeasureEffectiveRank(Tensor(Shape({2, 2}), &backend, std::vector<float>{0, 0, 0, 0}), false), std::invalid_argument);

    const Tensor x = Make(0.7, 50, 3);
    EXPECT_THROW((void)TwoNNDimension(x, 0.0), std::invalid_argument);
    EXPECT_THROW((void)TwoNNDimension(x, 1.0), std::invalid_argument);
    EXPECT_THROW((void)TwoNNDimension(Make(0.7, 4, 3), 0.9), std::invalid_argument);  // keeps 0 points
    EXPECT_THROW((void)TwoNNDimension(Tensor(Shape({3, 1}), &backend, std::vector<float>{1, 1, 2})), std::invalid_argument);
    // A 1D grid: every point's two neighbors are equally far.
    EXPECT_THROW((void)TwoNNDimension(Tensor(Shape({4, 1}), &backend, std::vector<float>{0, 1, 2, 3}), 0.5), std::invalid_argument);
}

TEST(SymmetricEigenvalues, AgreesWithSymmetricEigenInDouble) {
    CPUBackend backend;
    const std::vector<double> a = {4, 1, 0.5, 1, 3, 0.25, 0.5, 0.25, 1e-9};
    const std::vector<double> values = SymmetricEigenvalues(a, 3);
    const std::vector<float> a_float(a.begin(), a.end());
    const std::vector<float> reference = SymmetricEigen(Tensor(Shape({3, 3}), &backend, a_float)).values.to_host_vector();
    ASSERT_EQ(values.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(values[i], reference[i], 1e-5);
    EXPECT_GE(values[0], values[1]);
    EXPECT_GE(values[1], values[2]);
    // The trace and determinant hold to double precision, which a float copy couldn't give.
    EXPECT_NEAR(values[0] + values[1] + values[2], 4 + 3 + 1e-9, 1e-13);
    EXPECT_THROW((void)SymmetricEigenvalues({1, 2, 3}, 2), std::invalid_argument);
    EXPECT_THROW((void)SymmetricEigenvalues({1, 2, 3, 4}, 2), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
