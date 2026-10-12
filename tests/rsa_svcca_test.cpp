// INT-7: RSA and SVCCA, checked against independent numpy implementations of scipy's and Raghu et
// al.'s definitions (tools/generate_rsa_svcca_reference_values.py), their invariances, the
// permutation test's calibration, and SVCCA's known upward bias.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/representation_similarity.hpp"

namespace pulsatrix {
namespace {

class RsaSvccaTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The reference script's inputs: x[i][a] = sin(seed + 0.37 i + 1.13 a + 0.05 i a), float32.
    Tensor Make(double seed, int64_t n, int64_t p) {
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t a = 0; a < p; ++a) v.push_back(static_cast<float>(std::sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a)));
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    // The reference script's low_rank(): r strong directions under weak ones.
    Tensor LowRank(double seed, int64_t n, int64_t p, int64_t r) {
        const std::vector<float> a = Make(seed, n, r).to_host_vector();
        const std::vector<float> b = Make(seed + 5.0, r, p).to_host_vector();
        const std::vector<float> c = Make(seed + 9.0, n, p).to_host_vector();
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < p; ++j) {
                double s = 0;
                for (int64_t k = 0; k < r; ++k) s += static_cast<double>(a[static_cast<std::size_t>(i * r + k)]) * b[static_cast<std::size_t>(k * p + j)];
                v.push_back(static_cast<float>(s + 0.1 * c[static_cast<std::size_t>(i * p + j)]));
            }
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    std::vector<float> GaussianValues(int64_t count, std::mt19937& rng) {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(count));
        for (float& x : v) x = normal(rng);
        return v;
    }

    Tensor Gaussian(int64_t n, int64_t p, uint32_t seed) {
        std::mt19937 rng(seed);
        return Tensor(Shape({n, p}), &backend, GaussianValues(n * p, rng));
    }

    // x (n × p) times a p × q matrix.
    Tensor Times(const Tensor& x, const std::vector<float>& w, int64_t q) {
        const int64_t n = x.shape().dim(0), p = x.shape().dim(1);
        const std::vector<float> h = x.to_host_vector();
        std::vector<float> out(static_cast<std::size_t>(n * q), 0.0f);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t b = 0; b < q; ++b) {
                double s = 0;
                for (int64_t a = 0; a < p; ++a) s += static_cast<double>(h[static_cast<std::size_t>(i * p + a)]) * w[static_cast<std::size_t>(a * q + b)];
                out[static_cast<std::size_t>(i * q + b)] = static_cast<float>(s);
            }
        }
        return Tensor(Shape({n, q}), &backend, out);
    }

    static void ExpectNear(const std::vector<double>& got, const std::vector<double>& want, double tolerance) {
        ASSERT_EQ(got.size(), want.size());
        for (std::size_t k = 0; k < want.size(); ++k) EXPECT_NEAR(got[k], want[k], tolerance) << "at " << k;
    }
};

TEST_F(RsaSvccaTest, RdmsMatchScipysDefinitions) {
    const Tensor x = Make(0.3, 5, 4);
    ExpectNear(RepresentationalDissimilarity(x, RdmMetric::Correlation).values,
               {0.092030482062739338, 0.27131855219252543, 0.52058841764884833, 0.87817934562557032, 0.051458493857817644,
                0.19720006336484308, 0.47386071135225738, 0.049715262132835392, 0.23166073673132137, 0.070567850953407896},
               1e-12);
    ExpectNear(RepresentationalDissimilarity(x, RdmMetric::Euclidean).values,
               {0.6516888000599117, 1.1878961115458928, 1.5978701207086572, 1.9165959152808407, 0.57050421618442682,
                1.0626911396664724, 1.5149412818296495, 0.54098854973611721, 1.0889268198520046, 0.59208229309492089},
               1e-12);
    const Rdm cosine = RepresentationalDissimilarity(x, RdmMetric::Cosine);
    ExpectNear(cosine.values,
               {0.10733052139675225, 0.33146668339268093, 0.61726216867223971, 0.97473508391989283, 0.06868909171961457,
                0.24666528841738533, 0.544253967804741, 0.059754436488521079, 0.25708717488923205, 0.074418922215137018},
               1e-12);
    // pdist's order: (0,1) (0,2) (0,3) (0,4) (1,2) ...; at() reads it either way round.
    EXPECT_EQ(cosine.at(1, 2), cosine.values[4]);
    EXPECT_EQ(cosine.at(2, 1), cosine.values[4]);
    EXPECT_EQ(cosine.at(3, 3), 0.0);
    EXPECT_THROW((void)cosine.at(0, 5), std::out_of_range);
}

TEST_F(RsaSvccaTest, RsaMatchesReference) {
    const Tensor x = Make(0.0, 9, 6);
    const Tensor y = Make(2.0, 9, 4);
    const struct {
        RdmMetric metric;
        double spearman, pearson;
    } cases[] = {
        {RdmMetric::Correlation, 0.95675675675675675, 0.96926054108566173},
        {RdmMetric::Euclidean, 0.93513513513513513, 0.96470094525566508},
        {RdmMetric::Cosine, 0.95855855855855854, 0.97097618446779543},
    };
    for (const auto& c : cases) {
        EXPECT_NEAR(Rsa(x, y, c.metric, RdmComparison::Spearman), c.spearman, 1e-12);
        EXPECT_NEAR(Rsa(x, y, c.metric, RdmComparison::Pearson), c.pearson, 1e-12);
    }
}

TEST_F(RsaSvccaTest, SpearmanGivesTiesTheirAverageRank) {
    // The reference's ranks of {3, 1, 3, 2, 3, 1}: {5, 1.5, 5, 3, 5, 1.5}.
    const Rdm tied{4, {3, 1, 3, 2, 3, 1}};
    const Rdm ordered{4, {1, 2, 3, 4, 5, 6}};
    const std::vector<double> ranks = {5, 1.5, 5, 3, 5, 1.5};
    const Rdm as_ranks{4, ranks};
    EXPECT_NEAR(CompareRdms(tied, ordered, RdmComparison::Spearman), CompareRdms(as_ranks, ordered, RdmComparison::Pearson), 1e-15);
    // A constant RDM correlates with nothing.
    EXPECT_TRUE(std::isnan(CompareRdms(Rdm{4, std::vector<double>(6, 2.0)}, ordered)));
}

TEST_F(RsaSvccaTest, RsaIgnoresWhatItShould) {
    const Tensor x = Gaussian(20, 6, 1);
    // A non-orthogonal invertible map changes Euclidean distances, a rotation and scale only
    // scales them, and a monotone transform of the distances leaves their ranks alone.
    const double c = std::cos(0.7), s = std::sin(0.7);
    std::vector<float> rotate(36, 0.0f);
    for (int a = 0; a < 6; ++a) rotate[static_cast<std::size_t>(a * 6 + a)] = 3.0f;
    rotate[0] = rotate[7] = static_cast<float>(3 * c);
    rotate[1] = static_cast<float>(3 * s);
    rotate[6] = static_cast<float>(-3 * s);
    const Tensor rotated = Times(x, rotate, 6);
    EXPECT_NEAR(Rsa(x, rotated, RdmMetric::Euclidean, RdmComparison::Pearson), 1.0, 1e-6);
    EXPECT_NEAR(Rsa(x, rotated, RdmMetric::Euclidean, RdmComparison::Spearman), 1.0, 1e-12);
    Rdm squared = RepresentationalDissimilarity(x, RdmMetric::Euclidean);
    const Rdm plain = squared;
    for (double& d : squared.values) d = d * d;
    EXPECT_NEAR(CompareRdms(plain, squared, RdmComparison::Spearman), 1.0, 1e-12);
    EXPECT_LT(CompareRdms(plain, squared, RdmComparison::Pearson), 0.999);
}

TEST_F(RsaSvccaTest, PermutationTestFindsSharedStructureAndIsCalibrated) {
    // 24 inputs in 4 categories. x and y place the categories by unrelated linear maps of one
    // latent, each with its own noise: the same geometry, nothing else shared.
    const int64_t n = 24, d = 8;
    std::mt19937 rng(7);
    const std::vector<float> latent = GaussianValues(4 * d, rng);
    auto view = [&](uint32_t seed) {
        std::mt19937 r(seed);
        const std::vector<float> map = GaussianValues(d * 10, r);
        const std::vector<float> noise = GaussianValues(n * 10, r);
        std::vector<float> v(static_cast<std::size_t>(n * 10), 0.0f);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t b = 0; b < 10; ++b) {
                float t = 0.5f * noise[static_cast<std::size_t>(i * 10 + b)];
                for (int64_t a = 0; a < d; ++a) t += latent[static_cast<std::size_t>((i % 4) * d + a)] * map[static_cast<std::size_t>(a * 10 + b)];
                v[static_cast<std::size_t>(i * 10 + b)] = t;
            }
        }
        return Tensor(Shape({n, 10}), &backend, v);
    };
    const Rdm x = RepresentationalDissimilarity(view(11));
    const Rdm y = RepresentationalDissimilarity(view(12));
    const RsaTestResult shared = RsaPermutationTest(x, y, RdmComparison::Spearman, 199, 3);
    EXPECT_NEAR(shared.similarity, CompareRdms(x, y), 1e-12);
    std::printf("shared categories: RSA %.3f, p %.4f\n", shared.similarity, shared.p_value);
    EXPECT_GT(shared.similarity, 0.5);
    EXPECT_DOUBLE_EQ(shared.p_value, 1.0 / 200.0);
    EXPECT_EQ(shared.permutations, 199);
    // The same seed gives the same p-value.
    EXPECT_EQ(RsaPermutationTest(x, y, RdmComparison::Spearman, 199, 3).p_value, shared.p_value);

    // Unrelated representations: p below 0.05 about 1 time in 20.
    int significant = 0;
    for (uint32_t trial = 0; trial < 40; ++trial) {
        const Rdm a = RepresentationalDissimilarity(Gaussian(n, 10, 100 + trial));
        const Rdm b = RepresentationalDissimilarity(Gaussian(n, 10, 200 + trial));
        const RsaTestResult r = RsaPermutationTest(a, b, RdmComparison::Spearman, 199, trial);
        EXPECT_GE(r.p_value, 1.0 / 200.0);
        if (r.p_value < 0.05) ++significant;
    }
    std::printf("unrelated: p < 0.05 in %d of 40 trials\n", significant);
    EXPECT_LE(significant, 6);
}

TEST_F(RsaSvccaTest, RsaRejectsBadInputs) {
    std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 7, 7};  // the third row is constant
    const Tensor constant_row(Shape({3, 3}), &backend, v);
    EXPECT_THROW((void)RepresentationalDissimilarity(constant_row, RdmMetric::Correlation), std::invalid_argument);
    EXPECT_NO_THROW((void)RepresentationalDissimilarity(constant_row, RdmMetric::Cosine));
    v[6] = v[7] = v[8] = 0;
    EXPECT_THROW((void)RepresentationalDissimilarity(Tensor(Shape({3, 3}), &backend, v), RdmMetric::Cosine), std::invalid_argument);
    EXPECT_THROW((void)RepresentationalDissimilarity(Make(0, 2, 3)), std::invalid_argument);
    EXPECT_THROW((void)Rsa(Make(0, 5, 3), Make(1, 6, 3)), std::invalid_argument);
    const Rdm a = RepresentationalDissimilarity(Make(0, 5, 3));
    EXPECT_THROW((void)CompareRdms(a, RepresentationalDissimilarity(Make(0, 6, 3))), std::invalid_argument);
    EXPECT_THROW((void)CompareRdms(a, Rdm{5, {1, 2, 3}}), std::invalid_argument);
    Rdm with_nan = a;
    with_nan.values[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)CompareRdms(a, with_nan), std::invalid_argument);
    EXPECT_THROW((void)RsaPermutationTest(a, a, RdmComparison::Spearman, 0), std::invalid_argument);
}

TEST_F(RsaSvccaTest, CanonicalCorrelationsAndSvccaMatchReference) {
    ExpectNear(CanonicalCorrelations(Make(0.0, 30, 6), Make(2.0, 30, 4)),
               {0.9999992651472468, 0.99999640055349859, 0.99615395567797138, 0.97573103004449635}, 1e-9);
    ExpectNear(CanonicalCorrelations(Make(0.5, 25, 5), Make(1.5, 25, 5)),
               {0.99999999551672258, 0.9999999343154824, 0.99990173714693609, 0.99894121380243339, 0.59403243398141437}, 1e-9);

    const SvccaResult r = Svcca(Make(0.25, 60, 14), Make(1.75, 60, 10));
    EXPECT_EQ(r.x_directions, 10);
    EXPECT_EQ(r.y_directions, 8);
    ExpectNear(r.correlations,
               {0.9994570338461215, 0.99674093145190978, 0.9805624411007392, 0.7988826025488821, 0.40379163169781412,
                0.16600946861565369, 0.070735317160533745, 0.063690492015877856},
               1e-9);
    EXPECT_NEAR(r.similarity, 0.55998373980469152, 1e-9);

    // Wider than the inputs, through the Gram matrices, 6 directions of each.
    const Tensor wx = LowRank(1.0, 40, 50, 6);
    const Tensor wy = LowRank(3.0, 40, 30, 6);
    const SvccaResult wide = Svcca(wx, wy, SvccaOptions{0.99, 6});
    EXPECT_EQ(wide.x_directions, 6);
    EXPECT_EQ(wide.y_directions, 6);
    ExpectNear(wide.correlations,
               {0.99949761709822904, 0.99930424105765381, 0.99741519507530296, 0.98869967077441612, 0.88434634105542176,
                0.69655138948748474},
               1e-9);
    EXPECT_NEAR(wide.similarity, 0.92763574242475133, 1e-9);
    const SvccaResult by_rule = Svcca(wx, wy);
    EXPECT_EQ(by_rule.x_directions, 5);
    EXPECT_EQ(by_rule.y_directions, 4);
    // make()'s wide matrices have nearly flat spectra: the rule would keep 33 and 20 of 40
    // inputs' directions, and some correlations would be 1 whatever the data, so it refuses.
    EXPECT_THROW((void)Svcca(Make(1.0, 40, 50), Make(3.0, 40, 30)), std::invalid_argument);
}

TEST_F(RsaSvccaTest, SvccaIsInvariantToInvertibleMapsAndCkaIsNot) {
    const Tensor x = Gaussian(80, 5, 4);
    // Stretch one axis ten times and shear another: invertible, far from orthogonal.
    std::vector<float> map(25, 0.0f);
    for (int a = 0; a < 5; ++a) map[static_cast<std::size_t>(a * 5 + a)] = 1.0f;
    map[0] = 10.0f;
    map[1 * 5 + 2] = 3.0f;
    const Tensor y = Times(x, map, 5);
    for (double rho : CanonicalCorrelations(x, y)) EXPECT_NEAR(rho, 1.0, 1e-6);
    EXPECT_NEAR(Svcca(x, y, SvccaOptions{1.0}).similarity, 1.0, 1e-6);
    std::printf("stretched and sheared: CKA %.3f\n", LinearCka(x, y));
    EXPECT_LT(LinearCka(x, y), 0.6);

    // A shared 3-dimensional signal among unshared noise: three correlations near 1, the rest low.
    std::mt19937 rng(9);
    const std::vector<float> signal = GaussianValues(400 * 3, rng);
    auto embed = [&](uint32_t seed) {
        std::mt19937 r(seed);
        const std::vector<float> noise = GaussianValues(400 * 6, r);
        std::vector<float> v(static_cast<std::size_t>(400 * 6));
        for (int64_t i = 0; i < 400; ++i) {
            for (int64_t b = 0; b < 6; ++b) {
                v[static_cast<std::size_t>(i * 6 + b)] = b < 3 ? signal[static_cast<std::size_t>(i * 3 + b)] + 0.05f * noise[static_cast<std::size_t>(i * 6 + b)]
                                                               : noise[static_cast<std::size_t>(i * 6 + b)];
            }
        }
        return Tensor(Shape({400, 6}), &backend, v);
    };
    const std::vector<double> rho = CanonicalCorrelations(embed(20), embed(21));
    ASSERT_EQ(rho.size(), 6u);
    for (int k = 0; k < 3; ++k) EXPECT_GT(rho[static_cast<std::size_t>(k)], 0.99);
    for (int k = 3; k < 6; ++k) EXPECT_LT(rho[static_cast<std::size_t>(k)], 0.25);
}

TEST_F(RsaSvccaTest, SvccaOfUnrelatedRepresentationsNeedsManyInputs) {
    // The trap: canonical correlations are fitted to the inputs. Two unrelated 20-wide
    // representations of 60 inputs score far above 0, where debiased CKA doesn't; it takes
    // thousands of inputs to bring SVCCA down.
    const SvccaResult few = Svcca(Gaussian(60, 20, 30), Gaussian(60, 20, 31), SvccaOptions{0.99, 20});
    const SvccaResult many = Svcca(Gaussian(2000, 20, 30), Gaussian(2000, 20, 31), SvccaOptions{0.99, 20});
    std::printf("unrelated SVCCA: 60 inputs %.3f, 2000 inputs %.3f; debiased CKA at 60: %.3f\n", few.similarity, many.similarity,
                LinearCka(Gaussian(60, 20, 30), Gaussian(60, 20, 31)));
    EXPECT_GT(few.similarity, 0.4);
    EXPECT_LT(many.similarity, 0.15);
    EXPECT_LT(std::fabs(LinearCka(Gaussian(60, 20, 30), Gaussian(60, 20, 31))), 0.1);
}

TEST_F(RsaSvccaTest, SvccaRejectsBadInputs) {
    EXPECT_THROW((void)CanonicalCorrelations(Make(0, 10, 3), Make(1, 11, 3)), std::invalid_argument);
    EXPECT_THROW((void)CanonicalCorrelations(Make(0, 8, 4), Make(1, 8, 4)), std::invalid_argument);  // 4 + 4 directions, 8 inputs
    const Tensor constant(Shape({10, 2}), &backend, std::vector<float>(20, 1.5f));
    EXPECT_THROW((void)CanonicalCorrelations(constant, Make(1, 10, 2)), std::invalid_argument);
    std::vector<float> v(20, 0.0f);
    v[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)CanonicalCorrelations(Tensor(Shape({10, 2}), &backend, v), Make(1, 10, 2)), std::invalid_argument);
    EXPECT_THROW((void)Svcca(Make(0, 20, 3), Make(1, 20, 3), SvccaOptions{0.0}), std::invalid_argument);
    EXPECT_THROW((void)Svcca(Make(0, 20, 3), Make(1, 20, 3), SvccaOptions{1.5}), std::invalid_argument);
    EXPECT_THROW((void)Svcca(Make(0, 20, 3), Make(1, 20, 3), SvccaOptions{0.99, -1}), std::invalid_argument);
    // directions is capped at each representation's rank.
    const SvccaResult capped = Svcca(Make(0, 20, 3), Make(1, 20, 2), SvccaOptions{0.99, 10});
    EXPECT_EQ(capped.x_directions, 3);
    EXPECT_EQ(capped.y_directions, 2);
}

}  // namespace
}  // namespace pulsatrix
