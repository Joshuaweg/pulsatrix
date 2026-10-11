// INT-5: PCA, checked against scikit-learn's PCA (tools/generate_pca_reference_values.py), with
// projections, reconstructions, steering directions and the stability of a picture.
#include "pulsatrix/pca.hpp"

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

// scikit-learn 1.7.2's PCA (svd_solver="full"), three components, on Make(0.3, 40, 6, 0.5) and
// Make(1.1, 8, 30, 0.5).
const std::vector<double> kTallComponents = {
    0.21180318384454622, -0.33688821232126676, -0.30995953641355001, 0.72276190690434927,
    -0.094383318465276519, -0.46290154959997992, 0.17418417922969415, 0.34519974336862369,
    -0.37906292899889754, 0.24797605182904708, -0.55338417680158725, 0.5823075951144705,
    -0.20702288026115154, 0.63133916570976589, -0.56042970996618113, -0.017616501021694729,
    0.4013260823945981, -0.28826716697892107};
const std::vector<double> kTallVariance = {
    0.92822093076357082, 0.69948430450800714, 0.62733360221997603};
const std::vector<double> kTallRatio = {
    0.32365018683694902, 0.24389476507203831, 0.21873740489838986};
const std::vector<double> kTallProjection = {
    -0.42492571903952597, 0.37086538362736904, -0.20040438927130785, -0.73339114536456873,
    0.87672952241861568, 0.00042914328168297485, 1.2002914119657659, -0.9185937557895435,
    -0.25362222269725243};
const std::vector<double> kWideComponents = {
    -0.26753715123013255, -0.1049113581299624, 0.25725314147469819, 0.22669563559946604,
    -0.18264304671676232, -0.3247148936363618, 0.048053087001166672, 0.35875456494675151,
    0.11341109520249008, -0.30633722916959011, -0.24927511201001881, 0.17962954363005812,
    0.31189887916577452, -0.024757944595427947, -0.28432849333771537, -0.099361725911546489,
    0.19115581353977004, 0.15184462821238795, -0.08571037977993147, -0.1316832297597697,
    0.019826023623549505, 0.07631553517853118, -0.014368582000323626, -0.037005145953928523,
    0.049132977699553347, 0.046279153592774666, -0.078658941993257417, -0.098225158847417282,
    0.063671895225213535, 0.15425332880437415, -0.070639419350216642, 0.073542629553196348,
    0.10093631594408475, 0.031035056077614743, -0.044818011794863777, -0.088002536426769751,
    -0.063198689753632853, 0.054366873543282179, 0.15422041309651072, 0.062278828842464995,
    -0.16160179533768776, -0.20298196601316162, 0.059997430425479503, 0.28626511978892111,
    0.11563955391620336, -0.25270204737481039, -0.28408634087446405, 0.10067987417697807,
    0.35955351581581652, 0.10816469343166775, -0.29889201000201715, -0.27907694121330184,
    0.12729097940025513, 0.33446196685827612, 0.073476979246578436, -0.25616186797797819,
    -0.20890136937696385, 0.095087764431832075, 0.22320138020984878, 0.057996355659445477,
    -0.07226689572122659, -0.15711255803447233, 0.0061906624065477013, 0.14724696693896427,
    0.060190879631058923, -0.087641391844182778, -0.083837916790595143, 0.0077348682400619614,
    0.042519163168811082, 0.040943450017667748, 0.044941980792483634, -0.017853479357064793,
    -0.12538778061211403, -0.078001087305122008, 0.14159546675343782, 0.20340489163704906,
    -0.066760694182277355, -0.29356088583302836, -0.078721787542893246, 0.29762307809424976,
    0.23807591488315802, -0.20514036611082281, -0.34858734655762036, 0.048547773009482313,
    0.37223970264062867, 0.11706214946816236, -0.30916573381425821, -0.2420343419427646,
    0.18946893325458478, 0.30092502719106651};
const std::vector<double> kWideVariance = {
    5.2825748172721836, 4.321827585789916, 3.5090808394592896};
const std::vector<double> kWideRatio = {
    0.33236292475020612, 0.27191574305444388, 0.22078028448820838};
const std::vector<double> kWideProjection = {
    1.6807139286274013, 0.83148232681573842, -1.4441848731774978, -0.9843441760307774,
    -3.5937433711174096, -1.2369929240812643, 2.3761892008037395, 2.9665522484773392,
    0.39169725422754187};

class PcaTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The reference script's inputs: offset + sin(seed + 0.37 i + 1.13 a + 0.05 i a + 0.71 i²).
    Tensor Make(double seed, int64_t n, int64_t p, double offset = 0.0) {
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t a = 0; a < p; ++a) {
                const double di = static_cast<double>(i);
                const double da = static_cast<double>(a);
                v.push_back(static_cast<float>(offset + std::sin(seed + 0.37 * di + 1.13 * da + 0.05 * di * da + 0.71 * di * di)));
            }
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    // Uniform in [-1, 1) from mt19937's raw output, the same on every platform.
    static double Uniform(std::mt19937& rng) { return static_cast<double>(rng()) / 4294967296.0 * 2.0 - 1.0; }

    // Rows spread over a plane (spread 3 and 2 along two fixed orthonormal directions of a 50-wide
    // space), plus isotropic noise of the given size in every direction.
    Tensor Plane(int64_t n, double noise, uint32_t seed, std::vector<double>* plane = nullptr) {
        const int64_t p = 50;
        std::mt19937 rng(seed);
        std::vector<double> d1(p, 0.0), d2(p, 0.0);
        for (int64_t j = 0; j < p; ++j) {
            d1[static_cast<std::size_t>(j)] = j % 2 == 0 ? 1.0 : 0.0;
            d2[static_cast<std::size_t>(j)] = j % 4 == 1 ? 1.0 : (j % 4 == 3 ? -1.0 : 0.0);
        }
        for (double& v : d1) v /= 5.0;            // 25 entries of 1
        for (double& v : d2) v /= std::sqrt(25.0);  // 25 entries of +-1
        if (plane != nullptr) {
            plane->assign(d1.begin(), d1.end());
            plane->insert(plane->end(), d2.begin(), d2.end());
        }
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            const double a = 3.0 * Uniform(rng), b = 2.0 * Uniform(rng);
            for (int64_t j = 0; j < p; ++j) {
                v.push_back(static_cast<float>(a * d1[static_cast<std::size_t>(j)] + b * d2[static_cast<std::size_t>(j)] + noise * Uniform(rng)));
            }
        }
        return Tensor(Shape({n, p}), &backend, v);
    }
};

void ExpectNear(const std::vector<double>& actual, const std::vector<double>& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], tolerance) << "at " << i;
}

void ExpectNear(const Tensor& actual, const std::vector<double>& expected, double tolerance) {
    const std::vector<float> host = actual.to_host_vector();
    ExpectNear(std::vector<double>(host.begin(), host.end()), expected, tolerance);
}

// ---- reference values ----------------------------------------------------------------------------

TEST_F(PcaTest, MatchesScikitLearnOnTallData) {
    const Tensor x = Make(0.3, 40, 6, 0.5);
    const Pca pca = FitPca(x, 3);
    EXPECT_EQ(pca.width, 6);
    EXPECT_EQ(pca.samples, 40);
    EXPECT_EQ(pca.num_components(), 3);
    ExpectNear(pca.components, kTallComponents, 1e-10);
    ExpectNear(pca.explained_variance, kTallVariance, 1e-12);
    ExpectNear(pca.explained_variance_ratio, kTallRatio, 1e-12);
    const Tensor scores = PcaProject(pca, x);
    EXPECT_EQ(scores.shape().dim(0), 40);
    EXPECT_EQ(scores.shape().dim(1), 3);
    const std::vector<float> host = scores.to_host_vector();
    ExpectNear(std::vector<double>(host.begin(), host.begin() + 9), kTallProjection, 1e-6);
}

// Fewer rows than width: the n x n Gram matrix gives the components.
TEST_F(PcaTest, MatchesScikitLearnOnWideData) {
    const Tensor x = Make(1.1, 8, 30, 0.5);
    const Pca pca = FitPca(x, 3);
    ExpectNear(pca.components, kWideComponents, 1e-10);
    ExpectNear(pca.explained_variance, kWideVariance, 1e-12);
    ExpectNear(pca.explained_variance_ratio, kWideRatio, 1e-12);
    const std::vector<float> host = PcaProject(pca, x).to_host_vector();
    ExpectNear(std::vector<double>(host.begin(), host.begin() + 9), kWideProjection, 1e-5);
}

// ---- behavior ------------------------------------------------------------------------------------

TEST_F(PcaTest, ProjectionsHaveTheComponentsVariancesAndReconstructionInvertsThem) {
    const Tensor x = Make(0.3, 40, 6, 0.5);
    const Pca all = FitPca(x, 6);
    double ratio = 0;
    for (double r : all.explained_variance_ratio) ratio += r;
    EXPECT_NEAR(ratio, 1.0, 1e-12);

    // Each projection's sample variance is its component's; whitened, it is 1.
    for (bool whiten : {false, true}) {
        const std::vector<float> s = PcaProject(all, x, whiten).to_host_vector();
        for (int64_t c = 0; c < 6; ++c) {
            double sum = 0, sum2 = 0;
            for (int64_t i = 0; i < 40; ++i) {
                sum += s[static_cast<std::size_t>(i * 6 + c)];
                sum2 += static_cast<double>(s[static_cast<std::size_t>(i * 6 + c)]) * s[static_cast<std::size_t>(i * 6 + c)];
            }
            EXPECT_NEAR(sum / 40, 0.0, 1e-6);
            const double variance = sum2 / 39;
            EXPECT_NEAR(variance, whiten ? 1.0 : all.explained_variance[static_cast<std::size_t>(c)], 1e-5);
        }
    }

    // With every component, reconstruction gives the rows back.
    ExpectNear(PcaReconstruct(all, PcaProject(all, x)), [&] {
        const std::vector<float> h = x.to_host_vector();
        return std::vector<double>(h.begin(), h.end());
    }(), 1e-5);
}

// The falsifier: data spread over a plane, with noise, give that plane; noise alone gives a picture
// that two halves of the data don't agree on.
TEST_F(PcaTest, FindsAPlantedPlaneAndSplitHalvesAgreeOnlyWhenThereIsOne) {
    std::vector<double> plane;
    const Tensor x = Plane(600, 0.1, 3, &plane);
    const Pca pca = FitPca(x, 2);
    Pca truth;
    truth.width = 50;
    truth.components = plane;
    truth.explained_variance = {1.0, 1.0};
    EXPECT_GT(SubspaceOverlap(pca, truth, 2), 0.99);
    EXPECT_GT(pca.explained_variance_ratio[0] + pca.explained_variance_ratio[1], 0.9);
    EXPECT_GT(PcaSplitHalfOverlap(x, 2), 0.99);

    // Under noise as large as the plane, the plane still shows (0.92 here) ...
    EXPECT_GT(PcaSplitHalfOverlap(Plane(600, 1.0, 4), 2), 0.85);
    // ... but with noise alone the halves disagree (0.09 here): the picture is an accident.
    std::mt19937 rng(9);
    std::vector<float> iso(600 * 50);
    for (float& v : iso) v = static_cast<float>(Uniform(rng));
    EXPECT_LT(PcaSplitHalfOverlap(Tensor(Shape({600, 50}), &backend, iso), 2), 0.3);
}

TEST_F(PcaTest, AComponentIsAUnitSteeringDirection) {
    const Pca pca = FitPca(Make(0.3, 40, 6, 0.5), 2);
    for (int64_t c = 0; c < 2; ++c) {
        const std::vector<float> v = PcaComponent(pca, c);
        ASSERT_EQ(v.size(), 6u);
        double norm = 0;
        for (float t : v) norm += static_cast<double>(t) * t;
        EXPECT_NEAR(norm, 1.0, 1e-6);
    }
    EXPECT_THROW((void)PcaComponent(pca, 2), std::invalid_argument);
    EXPECT_THROW((void)PcaComponent(pca, -1), std::invalid_argument);
}

TEST_F(PcaTest, SubspaceOverlapIgnoresOrderAndRotationWithinTheSpan) {
    Pca a, b, c;
    a.width = b.width = c.width = 3;
    a.components = {1, 0, 0, 0, 1, 0};
    const double h = std::sqrt(0.5);
    b.components = {0, 1, 0, h, h, 0};  // the same plane, other vectors
    c.components = {0, 0, 1, 1, 0, 0};
    a.explained_variance = b.explained_variance = c.explained_variance = {2, 1};
    EXPECT_NEAR(SubspaceOverlap(a, b, 2), 1.0, 1e-12);
    EXPECT_NEAR(SubspaceOverlap(a, c, 2), 0.5, 1e-12);
    EXPECT_NEAR(SubspaceOverlap(a, c, 1), 0.0, 1e-12);
    EXPECT_THROW((void)SubspaceOverlap(a, b, 3), std::invalid_argument);
    Pca wide = a;
    wide.width = 2;
    EXPECT_THROW((void)SubspaceOverlap(a, wide, 1), std::invalid_argument);
}

TEST_F(PcaTest, RejectsWhatItCannotFit) {
    const Tensor x = Make(0.3, 40, 6, 0.5);
    EXPECT_THROW((void)FitPca(x, 0), std::invalid_argument);
    EXPECT_THROW((void)FitPca(x, 7), std::invalid_argument);
    EXPECT_THROW((void)FitPca(Make(0.3, 1, 6), 1), std::invalid_argument);
    EXPECT_THROW((void)FitPca(Tensor(Shape({4}), &backend, std::vector<float>{1, 2, 3, 4}), 1), std::invalid_argument);
    EXPECT_THROW((void)FitPca(Tensor(Shape({2, 2}), &backend, std::vector<float>{1, NAN, 3, 4}), 1), std::invalid_argument);
    // Centered, 8 rows span at most 7 directions.
    EXPECT_THROW((void)FitPca(Make(1.1, 8, 30, 0.5), 8), std::invalid_argument);
    // Rows that are all the same have no variance at all.
    EXPECT_THROW((void)FitPca(Tensor(Shape({3, 2}), &backend, std::vector<float>{1, 2, 1, 2, 1, 2}), 1), std::invalid_argument);

    const Pca pca = FitPca(x, 2);
    EXPECT_THROW((void)PcaProject(pca, Make(0.3, 5, 7)), std::invalid_argument);
    EXPECT_THROW((void)PcaReconstruct(pca, Make(0.3, 5, 3)), std::invalid_argument);
    EXPECT_THROW((void)PcaSplitHalfOverlap(Make(0.3, 3, 6), 1), std::invalid_argument);
}

TEST(SymmetricEigenInDouble, AgreesWithSymmetricEigen) {
    CPUBackend backend;
    const std::vector<double> a = {4, 1, 0.5, 1, 3, 0.25, 0.5, 0.25, 2};
    const DoubleEigenResult d = SymmetricEigenInDouble(a, 3);
    const EigenResult f = SymmetricEigen(Tensor(Shape({3, 3}), &backend, std::vector<float>(a.begin(), a.end())));
    const std::vector<float> values = f.values.to_host_vector(), vectors = f.vectors.to_host_vector();
    for (int64_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(d.values[static_cast<std::size_t>(j)], values[static_cast<std::size_t>(j)], 1e-5);
        for (int64_t i = 0; i < 3; ++i) {
            // SymmetricEigen's column j is SymmetricEigenInDouble's row j, with the same sign.
            EXPECT_NEAR(d.vectors[static_cast<std::size_t>(j * 3 + i)], vectors[static_cast<std::size_t>(i * 3 + j)], 1e-6);
        }
    }
    EXPECT_THROW((void)SymmetricEigenInDouble({1, 2, 3}, 2), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
