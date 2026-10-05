// CFS-4: Morris screening and Sobol indices. The analyzers are fed SALib 1.6's own samples and
// outputs (tools/generate_global_sensitivity_reference_values.py) and must reproduce its
// statistics; the samplers are checked end to end against the Ishigami function's analytic Sobol
// indices.

#include "pulsatrix/global_sensitivity.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

void ExpectNearAll(const std::vector<float>& actual, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol * std::max(1.0f, std::fabs(expected[i]))) << "element " << i;
    }
}

// ---- Reference values: tools/generate_global_sensitivity_reference_values.py output, verbatim ----
const std::vector<float> kMorrisSamples = {
    3.14159274f, 1.04719758f, -3.14159274f, -1.04719758f, 1.04719758f, -3.14159274f, -1.04719758f, -3.14159274f,
    -3.14159274f, -1.04719758f, -3.14159274f, 1.04719758f, 1.04719758f, -1.04719758f, 3.14159274f, 1.04719758f,
    -1.04719758f, -1.04719758f, 1.04719758f, 3.14159274f, -1.04719758f, -3.14159274f, 3.14159274f, -1.04719758f,
    -1.04719758f, -1.04719758f, -3.14159274f, -1.04719758f, 3.14159274f, -3.14159274f, 3.14159274f, 3.14159274f,
    -3.14159274f, 3.14159274f, 3.14159274f, 1.04719758f, 1.04719758f, 3.14159274f, 1.04719758f, 1.04719758f,
    -1.04719758f, 1.04719758f, -3.14159274f, -1.04719758f, 1.04719758f, -3.14159274f, -1.04719758f, -3.14159274f,
    -3.14159274f, 1.04719758f, -1.04719758f, 1.04719758f, 1.04719758f, -1.04719758f, 1.04719758f, 1.04719758f,
    3.14159274f, 1.04719758f, -3.14159274f, 3.14159274f, 3.14159274f, 3.14159274f, 3.14159274f, -1.04719758f,
    3.14159274f, 3.14159274f, -1.04719758f, 3.14159274f, -1.04719758f, -1.04719758f, -1.04719758f, -1.04719758f,
    1.04719758f, -1.04719758f, -3.14159274f, 1.04719758f, 3.14159274f, -3.14159274f, -3.14159274f, 3.14159274f,
    -3.14159274f, -3.14159274f, 3.14159274f, 1.04719758f, 1.04719758f, -3.14159274f, 3.14159274f, -3.14159274f,
    -3.14159274f, 3.14159274f, -3.14159274f, 1.04719758f, 3.14159274f, -3.14159274f, 1.04719758f, -1.04719758f,
    3.14159274f, -3.14159274f, -3.14159274f, -1.04719758f, -3.14159274f, -3.14159274f, -1.04719758f, -3.14159274f,
    1.04719758f, -1.04719758f, 1.04719758f, 1.04719758f, -3.14159274f, 1.04719758f, 3.14159274f, -3.14159274f,
    -3.14159274f, 3.14159274f, -3.14159274f, -3.14159274f, -1.04719758f, 1.04719758f, -3.14159274f, -1.04719758f,
    1.04719758f, -3.14159274f, -1.04719758f, 1.04719758f, 1.04719758f, -1.04719758f, -3.14159274f, 1.04719758f,
    -1.04719758f, -3.14159274f, 1.04719758f, 3.14159274f, -1.04719758f, -3.14159274f, -3.14159274f, -1.04719758f,
    -3.14159274f, 1.04719758f, 3.14159274f, -3.14159274f, 1.04719758f, 3.14159274f, 1.04719758f, 1.04719758f,
};
const std::vector<float> kMorrisOutputs = {
    5.24999952f, -4.05190182f, -9.30190277f, -0.970172048f, 14.5519028f, 6.22017241f, 0.970172048f, 9.79361374e-08f,
    -4.05190182f, -9.30190277f, -9.39000188e-07f, -9.79360237e-08f, 0.970172048f, 6.22017241f, 5.25000048f,
    5.25000143f, 5.25000048f, 6.22017241f, 14.5519028f, 9.30190277f, -9.39000188e-07f, -9.30190277f, -0.970172048f,
    4.27982855f, 14.5519028f, 9.30190277f, 9.39000302e-07f, 9.79361374e-08f, 9.30190277f, 9.39000302e-07f,
    5.25000143f, 5.25000048f, -9.39000188e-07f, -9.30190277f, -0.970172048f, 4.27982855f, 5.25000143f,
    9.39000302e-07f, 9.79361374e-08f, 0.970172048f, 0.970172048f, 6.22017241f, 5.25000048f, 5.25000143f,
    -9.30190277f, -0.970172048f, -9.79360237e-08f, 5.25000048f,
};
const std::vector<float> kMorrisMu = {
    7.70405527f, 1.31250015f, 4.1658655f,
};
const std::vector<float> kMorrisMuStar = {
    7.70405527f, 7.87500072f, 6.24879863f,
};
const std::vector<float> kMorrisSigma = {
    6.52665598f, 8.11012698f, 8.14017085f,
};
const std::vector<float> kMorrisMuStarConf = {
    3.49220092f, 2.44872269e-07f, 3.48179416f,
};
const std::vector<float> kSobolOutputs = {
    1.11411726f, 0.940432072f, 3.91420388f, 0.720862627f, 3.26652718f, 5.94397783f, 5.73091364f, 2.68980956f,
    10.6963711f, 6.16088915f, 9.45098114f, 5.33416939f, 5.9879508f, 8.16987705f, 2.71939158f, -1.98486495f,
    0.251299828f, -0.651324987f, -3.29587269f, 1.70784271f, -0.494397968f, 1.1819942f, 6.0759058f, -0.377470911f,
    7.48799515f, 7.15621948f, 5.656775f, 0.306595474f, 7.2117486f, -1.53409159f, 6.01498652f, 8.84216213f,
    7.16656494f, 5.10829258f, 6.53530312f, -2.84097981f, -2.11721897f, -4.84577131f, 0.94132483f, -0.899127841f,
    4.67918825f, 1.21901917f, -0.640355647f, 4.24803638f, -6.71469975f, 4.11257315f, 6.78695917f, 8.58872414f,
    2.21644783f, 7.24326324f, 0.628943622f, -0.308909595f, 0.718745351f, 5.08735132f, -2.41358137f, 6.98210573f,
    7.53420067f, 6.61583614f, 6.98214912f, 7.15770769f, 4.90284729f, 9.19867325f, -0.720267594f, 3.64435005f,
    4.89497852f, 1.62580371f, -1.62634516f, 5.19855785f, 0.789908826f, 2.79569006f, 2.39786291f, 2.25814128f,
    2.45701337f, 3.81770158f, 3.4664607f, 4.25192118f, 5.05514193f, 3.85171866f, 4.25189638f, 4.65493488f,
    -2.60679245f, 3.43522596f, -2.65487933f, 2.07410789f, 3.07390881f, 9.00009727f, 0.554267406f, 10.9188442f,
    6.41251373f, 4.29528856f, 7.52661657f, 7.36214399f, 3.5116396f, 7.48804569f, 3.31553936f, -0.785334587f,
    -0.78596133f, 3.85619473f, -3.75615406f, 0.882987022f, -0.248577207f, -0.35763678f, -0.00539711118f,
    -0.0147536546f, 0.157976687f, 7.86009693f, 7.42264128f, 8.13357544f, 12.9058743f, 10.812439f, 4.61813259f,
    3.019135f, 1.16134441f, 4.64498472f, -0.477581829f, 2.64297485f, 3.80142784f, 6.5891366f, 1.10449684f,
    13.074604f, 3.02522302f, 4.10964441f, 6.22394896f, 2.429281f, 7.78350544f, 3.63830614f, 3.06066871f,
    0.84258759f, 3.63911176f, 0.264277309f, 1.46681857f, 4.48313332f, 7.90631866f, 0.976922452f, 8.83179283f,
    3.88876247f, 3.35337925f, -2.0138464f, 5.86160088f, -0.182165831f, 5.12157965f, 6.1669488f, 3.5493474f,
    1.99442077f, 3.97208023f, 1.83053339f, 1.26700687f, 4.01114607f, 1.75316584f, 3.40177703f, 10.4301434f,
    -2.3089292f, 13.0085192f, 8.59682465f, 1.72981524f, -2.62936258f, 11.2362862f, -3.32067537f, 2.58303404f,
    4.39348936f, 5.88989878f, 7.804708f, -0.893242419f, 5.9824996f, 0.933739662f, 1.19560122f, -1.02398849f,
    7.73519659f, 1.81646705f, 4.90641069f, 5.67944384f, 2.35039878f, 4.53644466f, 2.21010852f, 0.306225687f,
    2.22891641f, 3.83292675f, 2.55787539f, 3.87393761f, 5.07891607f, 5.22478676f, 4.84750986f, 1.0973314f,
    4.91018057f, 0.21304971f, 1.44023669f, 1.90167212f, 5.14311266f, 3.15607023f, 9.69439507f, 0.607531905f,
    -1.19834983f, 1.81617475f, 0.397262394f, 0.616131783f, 5.25550652f, 7.35548639f, 1.88080883f, 3.84647655f,
    4.35161495f, -0.831784844f, 3.40433931f, 2.45532465f, -0.928973854f, 6.97664738f, 8.55463791f, 4.71146011f,
    3.1112895f, 7.41977167f, 0.685281098f, 6.20646429f, 6.18727732f, 5.47755003f, 7.41145515f, 6.62925482f,
    1.09965444f, 1.32277489f, 1.48039889f, 1.10935605f, 1.70575571f, -7.42023993f, -6.61573124f, -2.6992774f,
    -6.0463233f, -0.645522118f, 14.463994f, 14.7773714f, 8.85907936f, 6.63170481f, 1.06159675f, 7.81045914f,
    6.2755146f, 7.67903662f, 14.4916172f, 3.29004073f, -1.04786277f, 0.675804317f, -1.10905278f, -0.9472844f,
    0.572118819f, 3.02032208f, 4.60053396f, 3.83702207f, 5.04782343f, 6.42761135f, 3.76838732f, 1.92799687f,
    3.43977737f, 9.23575306f, 3.41224146f, 3.3005662f, 1.60135508f, 7.55628061f, 3.32674837f, 5.8217535f,
    3.34764171f, 5.17043495f, -0.440893471f, -2.38291407f, 6.35201073f, 3.45972943f, 5.84901857f, 5.00852489f,
    4.27955103f, 6.11555243f, 6.13657856f, -1.90608406f, 6.46974754f, 3.82673049f, 1.56733871f, 1.6367532f,
    1.99363732f, 5.70505285f, 1.35012901f, 5.6536932f, 5.96225786f, 4.41735125f, 2.58344007f, 5.89727354f,
    -1.23780453f, 0.179564893f, 0.561528623f, 2.1197648f, -1.56773448f, 2.21464515f, 6.42384434f, 6.9140687f,
    5.07705498f, 6.47470236f, 5.74738121f, 5.16538048f, 0.153526545f, 3.44898343f, 5.11347771f, -1.02571416f,
    -0.833711743f, 8.98248196f, 2.80028319f, 1.65708876f, 7.35900879f, 1.48991263f, 2.72958183f, -0.317165494f,
    0.62749809f, 1.36861551f, 5.55438614f, 4.26224804f, 7.76375103f, 5.5676651f, 6.46637774f, 8.1058445f,
    8.08681488f, 2.35900736f, 8.27266979f, 2.50494814f, -0.532689989f, -0.755056739f, 5.70674133f, -0.0212890804f,
    6.08452177f,
};
const std::vector<float> kSobolS1 = {
    0.322605691f, 0.474903237f, 0.048150559f,
};
const std::vector<float> kSobolST = {
    0.506982487f, 0.476270071f, 0.236404664f,
};
const std::vector<float> kSobolS1Conf = {
    0.175716356f, 0.245830697f, 0.148692402f,
};
const std::vector<float> kSobolSTConf = {
    0.339127508f, 0.174528737f, 0.118377195f,
};

constexpr float kPi = 3.14159265358979f;

class GlobalSensitivityTest : public ::testing::Test {
protected:
    CPUBackend backend;
    // Ishigami on a (1, 3) input.
    std::function<Tensor(const Tensor&)> ishigami = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        const float s1 = std::sin(v[0]);
        return Tensor(Shape({1, 1}), &backend, {s1 + 7.0f * std::sin(v[1]) * std::sin(v[1]) + 0.1f * std::pow(v[2], 4.0f) * s1});
    };
    GlobalSensitivityProblem Problem() {
        return {Tensor(Shape({1, 3}), &backend), {0, 1, 2}, {-kPi, -kPi, -kPi}, {kPi, kPi, kPi}};
    }
};

TEST_F(GlobalSensitivityTest, MorrisAnalyzerMatchesSalibOnItsSamples) {
    std::vector<std::vector<float>> rows;
    for (size_t i = 0; i < kMorrisSamples.size(); i += 3) {
        rows.push_back({kMorrisSamples[i], kMorrisSamples[i + 1], kMorrisSamples[i + 2]});
    }
    MorrisResult r = AnalyzeMorris(rows, kMorrisOutputs);
    EXPECT_EQ(r.num_trajectories, 12);
    ExpectNearAll(r.mu, kMorrisMu, 1e-5f);
    ExpectNearAll(r.mu_star, kMorrisMuStar, 1e-5f);
    ExpectNearAll(r.sigma, kMorrisSigma, 1e-5f);
    // Different bootstrap generators: the same size, not the same digits.
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(r.mu_star_conf[j], kMorrisMuStarConf[j], 0.25f * kMorrisMuStarConf[j] + 1e-3f) << "feature " << j;
    }
}

TEST_F(GlobalSensitivityTest, SobolAnalyzerMatchesSalibOnItsSamples) {
    SobolOptions o;
    o.num_resamples = 200;
    SobolResult r = AnalyzeSobol(kSobolOutputs, 3, o);
    EXPECT_EQ(r.num_samples, 64);
    ExpectNearAll(r.first_order, kSobolS1, 1e-5f);
    ExpectNearAll(r.total_order, kSobolST, 1e-5f);
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(r.first_order_conf[j], kSobolS1Conf[j], 0.3f * kSobolS1Conf[j]) << "feature " << j;
        EXPECT_NEAR(r.total_order_conf[j], kSobolSTConf[j], 0.3f * kSobolSTConf[j]) << "feature " << j;
    }
}

// Ishigami (a = 7, b = 0.1) has known indices: S1 = (0.3139, 0.4424, 0), ST = (0.5576, 0.4424,
// 0.2437). x3 matters only through its interaction with x1, so its first order is 0 and its
// total order is not.
TEST_F(GlobalSensitivityTest, SobolSamplerRecoversIshigamisAnalyticIndices) {
    SobolOptions o;
    o.num_samples = 8192;
    o.seed = 3;
    SobolResult r = ComputeSobol(ishigami, Problem(), 0, o);
    EXPECT_EQ(r.features, (std::vector<int64_t>{0, 1, 2}));
    const std::vector<float> s1{0.3139f, 0.4424f, 0.0f}, st{0.5576f, 0.4424f, 0.2437f};
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(r.first_order[j], s1[j], 0.04f) << "feature " << j;
        EXPECT_NEAR(r.total_order[j], st[j], 0.04f) << "feature " << j;
        // The analytic value is inside (or very near) the reported interval.
        EXPECT_LT(std::fabs(r.total_order[j] - st[j]), 2.0f * r.total_order_conf[j] + 0.01f);
    }
}

TEST_F(GlobalSensitivityTest, MorrisSamplerBuildsValidTrajectoriesAndRanksFeatures) {
    MorrisOptions o;
    o.num_trajectories = 50;
    o.seed = 5;
    const auto rows = MorrisSample(Problem(), o);
    ASSERT_EQ(rows.size(), 50u * 4u);
    // Values sit on the 4-level grid over [-pi, pi].
    for (const auto& row : rows) {
        for (float v : row) {
            const float level = (v + kPi) / (2.0f * kPi) * 3.0f;
            EXPECT_NEAR(level, std::round(level), 1e-4f);
        }
    }
    // A linear model: every elementary effect of x_j is its coefficient times the range times the
    // step's share of it, so sigma is 0 and mu = mu*.
    auto linear = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        return Tensor(Shape({1, 1}), &backend, {3.0f * v[0] - 1.0f * v[1] + 0.0f * v[2]});
    };
    MorrisResult r = ComputeMorris(linear, Problem(), 0, o);
    EXPECT_NEAR(r.mu[0], 3.0f * 2.0f * kPi, 1e-3f);
    EXPECT_NEAR(r.mu[1], -1.0f * 2.0f * kPi, 1e-3f);
    EXPECT_NEAR(r.mu_star[1], 2.0f * kPi, 1e-3f);
    EXPECT_NEAR(r.sigma[0], 0.0f, 1e-3f);
    EXPECT_NEAR(r.mu_star[2], 0.0f, 1e-6f);
    // The same seed gives the same design.
    EXPECT_EQ(MorrisSample(Problem(), o), rows);
}

TEST_F(GlobalSensitivityTest, UnvariedFeaturesKeepTheBaseValue) {
    GlobalSensitivityProblem p{Tensor(Shape({1, 3}), &backend, {0.0f, 2.0f, 0.0f}), {0}, {0.0f}, {1.0f}};
    auto product = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        return Tensor(Shape({1, 1}), &backend, {v[0] * v[1]});
    };
    MorrisResult r = ComputeMorris(product, p, 0, MorrisOptions{});
    EXPECT_NEAR(r.mu[0], 2.0f, 1e-5f);  // slope 2 over a range of 1
    EXPECT_EQ(r.features, (std::vector<int64_t>{0}));
}

TEST_F(GlobalSensitivityTest, RejectsBadInput) {
    GlobalSensitivityProblem p = Problem();
    p.features = {0, 0, 1};
    EXPECT_THROW((void)MorrisSample(p), std::invalid_argument);
    p = Problem();
    p.upper[1] = p.lower[1];
    EXPECT_THROW((void)ComputeSobol(ishigami, p, 0), std::invalid_argument);
    p = Problem();
    p.features = {0, 1, 3};
    EXPECT_THROW((void)MorrisSample(p), std::invalid_argument);
    MorrisOptions odd;
    odd.num_levels = 3;
    EXPECT_THROW((void)MorrisSample(Problem(), odd), std::invalid_argument);
    EXPECT_THROW((void)AnalyzeSobol({1.0f, 2.0f, 3.0f, 4.0f}, 3), std::invalid_argument);
    EXPECT_THROW((void)AnalyzeMorris({{0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}}, {0.0f, 1.0f, 2.0f}),
                 std::invalid_argument);  // a step that changes two features
    EXPECT_THROW((void)ComputeSobol(ishigami, Problem(), 1), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
