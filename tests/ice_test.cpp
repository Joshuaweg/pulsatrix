// CFS-1: ICE, centered and derivative ICE, two-feature partial dependence and grid construction.
// Reference values from tools/generate_ice_reference_values.py (scikit-learn 1.9.1
// partial_dependence on a fixed function, so both sides evaluate the same model).

#include "pulsatrix/ice.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/null_model_baseline.hpp"
#include "pulsatrix/pdp.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Det(int64_t n, int64_t mul, int64_t add, int64_t div) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>((i * mul + add) % 17 - 8) / static_cast<float>(div);
    }
    return v;
}

void ExpectNearAll(const std::vector<float>& actual, const std::vector<float>& expected, float tol = 1e-4f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol * std::max(1.0f, std::fabs(expected[i]))) << "element " << i;
    }
}

// ---- Reference values: tools/generate_ice_reference_values.py output, verbatim ----
const std::vector<float> kPercentileGrid = {
    -2.0f, -1.55555556f, -1.11111111f, -0.666666667f, -0.222222222f, 0.222222222f, 0.666666667f, 1.11111111f,
    1.55555556f, 2.0f,
};
const std::vector<float> kAverage = {
    1.01300788f, 0.987364948f, 0.92934233f, 0.842297733f, 0.73522836f, 0.621021628f, 0.513952196f, 0.426907688f,
    0.36888504f, 0.343242198f,
};
const std::vector<float> kIndividual = {
    3.28125f, 2.72569442f, 2.17013884f, 1.61458337f, 1.05902779f, 0.503472209f, -0.0520833731f, -0.607638955f,
    -1.16319442f, -1.71875f, 2.23589873f, 1.94785869f, 1.56267953f, 1.09043491f, 0.558115542f, 0.00438445807f,
    -0.527934909f, -1.00017953f, -1.38535869f, -1.67339873f, 1.44054747f, 1.42002273f, 1.20521998f, 0.816286504f,
    0.307203293f, -0.244703293f, -0.753786504f, -1.14271998f, -1.35752273f, -1.37804747f, 0.89519608f, 1.14218688f,
    1.09776056f, 0.79213804f, 0.306291074f, -0.243791074f, -0.72963804f, -1.03526056f, -1.07968688f, -0.83269608f,
    0.599844933f, 1.11435103f, 1.24030101f, 1.01798964f, 0.555378795f, 0.00712119043f, -0.455489635f, -0.677801013f,
    -0.551851034f, -0.0373448133f, -3.31002045f, -2.91299129f, -2.17597532f, -1.13423049f, 0.117776155f,
    1.44472384f, 2.69673061f, 3.73847532f, 4.47549152f, 4.87252045f, -3.10537171f, -2.44082713f, -1.53343475f,
    -0.408379018f, 0.866863906f, 2.19563603f, 3.47087908f, 4.59593487f, 5.50332737f, 6.16787148f, 5.31802702f,
    4.36119843f, 3.55007815f, 2.86955595f, 2.27914619f, 1.72085381f, 1.13044393f, 0.449921846f, -0.361198187f,
    -1.31802702f, 3.89767551f, 3.20836258f, 2.56761885f, 1.97040749f, 1.40323389f, 0.846766114f, 0.279592514f,
    -0.317618728f, -0.958362579f, -1.64767563f, 2.72732449f, 2.30552649f, 1.83515918f, 1.32125914f, 0.777321637f,
    0.222678348f, -0.321259141f, -0.835159183f, -1.30552649f, -1.72732437f, 1.80697298f, 1.65269065f, 1.35269976f,
    0.922110677f, 0.401409417f, -0.151409417f, -0.672110677f, -1.10269976f, -1.40269065f, -1.55697298f, 1.13662171f,
    1.2498548f, 1.12024021f, 0.772962272f, 0.275497168f, -0.275497168f, -0.772962272f, -1.12024021f, -1.2498548f,
    -1.13662171f, 0.716270447f, 1.09701908f, 1.13778079f, 0.873813748f, 0.399584949f, -0.149584949f, -0.623813748f,
    -0.887780786f, -0.847019076f, -0.466270447f, -3.31859493f, -3.05532336f, -2.40349555f, -1.40340638f,
    -0.163017705f, 1.16301775f, 2.40340638f, 3.40349555f, 4.0553236f, 4.31859493f, -3.23894596f, -2.70815945f,
    -1.8859551f, -0.802554727f, 0.461070031f, 1.78893006f, 3.05255461f, 4.13595486f, 4.95815945f, 5.48894596f,
    -2.90929747f, -2.11099505f, -1.11841452f, 0.0482968092f, 1.33515787f, 2.66484213f, 3.95170307f, 5.1184144f,
    6.11099529f, 6.90929747f, 4.57660103f, 3.7535305f, 3.02759838f, 2.38873196f, 1.8099401f, 1.2525599f,
    0.673768163f, 0.0349014997f, -0.691030502f, -1.51410127f, 3.28125f, 2.72569442f, 2.17013884f, 1.61458337f,
    1.05902779f, 0.503472209f, -0.0520833731f, -0.607638955f, -1.16319442f, -1.71875f, 2.23589873f, 1.94785869f,
    1.56267953f, 1.09043491f, 0.558115542f, 0.00438445807f, -0.527934909f, -1.00017953f, -1.38535869f, -1.67339873f,
    1.44054747f, 1.42002273f, 1.20521998f, 0.816286504f, 0.307203293f, -0.244703293f, -0.753786504f, -1.14271998f,
    -1.35752273f, -1.37804747f, 0.89519608f, 1.14218688f, 1.09776056f, 0.79213804f, 0.306291074f, -0.243791074f,
    -0.72963804f, -1.03526056f, -1.07968688f, -0.83269608f, 0.599844933f, 1.11435103f, 1.24030101f, 1.01798964f,
    0.555378795f, 0.00712119043f, -0.455489635f, -0.677801013f, -0.551851034f, -0.0373448133f, -3.31002045f,
    -2.91299129f, -2.17597532f, -1.13423049f, 0.117776155f, 1.44472384f, 2.69673061f, 3.73847532f, 4.47549152f,
    4.87252045f, -3.10537171f, -2.44082713f, -1.53343475f, -0.408379018f, 0.866863906f, 2.19563603f, 3.47087908f,
    4.59593487f, 5.50332737f, 6.16787148f, 5.31802702f, 4.36119843f, 3.55007815f, 2.86955595f, 2.27914619f,
    1.72085381f, 1.13044393f, 0.449921846f, -0.361198187f, -1.31802702f, 3.89767551f, 3.20836258f, 2.56761885f,
    1.97040749f, 1.40323389f, 0.846766114f, 0.279592514f, -0.317618728f, -0.958362579f, -1.64767563f, 2.72732449f,
    2.30552649f, 1.83515918f, 1.32125914f, 0.777321637f, 0.222678348f, -0.321259141f, -0.835159183f, -1.30552649f,
    -1.72732437f, 1.80697298f, 1.65269065f, 1.35269976f, 0.922110677f, 0.401409417f, -0.151409417f, -0.672110677f,
    -1.10269976f, -1.40269065f, -1.55697298f, 1.13662171f, 1.2498548f, 1.12024021f, 0.772962272f, 0.275497168f,
    -0.275497168f, -0.772962272f, -1.12024021f, -1.2498548f, -1.13662171f, 0.716270447f, 1.09701908f, 1.13778079f,
    0.873813748f, 0.399584949f, -0.149584949f, -0.623813748f, -0.887780786f, -0.847019076f, -0.466270447f,
};
const std::vector<float> kDistinctGrid = {
    -2.0f, -1.75f, -1.5f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f,
    2.0f,
};
const std::vector<float> kGrid2DFeature0 = {
    -2.0f, -1.2f, -0.4f, 0.4f, 1.2f, 2.0f,
};
const std::vector<float> kGrid2DFeature2 = {
    -1.7725f, -1.018f, -0.2635f, 0.491f, 1.2455f, 2.0f,
};
const std::vector<float> kAverage2D = {
    -1.82257247f, -1.36628854f, -0.340734303f, 1.25408995f, 3.41818476f, 6.15154934f, -0.400782108f, -0.548098147f,
    -0.126144022f, 0.865080416f, 2.42557526f, 4.55534029f, 0.926781118f, 0.175865069f, -0.00578082213f,
    0.381843567f, 1.33873808f, 2.86490297f, 2.21497512f, 0.86045897f, 0.0752130747f, -0.140762553f, 0.212532014f,
    1.13509691f, 3.54253817f, 1.58442223f, 0.19557628f, -0.623999417f, -0.874304831f, -0.555339932f, 4.96432829f,
    2.40261245f, 0.410166562f, -1.01300895f, -1.86691439f, -2.15154982f,
};

class IceTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::vector<Tensor> background;
    // f(x) = sin(x0) * x1 + 0.5 * x2^2 - x0 * x2, the generator's model, on one (3,) instance.
    std::function<Tensor(const Tensor&)> predict = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        const float y = std::sin(v[0]) * v[1] + 0.5f * v[2] * v[2] - v[0] * v[2];
        return Tensor(Shape({1}), &backend, {y});
    };

    void SetUp() override {
        const std::vector<float> all = Det(90, 5, 3, 4);
        for (size_t i = 0; i < 30; ++i) {
            background.emplace_back(Shape({3}), &backend, std::vector<float>(all.begin() + 3 * i, all.begin() + 3 * i + 3));
        }
    }
};

TEST_F(IceTest, GridMatchesScikitLearn) {
    ExpectNearAll(FeatureGrid(background, 0, 10), kPercentileGrid);
    ExpectNearAll(FeatureGrid(background, 1, 20), kDistinctGrid);   // 17 distinct values < 20
    ExpectNearAll(FeatureGrid(background, 2, 6), kGrid2DFeature2);  // interpolated percentiles
}

TEST_F(IceTest, CurvesAndAverageMatchScikitLearn) {
    IceResult r = ComputeIce(predict, background, 0, 0, kPercentileGrid);
    EXPECT_EQ(r.num_instances, 30);
    ExpectNearAll(r.curves, kIndividual);
    ExpectNearAll(r.partial_dependence(), kAverage);
    for (size_t i = 0; i < 30; ++i) {
        EXPECT_EQ(r.feature_values[i], background[i].read_element(0));
    }
}

// The average of the ICE curves is exactly what the existing PDP explainer computes.
TEST_F(IceTest, AverageEqualsPdpExplainer) {
    const std::vector<float> grid{-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    IceResult r = ComputeIce(predict, background, 2, 0, grid);
    Attribution pdp = PDP().explain(predict, background, 2, 0, -1.0f, 1.0f, 5, &backend);
    ExpectNearAll(r.partial_dependence(), pdp.values.to_host_vector(), 1e-5f);
}

TEST_F(IceTest, TwoFeaturePartialDependenceMatchesScikitLearn) {
    PartialDependence2D pd =
        ComputePartialDependence2D(predict, background, 0, 2, 0, kGrid2DFeature0, kGrid2DFeature2);
    ASSERT_EQ(pd.values.size(), 36u);
    // pulsatrix rows follow feature_y (2), scikit-learn's first axis follows feature 0.
    for (size_t r = 0; r < 6; ++r) {
        for (size_t c = 0; c < 6; ++c) {
            EXPECT_NEAR(pd.values[r * 6 + c], kAverage2D[c * 6 + r], 1e-4f) << "row " << r << ", col " << c;
        }
    }
}

// An additive model has parallel ICE curves: centered, they all coincide, and their slopes agree.
// With an interaction, centered curves fan out and d-ICE recovers the other feature.
TEST_F(IceTest, CenteredAndDerivativeIceSeparateAdditiveFromInteraction) {
    const std::vector<float> grid{-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    auto additive = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        return Tensor(Shape({1}), &backend, {3.0f * v[0] + v[1] * v[1]});
    };
    IceResult a = ComputeIce(additive, background, 0, 0, grid);
    const std::vector<float> ca = a.centered();
    const std::vector<float> da = a.derivative();
    for (size_t i = 1; i < a.curves.size() / grid.size(); ++i) {
        for (size_t k = 0; k < grid.size(); ++k) {
            EXPECT_NEAR(ca[i * grid.size() + k], ca[k], 1e-5f);
            EXPECT_NEAR(da[i * grid.size() + k], 3.0f, 1e-4f);
        }
    }

    auto product = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        return Tensor(Shape({1}), &backend, {v[0] * v[1]});
    };
    IceResult p = ComputeIce(product, background, 0, 0, grid);
    const std::vector<float> dp = p.derivative();
    for (size_t i = 0; i < background.size(); ++i) {
        for (size_t k = 0; k < grid.size(); ++k) {
            EXPECT_NEAR(dp[i * grid.size() + k], background[i].read_element(1), 1e-4f);
        }
    }
}

// numpy.gradient on an uneven grid: exact for a quadratic.
TEST_F(IceTest, DerivativeIsExactForAQuadraticOnAnUnevenGrid) {
    const std::vector<float> grid{-1.0f, 0.0f, 0.5f, 2.0f};
    auto square = [this](const Tensor& x) {
        const float v = x.read_element(0);
        return Tensor(Shape({1}), &backend, {v * v});
    };
    IceResult r = ComputeIce(square, {background[0]}, 0, 0, grid);
    const std::vector<float> d = r.derivative();
    EXPECT_NEAR(d[1], 0.0f, 1e-5f);
    EXPECT_NEAR(d[2], 1.0f, 1e-5f);
}

// Rule 1: ICE runs under NullModelBaseline like any analysis, and on a re-initialized model the
// curves change.
TEST_F(IceTest, RunsAgainstANullModel) {
    LinearModule model(3, 1, &backend);
    model.set_weight({2.0f, 0.0f, -1.0f});
    model.set_bias({0.5f});
    auto analysis = [&]() {
        auto f = [&](const Tensor& x) {
            Tensor batch = x;
            batch.reshape(Shape({1, 3}));
            return model.forward(batch);
        };
        return ComputeIce(f, background, 0, 0, {-1.0f, 0.0f, 1.0f});
    };
    auto report = NullModelBaseline(model, analysis, 7);
    EXPECT_NEAR(report.trained.derivative()[0], 2.0f, 1e-5f);
    EXPECT_NE(report.null_model.curves, report.trained.curves);
}

TEST_F(IceTest, RejectsBadInput) {
    EXPECT_THROW((void)ComputeIce(predict, {}, 0, 0, {1.0f}), std::invalid_argument);
    EXPECT_THROW((void)ComputeIce(predict, background, 3, 0, {1.0f}), std::invalid_argument);
    EXPECT_THROW((void)ComputeIce(predict, background, 0, 1, {1.0f}), std::invalid_argument);
    EXPECT_THROW((void)ComputeIce(predict, background, 0, 0, {}), std::invalid_argument);
    std::vector<Tensor> mixed = background;
    mixed.emplace_back(Shape({4}), &backend);
    EXPECT_THROW((void)ComputeIce(predict, mixed, 0, 0, {1.0f}), std::invalid_argument);
    EXPECT_THROW((void)ComputePartialDependence2D(predict, background, 1, 1, 0, {1.0f}, {1.0f}),
                 std::invalid_argument);
    EXPECT_THROW((void)FeatureGrid(background, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)FeatureGrid(background, 0, 10, 0.6f, 0.4f), std::invalid_argument);
    // 97 equal values and 3 others: 4 distinct values, but the 5th and 95th percentiles coincide.
    std::vector<Tensor> constant(97, Tensor(Shape({3}), &backend, {1.0f, 2.0f, 3.0f}));
    for (float v : {5.0f, 6.0f, 7.0f}) {
        constant.push_back(Tensor(Shape({3}), &backend, {v, 2.0f, 3.0f}));
    }
    EXPECT_THROW((void)FeatureGrid(constant, 0, 2), std::invalid_argument);
    IceResult r = ComputeIce(predict, background, 0, 0, {1.0f});
    EXPECT_THROW((void)r.derivative(), std::logic_error);
    EXPECT_THROW((void)r.centered(1), std::out_of_range);
}

}  // namespace
}  // namespace pulsatrix
