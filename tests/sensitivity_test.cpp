// CFS-3: one-at-a-time local sensitivity and occlusion. Occlusion and the bounds' statistics
// are checked against tools/generate_sensitivity_reference_values.py (Captum 0.9.0, numpy).

#include "pulsatrix/sensitivity.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Det(int64_t n, int64_t mul, int64_t add, int64_t div) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>((i * mul + add) % 17 - 8) / static_cast<float>(div);
    }
    return v;
}

void ExpectNearAll(const std::vector<float>& actual, const std::vector<float>& expected, float tol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol * std::max(1.0f, std::fabs(expected[i]))) << "element " << i;
    }
}

// ---- Reference values: tools/generate_sensitivity_reference_values.py output, verbatim ----
const std::vector<float> kOcclusion = {
    1.08410072f, 1.08410072f, 0.401111841f, -0.281877041f, -0.281877041f, 1.08410072f, 1.08410072f, 0.401111841f,
    -0.281877041f, -0.281877041f, 0.998020768f, 0.998020768f, 0.466430366f, -0.0651600361f, -0.0651600361f,
    0.911940813f, 0.911940813f, 0.531748891f, 0.151556969f, 0.151556969f, -0.151575565f, -0.151575565f,
    0.799025416f, 1.7496264f, 1.7496264f, -0.151575565f, -0.151575565f, 0.799025416f, 1.7496264f, 1.7496264f,
    -0.0678701401f, -0.0678701401f, 0.433463156f, 0.934796453f, 0.934796453f, 0.0158352852f, 0.0158352852f,
    0.0679008961f, 0.119966507f, 0.119966507f,
};
const std::vector<float> kPercentile10 = {
    -2.23333335f, -2.23333335f, -2.23333335f,
};
const std::vector<float> kPercentile80 = {
    1.79999995f, 1.66666663f, 1.4666667f,
};
const std::vector<float> kStd = {
    1.63293409f, 1.68044078f, 1.61780226f,
};

class SensitivityTest : public ::testing::Test {
protected:
    CPUBackend backend;
    // f(x) = 3 x0 - x1^2 + 0 * x2, with a second output 2 x2.
    std::function<Tensor(const Tensor&)> predict = [this](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        return Tensor(Shape({2}), &backend, {3.0f * v[0] - v[1] * v[1], 2.0f * v[2]});
    };
    Tensor input{Shape({3}), &backend, {1.0f, 2.0f, -1.0f}};
};

TEST_F(SensitivityTest, OneAtATimeChangesEachFeatureAlone) {
    LocalSensitivityResult r = ComputeLocalSensitivity(predict, input, 0, DeltaBounds(input, 0.5f));
    EXPECT_FLOAT_EQ(r.output, -1.0f);
    ASSERT_EQ(r.features.size(), 3u);
    const FeatureSensitivity& a = r.features[0];
    EXPECT_FLOAT_EQ(a.low, 0.5f);
    EXPECT_FLOAT_EQ(a.output_low, -2.5f);
    EXPECT_FLOAT_EQ(a.output_high, 0.5f);
    EXPECT_FLOAT_EQ(a.swing(), 3.0f);
    EXPECT_FLOAT_EQ(a.slope(), 3.0f);
    // x1^2 around 2: (1.5^2, 2.5^2), slope -4 = the derivative at 2 (exact for a quadratic).
    EXPECT_FLOAT_EQ(r.features[1].output_low, 3.0f - 2.25f);
    EXPECT_FLOAT_EQ(r.features[1].slope(), -4.0f);
    EXPECT_FLOAT_EQ(r.features[2].swing(), 0.0f);  // output 0 ignores x2

    LocalSensitivityResult other = ComputeLocalSensitivity(predict, input, 1, DeltaBounds(input, 0.5f), {2});
    ASSERT_EQ(other.features.size(), 1u);
    EXPECT_FLOAT_EQ(other.features[0].slope(), 2.0f);
}

TEST_F(SensitivityTest, BoundsMatchNumpy) {
    std::vector<Tensor> background;
    const std::vector<float> all = Det(72, 5, 1, 3);
    for (size_t i = 0; i < 24; ++i) {
        background.emplace_back(Shape({3}), &backend, std::vector<float>(all.begin() + 3 * i, all.begin() + 3 * i + 3));
    }
    SensitivityBounds range = RangeBounds(background, 0.1f, 0.8f);
    ExpectNearAll(range.low, kPercentile10);
    ExpectNearAll(range.high, kPercentile80);
    SensitivityBounds scaled = ScaledDeltaBounds(input, background, 2.0f);
    for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(scaled.high[j] - input.read_element(static_cast<int64_t>(j)), 2.0f * kStd[j], 1e-5f);
        EXPECT_NEAR(input.read_element(static_cast<int64_t>(j)) - scaled.low[j], 2.0f * kStd[j], 1e-5f);
    }
}

TEST_F(SensitivityTest, OcclusionMatchesCaptum) {
    const std::vector<float> w = Det(40, 7, 2, 9);
    auto model = [&](const Tensor& x) {
        const std::vector<float> v = x.to_host_vector();
        float base = 0.0f;
        for (size_t i = 0; i < 40; ++i) base += w[i] * std::tanh(v[i]);
        // x[0,1,2] is flat 7; x[1,3,4] is flat 20 + 15 + 4 = 39.
        return Tensor(Shape({2}), &backend, {base, base + v[7] * v[39]});
    };
    Tensor x(Shape({2, 4, 5}), &backend, Det(40, 5, 3, 4));
    Attribution a = Occlusion(model, x, 1, {1, 3, 3}, {1, 2, 2}, 0.25f);
    EXPECT_EQ(a.method, "occlusion");
    EXPECT_EQ(a.values.shape(), x.shape());
    ExpectNearAll(a.values.to_host_vector(), kOcclusion, 1e-4f);
    EXPECT_EQ(a.metadata.at("window"), "1x3x3");
}

// A window of one element is feature ablation: f(x) - f(x with that feature at the baseline).
TEST_F(SensitivityTest, SingleElementOcclusionIsFeatureAblation) {
    Attribution a = Occlusion(predict, input, 0, {1}, {1}, 0.0f);
    EXPECT_FLOAT_EQ(a.values.read_element(0), -1.0f - (-4.0f));  // x0 -> 0
    EXPECT_FLOAT_EQ(a.values.read_element(1), -1.0f - 3.0f);     // x1 -> 0
    EXPECT_FLOAT_EQ(a.values.read_element(2), 0.0f);
}

TEST_F(SensitivityTest, RejectsBadInput) {
    EXPECT_THROW((void)DeltaBounds(input, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)ScaledDeltaBounds(input, {}, 1.0f), std::invalid_argument);
    EXPECT_THROW((void)ScaledDeltaBounds(input, {Tensor(Shape({4}), &backend)}, 1.0f), std::invalid_argument);
    EXPECT_THROW((void)RangeBounds({input}, 0.9f, 0.1f), std::invalid_argument);
    EXPECT_THROW((void)ComputeLocalSensitivity(predict, input, 0, SensitivityBounds{{1.0f}, {2.0f}}),
                 std::invalid_argument);
    EXPECT_THROW((void)ComputeLocalSensitivity(predict, input, 0, DeltaBounds(input, 1.0f), {3}),
                 std::invalid_argument);
    EXPECT_THROW((void)ComputeLocalSensitivity(predict, input, 2, DeltaBounds(input, 1.0f)), std::invalid_argument);
    EXPECT_THROW((void)Occlusion(predict, input, 0, {1, 1}, {1}, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)Occlusion(predict, input, 0, {4}, {1}, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)Occlusion(predict, input, 0, {2}, {3}, 0.0f), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
