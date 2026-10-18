#include "pulsatrix/calibration.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace pulsatrix {
namespace {

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Logit(double p) { return std::log(p / (1.0 - p)); }

double BinaryLoss(const std::vector<double>& z, const std::vector<int64_t>& y, double t) {
    double loss = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        const double p = Sigmoid(z[i] / t);
        loss -= y[i] == 1 ? std::log(p) : std::log1p(-p);
    }
    return loss / static_cast<double>(z.size());
}

// Labels drawn from sigmoid(z / 2.5): the fitted temperature should be near 2.5, and no
// temperature on a fine grid should do better.
TEST(CalibrationTest, BinaryTemperatureRecoversTheTrueOne) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> normal(0.0, 3.0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> z;
    std::vector<int64_t> y;
    for (int i = 0; i < 100000; ++i) {
        z.push_back(normal(rng));
        y.push_back(uniform(rng) < Sigmoid(z.back() / 2.5) ? 1 : 0);
    }
    const TemperatureFit fit = FitBinaryTemperature(z, y);
    EXPECT_TRUE(fit.converged);
    EXPECT_NEAR(fit.temperature, 2.5, 0.08);
    EXPECT_NEAR(fit.log_loss_before, BinaryLoss(z, y, 1.0), 1e-12);
    EXPECT_NEAR(fit.log_loss_after, BinaryLoss(z, y, fit.temperature), 1e-12);
    for (double t = 2.0; t <= 3.0; t += 0.01) EXPECT_LE(fit.log_loss_after, BinaryLoss(z, y, t) + 1e-12) << "T = " << t;
    EXPECT_LT(fit.iterations, 30);
}

// Two classes with logits (0, z) are the binary case with logit z.
TEST(CalibrationTest, MultiClassMatchesBinaryOnTwoClasses) {
    const std::vector<double> z = {2.0, -1.0, 0.5, 3.0, -2.5, 1.5, -0.3, 0.8};
    const std::vector<int64_t> y = {1, 0, 0, 1, 1, 1, 0, 0};
    std::vector<std::vector<double>> rows;
    for (const double v : z) rows.push_back({0.0, v});
    const TemperatureFit binary = FitBinaryTemperature(z, y);
    const TemperatureFit multi = FitTemperature(rows, y);
    EXPECT_NEAR(multi.temperature, binary.temperature, 1e-9);
    EXPECT_NEAR(multi.log_loss_after, binary.log_loss_after, 1e-12);
}

TEST(CalibrationTest, MultiClassBeatsEveryTemperatureOnAGrid) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> normal(0.0, 2.0);
    std::vector<std::vector<double>> rows;
    std::vector<int64_t> y;
    for (int i = 0; i < 2000; ++i) {
        std::vector<double> r = {normal(rng), normal(rng), normal(rng), normal(rng)};
        y.push_back(static_cast<int64_t>(i % 4));
        r[static_cast<size_t>(y.back())] += 1.0;  // informative, but overconfident at T = 1
        rows.push_back(r);
    }
    const TemperatureFit fit = FitTemperature(rows, y);
    ASSERT_TRUE(fit.converged);
    EXPECT_GT(fit.temperature, 1.0);
    EXPECT_LT(fit.log_loss_after, fit.log_loss_before);
    auto loss = [&](double t) {
        double total = 0.0;
        for (size_t i = 0; i < rows.size(); ++i) {
            double sum = 0.0;
            for (const double v : rows[i]) sum += std::exp(v / t);
            total -= rows[i][static_cast<size_t>(y[i])] / t - std::log(sum);
        }
        return total / static_cast<double>(rows.size());
    };
    EXPECT_NEAR(fit.log_loss_after, loss(fit.temperature), 1e-12);
    for (double t = 0.5; t <= 6.0; t += 0.01) EXPECT_LE(fit.log_loss_after, loss(t) + 1e-12) << "T = " << t;
}

TEST(CalibrationTest, AWeightCountsAsRepeats) {
    const std::vector<double> z = {2.0, -1.0, 0.5, 3.0, -2.5};
    const std::vector<int64_t> y = {1, 1, 0, 0, 0};
    const TemperatureFit weighted = FitBinaryTemperature(z, y, {1.0, 3.0, 1.0, 2.0, 1.0});
    const TemperatureFit repeated =
        FitBinaryTemperature({2.0, -1.0, -1.0, -1.0, 0.5, 3.0, 3.0, -2.5}, {1, 1, 1, 1, 0, 0, 0, 0});
    EXPECT_NEAR(weighted.temperature, repeated.temperature, 1e-9);
    EXPECT_NEAR(weighted.log_loss_after, repeated.log_loss_after, 1e-12);
}

TEST(CalibrationTest, ReportsAMinimumAtABound) {
    // Perfectly separated: sharper is always better.
    const TemperatureFit sharp = FitBinaryTemperature({-2.0, -1.0, 1.0, 2.0}, {0, 0, 1, 1});
    EXPECT_FALSE(sharp.converged);
    EXPECT_NEAR(sharp.temperature, 1e-6, 1e-12);
    // Pointing the wrong way: the best positive temperature is infinite.
    const TemperatureFit wrong = FitBinaryTemperature({-2.0, -1.0, 1.0, 2.0}, {1, 1, 0, 0});
    EXPECT_FALSE(wrong.converged);
    EXPECT_NEAR(wrong.temperature, 1e6, 1e-3);
}

TEST(CalibrationTest, TemperatureRejectsBadInput) {
    EXPECT_THROW((void)FitBinaryTemperature({}, {}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({1.0}, {}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({1.0}, {2}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({NAN}, {1}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({1.0}, {1}, {-1.0}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({1.0}, {1}, {0.0}), std::invalid_argument);
    EXPECT_THROW((void)FitBinaryTemperature({1.0}, {1}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW((void)FitTemperature({{1.0}}, {0}), std::invalid_argument);
    EXPECT_THROW((void)FitTemperature({{1.0, 2.0}, {1.0}}, {0, 0}), std::invalid_argument);
    EXPECT_THROW((void)FitTemperature({{1.0, 2.0}}, {2}), std::invalid_argument);
}

TEST(CalibrationTest, LogOddsErrorByBand) {
    const std::vector<double> predicted = {0.01, 0.5, 1e-5, 0.2};
    const std::vector<double> reference = {0.001, 0.5, 1e-4, 0.1};
    const LogOddsErrorReport r = LogOddsError(predicted, reference);
    ASSERT_EQ(r.bands.size(), 2u);
    const double e0 = Logit(0.01) - Logit(0.001), e2 = Logit(1e-5) - Logit(1e-4), e3 = Logit(0.2) - Logit(0.1);
    // [0, 1e-3) holds the 1e-4 reference only; 1e-3 itself is in the upper band.
    EXPECT_EQ(r.bands[0].count, 1);
    EXPECT_NEAR(r.bands[0].bias, e2, 1e-12);
    EXPECT_NEAR(r.bands[0].mean_abs, std::abs(e2), 1e-12);
    EXPECT_EQ(r.bands[1].count, 3);
    EXPECT_DOUBLE_EQ(r.bands[1].lower, 1e-3);
    EXPECT_DOUBLE_EQ(r.bands[1].upper, 1.0);
    EXPECT_NEAR(r.bands[1].mean_abs, (e0 + 0.0 + e3) / 3.0, 1e-12);
    EXPECT_NEAR(r.bands[1].rms, std::sqrt((e0 * e0 + e3 * e3) / 3.0), 1e-12);
    EXPECT_NEAR(r.bands[1].median_abs, e3, 1e-12);
    EXPECT_NEAR(r.bands[1].max_abs, e0, 1e-12);
    EXPECT_EQ(r.overall.count, 4);
    EXPECT_NEAR(r.overall.bias, (e0 + e2 + e3) / 4.0, 1e-12);
}

TEST(CalibrationTest, LogOddsErrorWeightsAndClips) {
    // Weights act as repeats, quantiles included.
    const LogOddsErrorReport w = LogOddsError({0.2, 0.4}, {0.1, 0.1}, {3.0, 1.0}, {});
    ASSERT_EQ(w.bands.size(), 1u);
    const double a = Logit(0.2) - Logit(0.1), b = Logit(0.4) - Logit(0.1);
    EXPECT_NEAR(w.overall.mean_abs, (3 * a + b) / 4.0, 1e-12);
    EXPECT_NEAR(w.overall.median_abs, a, 1e-12);
    EXPECT_NEAR(w.overall.p90_abs, b, 1e-12);
    EXPECT_DOUBLE_EQ(w.overall.weight, 4.0);
    // Both sides are clipped: 0 against 1e-9 is no error at the default clip, and an exact 0 or 1
    // gives a finite one.
    const LogOddsErrorReport c = LogOddsError({1e-9, 0.5, 1.0}, {0.0, 0.0, 0.0});
    EXPECT_EQ(c.bands[0].count, 3);
    EXPECT_NEAR(c.bands[0].max_abs, 2.0 * Logit(1.0 - 1e-6), 1e-9);
    EXPECT_NEAR(c.bands[0].median_abs, -Logit(1e-6), 1e-9);
    const JsonValue j = ToJson(c);
    EXPECT_DOUBLE_EQ(j.find("clip")->as_double(), 1e-6);
    EXPECT_EQ(j.find("bands")->as_array().size(), 2u);
    EXPECT_EQ(j.find("overall")->find("count")->as_int64(), 3);
}

TEST(CalibrationTest, LogOddsErrorRejectsBadInput) {
    EXPECT_THROW((void)LogOddsError({}, {}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({1.1}, {0.1}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {NAN}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {0.1}, {}, {0.5, 0.2}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {0.1}, {}, {0.0}), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {0.1}, {}, {1e-3}, 0.0), std::invalid_argument);
    EXPECT_THROW((void)LogOddsError({0.1}, {0.1}, {0.0}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
