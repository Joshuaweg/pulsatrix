// KS-3: split conformal prediction. LAC sets and absolute-residual intervals are checked against
// MAPIE 1.1.0, APS sets against Angelopoulos and Bates' reference code
// (tools/generate_conformal_reference_values.py), and the guarantee on cases whose answer is known.

#include "pulsatrix/conformal.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/performance.hpp"

namespace pulsatrix {
namespace {

// ---- Reference values: tools/generate_conformal_reference_values.py output, verbatim ----
// Each set is one character, the bitmask of its classes: "0123456789ABCDEFGHIJKLMNOPQRSTUV"[mask].
struct SetReference {
    ConformalScore score;
    double alpha, quantile, coverage, mean_size;
    const char* sets;
};
const std::vector<SetReference> kSets = {
    {ConformalScore::Lac, 0.1, 0.7922570109367371, 0.91, 1.7466666666666666,
     "1O4OHJH4HI2I9I454I1A2A1I555LA5L5929559I944DI462482HHH2O466O1GOOH71GG3C3OOOOO331O16C61CGGGC164428J2IH49II948IIA5AGK2AKLLL9IA8AI1KII494448I8448J846CH4333O1GGG3O333O13333O111HCH11HC1223O24C4ICHI9DI454G9IKGA5AIAALA91AG8551A4IIMCI989J42H36O422CHC66SHS337O330O7O333G7OGOC16C43O366J8I66I9I44KA4195G9AIAALLAK"},
    {ConformalScore::Lac, 0.05, 0.8414934724569321, 0.9433333333333334, 2.0533333333333332,
     "HO6OHJH4HIII9I454I1A2A1I55LLALLL9I9559I944DI46I682HPH2O466O3GOOH71OO3U7UVVVV333O36E71CGHGC164438J2IH49II949IIA5AGKQALLLL9IA8AI1KII4B444DI8448J846CHC333O1GOO3O737UVF773OJ33HCH13HC1363O24C4ICPI9DI455G9IKGA5AQAALA99AG9551A4IIMCI989J42H36O426CHC67SHS33FO7FVSVS333O7OGOE16C63O366J8J66I9I44MAK195K9AIAALLAK"},
    {ConformalScore::Aps, 0.1, 0.9538067728281021, 0.93, 2.0633333333333335,
     "HO6OHHH4P2IG8I455G58A29M5D5K2555DI911121449JC6I496HHG68666O1GOGP7NSUFUFUUUUT77NO3EE73EGHGCH4C43CHI2P49G294922812KKQ8K5KKFM8A8I5KGI4B5449I944CHC648HC113O1OGP3OF7FUFFFFFSN33RSP11HCH363O6444ICHIPCG454I92GK252UU2K8DD8I9155A4IGICG899H43H34O666CHC77SRS77FUFFUUFT773S7OGGCH6C62G3223CI6628244KB4911K92Q2AKKUN"},
    {ConformalScore::Aps, 0.05, 0.9725767523050308, 0.9733333333333334, 2.3466666666666667,
     "HO6OHHH4PIIG8I555G5AA29M5T5K2L55DIB55929559JC6I496HHH6O666O1GOOR7NSUFUFUUUUTF7NO3EU73EGHGCH6C63DHIIP4PII959IIA52KKQAL5LKFU8A8I5KGI5B5649I94CCHC66CHC313O1POT3SFFFUFFFFFSN33RSP11JCH363O6C46ICPJPCG555I9IKK252UU2K8FF8I9555Q5IGICI999H4JH36O666EPS77SRU77FUFFUUFT777T7OOOEH6C63O3663CI66I82KKKBK995K9AQ2BKLUN"},
};
struct IntervalReference {
    double alpha, quantile, coverage;
};
const std::vector<IntervalReference> kIntervals = {
    {0.1, 0.4964015870757183, 0.9033333333333333},
    {0.05, 0.5465729545927484, 0.9433333333333334},
};
// ---- end of reference values ----

constexpr int64_t kK = 5;

class ConformalTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The generator's classification data (its docstring gives the formula).
    struct Data {
        Tensor probabilities;
        std::vector<int64_t> labels;
    };
    Data Make(int64_t n, double o) {
        std::vector<float> p;
        std::vector<int64_t> labels;
        for (int64_t i = 0; i < n; ++i) {
            const double di = static_cast<double>(i);
            double logits[kK], peak = -1e300, sum = 0;
            for (int64_t k = 0; k < kK; ++k) {
                const double dk = static_cast<double>(k);
                logits[k] = 3.0 * std::sin(o + 0.53 * di + 1.21 * dk + 0.07 * di * dk + 0.61 * di * di);
                peak = std::max(peak, logits[k]);
            }
            for (double l : logits) sum += std::exp(l - peak);
            std::vector<float> row;
            for (double l : logits) row.push_back(static_cast<float>(std::exp(l - peak) / sum));
            // The two most probable classes, by the stored float32 values, ties to the lower index.
            int64_t first = 0, second = -1;
            for (int64_t k = 1; k < kK; ++k) {
                if (row[k] > row[first]) first = k;
            }
            for (int64_t k = 0; k < kK; ++k) {
                if (k != first && (second < 0 || row[k] > row[second])) second = k;
            }
            labels.push_back(std::sin(2.3 * di + o) < -0.6 ? second : first);
            p.insert(p.end(), row.begin(), row.end());
        }
        return {Tensor(Shape({n, kK}), &backend, p), labels};
    }

    // The generator's regression data: calibration (o = 0.2) or test (o = 1.9).
    static void Regression(int64_t n, double o, std::vector<double>& prediction, std::vector<double>& target) {
        for (int64_t j = 0; j < n; ++j) {
            const double dj = static_cast<double>(j);
            prediction.push_back(std::sin(0.8 * dj + o));
            target.push_back(prediction.back() + 0.4 * std::sin(3.1 * dj + 2 * o) * (1 + 0.5 * std::cos(dj)));
        }
    }

    static std::string Masks(const std::vector<std::vector<int64_t>>& sets) {
        static const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
        std::string out;
        for (const std::vector<int64_t>& s : sets) {
            int mask = 0;
            for (int64_t c : s) mask |= 1 << c;
            out += digits[mask];
        }
        return out;
    }
};

TEST_F(ConformalTest, LacMatchesMapieAndApsTheReferenceCode) {
    const Data cal = Make(500, 0.2), test = Make(300, 1.9);
    for (const SetReference& want : kSets) {
        ConformalOptions options;
        options.alpha = want.alpha;
        options.score = want.score;
        const ConformalClassifier c = CalibrateConformalClassifier(cal.probabilities, cal.labels, options);
        EXPECT_EQ(c.n, 500);
        EXPECT_NEAR(c.quantile, want.quantile, 1e-15) << want.alpha;
        const std::vector<std::vector<int64_t>> sets = PredictionSets(c, test.probabilities);
        EXPECT_EQ(Masks(sets), want.sets) << want.alpha;
        const CoverageReport r = EvaluateCoverage(sets, test.labels);
        EXPECT_DOUBLE_EQ(r.coverage, want.coverage);
        EXPECT_NEAR(r.mean_size, want.mean_size, 1e-15);
        // Each set lists its classes from most to least probable.
        const std::vector<float> p = test.probabilities.to_host_vector();
        for (size_t i = 0; i < sets.size(); ++i) {
            for (size_t j = 1; j < sets[i].size(); ++j) {
                EXPECT_GE(p[i * kK + static_cast<size_t>(sets[i][j - 1])], p[i * kK + static_cast<size_t>(sets[i][j])]);
            }
        }
    }
}

TEST_F(ConformalTest, AbsoluteResidualIntervalsMatchMapie) {
    std::vector<double> cal_prediction, cal_target, prediction, target;
    Regression(500, 0.2, cal_prediction, cal_target);
    Regression(300, 1.9, prediction, target);
    for (const IntervalReference& want : kIntervals) {
        const ConformalRegressor r = CalibrateConformalRegressor(cal_prediction, cal_target, want.alpha);
        EXPECT_EQ(r.score, "absolute_residual");
        EXPECT_NEAR(r.quantile, want.quantile, 1e-15);
        const std::vector<PredictionInterval> intervals = PredictionIntervals(r, prediction);
        EXPECT_DOUBLE_EQ(intervals[0].upper - prediction[0], r.quantile);
        const CoverageReport c = EvaluateCoverage(intervals, target);
        EXPECT_DOUBLE_EQ(c.coverage, want.coverage);
        EXPECT_NEAR(c.mean_size, 2 * r.quantile, 1e-12);
        EXPECT_TRUE(c.size_counts.empty());
    }
}

// CQR: the noise in the generator's targets grows with 1 + 0.5 cos(j), so intervals from a
// quantile regressor that knows it are narrower where the noise is, and still cover.
TEST_F(ConformalTest, CqrWidensAndNarrowsWithTheQuantileRegressor) {
    std::vector<double> cal_prediction, cal_target, prediction, target;
    Regression(500, 0.2, cal_prediction, cal_target);
    Regression(300, 1.9, prediction, target);
    const auto bands = [](const std::vector<double>& mid, std::vector<double>& lo, std::vector<double>& hi) {
        for (size_t j = 0; j < mid.size(); ++j) {
            const double half = 0.3 * (1 + 0.5 * std::cos(static_cast<double>(j)));
            lo.push_back(mid[j] - half);
            hi.push_back(mid[j] + half);
        }
    };
    std::vector<double> cal_lo, cal_hi, lo, hi;
    bands(cal_prediction, cal_lo, cal_hi);
    bands(prediction, lo, hi);
    const ConformalRegressor r = CalibrateConformalQuantileRegressor(cal_lo, cal_hi, cal_target, 0.1);
    EXPECT_EQ(r.score, "cqr");
    // q is the 451st smallest of max(lo - y, y - hi).
    std::vector<double> scores;
    for (size_t i = 0; i < cal_target.size(); ++i) scores.push_back(std::max(cal_lo[i] - cal_target[i], cal_target[i] - cal_hi[i]));
    std::sort(scores.begin(), scores.end());
    EXPECT_DOUBLE_EQ(r.quantile, scores[450]);
    const std::vector<PredictionInterval> intervals = PredictionIntervals(r, lo, hi);
    EXPECT_DOUBLE_EQ(intervals[3].lower, lo[3] - r.quantile);
    EXPECT_DOUBLE_EQ(intervals[3].upper, hi[3] + r.quantile);
    const CoverageReport c = EvaluateCoverage(intervals, target);
    EXPECT_GE(c.coverage, 0.85);
    const ConformalRegressor plain = CalibrateConformalRegressor(cal_prediction, cal_target, 0.1);
    EXPECT_LT(c.mean_size, EvaluateCoverage(PredictionIntervals(plain, prediction), target).mean_size);
    EXPECT_THROW((void)PredictionIntervals(r, prediction), std::invalid_argument);
    EXPECT_THROW((void)PredictionIntervals(plain, lo, hi), std::invalid_argument);
}

TEST_F(ConformalTest, RandomizedApsCoversCloserToTheTarget) {
    const Data cal = Make(500, 0.2), test = Make(300, 1.9);
    ConformalOptions options;
    options.alpha = 0.1;
    const CoverageReport plain =
        EvaluateCoverage(PredictionSets(CalibrateConformalClassifier(cal.probabilities, cal.labels, options), test.probabilities), test.labels);
    options.randomized = true;
    options.seed = 11;
    const ConformalClassifier c = CalibrateConformalClassifier(cal.probabilities, cal.labels, options);
    const std::vector<std::vector<int64_t>> sets = PredictionSets(c, test.probabilities);
    const CoverageReport randomized = EvaluateCoverage(sets, test.labels);
    EXPECT_LT(randomized.mean_size, plain.mean_size);
    EXPECT_LT(std::fabs(randomized.coverage - 0.9), std::fabs(plain.coverage - 0.9));
    EXPECT_EQ(Masks(PredictionSets(c, test.probabilities)), Masks(sets));  // reproducible
    options.score = ConformalScore::Lac;
    EXPECT_THROW((void)CalibrateConformalClassifier(cal.probabilities, cal.labels, options), std::invalid_argument);
}

// The guarantee in its simplest case: n calibration scores, and a test score exchangeable with them,
// ranks below the ⌈(n + 1)(1 − alpha)⌉-th with probability at least 1 − alpha.
TEST_F(ConformalTest, TooFewCalibrationExamplesGiveEverything) {
    // Rank ⌈6 * 0.9⌉ = 6 > 5: no finite quantile is enough, so every set holds every class.
    const Tensor five(Shape({5, 2}), &backend, {0.9f, 0.1f, 0.8f, 0.2f, 0.3f, 0.7f, 0.6f, 0.4f, 0.1f, 0.9f});
    ConformalOptions options;
    options.score = ConformalScore::Lac;
    const ConformalClassifier c = CalibrateConformalClassifier(five, {0, 0, 1, 0, 1}, options);
    EXPECT_TRUE(std::isinf(c.quantile));
    EXPECT_EQ(PredictionSets(c, five)[0].size(), 2u);
    // With alpha = 0.5 the rank is ⌈3⌉ = 3: the third smallest of 1 - p(label).
    options.alpha = 0.5;
    EXPECT_NEAR(CalibrateConformalClassifier(five, {0, 0, 1, 0, 1}, options).quantile, 0.2, 1e-7);
    const ConformalRegressor r = CalibrateConformalRegressor({0, 0, 0}, {1, 2, 3}, 0.1);
    EXPECT_TRUE(std::isinf(r.quantile));
    const CoverageReport wide = EvaluateCoverage(PredictionIntervals(r, {0.0}), {5.0});
    EXPECT_DOUBLE_EQ(wide.coverage, 1.0);
    EXPECT_TRUE(ToJson(wide).find("mean_size")->is_null());
}

TEST_F(ConformalTest, CoverageBySizeAndSlice) {
    const CoverageReport r = EvaluateCoverage(std::vector<std::vector<int64_t>>{{0}, {1, 2}, {}, {2, 0}, {1}}, {0, 0, 1, 0, 1});
    EXPECT_EQ(r.n, 5);
    EXPECT_DOUBLE_EQ(r.coverage, 0.6);
    EXPECT_DOUBLE_EQ(r.mean_size, 1.2);
    EXPECT_EQ(r.empty, 1);
    EXPECT_EQ(r.size_counts, (std::vector<int64_t>{1, 2, 2}));
    EXPECT_EQ(r.coverage_by_size, (std::vector<double>{0.0, 1.0, 0.5}));
    EXPECT_EQ(r.misses, (std::vector<double>{0, 1, 1, 0, 0}));
    // The misses are 0/1 errors: PerSliceError tests each slice's coverage.
    const SliceErrorReport slices = PerSliceError(r.misses, {"x", "y", "y", "x", "x"});
    EXPECT_EQ(slices.slices.front().name, "y");
    EXPECT_DOUBLE_EQ(slices.slices.front().error, 1.0);
    const JsonValue doc = ToJson(r);
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.coverage.v1");
    EXPECT_EQ(doc.find("size_counts")->as_array().size(), 3u);
}

TEST_F(ConformalTest, InputsAreChecked) {
    const Tensor p(Shape({2, 2}), &backend, {0.9f, 0.1f, 0.4f, 0.6f});
    ConformalOptions bad;
    bad.alpha = 0.0;
    EXPECT_THROW((void)CalibrateConformalClassifier(p, {0, 1}, bad), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalClassifier(p, {0}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalClassifier(p, {0, 2}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalClassifier(Tensor(Shape({2, 1}), &backend, {1, 1}), {0, 0}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalClassifier(Tensor(Shape({1, 2}), &backend, {NAN, 1}), {0}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalRegressor({}, {}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalRegressor({1, 2}, {1}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalRegressor({1}, {INFINITY}), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalRegressor({1}, {1}, 1.0), std::invalid_argument);
    EXPECT_THROW((void)CalibrateConformalQuantileRegressor({1}, {2, 3}, {1}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateCoverage(std::vector<std::vector<int64_t>>{}, {}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateCoverage(std::vector<PredictionInterval>{{0, 1}}, {}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
