// KS-3: EvaluateClassifier, EvaluateRegressor and PerSliceError, checked against scikit-learn 1.9.1
// and SciPy 1.18.1 (tools/generate_performance_reference_values.py), then on cases whose answer is
// known.

#include "pulsatrix/performance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

// ---- Reference values: tools/generate_performance_reference_values.py output, verbatim ----
constexpr double kAccuracy = 0.575;
constexpr double kLogLoss = 1.4107772098729674;
constexpr double kBrier = 0.6619257712236681;
const std::vector<double> kPrecision = {0.6101694915254238, 0.5689655172413793, 0.6176470588235294, 0.5102040816326531};
const std::vector<double> kRecall = {0.6, 0.6470588235294118, 0.525, 0.5102040816326531};
const std::vector<double> kF1 = {0.6050420168067226, 0.6055045871559633, 0.5675675675675675, 0.5102040816326531};
const std::vector<int64_t> kSupport = {60, 51, 40, 49};
constexpr double kMacroF1 = 0.5720795632907266;
const std::vector<int64_t> kConfusion = {36, 14, 3, 7, 7, 33, 4, 7, 5, 4, 21, 10, 11, 7, 6, 25};
constexpr double kEce15 = 0.15156397089362145, kMce15 = 0.455288864672184;
const std::vector<int64_t> kBins15Count = {0, 0, 0, 1, 4, 9, 9, 27, 28, 26, 28, 17, 28, 23, 0};
const std::vector<double> kBins15Confidence = {0.0, 0.0, 0.0, 0.2527056634426117, 0.294711135327816, 0.3660157322883606, 0.4456508722570207, 0.5028371788837291, 0.5668013351304191, 0.6279356387945322, 0.7015515693596431, 0.7688617846545052, 0.8343367619173867, 0.9038169487662937, 0.0};
const std::vector<double> kBins15Accuracy = {0.0, 0.0, 0.0, 0.0, 0.75, 0.4444444444444444, 0.6666666666666666, 0.5555555555555556, 0.39285714285714285, 0.6538461538461539, 0.7142857142857143, 0.47058823529411764, 0.5357142857142857, 0.6956521739130435, 0.0};
constexpr double kEce10 = 0.15690832570195196, kMce10 = 0.2723958952086313;
const std::vector<int64_t> kBins10Count = {0, 0, 4, 10, 21, 43, 40, 31, 35, 16};

struct SliceReference {
    const char* name;
    int64_t count;
    int64_t errors;
    double error, lower, upper, rest_error, p_value;
};
const std::vector<SliceReference> kSlices = {
    {"a", 67, 24, 0.3582089552238806, 0.25398057076391606, 0.4778149016007814, 0.45864661654135336, 0.934759646706865},
    {"b", 67, 22, 0.3283582089552239, 0.22791317376089593, 0.44741818752268525, 0.47368421052631576, 0.9833219886565969},
    {"c", 66, 39, 0.5909090909090909, 0.4704918970000326, 0.7013258199065963, 0.34328358208955223, 0.0007510779309894813},
};
constexpr double kOverallError = 0.425;

constexpr double kMse = 0.04521501209578994, kMae = 0.19204055571115286, kR2 = 0.9166693062155595;
// ---- end of reference values ----

constexpr int64_t kN = 200, kK = 4;

// The generator's data (its docstring gives the formulas).
struct Data {
    std::vector<float> probabilities;
    std::vector<int64_t> labels;
    std::vector<std::string> slices;
};

Data MakeData() {
    Data d;
    for (int64_t i = 0; i < kN; ++i) {
        std::vector<double> logits(kK);
        for (int64_t k = 0; k < kK; ++k) {
            const double di = static_cast<double>(i), dk = static_cast<double>(k);
            logits[k] = 2.5 * std::sin(0.3 + 0.37 * di + 1.13 * dk + 0.05 * di * dk + 0.71 * di * di);
        }
        int64_t top = 0;
        double peak = logits[0], total = 0;
        for (int64_t k = 1; k < kK; ++k) {
            if (logits[k] > peak) peak = logits[k], top = k;
        }
        for (double l : logits) total += std::exp(l - peak);
        for (double l : logits) d.probabilities.push_back(static_cast<float>(std::exp(l - peak) / total));
        const std::string slice(1, "abc"[i % 3]);
        const double threshold = slice == "c" ? 0.3 : -0.5;
        d.labels.push_back(std::sin(1.7 * static_cast<double>(i)) < threshold ? (top + 1 + i % 3) % kK : top);
        d.slices.push_back(slice);
    }
    return d;
}

class PerformanceTest : public ::testing::Test {
protected:
    CPUBackend backend;
    Data data = MakeData();
    Tensor probabilities = Tensor(Shape({kN, kK}), &backend, data.probabilities);

    Tensor Probs(int64_t n, int64_t k, std::vector<float> values) { return Tensor(Shape({n, k}), &backend, values); }
};

TEST_F(PerformanceTest, ClassifierMetricsMatchScikitLearn) {
    const ClassificationReport r = EvaluateClassifier(probabilities, data.labels);
    EXPECT_EQ(r.n, kN);
    EXPECT_EQ(r.classes, kK);
    EXPECT_DOUBLE_EQ(r.accuracy, kAccuracy);
    EXPECT_NEAR(r.log_loss, kLogLoss, 1e-12);
    EXPECT_NEAR(r.brier, kBrier, 1e-12);
    for (size_t c = 0; c < kPrecision.size(); ++c) {
        EXPECT_NEAR(r.precision[c], kPrecision[c], 1e-15) << c;
        EXPECT_NEAR(r.recall[c], kRecall[c], 1e-15) << c;
        EXPECT_NEAR(r.f1[c], kF1[c], 1e-15) << c;
        EXPECT_EQ(r.support[c], kSupport[c]) << c;
    }
    EXPECT_NEAR(r.macro_f1, kMacroF1, 1e-15);
    std::vector<int64_t> confusion;
    for (const std::vector<int64_t>& row : r.confusion) confusion.insert(confusion.end(), row.begin(), row.end());
    EXPECT_EQ(confusion, kConfusion);
}

TEST_F(PerformanceTest, CalibrationErrorMatchesItsDefinition) {
    const ClassificationReport r = EvaluateClassifier(probabilities, data.labels);
    EXPECT_NEAR(r.ece, kEce15, 1e-12);
    EXPECT_NEAR(r.mce, kMce15, 1e-12);
    ASSERT_EQ(r.bins.size(), 15u);
    for (size_t b = 0; b < r.bins.size(); ++b) {
        EXPECT_EQ(r.bins[b].count, kBins15Count[b]) << b;
        EXPECT_NEAR(r.bins[b].confidence, kBins15Confidence[b], 1e-12) << b;
        EXPECT_NEAR(r.bins[b].accuracy, kBins15Accuracy[b], 1e-15) << b;
        EXPECT_NEAR(r.bins[b].upper - r.bins[b].lower, 1.0 / 15.0, 1e-15) << b;
    }
    ClassificationOptions ten;
    ten.bins = 10;
    const ClassificationReport r10 = EvaluateClassifier(probabilities, data.labels, ten);
    EXPECT_NEAR(r10.ece, kEce10, 1e-12);
    EXPECT_NEAR(r10.mce, kMce10, 1e-12);
    for (size_t b = 0; b < r10.bins.size(); ++b) EXPECT_EQ(r10.bins[b].count, kBins10Count[b]) << b;
}

TEST_F(PerformanceTest, APerfectConfidentClassifierScoresPerfectly) {
    const ClassificationReport r = EvaluateClassifier(Probs(3, 2, {1, 0, 0, 1, 1, 0}), {0, 1, 0});
    EXPECT_DOUBLE_EQ(r.accuracy, 1.0);
    EXPECT_DOUBLE_EQ(r.brier, 0.0);
    EXPECT_NEAR(r.log_loss, 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(r.ece, 0.0);
    EXPECT_EQ(r.bins.back().count, 3);  // a confidence of exactly 1 is in the last bin
    // A probability of 0 for the label counts as 1e-15, not infinity.
    const ClassificationReport wrong = EvaluateClassifier(Probs(1, 2, {1, 0}), {1});
    EXPECT_NEAR(wrong.log_loss, -std::log(1e-15), 1e-9);
    EXPECT_DOUBLE_EQ(wrong.ece, 1.0);
    // A class never predicted has precision 0, not NaN; ties go to the lower class.
    const ClassificationReport tie = EvaluateClassifier(Probs(2, 2, {0.5f, 0.5f, 0.5f, 0.5f}), {0, 1});
    EXPECT_DOUBLE_EQ(tie.accuracy, 0.5);
    EXPECT_DOUBLE_EQ(tie.precision[1], 0.0);
    EXPECT_DOUBLE_EQ(tie.f1[1], 0.0);
}

TEST_F(PerformanceTest, ClassifierInputsAreChecked) {
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 1, {1, 1}), {0, 0}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 2, {1, 0, 0, 1}), {0}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 2, {1, 0, 0, 1}), {0, 2}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 2, {1, 0, -0.1f, 1}), {0, 1}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 2, {1, 0, NAN, 1}), {0, 1}), std::invalid_argument);
    ClassificationOptions none;
    none.bins = 0;
    EXPECT_THROW((void)EvaluateClassifier(Probs(2, 2, {1, 0, 0, 1}), {0, 1}, none), std::invalid_argument);
}

TEST_F(PerformanceTest, RegressorMetricsMatchScikitLearn) {
    std::vector<double> predictions, targets;
    for (int64_t i = 0; i < kN; ++i) {
        const double p = std::sin(0.9 * static_cast<double>(i));
        predictions.push_back(p);
        targets.push_back(p + 0.3 * std::cos(2.1 * static_cast<double>(i)));
    }
    const RegressionReport r = EvaluateRegressor(predictions, targets);
    EXPECT_NEAR(r.mse, kMse, 1e-15);
    EXPECT_NEAR(r.rmse, std::sqrt(kMse), 1e-15);
    EXPECT_NEAR(r.mae, kMae, 1e-15);
    EXPECT_NEAR(r.r2, kR2, 1e-14);
    // Constant targets: 1 for a perfect fit, 0 otherwise, as scikit-learn.
    EXPECT_DOUBLE_EQ(EvaluateRegressor({2, 2}, {2, 2}).r2, 1.0);
    EXPECT_DOUBLE_EQ(EvaluateRegressor({1, 2}, {2, 2}).r2, 0.0);
    EXPECT_THROW((void)EvaluateRegressor({}, {}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateRegressor({1}, {1, 2}), std::invalid_argument);
    EXPECT_THROW((void)EvaluateRegressor({INFINITY}, {1}), std::invalid_argument);
}

TEST_F(PerformanceTest, PerSliceErrorMatchesFishersExactTest) {
    const SliceErrorReport r = PerSliceError(probabilities, data.labels, data.slices);
    EXPECT_EQ(r.test, "fisher_exact");
    EXPECT_EQ(r.n, kN);
    EXPECT_DOUBLE_EQ(r.error, kOverallError);
    ASSERT_EQ(r.slices.size(), 3u);
    // Worst first: c, then a, then b.
    const std::vector<size_t> order = {2, 0, 1};
    for (size_t s = 0; s < order.size(); ++s) {
        const SliceReference& want = kSlices[order[s]];
        const SliceError& got = r.slices[s];
        EXPECT_EQ(got.name, want.name);
        EXPECT_EQ(got.count, want.count);
        EXPECT_DOUBLE_EQ(got.error, want.error);
        EXPECT_NEAR(got.lower, want.lower, 1e-9) << got.name;  // z from NormalQuantile, good to ~1e-9
        EXPECT_NEAR(got.upper, want.upper, 1e-9) << got.name;
        EXPECT_NEAR(got.rest_error, want.rest_error, 1e-15) << got.name;
        EXPECT_NEAR(got.p_value, want.p_value, 1e-12 + 1e-10 * want.p_value) << got.name;
    }
    // c's p of 0.00075 is below 0.05 / 3; the others are nowhere near.
    EXPECT_TRUE(r.slices[0].worse);
    EXPECT_FALSE(r.slices[1].worse);
    EXPECT_FALSE(r.slices[2].worse);
    EXPECT_TRUE(r.any_worse());
    const std::string text = DescribeSliceError(r);
    EXPECT_NE(text.find("Fisher's exact test"), std::string::npos);
    EXPECT_NE(text.find("WORSE"), std::string::npos);
    EXPECT_NE(text.find("95% interval"), std::string::npos);
}

TEST_F(PerformanceTest, BonferroniDecidesWhichSlicesAreWorse) {
    // c's p of 0.00075 survives any correction up to 66 slices; at alpha = 0.002 / 3 it doesn't.
    SliceErrorOptions strict;
    strict.alpha = 0.002;
    EXPECT_FALSE(PerSliceError(probabilities, data.labels, data.slices, strict).any_worse());
}

TEST_F(PerformanceTest, LossesAreTestedByPermutation) {
    // Slice "high" has losses 1 higher than the rest; "same" matches "rest".
    std::vector<double> errors;
    std::vector<std::string> slices;
    for (int i = 0; i < 90; ++i) {
        const double noise = 0.3 * std::sin(1.3 * i);
        const std::string s = i % 3 == 0 ? "high" : i % 3 == 1 ? "same" : "rest";
        errors.push_back(noise + (s == "high" ? 1.0 : 0.5));
        slices.push_back(s);
    }
    const SliceErrorReport r = PerSliceError(errors, slices);
    EXPECT_EQ(r.test, "permutation");
    EXPECT_EQ(r.slices[0].name, "high");
    EXPECT_TRUE(r.slices[0].worse);
    EXPECT_DOUBLE_EQ(r.slices[0].p_value, 1.0 / 1000.0);  // beats every relabeling
    EXPECT_FALSE(r.slices[1].worse);
    EXPECT_FALSE(r.slices[2].worse);
    EXPECT_LT(r.slices[0].lower, r.slices[0].error);
    EXPECT_GT(r.slices[0].upper, r.slices[0].error);
    // The same seed gives the same p-values.
    EXPECT_DOUBLE_EQ(PerSliceError(errors, slices).slices[1].p_value, r.slices[1].p_value);
    // Forcing the permutation test on 0/1 errors works too.
    SliceErrorOptions permutation;
    permutation.test = SliceTest::Permutation;
    EXPECT_EQ(PerSliceError(probabilities, data.labels, data.slices, permutation).test, "permutation");
}

TEST_F(PerformanceTest, OneSliceIsTheWholeDataset) {
    const SliceErrorReport r = PerSliceError({0, 1, 1, 0}, {"all", "all", "all", "all"});
    ASSERT_EQ(r.slices.size(), 1u);
    EXPECT_DOUBLE_EQ(r.slices[0].error, 0.5);
    EXPECT_DOUBLE_EQ(r.slices[0].p_value, 1.0);
    EXPECT_FALSE(r.any_worse());
}

TEST_F(PerformanceTest, SliceInputsAreChecked) {
    EXPECT_THROW((void)PerSliceError(std::vector<double>{}, {}), std::invalid_argument);
    EXPECT_THROW((void)PerSliceError({0, 1}, {"a"}), std::invalid_argument);
    EXPECT_THROW((void)PerSliceError({0, NAN}, {"a", "b"}), std::invalid_argument);
    SliceErrorOptions fisher;
    fisher.test = SliceTest::FisherExact;
    EXPECT_THROW((void)PerSliceError({0, 0.5}, {"a", "b"}, fisher), std::invalid_argument);
    SliceErrorOptions bad;
    bad.alpha = 1.0;
    EXPECT_THROW((void)PerSliceError({0, 1}, {"a", "b"}, bad), std::invalid_argument);
    bad.alpha = 0.05;
    bad.permutations = 0;
    EXPECT_THROW((void)PerSliceError({0, 1}, {"a", "b"}, bad), std::invalid_argument);
}

TEST_F(PerformanceTest, ReportsBecomeDocuments) {
    const JsonValue c = ToJson(EvaluateClassifier(probabilities, data.labels));
    EXPECT_EQ(c.find("schema")->as_string(), "pulsatrix.classification_report.v1");
    EXPECT_EQ(c.find("reliability")->as_array().size(), 15u);
    EXPECT_EQ(c.find("confusion")->as_array().size(), static_cast<size_t>(kK));
    EXPECT_EQ(c.find("per_class")->as_array().size(), static_cast<size_t>(kK));
    const JsonValue s = ToJson(PerSliceError(probabilities, data.labels, data.slices));
    EXPECT_EQ(s.find("schema")->as_string(), "pulsatrix.slice_error.v1");
    EXPECT_EQ(s.find("slices")->as_array().front().find("name")->as_string(), "c");
    EXPECT_TRUE(s.find("slices")->as_array().front().find("worse")->as_bool());
    const JsonValue g = ToJson(EvaluateRegressor({1, 2}, {1, 3}));
    EXPECT_EQ(g.find("schema")->as_string(), "pulsatrix.regression_report.v1");
    EXPECT_DOUBLE_EQ(g.find("mae")->as_double(), 0.5);
}

}  // namespace
}  // namespace pulsatrix
