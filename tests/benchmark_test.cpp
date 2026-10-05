#include "pulsatrix/benchmark.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {
namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

TEST(BenchmarkStatsTest, SummarizesWithInterpolatedPercentiles) {
    TimingStats s = SummarizeTimings({5.0, 1.0, 3.0, 2.0, 4.0});
    EXPECT_EQ(s.repeats, 5);
    EXPECT_DOUBLE_EQ(s.min_ms, 1.0);
    EXPECT_DOUBLE_EQ(s.median_ms, 3.0);
    EXPECT_DOUBLE_EQ(s.mean_ms, 3.0);
    EXPECT_DOUBLE_EQ(s.p10_ms, 1.4);  // position 0.1 * 4 = 0.4 between 1 and 2
    EXPECT_DOUBLE_EQ(s.p90_ms, 4.6);
    TimingStats even = SummarizeTimings({4.0, 1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(even.median_ms, 2.5);
    TimingStats one = SummarizeTimings({7.0});
    EXPECT_DOUBLE_EQ(one.p10_ms, 7.0);
    EXPECT_DOUBLE_EQ(one.p90_ms, 7.0);
}

TEST(BenchmarkStatsTest, RejectsUnusableSamples) {
    EXPECT_THROW((void)SummarizeTimings({}), std::invalid_argument);
    EXPECT_THROW((void)SummarizeTimings({1.0, -0.5}), std::invalid_argument);
    EXPECT_THROW((void)SummarizeTimings({1.0, kNaN}), std::invalid_argument);
}

TEST(BenchmarkTimeItTest, RunsWarmupUntimedThenTimesEachRepeat) {
    int calls = 0;
    TimingStats s = TimeIt(
        [&] {
            ++calls;
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(2000);
            while (std::chrono::steady_clock::now() < until) {
            }
        },
        2, 5);
    EXPECT_EQ(calls, 7);
    EXPECT_EQ(s.repeats, 5);
    EXPECT_GE(s.min_ms, 1.9);
    EXPECT_LT(s.median_ms, 200.0);
    EXPECT_THROW((void)TimeIt([] {}, -1, 1), std::invalid_argument);
    EXPECT_THROW((void)TimeIt([] {}, 0, 0), std::invalid_argument);
}

BenchmarkRecord Time(const std::string& name, const std::string& device, double median) {
    BenchmarkRecord r;
    r.name = name;
    r.device = device;
    r.kind = BenchmarkRecord::Kind::Time;
    r.stats = SummarizeTimings({median * 0.9, median, median * 1.2});
    r.unit = "ms per step";
    return r;
}

BenchmarkRecord Metric(const std::string& name, const std::string& device, double value) {
    BenchmarkRecord r;
    r.name = name;
    r.device = device;
    r.kind = BenchmarkRecord::Kind::Metric;
    r.value = value;
    r.unit = "relative error";
    return r;
}

BenchmarkReport Report(std::vector<BenchmarkRecord> records, const std::string& label = "run") {
    BenchmarkReport r;
    r.label = label;
    r.environment = {{"build_type", "Release"}, {"version", "1.0.0"}};
    r.records = std::move(records);
    return r;
}

TEST(BenchmarkReportTest, RoundTripsThroughJson) {
    BenchmarkReport report = Report({Time("train.mlp", "cpu", 1.25), Metric("conservation.lrp_epsilon.mlp", "cpu", 3e-7),
                                     Metric("conservation.lrp_epsilon.mlp", "hip", kNaN)});
    const std::string json = ToJson(report);
    EXPECT_NE(json.find("\"schema\": \"pulsatrix.benchmark.v1\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"nonfinite\""), std::string::npos) << json;
    BenchmarkReport back = ParseBenchmarkReport(json);
    EXPECT_EQ(back.label, "run");
    EXPECT_EQ(back.environment, report.environment);
    ASSERT_EQ(back.records.size(), 3u);
    EXPECT_EQ(back.records[0].name, "train.mlp");
    EXPECT_EQ(back.records[0].kind, BenchmarkRecord::Kind::Time);
    EXPECT_DOUBLE_EQ(back.records[0].stats.median_ms, report.records[0].stats.median_ms);
    EXPECT_DOUBLE_EQ(back.records[0].stats.p90_ms, report.records[0].stats.p90_ms);
    EXPECT_EQ(back.records[0].stats.repeats, 3);
    EXPECT_EQ(back.records[0].unit, "ms per step");
    EXPECT_EQ(back.records[1].kind, BenchmarkRecord::Kind::Metric);
    EXPECT_DOUBLE_EQ(back.records[1].value, 3e-7);
    EXPECT_TRUE(std::isnan(back.records[2].value));
    EXPECT_EQ(back.records[2].device, "hip");
    EXPECT_EQ(ToJson(back), json);
}

TEST(BenchmarkReportTest, RejectsDuplicatesAndOtherDocuments) {
    EXPECT_THROW((void)ToJson(Report({Time("a", "cpu", 1.0), Time("a", "cpu", 2.0)})), std::invalid_argument);
    EXPECT_NO_THROW((void)ToJson(Report({Time("a", "cpu", 1.0), Time("a", "hip", 2.0)})));
    EXPECT_THROW((void)ParseBenchmarkReport(R"({"schema": "pulsatrix.heatmap.v1"})"), std::invalid_argument);
    try {
        (void)ParseBenchmarkReport(R"({"schema": "pulsatrix.benchmark.v1", "label": "x", "environment": {},
                                      "records": [{"name": "a", "device": "cpu", "kind": "speed", "unit": ""}]})");
        FAIL() << "accepted an unknown kind";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("/records/0/kind"), std::string::npos) << e.what();
    }
}

const BenchmarkComparison& Find(const std::vector<BenchmarkComparison>& cs, const std::string& name, const std::string& device) {
    for (const auto& c : cs) {
        if (c.name == name && c.device == device) return c;
    }
    throw std::runtime_error("no comparison for " + name + " on " + device);
}

TEST(BenchmarkCompareTest, TimesRegressBeyondTheTolerance) {
    auto cs = CompareBenchmarks({Report({Time("fast", "cpu", 10.0), Time("slow", "cpu", 10.0), Time("better", "cpu", 10.0)})},
                                {Report({Time("fast", "cpu", 10.5), Time("slow", "cpu", 12.0), Time("better", "cpu", 5.0)})});
    ASSERT_EQ(cs.size(), 3u);
    EXPECT_FALSE(Find(cs, "fast", "cpu").regression);
    EXPECT_NEAR(Find(cs, "fast", "cpu").change, 0.05, 1e-12);
    EXPECT_TRUE(Find(cs, "slow", "cpu").regression);
    EXPECT_NEAR(Find(cs, "slow", "cpu").change, 0.2, 1e-12);
    EXPECT_FALSE(Find(cs, "better", "cpu").regression);
    EXPECT_NEAR(Find(cs, "better", "cpu").change, -0.5, 1e-12);
    CompareOptions loose;
    loose.time_tolerance = 0.25;
    EXPECT_FALSE(Find(CompareBenchmarks({Report({Time("slow", "cpu", 10.0)})}, {Report({Time("slow", "cpu", 12.0)})}, loose),
                      "slow", "cpu")
                     .regression);
}

TEST(BenchmarkCompareTest, RoundsAreCombinedByMedianSoOneNoisyRoundDoesNotDecide) {
    std::vector<BenchmarkReport> baseline = {Report({Time("t", "cpu", 10.0)}), Report({Time("t", "cpu", 10.2)}),
                                             Report({Time("t", "cpu", 9.8)})};
    std::vector<BenchmarkReport> candidate = {Report({Time("t", "cpu", 10.1)}), Report({Time("t", "cpu", 30.0)}),
                                              Report({Time("t", "cpu", 10.0)})};
    const BenchmarkComparison& c = Find(CompareBenchmarks(baseline, candidate), "t", "cpu");
    EXPECT_DOUBLE_EQ(c.baseline, 10.0);
    EXPECT_DOUBLE_EQ(c.candidate, 10.1);
    EXPECT_FALSE(c.regression);
}

std::vector<BenchmarkReport> Rounds(const std::vector<double>& medians) {
    std::vector<BenchmarkReport> runs;
    for (double m : medians) runs.push_back(Report({Time("t", "cpu", m)}));
    return runs;
}

TEST(BenchmarkCompareTest, OverlappingNoisyRoundsAreNotARegression) {
    // Measured A/A noise on this kind of benchmark: whole rounds jump between two speeds. The
    // medians differ by more than the tolerance, but the rounds interleave.
    const BenchmarkComparison& c = Find(CompareBenchmarks(Rounds({0.63, 0.90, 0.65, 0.66, 0.92, 0.64}),
                                                          Rounds({0.95, 0.66, 0.93, 0.64, 0.97, 0.90})),
                                        "t", "cpu");
    EXPECT_GT(c.change, 0.10);
    EXPECT_GT(c.p_value, 0.05);
    EXPECT_FALSE(c.regression);
}

TEST(BenchmarkCompareTest, AConsistentSlowdownAcrossRoundsIsARegression) {
    const BenchmarkComparison& c = Find(CompareBenchmarks(Rounds({10.0, 10.3, 9.9, 10.1, 10.2, 9.8}),
                                                          Rounds({12.1, 12.4, 11.9, 12.0, 12.6, 12.2})),
                                        "t", "cpu");
    EXPECT_NEAR(c.p_value, 1.0 / 924.0, 1e-12);  // complete separation, 6 vs 6: 1 / C(12, 6)
    EXPECT_TRUE(c.regression);
    // Consistently slower but within the tolerance: not a regression.
    const BenchmarkComparison& small = Find(CompareBenchmarks(Rounds({10.0, 10.1, 10.0, 10.1}), Rounds({10.4, 10.5, 10.4, 10.5})),
                                            "t", "cpu");
    EXPECT_LT(small.p_value, 0.05);
    EXPECT_FALSE(small.regression);
}

TEST(BenchmarkCompareTest, FewRoundsFallBackToTheToleranceAlone) {
    const BenchmarkComparison& c = Find(CompareBenchmarks(Rounds({10.0, 10.0}), Rounds({12.0, 12.0})), "t", "cpu");
    EXPECT_TRUE(std::isnan(c.p_value));
    EXPECT_TRUE(c.regression);
}

TEST(BenchmarkCompareTest, MetricsRegressBeyondAnAbsoluteTolerance) {
    auto cs = CompareBenchmarks({Report({Metric("small", "cpu", 1e-7), Metric("big", "cpu", 1e-7), Metric("broken", "cpu", 1e-7)})},
                                {Report({Metric("small", "cpu", 2e-6), Metric("big", "cpu", 1e-4), Metric("broken", "cpu", kNaN)})});
    EXPECT_FALSE(Find(cs, "small", "cpu").regression);
    EXPECT_NEAR(Find(cs, "small", "cpu").change, 1.9e-6, 1e-18);
    EXPECT_TRUE(Find(cs, "big", "cpu").regression);
    EXPECT_TRUE(Find(cs, "broken", "cpu").regression);
}

TEST(BenchmarkCompareTest, MatchesOnDeviceAndReportsMissingRecords) {
    auto cs = CompareBenchmarks({Report({Time("t", "cpu", 1.0), Time("t", "hip", 1.0), Time("gone", "cpu", 1.0)})},
                                {Report({Time("t", "cpu", 1.0), Time("t", "hip", 5.0), Time("new", "cpu", 1.0)})});
    EXPECT_FALSE(Find(cs, "t", "cpu").regression);
    EXPECT_TRUE(Find(cs, "t", "hip").regression);
    EXPECT_EQ(Find(cs, "gone", "cpu").missing_from, "candidate");
    EXPECT_FALSE(Find(cs, "gone", "cpu").regression);
    EXPECT_EQ(Find(cs, "new", "cpu").missing_from, "baseline");
    EXPECT_FALSE(Find(cs, "new", "cpu").regression);
}

TEST(BenchmarkCompareTest, RejectsMissingSidesAndNegativeTolerances) {
    EXPECT_THROW((void)CompareBenchmarks({}, {Report({})}), std::invalid_argument);
    EXPECT_THROW((void)CompareBenchmarks({Report({})}, {}), std::invalid_argument);
    CompareOptions bad;
    bad.time_tolerance = -0.1;
    EXPECT_THROW((void)CompareBenchmarks({Report({})}, {Report({})}, bad), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
