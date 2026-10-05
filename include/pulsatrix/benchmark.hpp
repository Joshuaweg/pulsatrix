/** @file benchmark.hpp
 *  @brief Timing, benchmark reports (`pulsatrix.benchmark.v1`) and regression comparison.
 *
 *  The `pulsatrix_bench` tool (tools/bench/) uses these to measure training step time,
 *  explanation time and LRP conservation error on every backend a build has, and to compare two
 *  builds. They are public so a project can benchmark its own models the same way.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace pulsatrix {

/** @brief Summary of repeated timings, in milliseconds. */
struct TimingStats {
    int64_t repeats = 0;
    double min_ms = 0.0;
    double p10_ms = 0.0;
    double median_ms = 0.0;
    double p90_ms = 0.0;
    double mean_ms = 0.0;
};

/**
 * @brief Summarizes timings. Percentiles interpolate linearly between the sorted samples.
 * @throws std::invalid_argument if @p samples_ms is empty or has a negative or non-finite value.
 */
[[nodiscard]] TimingStats SummarizeTimings(std::vector<double> samples_ms);

/**
 * @brief Runs @p fn @p warmup times untimed, then @p repeats times timed with a steady clock.
 * @note A GPU op returns before the GPU finishes, so @p fn must end by waiting for its device
 *       work, for example by reading a result to the host. Otherwise this times kernel launches.
 * @throws std::invalid_argument if @p warmup < 0 or @p repeats < 1.
 */
[[nodiscard]] TimingStats TimeIt(const std::function<void()>& fn, int warmup, int repeats);

/** @brief One measured quantity. */
struct BenchmarkRecord {
    enum class Kind {
        /** @brief A duration: stats holds the timings and lower is better. */
        Time,
        /** @brief A single number where lower is better, such as a relative error: value holds it. */
        Metric,
    };
    /** @brief What was measured, for example `"train.mlp"` or `"conservation.lrp_epsilon.mlp"`. */
    std::string name;
    /** @brief Where it ran: `"cpu"`, `"hip"` or `"cuda"`. */
    std::string device;
    Kind kind = Kind::Time;
    /** @brief Set for Kind::Time. */
    TimingStats stats;
    /** @brief Set for Kind::Metric. May be NaN or infinite when the measured thing is broken. */
    double value = 0.0;
    /** @brief What value or stats measure, for example `"ms per step"` or `"relative error"`. */
    std::string unit;
};

/** @brief A run of benchmarks: `pulsatrix.benchmark.v1`. */
struct BenchmarkReport {
    /** @brief Free-form name for the run, for example a branch or commit. */
    std::string label;
    /** @brief Build and machine facts: version, build type, compiler, device names. */
    std::map<std::string, std::string> environment;
    std::vector<BenchmarkRecord> records;
};

/** @throws std::invalid_argument if two records share a name and device. */
[[nodiscard]] std::string ToJson(const BenchmarkReport& report);
/** @brief Reads a report, following the viz document rules (docs/visualization): schema and
 *         version checks, unknown fields ignored, "nonfinite" for NaN and infinity, and errors
 *         that name the JSON Pointer of the problem. */
[[nodiscard]] BenchmarkReport ParseBenchmarkReport(std::string_view json);

/** @brief How CompareBenchmarks decides that a candidate got worse. */
struct CompareOptions {
    /** @brief A time regresses when the candidate's median exceeds the baseline's by more than
     *         this fraction. */
    double time_tolerance = 0.10;
    /** @brief A metric regresses when the candidate exceeds the baseline by more than this
     *         absolute amount (metrics such as conservation error are near zero, so a fraction
     *         of the baseline would be meaningless). */
    double metric_tolerance = 1e-5;
    /** @brief With at least three rounds on each side, a time also needs the candidate's rounds
     *         to be slower than the baseline's with this one-sided Mann-Whitney p-value. Separate
     *         processes of the same build can differ by 20% on sub-millisecond work, so medians
     *         alone raise false alarms; a real slowdown makes every round slower. */
    double significance = 0.05;
};

/** @brief One record compared across two builds. */
struct BenchmarkComparison {
    std::string name;
    std::string device;
    BenchmarkRecord::Kind kind = BenchmarkRecord::Kind::Time;
    /** @brief The baseline's and candidate's median time (median over runs of each run's
     *         median), or their metric value (median over runs). */
    double baseline = 0.0;
    double candidate = 0.0;
    /** @brief candidate / baseline - 1 for times; candidate - baseline for metrics. */
    double change = 0.0;
    bool regression = false;
    /** @brief For times with at least three rounds per side: the one-sided Mann-Whitney U test's
     *         p-value that the candidate's rounds are no slower than the baseline's. NaN
     *         otherwise. */
    double p_value = 0.0;
    /** @brief Set when the record is in only one of the two builds' reports. */
    std::string missing_from;
};

/**
 * @brief Compares a candidate build with a baseline, record by record (matched on name and
 *        device). Several reports per side are the rounds of an interleaved A/B run; each
 *        side's value is the median over its rounds, so one noisy round doesn't decide, and
 *        with three or more rounds per side a time regresses only if its rounds also separate
 *        (CompareOptions::significance).
 * @note A metric that is NaN or infinite in the candidate but finite in the baseline is a
 *       regression. A record missing from either side is reported, not a regression.
 * @throws std::invalid_argument if either side has no reports, a tolerance is negative, or the
 *         significance is outside (0, 1].
 */
[[nodiscard]] std::vector<BenchmarkComparison> CompareBenchmarks(const std::vector<BenchmarkReport>& baseline_runs,
                                                                  const std::vector<BenchmarkReport>& candidate_runs,
                                                                  const CompareOptions& options = {});

}  // namespace pulsatrix
