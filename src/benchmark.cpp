#include "pulsatrix/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include "pulsatrix/json.hpp"
#include "viz/document_io.hpp"

namespace pulsatrix {

using namespace document_io;

namespace {

// Linear interpolation between sorted samples at fraction q in [0, 1].
double Percentile(const std::vector<double>& sorted, double q) {
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return Percentile(v, 0.5);
}

// One-sided Mann-Whitney U test: the probability, with no difference between the two sides,
// of a U at least as large as observed, where U counts the (baseline, candidate) pairs in which
// the candidate is slower (ties count one half). Exact null distribution by dynamic programming.
double MannWhitneyGreater(const std::vector<double>& base, const std::vector<double>& cand) {
    const size_t n = base.size(), m = cand.size();
    double u = 0.0;
    for (double b : base) {
        for (double c : cand) {
            u += c > b ? 1.0 : (c == b ? 0.5 : 0.0);
        }
    }
    // ways[i][j][k]: orderings of i baseline and j candidate values with U = k.
    const size_t max_u = n * m;
    std::vector<std::vector<std::vector<double>>> ways(n + 1, std::vector<std::vector<double>>(m + 1, std::vector<double>(max_u + 1, 0.0)));
    for (size_t i = 0; i <= n; ++i) {
        for (size_t j = 0; j <= m; ++j) {
            if (i == 0 || j == 0) {
                ways[i][j][0] = 1.0;
                continue;
            }
            for (size_t k = 0; k <= i * j; ++k) {
                // The largest value is a candidate (beating all i baseline values) or a baseline.
                ways[i][j][k] = (k >= i ? ways[i][j - 1][k - i] : 0.0) + ways[i - 1][j][k];
            }
        }
    }
    double total = 0.0, tail = 0.0;
    const auto threshold = static_cast<size_t>(std::ceil(u - 1e-9));
    for (size_t k = 0; k <= max_u; ++k) {
        total += ways[n][m][k];
        if (k >= threshold) tail += ways[n][m][k];
    }
    return tail / total;
}

constexpr size_t kMinRoundsForTest = 3;
constexpr size_t kMaxRoundsForTest = 60;

const char* KindName(BenchmarkRecord::Kind k) { return k == BenchmarkRecord::Kind::Time ? "time" : "metric"; }

}  // namespace

TimingStats SummarizeTimings(std::vector<double> samples_ms) {
    if (samples_ms.empty()) {
        throw std::invalid_argument("SummarizeTimings: no samples");
    }
    double total = 0.0;
    for (double s : samples_ms) {
        if (!std::isfinite(s) || s < 0.0) {
            throw std::invalid_argument("SummarizeTimings: a sample is negative or not finite");
        }
        total += s;
    }
    std::sort(samples_ms.begin(), samples_ms.end());
    TimingStats stats;
    stats.repeats = static_cast<int64_t>(samples_ms.size());
    stats.min_ms = samples_ms.front();
    stats.p10_ms = Percentile(samples_ms, 0.1);
    stats.median_ms = Percentile(samples_ms, 0.5);
    stats.p90_ms = Percentile(samples_ms, 0.9);
    stats.mean_ms = total / static_cast<double>(samples_ms.size());
    return stats;
}

TimingStats TimeIt(const std::function<void()>& fn, int warmup, int repeats) {
    if (warmup < 0 || repeats < 1) {
        throw std::invalid_argument("TimeIt: needs warmup >= 0 and repeats >= 1");
    }
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return SummarizeTimings(std::move(samples));
}

// ---- pulsatrix.benchmark.v1 ----------------------------------------------------------------

std::string ToJson(const BenchmarkReport& report) {
    std::set<std::pair<std::string, std::string>> seen;
    for (const BenchmarkRecord& r : report.records) {
        if (!seen.insert({r.name, r.device}).second) {
            Invalid("benchmark", "/records: \"" + r.name + "\" on " + r.device + " appears twice");
        }
    }
    DocWriter w("benchmark");
    w.root().add("label", report.label);
    JsonValue env{JsonValue::Object{}};
    for (const auto& [k, v] : report.environment) {
        env.add(k, v);
    }
    w.root().add("environment", std::move(env));
    JsonValue records{JsonValue::Array{}};
    for (size_t i = 0; i < report.records.size(); ++i) {
        const BenchmarkRecord& r = report.records[i];
        const std::string p = Index("/records", i);
        JsonValue rec{JsonValue::Object{}};
        rec.add("name", r.name);
        rec.add("device", r.device);
        rec.add("kind", KindName(r.kind));
        rec.add("unit", r.unit);
        if (r.kind == BenchmarkRecord::Kind::Time) {
            JsonValue s{JsonValue::Object{}};
            s.add("repeats", JsonValue(r.stats.repeats));
            s.add("min_ms", w.num_double(r.stats.min_ms, p + "/stats/min_ms"));
            s.add("p10_ms", w.num_double(r.stats.p10_ms, p + "/stats/p10_ms"));
            s.add("median_ms", w.num_double(r.stats.median_ms, p + "/stats/median_ms"));
            s.add("p90_ms", w.num_double(r.stats.p90_ms, p + "/stats/p90_ms"));
            s.add("mean_ms", w.num_double(r.stats.mean_ms, p + "/stats/mean_ms"));
            rec.add("stats", std::move(s));
        } else {
            rec.add("value", w.num_double(r.value, p + "/value"));
        }
        records.push_back(std::move(rec));
    }
    w.root().add("records", std::move(records));
    return w.finish();
}

BenchmarkReport ParseBenchmarkReport(std::string_view json) {
    DocReader r(json, "benchmark");
    const JsonValue& root = r.root();
    BenchmarkReport report;
    report.label = r.string(r.member(root, "", "label"), "/label");
    for (const auto& [k, v] : r.object(r.member(root, "", "environment"), "/environment")) {
        report.environment.emplace(k, r.string(v, "/environment/" + k));
    }
    const JsonValue::Array& records = r.array(r.member(root, "", "records"), "/records");
    std::set<std::pair<std::string, std::string>> seen;
    for (size_t i = 0; i < records.size(); ++i) {
        const std::string p = Index("/records", i);
        const JsonValue& rec = records[i];
        r.object(rec, p);
        BenchmarkRecord out;
        out.name = r.string(r.member(rec, p, "name"), p + "/name");
        out.device = r.string(r.member(rec, p, "device"), p + "/device");
        out.unit = r.string(r.member(rec, p, "unit"), p + "/unit");
        const std::string& kind = r.string(r.member(rec, p, "kind"), p + "/kind");
        if (kind == "time") {
            out.kind = BenchmarkRecord::Kind::Time;
            const JsonValue& s = r.member(rec, p, "stats");
            r.object(s, p + "/stats");
            const std::string sp = p + "/stats";
            out.stats.repeats = r.integer(r.member(s, sp, "repeats"), sp + "/repeats", 0, kMaxIndex);
            out.stats.min_ms = r.number_double(r.member(s, sp, "min_ms"), sp + "/min_ms");
            out.stats.p10_ms = r.number_double(r.member(s, sp, "p10_ms"), sp + "/p10_ms");
            out.stats.median_ms = r.number_double(r.member(s, sp, "median_ms"), sp + "/median_ms");
            out.stats.p90_ms = r.number_double(r.member(s, sp, "p90_ms"), sp + "/p90_ms");
            out.stats.mean_ms = r.number_double(r.member(s, sp, "mean_ms"), sp + "/mean_ms");
        } else if (kind == "metric") {
            out.kind = BenchmarkRecord::Kind::Metric;
            out.value = r.number_double(r.member(rec, p, "value"), p + "/value");
        } else {
            r.fail(p + "/kind", "must be \"time\" or \"metric\", got \"" + kind + "\"");
        }
        if (!seen.insert({out.name, out.device}).second) {
            r.fail(p + "/name", "\"" + out.name + "\" on " + out.device + " appears twice");
        }
        report.records.push_back(std::move(out));
    }
    r.finish();
    return report;
}

// ---- comparison ----------------------------------------------------------------------------

std::vector<BenchmarkComparison> CompareBenchmarks(const std::vector<BenchmarkReport>& baseline_runs,
                                                   const std::vector<BenchmarkReport>& candidate_runs,
                                                   const CompareOptions& options) {
    if (baseline_runs.empty() || candidate_runs.empty()) {
        throw std::invalid_argument("CompareBenchmarks: each side needs at least one report");
    }
    if (!(options.time_tolerance >= 0.0) || !(options.metric_tolerance >= 0.0)) {
        throw std::invalid_argument("CompareBenchmarks: tolerances must be non-negative");
    }
    if (!(options.significance > 0.0 && options.significance <= 1.0)) {
        throw std::invalid_argument("CompareBenchmarks: significance must be in (0, 1]");
    }
    using Key = std::pair<std::string, std::string>;
    struct Side {
        std::vector<double> values;
        BenchmarkRecord::Kind kind = BenchmarkRecord::Kind::Time;
    };
    // Each side's value per record: one entry per round, combined by median below.
    auto collect = [](const std::vector<BenchmarkReport>& runs, std::vector<Key>& order) {
        std::map<Key, Side> sides;
        for (const BenchmarkReport& run : runs) {
            for (const BenchmarkRecord& rec : run.records) {
                Key key{rec.name, rec.device};
                auto [it, inserted] = sides.try_emplace(key);
                if (inserted) {
                    order.push_back(key);
                    it->second.kind = rec.kind;
                }
                it->second.values.push_back(rec.kind == BenchmarkRecord::Kind::Time ? rec.stats.median_ms : rec.value);
            }
        }
        return sides;
    };
    std::vector<Key> order;
    std::map<Key, Side> base = collect(baseline_runs, order);
    std::map<Key, Side> cand = collect(candidate_runs, order);

    // A median over values that include NaN must keep the NaN visible: a broken round is news.
    auto combine = [](const std::vector<double>& v) {
        for (double x : v) {
            if (std::isnan(x)) return x;
        }
        return Median(v);
    };

    std::vector<BenchmarkComparison> out;
    std::set<Key> done;
    for (const Key& key : order) {
        if (!done.insert(key).second) {
            continue;
        }
        BenchmarkComparison c;
        c.name = key.first;
        c.device = key.second;
        auto b = base.find(key);
        auto k = cand.find(key);
        if (b == base.end() || k == cand.end()) {
            c.kind = (b != base.end() ? b->second : k->second).kind;
            c.missing_from = b == base.end() ? "baseline" : "candidate";
            if (b != base.end()) c.baseline = combine(b->second.values);
            if (k != cand.end()) c.candidate = combine(k->second.values);
            out.push_back(std::move(c));
            continue;
        }
        c.kind = k->second.kind;
        c.baseline = combine(b->second.values);
        c.candidate = combine(k->second.values);
        if (c.kind == BenchmarkRecord::Kind::Time) {
            c.change = c.baseline > 0.0 ? c.candidate / c.baseline - 1.0 : 0.0;
            c.regression = c.candidate > c.baseline * (1.0 + options.time_tolerance);
            const std::vector<double>& bv = b->second.values;
            const std::vector<double>& kv = k->second.values;
            c.p_value = std::numeric_limits<double>::quiet_NaN();
            if (bv.size() >= kMinRoundsForTest && kv.size() >= kMinRoundsForTest && bv.size() <= kMaxRoundsForTest &&
                kv.size() <= kMaxRoundsForTest) {
                c.p_value = MannWhitneyGreater(bv, kv);
                c.regression = c.regression && c.p_value < options.significance;
            }
        } else {
            c.change = c.candidate - c.baseline;
            const bool broke = !std::isfinite(c.candidate) && std::isfinite(c.baseline);
            c.regression = broke || c.candidate > c.baseline + options.metric_tolerance;
        }
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace pulsatrix
