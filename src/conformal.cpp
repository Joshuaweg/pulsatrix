// KS-3: split conformal prediction. The quantile is the ⌈(n + 1)(1 − α)⌉-th smallest calibration
// score (Angelopoulos and Bates, arXiv 2107.07511, section 1.1); a prediction set holds every class
// whose score would be at most that.

#include "pulsatrix/conformal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "portable_random.hpp"

namespace pulsatrix {
namespace {

void CheckAlpha(double alpha, const char* who) {
    if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument(std::string(who) + ": alpha must be in (0, 1)");
}

// The probabilities as rows, checked; labels too when given.
std::vector<std::vector<double>> Rows(const Tensor& probabilities, const std::vector<int64_t>* labels, const char* who) {
    const Shape& s = probabilities.shape();
    if (s.rank() != 2 || s.dim(0) < 1 || s.dim(1) < 2) {
        throw std::invalid_argument(std::string(who) + ": probabilities must be (N, K) with N >= 1 and K >= 2");
    }
    const int64_t n = s.dim(0), k = s.dim(1);
    if (labels != nullptr && static_cast<int64_t>(labels->size()) != n) {
        throw std::invalid_argument(std::string(who) + ": " + std::to_string(labels->size()) + " labels for " +
                                    std::to_string(n) + " examples");
    }
    const std::vector<float> flat = probabilities.to_host_vector();
    std::vector<std::vector<double>> rows(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n; ++i) {
        if (labels != nullptr && ((*labels)[static_cast<size_t>(i)] < 0 || (*labels)[static_cast<size_t>(i)] >= k)) {
            throw std::invalid_argument(std::string(who) + ": a label isn't a class in [0, " + std::to_string(k) + ")");
        }
        for (int64_t j = 0; j < k; ++j) {
            const double p = flat[static_cast<size_t>(i * k + j)];
            if (!std::isfinite(p) || p < 0.0) throw std::invalid_argument(std::string(who) + ": a probability is negative or not finite");
            rows[static_cast<size_t>(i)][static_cast<size_t>(j)] = p;
        }
    }
    return rows;
}

// The classes from most to least probable, the lower index first on a tie.
std::vector<int64_t> Ranked(const std::vector<double>& p) {
    std::vector<int64_t> order(p.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return p[static_cast<size_t>(a)] > p[static_cast<size_t>(b)]; });
    return order;
}

// Every class's score for one example; u is APS's randomization, 0 without it.
std::vector<double> Scores(ConformalScore score, const std::vector<double>& p, double u) {
    std::vector<double> s(p.size());
    if (score == ConformalScore::Lac) {
        for (size_t j = 0; j < p.size(); ++j) s[j] = 1.0 - p[j];
        return s;
    }
    double mass = 0.0;
    for (int64_t c : Ranked(p)) {
        mass += p[static_cast<size_t>(c)];
        s[static_cast<size_t>(c)] = mass - u * p[static_cast<size_t>(c)];
    }
    return s;
}

// The ⌈(n + 1)(1 − alpha)⌉-th smallest score, or +infinity when that rank is above n.
double Quantile(std::vector<double> scores, double alpha) {
    const double n = static_cast<double>(scores.size());
    // The rank, with a tolerance so (n + 1)(1 − alpha) that is a whole number in exact arithmetic
    // isn't rounded up past it.
    const double target = (n + 1.0) * (1.0 - alpha);
    const auto rank = static_cast<int64_t>(std::ceil(target - 1e-9 * std::max(1.0, target)));
    if (rank > static_cast<int64_t>(scores.size())) return std::numeric_limits<double>::infinity();
    const auto kth = scores.begin() + (std::max<int64_t>(rank, 1) - 1);
    std::nth_element(scores.begin(), kth, scores.end());
    return *kth;
}

void CheckValues(const std::vector<double>& v, const char* who) {
    for (double x : v) {
        if (!std::isfinite(x)) throw std::invalid_argument(std::string(who) + ": a value isn't finite");
    }
}

ConformalRegressor Regressor(std::string score, std::vector<double> scores, double alpha) {
    ConformalRegressor r;
    r.score = std::move(score);
    r.alpha = alpha;
    r.n = static_cast<int64_t>(scores.size());
    r.quantile = Quantile(std::move(scores), alpha);
    return r;
}

}  // namespace

ConformalClassifier CalibrateConformalClassifier(const Tensor& probabilities, const std::vector<int64_t>& labels,
                                                 const ConformalOptions& options) {
    CheckAlpha(options.alpha, "CalibrateConformalClassifier");
    if (options.randomized && options.score == ConformalScore::Lac) {
        throw std::invalid_argument("CalibrateConformalClassifier: only APS is randomized");
    }
    const std::vector<std::vector<double>> rows = Rows(probabilities, &labels, "CalibrateConformalClassifier");
    PortableRng rng{options.seed};
    std::vector<double> scores;
    scores.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const double u = options.randomized ? rng.uniform() : 0.0;
        scores.push_back(Scores(options.score, rows[i], u)[static_cast<size_t>(labels[i])]);
    }
    ConformalClassifier c;
    c.score = options.score;
    c.alpha = options.alpha;
    c.randomized = options.randomized;
    // Prediction draws a different stream from calibration, so the two u's are independent.
    c.seed = options.seed ^ 0x9E3779B97F4A7C15ULL;
    c.n = static_cast<int64_t>(rows.size());
    c.quantile = Quantile(std::move(scores), options.alpha);
    return c;
}

std::vector<std::vector<int64_t>> PredictionSets(const ConformalClassifier& classifier, const Tensor& probabilities) {
    const std::vector<std::vector<double>> rows = Rows(probabilities, nullptr, "PredictionSets");
    PortableRng rng{classifier.seed};
    std::vector<std::vector<int64_t>> sets;
    sets.reserve(rows.size());
    for (const std::vector<double>& p : rows) {
        const double u = classifier.randomized ? rng.uniform() : 0.0;
        const std::vector<double> s = Scores(classifier.score, p, u);
        std::vector<int64_t> set;
        for (int64_t c : Ranked(p)) {
            if (s[static_cast<size_t>(c)] <= classifier.quantile) set.push_back(c);
        }
        sets.push_back(std::move(set));
    }
    return sets;
}

ConformalRegressor CalibrateConformalRegressor(const std::vector<double>& predictions, const std::vector<double>& targets,
                                               double alpha) {
    CheckAlpha(alpha, "CalibrateConformalRegressor");
    if (predictions.empty() || predictions.size() != targets.size()) {
        throw std::invalid_argument("CalibrateConformalRegressor: needs as many predictions as targets, at least one");
    }
    CheckValues(predictions, "CalibrateConformalRegressor");
    CheckValues(targets, "CalibrateConformalRegressor");
    std::vector<double> scores(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) scores[i] = std::fabs(targets[i] - predictions[i]);
    return Regressor("absolute_residual", std::move(scores), alpha);
}

ConformalRegressor CalibrateConformalQuantileRegressor(const std::vector<double>& lower, const std::vector<double>& upper,
                                                       const std::vector<double>& targets, double alpha) {
    CheckAlpha(alpha, "CalibrateConformalQuantileRegressor");
    if (targets.empty() || lower.size() != targets.size() || upper.size() != targets.size()) {
        throw std::invalid_argument("CalibrateConformalQuantileRegressor: needs a lower and upper prediction per target, at least one");
    }
    CheckValues(lower, "CalibrateConformalQuantileRegressor");
    CheckValues(upper, "CalibrateConformalQuantileRegressor");
    CheckValues(targets, "CalibrateConformalQuantileRegressor");
    std::vector<double> scores(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) scores[i] = std::max(lower[i] - targets[i], targets[i] - upper[i]);
    return Regressor("cqr", std::move(scores), alpha);
}

std::vector<PredictionInterval> PredictionIntervals(const ConformalRegressor& regressor, const std::vector<double>& predictions) {
    if (regressor.score != "absolute_residual") {
        throw std::invalid_argument("PredictionIntervals: a " + regressor.score + " regressor needs lower and upper predictions");
    }
    CheckValues(predictions, "PredictionIntervals");
    std::vector<PredictionInterval> out;
    out.reserve(predictions.size());
    for (double y : predictions) out.push_back({y - regressor.quantile, y + regressor.quantile});
    return out;
}

std::vector<PredictionInterval> PredictionIntervals(const ConformalRegressor& regressor, const std::vector<double>& lower,
                                                    const std::vector<double>& upper) {
    if (regressor.score != "cqr") throw std::invalid_argument("PredictionIntervals: only a cqr regressor takes lower and upper predictions");
    if (lower.size() != upper.size()) throw std::invalid_argument("PredictionIntervals: as many lower as upper predictions");
    CheckValues(lower, "PredictionIntervals");
    CheckValues(upper, "PredictionIntervals");
    std::vector<PredictionInterval> out;
    out.reserve(lower.size());
    for (size_t i = 0; i < lower.size(); ++i) out.push_back({lower[i] - regressor.quantile, upper[i] + regressor.quantile});
    return out;
}

CoverageReport EvaluateCoverage(const std::vector<std::vector<int64_t>>& sets, const std::vector<int64_t>& labels) {
    if (sets.empty() || sets.size() != labels.size()) {
        throw std::invalid_argument("EvaluateCoverage: needs a label for each set, at least one");
    }
    CoverageReport r;
    r.n = static_cast<int64_t>(sets.size());
    std::vector<int64_t> held_by_size;
    double held = 0.0, size = 0.0;
    for (size_t i = 0; i < sets.size(); ++i) {
        const size_t s = sets[i].size();
        if (r.size_counts.size() <= s) r.size_counts.resize(s + 1, 0), held_by_size.resize(s + 1, 0);
        const bool hit = std::find(sets[i].begin(), sets[i].end(), labels[i]) != sets[i].end();
        ++r.size_counts[s];
        held_by_size[s] += hit ? 1 : 0;
        held += hit ? 1.0 : 0.0;
        size += static_cast<double>(s);
        r.empty += s == 0 ? 1 : 0;
        r.misses.push_back(hit ? 0.0 : 1.0);
    }
    for (size_t s = 0; s < r.size_counts.size(); ++s) {
        r.coverage_by_size.push_back(r.size_counts[s] > 0 ? static_cast<double>(held_by_size[s]) / static_cast<double>(r.size_counts[s]) : 0.0);
    }
    r.coverage = held / static_cast<double>(r.n);
    r.mean_size = size / static_cast<double>(r.n);
    return r;
}

CoverageReport EvaluateCoverage(const std::vector<PredictionInterval>& intervals, const std::vector<double>& targets) {
    if (intervals.empty() || intervals.size() != targets.size()) {
        throw std::invalid_argument("EvaluateCoverage: needs a target for each interval, at least one");
    }
    CoverageReport r;
    r.n = static_cast<int64_t>(intervals.size());
    double held = 0.0, width = 0.0;
    for (size_t i = 0; i < intervals.size(); ++i) {
        const bool hit = intervals[i].lower <= targets[i] && targets[i] <= intervals[i].upper;
        held += hit ? 1.0 : 0.0;
        width += intervals[i].upper - intervals[i].lower;
        r.misses.push_back(hit ? 0.0 : 1.0);
    }
    r.coverage = held / static_cast<double>(r.n);
    r.mean_size = width / static_cast<double>(r.n);
    return r;
}

JsonValue ToJson(const CoverageReport& r) {
    JsonValue::Array counts, by_size;
    for (int64_t c : r.size_counts) counts.emplace_back(c);
    for (double c : r.coverage_by_size) by_size.emplace_back(c);
    // An infinite interval width (a quantile above the calibration's largest score) is written as null.
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.coverage.v1"},
        {"n", JsonValue(r.n)},
        {"coverage", JsonValue(r.coverage)},
        {"mean_size", std::isfinite(r.mean_size) ? JsonValue(r.mean_size) : JsonValue()},
        {"empty", JsonValue(r.empty)},
        {"size_counts", JsonValue(std::move(counts))},
        {"coverage_by_size", JsonValue(std::move(by_size))},
    });
}

}  // namespace pulsatrix
