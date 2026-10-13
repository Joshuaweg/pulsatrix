// KS-3: how well a model does. The metrics follow scikit-learn's definitions, the calibration error
// Guo, Pleiss, Sun and Weinberger (ICML 2017), and the per-slice test Fisher's exact test, as SciPy's
// fisher_exact(alternative="greater").

#include "pulsatrix/performance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "normal_quantile.hpp"
#include "portable_random.hpp"

namespace pulsatrix {
namespace {

std::string Fmt(double v, const char* format = "%.4g") {
    char buf[40];
    std::snprintf(buf, sizeof buf, format, v);
    return buf;
}

// The probabilities as N rows of K, checked.
std::vector<std::vector<double>> Rows(const Tensor& probabilities, const std::vector<int64_t>& labels, const char* who) {
    const Shape& s = probabilities.shape();
    if (s.rank() != 2 || s.dim(0) < 1 || s.dim(1) < 2) {
        throw std::invalid_argument(std::string(who) + ": probabilities must be (N, K) with N >= 1 and K >= 2");
    }
    const int64_t n = s.dim(0), k = s.dim(1);
    if (static_cast<int64_t>(labels.size()) != n) {
        throw std::invalid_argument(std::string(who) + ": " + std::to_string(labels.size()) + " labels for " +
                                    std::to_string(n) + " inputs");
    }
    const std::vector<float> flat = probabilities.to_host_vector();
    std::vector<std::vector<double>> rows(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(k)));
    for (int64_t i = 0; i < n; ++i) {
        if (labels[static_cast<size_t>(i)] < 0 || labels[static_cast<size_t>(i)] >= k) {
            throw std::invalid_argument(std::string(who) + ": label " + std::to_string(labels[static_cast<size_t>(i)]) +
                                        " isn't a class in [0, " + std::to_string(k) + ")");
        }
        for (int64_t j = 0; j < k; ++j) {
            const double p = flat[static_cast<size_t>(i * k + j)];
            if (!std::isfinite(p) || p < 0.0) {
                throw std::invalid_argument(std::string(who) + ": a probability is negative or not finite");
            }
            rows[static_cast<size_t>(i)][static_cast<size_t>(j)] = p;
        }
    }
    return rows;
}

int64_t ArgMax(const std::vector<double>& row) {
    return static_cast<int64_t>(std::max_element(row.begin(), row.end()) - row.begin());
}

double Ratio(double a, double b) { return b > 0.0 ? a / b : 0.0; }

// log of the hypergeometric probability of x errors in a slice of n, from N inputs with K errors.
double LogHypergeometric(int64_t x, int64_t n, int64_t K, int64_t N) {
    const auto lchoose = [](double a, double b) { return std::lgamma(a + 1) - std::lgamma(b + 1) - std::lgamma(a - b + 1); };
    return lchoose(static_cast<double>(K), static_cast<double>(x)) +
           lchoose(static_cast<double>(N - K), static_cast<double>(n - x)) -
           lchoose(static_cast<double>(N), static_cast<double>(n));
}

// One-sided Fisher exact p-value: P(X >= errors) for X the errors a random slice of its size gets.
double FisherGreater(int64_t errors, int64_t n, int64_t total_errors, int64_t total) {
    const int64_t hi = std::min(n, total_errors);
    double p = 0.0;
    for (int64_t x = errors; x <= hi; ++x) p += std::exp(LogHypergeometric(x, n, total_errors, total));
    return std::min(1.0, p);
}

}  // namespace

ClassificationReport EvaluateClassifier(const Tensor& probabilities, const std::vector<int64_t>& labels,
                                        const ClassificationOptions& options) {
    if (options.bins < 1) throw std::invalid_argument("EvaluateClassifier: bins must be at least 1");
    const std::vector<std::vector<double>> rows = Rows(probabilities, labels, "EvaluateClassifier");
    const int64_t n = static_cast<int64_t>(rows.size()), k = static_cast<int64_t>(rows.front().size());
    ClassificationReport r;
    r.n = n;
    r.classes = k;
    r.confusion.assign(static_cast<size_t>(k), std::vector<int64_t>(static_cast<size_t>(k), 0));
    r.bins.resize(static_cast<size_t>(options.bins));
    for (int64_t b = 0; b < options.bins; ++b) {
        r.bins[static_cast<size_t>(b)].lower = static_cast<double>(b) / static_cast<double>(options.bins);
        r.bins[static_cast<size_t>(b)].upper = static_cast<double>(b + 1) / static_cast<double>(options.bins);
    }
    std::vector<double> bin_correct(static_cast<size_t>(options.bins), 0.0);
    int64_t correct = 0;
    double nll = 0.0, brier = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const std::vector<double>& p = rows[static_cast<size_t>(i)];
        const int64_t y = labels[static_cast<size_t>(i)], predicted = ArgMax(p);
        const bool right = predicted == y;
        correct += right ? 1 : 0;
        ++r.confusion[static_cast<size_t>(y)][static_cast<size_t>(predicted)];
        nll -= std::log(std::max(p[static_cast<size_t>(y)], 1e-15));
        for (int64_t j = 0; j < k; ++j) {
            const double d = p[static_cast<size_t>(j)] - (j == y ? 1.0 : 0.0);
            brier += d * d;
        }
        // Bin m holds ((m-1)/M, m/M]; ceil puts a confidence on an edge in the bin below it.
        const double confidence = p[static_cast<size_t>(predicted)];
        int64_t b = static_cast<int64_t>(std::ceil(confidence * static_cast<double>(options.bins))) - 1;
        b = std::min(std::max<int64_t>(b, 0), options.bins - 1);
        ReliabilityBin& bin = r.bins[static_cast<size_t>(b)];
        ++bin.count;
        bin.confidence += confidence;
        bin_correct[static_cast<size_t>(b)] += right ? 1.0 : 0.0;
    }
    r.accuracy = static_cast<double>(correct) / static_cast<double>(n);
    r.log_loss = nll / static_cast<double>(n);
    r.brier = brier / static_cast<double>(n);
    for (size_t b = 0; b < r.bins.size(); ++b) {
        ReliabilityBin& bin = r.bins[b];
        if (bin.count == 0) continue;
        bin.confidence /= static_cast<double>(bin.count);
        bin.accuracy = bin_correct[b] / static_cast<double>(bin.count);
        const double gap = std::fabs(bin.accuracy - bin.confidence);
        r.ece += static_cast<double>(bin.count) / static_cast<double>(n) * gap;
        r.mce = std::max(r.mce, gap);
    }
    for (int64_t c = 0; c < k; ++c) {
        int64_t predicted = 0, actual = 0;
        for (int64_t j = 0; j < k; ++j) {
            predicted += r.confusion[static_cast<size_t>(j)][static_cast<size_t>(c)];
            actual += r.confusion[static_cast<size_t>(c)][static_cast<size_t>(j)];
        }
        const double hit = static_cast<double>(r.confusion[static_cast<size_t>(c)][static_cast<size_t>(c)]);
        const double precision = Ratio(hit, static_cast<double>(predicted));
        const double recall = Ratio(hit, static_cast<double>(actual));
        r.precision.push_back(precision);
        r.recall.push_back(recall);
        r.f1.push_back(Ratio(2.0 * precision * recall, precision + recall));
        r.support.push_back(actual);
    }
    r.macro_f1 = std::accumulate(r.f1.begin(), r.f1.end(), 0.0) / static_cast<double>(k);
    return r;
}

RegressionReport EvaluateRegressor(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.empty() || predictions.size() != targets.size()) {
        throw std::invalid_argument("EvaluateRegressor: needs as many predictions as targets, at least one");
    }
    const double n = static_cast<double>(targets.size());
    double mean = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (!std::isfinite(predictions[i]) || !std::isfinite(targets[i])) {
            throw std::invalid_argument("EvaluateRegressor: a prediction or target isn't finite");
        }
        mean += targets[i];
    }
    mean /= n;
    RegressionReport r;
    r.n = static_cast<int64_t>(targets.size());
    double sse = 0.0, sst = 0.0, sae = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        const double e = targets[i] - predictions[i];
        sse += e * e;
        sae += std::fabs(e);
        sst += (targets[i] - mean) * (targets[i] - mean);
    }
    r.mse = sse / n;
    r.rmse = std::sqrt(r.mse);
    r.mae = sae / n;
    r.r2 = sst > 0.0 ? 1.0 - sse / sst : (sse == 0.0 ? 1.0 : 0.0);
    return r;
}

bool SliceErrorReport::any_worse() const {
    return std::any_of(slices.begin(), slices.end(), [](const SliceError& s) { return s.worse; });
}

SliceErrorReport PerSliceError(const std::vector<double>& errors, const std::vector<std::string>& slices,
                               const SliceErrorOptions& options) {
    if (errors.empty()) throw std::invalid_argument("PerSliceError: needs at least one input");
    if (slices.size() != errors.size()) {
        throw std::invalid_argument("PerSliceError: " + std::to_string(slices.size()) + " slice names for " +
                                    std::to_string(errors.size()) + " errors");
    }
    if (!(options.alpha > 0.0 && options.alpha < 1.0)) throw std::invalid_argument("PerSliceError: alpha must be in (0, 1)");
    if (options.permutations < 1) throw std::invalid_argument("PerSliceError: permutations must be at least 1");
    bool binary = true;
    for (double e : errors) {
        if (!std::isfinite(e)) throw std::invalid_argument("PerSliceError: an error isn't finite");
        binary = binary && (e == 0.0 || e == 1.0);
    }
    if (options.test == SliceTest::FisherExact && !binary) {
        throw std::invalid_argument("PerSliceError: Fisher's exact test needs every error to be 0 or 1");
    }
    const bool fisher = options.test == SliceTest::FisherExact || (options.test == SliceTest::Auto && binary);

    // Slices in the order they first appear, and each input's slice.
    std::vector<std::string> names;
    std::vector<size_t> slice_of(errors.size());
    for (size_t i = 0; i < errors.size(); ++i) {
        const auto it = std::find(names.begin(), names.end(), slices[i]);
        slice_of[i] = static_cast<size_t>(it - names.begin());
        if (it == names.end()) names.push_back(slices[i]);
    }
    const size_t m = names.size();
    const int64_t total = static_cast<int64_t>(errors.size());
    std::vector<int64_t> count(m, 0);
    std::vector<double> sum(m, 0.0), sum_sq(m, 0.0);
    double all = 0.0;
    for (size_t i = 0; i < errors.size(); ++i) {
        ++count[slice_of[i]];
        sum[slice_of[i]] += errors[i];
        sum_sq[slice_of[i]] += errors[i] * errors[i];
        all += errors[i];
    }

    // Permutation test: shuffle which inputs belong to which slice, keeping the slice sizes.
    std::vector<int64_t> exceed(m, 0);
    if (!fisher) {
        PortableRng rng{options.seed};
        std::vector<size_t> shuffled = slice_of;
        std::vector<double> perm_sum(m);
        for (int64_t b = 0; b < options.permutations; ++b) {
            for (size_t i = shuffled.size(); i > 1; --i) {
                const size_t j = std::min(static_cast<size_t>(rng.uniform() * static_cast<double>(i)), i - 1);
                std::swap(shuffled[i - 1], shuffled[j]);
            }
            std::fill(perm_sum.begin(), perm_sum.end(), 0.0);
            for (size_t i = 0; i < errors.size(); ++i) perm_sum[shuffled[i]] += errors[i];
            // A slice's sum orders its mean against the rest's, the total being fixed; the tolerance
            // counts sums equal up to rounding as ties.
            for (size_t s = 0; s < m; ++s) exceed[s] += perm_sum[s] >= sum[s] - 1e-9 * (1.0 + std::fabs(sum[s])) ? 1 : 0;
        }
    }

    const double z = NormalQuantile(1.0 - options.alpha / 2.0);
    SliceErrorReport r;
    r.n = total;
    r.error = all / static_cast<double>(total);
    r.test = fisher ? "fisher_exact" : "permutation";
    r.alpha = options.alpha;
    for (size_t s = 0; s < m; ++s) {
        SliceError e;
        e.name = names[s];
        e.count = count[s];
        const double n = static_cast<double>(count[s]);
        e.error = sum[s] / n;
        const int64_t rest = total - count[s];
        e.rest_error = rest > 0 ? (all - sum[s]) / static_cast<double>(rest) : 0.0;
        if (binary) {
            const double centre = (e.error + z * z / (2 * n)) / (1 + z * z / n);
            const double half = z / (1 + z * z / n) * std::sqrt(e.error * (1 - e.error) / n + z * z / (4 * n * n));
            e.lower = centre - half;
            e.upper = centre + half;
        } else {
            const double variance = n > 1 ? std::max(0.0, (sum_sq[s] - n * e.error * e.error) / (n - 1)) : 0.0;
            e.lower = e.error - z * std::sqrt(variance / n);
            e.upper = e.error + z * std::sqrt(variance / n);
        }
        if (rest == 0) {
            e.p_value = 1.0;
        } else if (fisher) {
            e.p_value = FisherGreater(static_cast<int64_t>(std::llround(sum[s])), count[s],
                                      static_cast<int64_t>(std::llround(all)), total);
        } else {
            e.p_value = static_cast<double>(1 + exceed[s]) / static_cast<double>(1 + options.permutations);
        }
        e.worse = e.p_value <= options.alpha / static_cast<double>(m);
        r.slices.push_back(std::move(e));
    }
    std::stable_sort(r.slices.begin(), r.slices.end(), [](const SliceError& a, const SliceError& b) { return a.error > b.error; });
    return r;
}

SliceErrorReport PerSliceError(const Tensor& probabilities, const std::vector<int64_t>& labels,
                               const std::vector<std::string>& slices, const SliceErrorOptions& options) {
    const std::vector<std::vector<double>> rows = Rows(probabilities, labels, "PerSliceError");
    std::vector<double> errors(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) errors[i] = ArgMax(rows[i]) == labels[i] ? 0.0 : 1.0;
    return PerSliceError(errors, slices, options);
}

JsonValue ToJson(const ClassificationReport& r) {
    JsonValue::Array bins, confusion, classes;
    for (const ReliabilityBin& b : r.bins) {
        bins.push_back(JsonValue(JsonValue::Object{
            {"lower", JsonValue(b.lower)},
            {"upper", JsonValue(b.upper)},
            {"count", JsonValue(b.count)},
            {"confidence", JsonValue(b.confidence)},
            {"accuracy", JsonValue(b.accuracy)},
        }));
    }
    for (const std::vector<int64_t>& row : r.confusion) {
        JsonValue::Array cells;
        for (int64_t v : row) cells.emplace_back(v);
        confusion.emplace_back(std::move(cells));
    }
    for (size_t c = 0; c < r.f1.size(); ++c) {
        classes.push_back(JsonValue(JsonValue::Object{
            {"precision", JsonValue(r.precision[c])},
            {"recall", JsonValue(r.recall[c])},
            {"f1", JsonValue(r.f1[c])},
            {"support", JsonValue(r.support[c])},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.classification_report.v1"},
        {"n", JsonValue(r.n)},
        {"classes", JsonValue(r.classes)},
        {"accuracy", JsonValue(r.accuracy)},
        {"log_loss", JsonValue(r.log_loss)},
        {"brier", JsonValue(r.brier)},
        {"ece", JsonValue(r.ece)},
        {"mce", JsonValue(r.mce)},
        {"macro_f1", JsonValue(r.macro_f1)},
        {"reliability", JsonValue(std::move(bins))},
        {"confusion", JsonValue(std::move(confusion))},
        {"per_class", JsonValue(std::move(classes))},
    });
}

JsonValue ToJson(const RegressionReport& r) {
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.regression_report.v1"},
        {"n", JsonValue(r.n)},
        {"mse", JsonValue(r.mse)},
        {"rmse", JsonValue(r.rmse)},
        {"mae", JsonValue(r.mae)},
        {"r2", JsonValue(r.r2)},
    });
}

JsonValue ToJson(const SliceErrorReport& r) {
    JsonValue::Array slices;
    for (const SliceError& s : r.slices) {
        slices.push_back(JsonValue(JsonValue::Object{
            {"name", s.name},
            {"count", JsonValue(s.count)},
            {"error", JsonValue(s.error)},
            {"lower", JsonValue(s.lower)},
            {"upper", JsonValue(s.upper)},
            {"rest_error", JsonValue(s.rest_error)},
            {"p_value", JsonValue(s.p_value)},
            {"worse", JsonValue(s.worse)},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.slice_error.v1"},
        {"n", JsonValue(r.n)},
        {"error", JsonValue(r.error)},
        {"test", r.test},
        {"alpha", JsonValue(r.alpha)},
        {"slices", JsonValue(std::move(slices))},
    });
}

std::string DescribeSliceError(const SliceErrorReport& r) {
    size_t width = 5;
    for (const SliceError& s : r.slices) width = std::max(width, s.name.size());
    const auto pad = [](std::string s, size_t w) { return s.size() < w ? s + std::string(w - s.size(), ' ') : s; };
    const int level = static_cast<int>(std::lround(100.0 * (1.0 - r.alpha)));
    std::string out = "error " + Fmt(r.error) + " over " + std::to_string(r.n) + " inputs; each slice against the rest (" +
                      (r.test == "fisher_exact" ? "Fisher's exact test" : "permutation test") + ", alpha " + Fmt(r.alpha) +
                      " over " + std::to_string(r.slices.size()) + " slices)\n";
    out += "  " + pad("slice", width) + "  " + pad("n", 6) + "  " + pad("error", 7) + "  " + pad(std::to_string(level) + "% interval", 16) +
           "  " + pad("rest", 7) + "  p\n";
    for (const SliceError& s : r.slices) {
        out += "  " + pad(s.name, width) + "  " + pad(std::to_string(s.count), 6) + "  " + pad(Fmt(s.error, "%.3f"), 7) + "  " +
               pad(Fmt(s.lower, "%.3f") + " to " + Fmt(s.upper, "%.3f"), 16) + "  " + pad(Fmt(s.rest_error, "%.3f"), 7) + "  " +
               Fmt(s.p_value, "%.3g") + (s.worse ? "  WORSE" : "") + "\n";
    }
    out += r.any_worse() ? "at least one slice is worse than the rest\n" : "no slice is worse than the rest\n";
    return out;
}

}  // namespace pulsatrix
