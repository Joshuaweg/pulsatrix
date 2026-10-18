#include "pulsatrix/calibration.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsatrix {

namespace {

constexpr double kMinBeta = 1e-6;  // 1/T: T at most 1e6
constexpr double kMaxBeta = 1e6;   // T at least 1e-6

std::vector<double> CheckWeights(const std::vector<double>& weights, size_t n, const char* who) {
    if (weights.empty()) return std::vector<double>(n, 1.0);
    if (weights.size() != n) throw std::invalid_argument(std::string(who) + ": weights and inputs differ in length");
    double total = 0.0;
    for (const double w : weights) {
        if (!std::isfinite(w) || w < 0.0) throw std::invalid_argument(std::string(who) + ": a weight is negative or not finite");
        total += w;
    }
    if (total <= 0.0) throw std::invalid_argument(std::string(who) + ": every weight is 0");
    return weights;
}

// log(1 + exp(x)) without overflow.
double Softplus(double x) { return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); }

double Sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

/** Per example at inverse temperature beta: loss, d loss / d beta and d^2 loss / d beta^2. */
using Term = std::function<void(size_t i, double beta, double& loss, double& g, double& h)>;

/** Per example: whether the label's logit is (weakly) the largest, and whether it is strictly
 *  larger than some other logit. */
using Separation = std::function<std::pair<bool, bool>(size_t i)>;

/**
 * Minimizes the weighted mean loss over beta = 1/T, which is convex. The two ends are decided
 * exactly first: the minimum is at T -> 0 when every label has the largest logit (the loss falls
 * forever as beta grows), and at T -> infinity when the loss already rises at beta = 0. Otherwise
 * it is interior, and a safeguarded Newton iteration finds it.
 */
TemperatureFit Fit(size_t n, const std::vector<double>& weights, const Term& term, const Separation& separation) {
    double total_weight = 0.0;
    for (const double w : weights) total_weight += w;
    auto evaluate = [&](double beta, double& loss, double& g, double& h) {
        loss = g = h = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (weights[i] == 0.0) continue;
            double l = 0.0, gi = 0.0, hi = 0.0;
            term(i, beta, l, gi, hi);
            loss += weights[i] * l;
            g += weights[i] * gi;
            h += weights[i] * hi;
        }
        loss /= total_weight;
        g /= total_weight;
        h /= total_weight;
    };

    TemperatureFit fit;
    double loss = 0.0, g = 0.0, h = 0.0;
    evaluate(1.0, loss, g, h);
    fit.log_loss_before = loss;
    fit.log_loss_after = loss;
    fit.iterations = 0;

    bool separated = true, strict = false;
    for (size_t i = 0; i < n && separated; ++i) {
        if (weights[i] == 0.0) continue;
        const auto [top, gap] = separation(i);
        separated = top;
        strict = strict || gap;
    }
    if (separated && !strict) return fit;  // every logit ties: T changes nothing
    double beta = 1.0;
    double g0 = 0.0, unused = 0.0;
    evaluate(0.0, unused, g0, unused);
    if (separated || g0 >= 0.0) {
        beta = separated ? kMaxBeta : kMinBeta;
        evaluate(beta, loss, g, h);
        fit.converged = false;
    } else {
        double lo = 0.0, hi = kMaxBeta;  // g(lo) < 0, and g < 0 at every beta below the root
        for (fit.iterations = 1; fit.iterations <= 200; ++fit.iterations) {
            if (g < 0.0) lo = beta;
            else if (g > 0.0) hi = beta;
            else break;
            double next = h > 0.0 ? beta - g / h : -1.0;
            if (!(next > lo && next < hi)) next = hi >= kMaxBeta ? 2.0 * beta : 0.5 * (lo + hi);
            if (std::abs(next - beta) <= 1e-13 * beta) break;
            beta = std::min(next, kMaxBeta);
            evaluate(beta, loss, g, h);
        }
        if (beta < kMinBeta || beta >= kMaxBeta) {
            beta = std::min(std::max(beta, kMinBeta), kMaxBeta);
            evaluate(beta, loss, g, h);
            fit.converged = false;
        }
    }
    fit.temperature = 1.0 / beta;
    fit.log_loss_after = loss;
    return fit;
}

double Logit(double p, double clip) {
    const double c = std::min(std::max(p, clip), 1.0 - clip);
    return std::log(c) - std::log1p(-c);
}

// The smallest value with at least fraction q of the weight at or below it.
double WeightedQuantile(std::vector<std::pair<double, double>> values, double q) {
    std::sort(values.begin(), values.end());
    double total = 0.0;
    for (const auto& v : values) total += v.second;
    double running = 0.0;
    for (const auto& v : values) {
        running += v.second;
        if (running >= q * total * (1.0 - 1e-12)) return v.first;
    }
    return values.empty() ? 0.0 : values.back().first;
}

LogOddsBand Summarize(double lower, double upper, const std::vector<size_t>& members, const std::vector<double>& errors,
                      const std::vector<double>& weights) {
    LogOddsBand band;
    band.lower = lower;
    band.upper = upper;
    std::vector<std::pair<double, double>> abs_errors;
    double sum_abs = 0.0, sum = 0.0, sum_sq = 0.0;
    for (const size_t i : members) {
        const double w = weights[i], e = errors[i];
        ++band.count;
        band.weight += w;
        sum_abs += w * std::abs(e);
        sum += w * e;
        sum_sq += w * e * e;
        band.max_abs = std::max(band.max_abs, std::abs(e));
        if (w > 0.0) abs_errors.emplace_back(std::abs(e), w);
    }
    if (band.weight > 0.0) {
        band.mean_abs = sum_abs / band.weight;
        band.bias = sum / band.weight;
        band.rms = std::sqrt(sum_sq / band.weight);
        band.median_abs = WeightedQuantile(abs_errors, 0.5);
        band.p90_abs = WeightedQuantile(abs_errors, 0.9);
    }
    return band;
}

JsonValue BandJson(const LogOddsBand& b) {
    return JsonValue(JsonValue::Object{{"lower", b.lower},       {"upper", b.upper},           {"count", b.count},
                                       {"weight", b.weight},     {"mean_abs", b.mean_abs},     {"bias", b.bias},
                                       {"rms", b.rms},           {"median_abs", b.median_abs}, {"p90_abs", b.p90_abs},
                                       {"max_abs", b.max_abs}});
}

}  // namespace

TemperatureFit FitBinaryTemperature(const std::vector<double>& logits, const std::vector<int64_t>& labels,
                                    const std::vector<double>& weights) {
    if (logits.empty()) throw std::invalid_argument("FitBinaryTemperature: no examples");
    if (labels.size() != logits.size()) throw std::invalid_argument("FitBinaryTemperature: logits and labels differ in length");
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) throw std::invalid_argument("FitBinaryTemperature: a logit isn't finite");
        if (labels[i] != 0 && labels[i] != 1) throw std::invalid_argument("FitBinaryTemperature: a label isn't 0 or 1");
    }
    const std::vector<double> w = CheckWeights(weights, logits.size(), "FitBinaryTemperature");
    return Fit(logits.size(), w, [&](size_t i, double beta, double& loss, double& g, double& h) {
        const double z = logits[i], x = beta * z;
        const double y = static_cast<double>(labels[i]);
        loss = Softplus(x) - y * x;
        const double p = Sigmoid(x);
        g = (p - y) * z;
        h = p * (1.0 - p) * z * z;
    }, [&](size_t i) {
        const double signed_z = labels[i] == 1 ? logits[i] : -logits[i];
        return std::make_pair(signed_z >= 0.0, signed_z > 0.0);
    });
}

TemperatureFit FitTemperature(const std::vector<std::vector<double>>& logits, const std::vector<int64_t>& labels,
                              const std::vector<double>& weights) {
    if (logits.empty()) throw std::invalid_argument("FitTemperature: no examples");
    if (labels.size() != logits.size()) throw std::invalid_argument("FitTemperature: logits and labels differ in length");
    const size_t k = logits.front().size();
    if (k < 2) throw std::invalid_argument("FitTemperature: rows need at least two classes");
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i].size() != k) throw std::invalid_argument("FitTemperature: rows differ in length");
        for (const double z : logits[i]) {
            if (!std::isfinite(z)) throw std::invalid_argument("FitTemperature: a logit isn't finite");
        }
        if (labels[i] < 0 || static_cast<size_t>(labels[i]) >= k) throw std::invalid_argument("FitTemperature: a label is outside [0, K)");
    }
    const std::vector<double> w = CheckWeights(weights, logits.size(), "FitTemperature");
    return Fit(logits.size(), w, [&](size_t i, double beta, double& loss, double& g, double& h) {
        const std::vector<double>& z = logits[i];
        const double top = beta * *std::max_element(z.begin(), z.end());
        double sum = 0.0, mean = 0.0, second = 0.0;
        for (const double v : z) {
            const double e = std::exp(beta * v - top);
            sum += e;
            mean += e * v;
            second += e * v * v;
        }
        mean /= sum;
        second /= sum;
        const double zy = z[static_cast<size_t>(labels[i])];
        loss = top + std::log(sum) - beta * zy;
        g = mean - zy;
        h = std::max(0.0, second - mean * mean);
    }, [&](size_t i) {
        const std::vector<double>& z = logits[i];
        const double zy = z[static_cast<size_t>(labels[i])];
        const double top = *std::max_element(z.begin(), z.end());
        const double bottom = *std::min_element(z.begin(), z.end());
        return std::make_pair(zy >= top, zy > bottom);
    });
}

LogOddsErrorReport LogOddsError(const std::vector<double>& predicted, const std::vector<double>& reference,
                                const std::vector<double>& weights, const std::vector<double>& band_edges, double clip) {
    if (predicted.empty()) throw std::invalid_argument("LogOddsError: no examples");
    if (reference.size() != predicted.size()) throw std::invalid_argument("LogOddsError: predicted and reference differ in length");
    if (!(clip > 0.0 && clip < 0.5)) throw std::invalid_argument("LogOddsError: clip must be in (0, 0.5)");
    for (size_t b = 0; b < band_edges.size(); ++b) {
        if (!(band_edges[b] > 0.0 && band_edges[b] < 1.0) || (b > 0 && !(band_edges[b] > band_edges[b - 1]))) {
            throw std::invalid_argument("LogOddsError: band edges must increase strictly inside (0, 1)");
        }
    }
    for (size_t i = 0; i < predicted.size(); ++i) {
        for (const double p : {predicted[i], reference[i]}) {
            if (!std::isfinite(p) || p < 0.0 || p > 1.0) throw std::invalid_argument("LogOddsError: a probability is outside [0, 1]");
        }
    }
    const std::vector<double> w = CheckWeights(weights, predicted.size(), "LogOddsError");

    std::vector<double> errors(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) errors[i] = Logit(predicted[i], clip) - Logit(reference[i], clip);
    std::vector<double> edges = {0.0};
    edges.insert(edges.end(), band_edges.begin(), band_edges.end());
    edges.push_back(1.0);
    std::vector<std::vector<size_t>> members(edges.size() - 1);
    std::vector<size_t> all(predicted.size());
    std::iota(all.begin(), all.end(), size_t{0});
    for (size_t i = 0; i < predicted.size(); ++i) {
        size_t b = 0;
        while (b + 1 < members.size() && reference[i] >= edges[b + 1]) ++b;
        members[b].push_back(i);
    }
    LogOddsErrorReport report;
    report.clip = clip;
    report.overall = Summarize(0.0, 1.0, all, errors, w);
    for (size_t b = 0; b < members.size(); ++b) report.bands.push_back(Summarize(edges[b], edges[b + 1], members[b], errors, w));
    return report;
}

JsonValue ToJson(const LogOddsErrorReport& report) {
    JsonValue::Array bands;
    for (const LogOddsBand& b : report.bands) bands.push_back(BandJson(b));
    return JsonValue(JsonValue::Object{{"clip", report.clip}, {"overall", BandJson(report.overall)}, {"bands", JsonValue(std::move(bands))}});
}

}  // namespace pulsatrix
