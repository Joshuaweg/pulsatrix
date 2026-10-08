#include "pulsatrix/fitness_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pulsatrix {
namespace {

void CheckSizes(size_t a, size_t b, const char* who) {
    if (a != b) throw std::invalid_argument(std::string(who) + ": the inputs differ in length");
    if (a < 2) throw std::invalid_argument(std::string(who) + ": needs at least two values");
}

/** @brief Ranks from 1, ties sharing their average rank. */
std::vector<double> AverageRanks(const std::vector<double>& v) {
    std::vector<size_t> order(v.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> ranks(v.size());
    for (size_t i = 0; i < order.size();) {
        size_t j = i;
        while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) ++j;
        const double r = 0.5 * static_cast<double>(i + j) + 1.0;
        for (size_t k = i; k <= j; ++k) ranks[order[k]] = r;
        i = j + 1;
    }
    return ranks;
}

double Pearson(const std::vector<double>& x, const std::vector<double>& y) {
    const double n = static_cast<double>(x.size());
    const double mx = std::accumulate(x.begin(), x.end(), 0.0) / n, my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    if (sxx == 0 || syy == 0) return std::numeric_limits<double>::quiet_NaN();
    return sxy / std::sqrt(sxx * syy);
}

/** @brief 1-based ranks by descending score, ties in input order (argsort(argsort(-s)) + 1). */
std::vector<int64_t> DescendingRanks(const std::vector<double>& s) {
    std::vector<size_t> order(s.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return s[a] > s[b]; });
    std::vector<int64_t> ranks(s.size());
    for (size_t r = 0; r < order.size(); ++r) ranks[order[r]] = static_cast<int64_t>(r) + 1;
    return ranks;
}

}  // namespace

double SpearmanCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    CheckSizes(x.size(), y.size(), "SpearmanCorrelation");
    return Pearson(AverageRanks(x), AverageRanks(y));
}

double RocAuc(const std::vector<int>& labels, const std::vector<double>& scores) {
    CheckSizes(labels.size(), scores.size(), "RocAuc");
    const std::vector<double> ranks = AverageRanks(scores);
    double positives = 0, rank_sum = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != 0 && labels[i] != 1) throw std::invalid_argument("RocAuc: labels must be 0 or 1");
        if (labels[i] == 1) {
            positives += 1;
            rank_sum += ranks[i];
        }
    }
    const double negatives = static_cast<double>(labels.size()) - positives;
    if (positives == 0 || negatives == 0) return std::numeric_limits<double>::quiet_NaN();
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives);
}

double MatthewsCorrelation(const std::vector<int>& labels, const std::vector<int>& predicted) {
    CheckSizes(labels.size(), predicted.size(), "MatthewsCorrelation");
    double tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        const bool t = labels[i] == 1, p = predicted[i] == 1;
        tp += t && p;
        tn += !t && !p;
        fp += !t && p;
        fn += t && !p;
    }
    const double d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn);
    return d == 0 ? 0.0 : (tp * tn - fp * fn) / std::sqrt(d);
}

double Percentile(std::vector<double> values, double q) {
    if (values.empty()) throw std::invalid_argument("Percentile: no values");
    std::sort(values.begin(), values.end());
    const double pos = q / 100.0 * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (pos - static_cast<double>(lo)) * (values[hi] - values[lo]);
}

double MedianSplitMcc(const std::vector<int>& labels, const std::vector<double>& scores) {
    CheckSizes(labels.size(), scores.size(), "MedianSplitMcc");
    const double median = Percentile(scores, 50.0);
    std::vector<int> predicted(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) predicted[i] = scores[i] >= median ? 1 : 0;
    return MatthewsCorrelation(labels, predicted);
}

double Ndcg(const std::vector<double>& true_scores, const std::vector<double>& predicted, double top_percent) {
    CheckSizes(true_scores.size(), predicted.size(), "Ndcg");
    const auto k = static_cast<int64_t>(std::floor(static_cast<double>(true_scores.size()) * (top_percent / 100.0)));
    const auto [lo, hi] = std::minmax_element(true_scores.begin(), true_scores.end());
    std::vector<double> gains(true_scores.size());
    for (size_t i = 0; i < gains.size(); ++i) gains[i] = (true_scores[i] - *lo) / (*hi - *lo);
    const auto dcg_of = [&](const std::vector<int64_t>& ranks) {
        double s = 0;
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (ranks[i] <= k && gains[i] != 0) s += gains[i] / std::log2(static_cast<double>(ranks[i]) + 1);
        }
        return s;
    };
    const double dcg = dcg_of(DescendingRanks(predicted));
    if (dcg == 0) return 0.0;
    return dcg / dcg_of(DescendingRanks(gains));
}

double TopRecall(const std::vector<double>& true_scores, const std::vector<double>& predicted, double top_true, double top_model) {
    CheckSizes(true_scores.size(), predicted.size(), "TopRecall");
    const double t = Percentile(true_scores, 100.0 - top_true), m = Percentile(predicted, 100.0 - top_model);
    double top = 0, hit = 0;
    for (size_t i = 0; i < true_scores.size(); ++i) {
        if (true_scores[i] >= t) {
            top += 1;
            hit += predicted[i] >= m;
        }
    }
    return top > 0 ? hit / top : 0.0;
}

FitnessMetrics EvaluateFitness(const std::vector<double>& dms_scores, const std::vector<int>& dms_bins, const std::vector<double>& predicted) {
    return {SpearmanCorrelation(dms_scores, predicted), RocAuc(dms_bins, predicted), MedianSplitMcc(dms_bins, predicted), Ndcg(dms_scores, predicted),
            TopRecall(dms_scores, predicted)};
}

}  // namespace pulsatrix
