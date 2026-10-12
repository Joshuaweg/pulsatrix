#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "portable_random.hpp"
#include "pulsatrix/representation_similarity.hpp"

namespace pulsatrix {
namespace {

std::size_t PairIndex(int64_t n, int64_t i, int64_t j) {
    // pdist's order: row i's pairs (i, i+1), ..., (i, n-1) follow every earlier row's.
    return static_cast<std::size_t>(n * i - i * (i + 1) / 2 + (j - i - 1));
}

void CheckRdm(const Rdm& r, const char* function, const char* what) {
    if (r.n < 3) throw std::invalid_argument(std::string(function) + ": " + what + " must describe at least 3 inputs");
    if (r.values.size() != static_cast<std::size_t>(r.n * (r.n - 1) / 2)) {
        throw std::invalid_argument(std::string(function) + ": " + what + " holds " + std::to_string(r.values.size()) +
                                    " values; " + std::to_string(r.n) + " inputs have " + std::to_string(r.n * (r.n - 1) / 2) + " pairs");
    }
    for (double v : r.values) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(function) + ": " + what + " has a NaN or infinite value");
    }
}

// Ranks from 1, ties given the average of the ranks they span.
std::vector<double> AverageRanks(const std::vector<double>& v) {
    std::vector<std::size_t> order(v.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> rank(v.size());
    for (std::size_t i = 0; i < order.size();) {
        std::size_t j = i;
        while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) ++j;
        const double average = 0.5 * static_cast<double>(i + j) + 1.0;
        for (std::size_t t = i; t <= j; ++t) rank[order[t]] = average;
        i = j + 1;
    }
    return rank;
}

// The values centered and scaled to unit norm, so that a dot product of two is their Pearson
// correlation; empty if they are constant.
std::vector<double> Standardize(std::vector<double> v) {
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    double norm = 0;
    for (double& t : v) {
        t -= mean;
        norm += t * t;
    }
    norm = std::sqrt(norm);
    if (!(norm > 0)) return {};
    for (double& t : v) t /= norm;
    return v;
}

std::vector<double> Prepare(const Rdm& r, RdmComparison comparison) {
    return Standardize(comparison == RdmComparison::Spearman ? AverageRanks(r.values) : r.values);
}

void CheckPair(const Rdm& a, const Rdm& b, const char* function) {
    CheckRdm(a, function, "the first RDM");
    CheckRdm(b, function, "the second RDM");
    if (a.n != b.n) {
        throw std::invalid_argument(std::string(function) + ": the RDMs describe " + std::to_string(a.n) + " and " + std::to_string(b.n) +
                                    " inputs; they must describe the same ones");
    }
}

}  // namespace

double Rdm::at(int64_t i, int64_t j) const {
    if (i < 0 || j < 0 || i >= n || j >= n) {
        throw std::out_of_range("Rdm::at: (" + std::to_string(i) + ", " + std::to_string(j) + ") is outside " + std::to_string(n) + " inputs");
    }
    if (i == j) return 0.0;
    return values[i < j ? PairIndex(n, i, j) : PairIndex(n, j, i)];
}

Rdm RepresentationalDissimilarity(const Tensor& x, RdmMetric metric) {
    if (x.rank() < 2) throw std::invalid_argument("RepresentationalDissimilarity: x must have shape (n, ...), a row per input");
    const int64_t n = x.shape().dim(0);
    if (n < 3) throw std::invalid_argument("RepresentationalDissimilarity: x needs at least 3 rows");
    const int64_t p = static_cast<int64_t>(x.numel()) / n;
    const std::vector<float> host = x.to_host_vector();
    std::vector<double> rows(host.begin(), host.end());
    for (double v : rows) {
        if (!std::isfinite(v)) throw std::invalid_argument("RepresentationalDissimilarity: x has a NaN or infinite value");
    }
    // Correlation distance is cosine distance between rows with their own means removed.
    if (metric == RdmMetric::Correlation) {
        for (int64_t i = 0; i < n; ++i) {
            double* row = &rows[static_cast<std::size_t>(i * p)];
            const double mean = std::accumulate(row, row + p, 0.0) / static_cast<double>(p);
            for (int64_t a = 0; a < p; ++a) row[a] -= mean;
        }
    }
    std::vector<double> norm(static_cast<std::size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        const double* row = &rows[static_cast<std::size_t>(i * p)];
        for (int64_t a = 0; a < p; ++a) norm[static_cast<std::size_t>(i)] += row[a] * row[a];
        norm[static_cast<std::size_t>(i)] = std::sqrt(norm[static_cast<std::size_t>(i)]);
        if (metric != RdmMetric::Euclidean && !(norm[static_cast<std::size_t>(i)] > 0)) {
            throw std::invalid_argument("RepresentationalDissimilarity: row " + std::to_string(i) + " is " +
                                        (metric == RdmMetric::Correlation ? "constant" : "zero") + ", so its " +
                                        (metric == RdmMetric::Correlation ? "correlation" : "cosine") + " distance is undefined");
        }
    }
    Rdm r;
    r.n = n;
    r.values.reserve(static_cast<std::size_t>(n * (n - 1) / 2));
    for (int64_t i = 0; i < n; ++i) {
        const double* u = &rows[static_cast<std::size_t>(i * p)];
        for (int64_t j = i + 1; j < n; ++j) {
            const double* v = &rows[static_cast<std::size_t>(j * p)];
            double d = 0;
            if (metric == RdmMetric::Euclidean) {
                for (int64_t a = 0; a < p; ++a) d += (u[a] - v[a]) * (u[a] - v[a]);
                d = std::sqrt(d);
            } else {
                for (int64_t a = 0; a < p; ++a) d += u[a] * v[a];
                d = std::clamp(1.0 - d / (norm[static_cast<std::size_t>(i)] * norm[static_cast<std::size_t>(j)]), 0.0, 2.0);
            }
            r.values.push_back(d);
        }
    }
    return r;
}

double CompareRdms(const Rdm& a, const Rdm& b, RdmComparison comparison) {
    CheckPair(a, b, "CompareRdms");
    const std::vector<double> za = Prepare(a, comparison);
    const std::vector<double> zb = Prepare(b, comparison);
    if (za.empty() || zb.empty()) return std::numeric_limits<double>::quiet_NaN();
    double r = 0;
    for (std::size_t k = 0; k < za.size(); ++k) r += za[k] * zb[k];
    return std::clamp(r, -1.0, 1.0);
}

double Rsa(const Tensor& x, const Tensor& y, RdmMetric metric, RdmComparison comparison) {
    if (x.rank() >= 1 && y.rank() >= 1 && x.shape().dim(0) != y.shape().dim(0)) {
        throw std::invalid_argument("Rsa: the representations have " + std::to_string(x.shape().dim(0)) + " and " +
                                    std::to_string(y.shape().dim(0)) + " rows; they must describe the same inputs");
    }
    return CompareRdms(RepresentationalDissimilarity(x, metric), RepresentationalDissimilarity(y, metric), comparison);
}

RsaTestResult RsaPermutationTest(const Rdm& a, const Rdm& b, RdmComparison comparison, int64_t permutations, uint64_t seed) {
    CheckPair(a, b, "RsaPermutationTest");
    if (permutations < 1) throw std::invalid_argument("RsaPermutationTest: permutations must be at least 1");
    RsaTestResult result;
    result.permutations = permutations;
    // Relabeling b's inputs permutes its values, which changes neither their ranks' multiset nor
    // their mean and norm, so both are standardized once and each relabeling costs one dot product.
    const std::vector<double> za = Prepare(a, comparison);
    const std::vector<double> zb = Prepare(b, comparison);
    if (za.empty() || zb.empty()) {
        result.similarity = std::numeric_limits<double>::quiet_NaN();
        result.p_value = std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    const int64_t n = a.n;
    auto score = [&](const std::vector<int64_t>& label) {
        double r = 0;
        std::size_t k = 0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j, ++k) {
                const int64_t s = label[static_cast<std::size_t>(i)], t = label[static_cast<std::size_t>(j)];
                r += za[k] * zb[s < t ? PairIndex(n, s, t) : PairIndex(n, t, s)];
            }
        }
        return r;
    };
    std::vector<int64_t> label(static_cast<std::size_t>(n));
    std::iota(label.begin(), label.end(), int64_t{0});
    const double observed = score(label);
    result.similarity = std::clamp(observed, -1.0, 1.0);
    // Relabelings that tie with the real one count as reaching it, whatever the rounding.
    const double reach = observed - 1e-12;
    PortableRng rng{seed};
    int64_t at_least = 1;
    for (int64_t r = 0; r < permutations; ++r) {
        for (std::size_t i = label.size(); i > 1; --i) std::swap(label[i - 1], label[static_cast<std::size_t>(rng.next() % i)]);
        if (score(label) >= reach) ++at_least;
    }
    result.p_value = static_cast<double>(at_least) / static_cast<double>(permutations + 1);
    return result;
}

}  // namespace pulsatrix
