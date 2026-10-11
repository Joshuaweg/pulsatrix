#include "pulsatrix/representation_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include "pulsatrix/matrix_decompositions.hpp"

namespace pulsatrix {
namespace {

// Rows of a tensor as doubles: n rows of width p, row-major.
struct Rows {
    int64_t n = 0;
    int64_t p = 0;
    std::vector<double> x;

    [[nodiscard]] const double* row(int64_t i) const { return &x[static_cast<std::size_t>(i * p)]; }
    [[nodiscard]] double* row(int64_t i) { return &x[static_cast<std::size_t>(i * p)]; }
};

Rows ToRows(const Tensor& t, const char* function, int64_t min_rows) {
    if (t.rank() < 2) {
        throw std::invalid_argument(std::string(function) + ": x must have shape (n, ...), a row per input");
    }
    Rows r;
    r.n = t.shape().dim(0);
    if (r.n < min_rows) {
        throw std::invalid_argument(std::string(function) + ": x needs at least " + std::to_string(min_rows) + " rows");
    }
    r.p = static_cast<int64_t>(t.numel()) / r.n;
    const std::vector<float> host = t.to_host_vector();
    r.x.reserve(host.size());
    for (float v : host) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(function) + ": x has a NaN or infinite value");
        r.x.push_back(v);
    }
    return r;
}

void Center(Rows& r) {
    std::vector<double> mean(static_cast<std::size_t>(r.p), 0.0);
    for (int64_t i = 0; i < r.n; ++i) {
        for (int64_t j = 0; j < r.p; ++j) mean[static_cast<std::size_t>(j)] += r.row(i)[j];
    }
    for (double& m : mean) m /= static_cast<double>(r.n);
    for (int64_t i = 0; i < r.n; ++i) {
        for (int64_t j = 0; j < r.p; ++j) r.row(i)[j] -= mean[static_cast<std::size_t>(j)];
    }
}

double SquaredDistance(const double* a, const double* b, int64_t p) {
    double s = 0;
    for (int64_t j = 0; j < p; ++j) s += (a[j] - b[j]) * (a[j] - b[j]);
    return s;
}

}  // namespace

double Anisotropy(const Tensor& x, bool center) {
    Rows r = ToRows(x, "Anisotropy", 2);
    if (center) Center(r);
    // The mean of u_i . u_j over i != j is (|sum u|^2 - sum |u_i|^2) / (n (n - 1)), and |u_i| = 1.
    std::vector<double> sum(static_cast<std::size_t>(r.p), 0.0);
    for (int64_t i = 0; i < r.n; ++i) {
        const double* row = r.row(i);
        double norm2 = 0;
        for (int64_t j = 0; j < r.p; ++j) norm2 += row[j] * row[j];
        if (norm2 == 0) throw std::invalid_argument("Anisotropy: x has a row of zeros, which has no direction");
        const double inv = 1.0 / std::sqrt(norm2);
        for (int64_t j = 0; j < r.p; ++j) sum[static_cast<std::size_t>(j)] += row[j] * inv;
    }
    double total = 0;
    for (double s : sum) total += s * s;
    const auto n = static_cast<double>(r.n);
    return (total - n) / (n * (n - 1));
}

EffectiveRank MeasureEffectiveRank(const Tensor& x, bool center) {
    Rows r = ToRows(x, "MeasureEffectiveRank", 2);
    if (center) Center(r);
    // The nonzero eigenvalues of x x^T and x^T x are the same squared singular values; use the
    // smaller matrix.
    const bool by_rows = r.n <= r.p;
    const int64_t k = by_rows ? r.n : r.p;
    std::vector<double> gram(static_cast<std::size_t>(k * k), 0.0);
    if (by_rows) {
        for (int64_t i = 0; i < r.n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s = 0;
                for (int64_t c = 0; c < r.p; ++c) s += r.row(i)[c] * r.row(j)[c];
                gram[static_cast<std::size_t>(i * k + j)] = gram[static_cast<std::size_t>(j * k + i)] = s;
            }
        }
    } else {
        for (int64_t i = 0; i < r.n; ++i) {
            const double* row = r.row(i);
            for (int64_t a = 0; a < r.p; ++a) {
                for (int64_t b = 0; b <= a; ++b) gram[static_cast<std::size_t>(a * k + b)] += row[a] * row[b];
            }
        }
        for (int64_t a = 0; a < k; ++a) {
            for (int64_t b = 0; b < a; ++b) gram[static_cast<std::size_t>(b * k + a)] = gram[static_cast<std::size_t>(a * k + b)];
        }
    }
    const std::vector<double> eigenvalues = SymmetricEigenvalues(std::move(gram), k);
    if (!(eigenvalues.front() > 0)) throw std::invalid_argument("MeasureEffectiveRank: x is all zeros");
    // Eigenvalues the size of rounding error are zero singular values: without this, a missing
    // direction would come back as a singular value about 1e-7 of the largest.
    const double cutoff = eigenvalues.front() * static_cast<double>(std::max(r.n, r.p)) * std::numeric_limits<double>::epsilon();
    EffectiveRank out;
    out.singular_values.reserve(eigenvalues.size());
    for (double lambda : eigenvalues) out.singular_values.push_back(lambda > cutoff ? std::sqrt(lambda) : 0.0);
    const double sum = std::accumulate(out.singular_values.begin(), out.singular_values.end(), 0.0);
    double entropy = 0, sum2 = 0, sum4 = 0;
    for (double s : out.singular_values) {
        if (s > 0) {
            const double q = s / sum;
            entropy -= q * std::log(q);
        }
        sum2 += s * s;
        sum4 += s * s * s * s;
    }
    out.effective_rank = std::exp(entropy);
    out.participation_ratio = sum2 * sum2 / sum4;
    return out;
}

TwoNNResult TwoNNDimension(const Tensor& x, double discard_fraction) {
    if (!(discard_fraction > 0.0 && discard_fraction < 1.0)) {
        throw std::invalid_argument("TwoNNDimension: discard_fraction must be in (0, 1)");
    }
    const Rows all = ToRows(x, "TwoNNDimension", 1);
    // Drop exact repeats, keeping the first of each.
    std::vector<int64_t> order(static_cast<std::size_t>(all.n));
    std::iota(order.begin(), order.end(), int64_t{0});
    const auto less = [&](int64_t a, int64_t b) {
        if (std::lexicographical_compare(all.row(a), all.row(a) + all.p, all.row(b), all.row(b) + all.p)) return true;
        if (std::lexicographical_compare(all.row(b), all.row(b) + all.p, all.row(a), all.row(a) + all.p)) return false;
        return a < b;
    };
    std::sort(order.begin(), order.end(), less);
    std::vector<int64_t> keep;
    for (std::size_t k = 0; k < order.size(); ++k) {
        if (k > 0 && std::equal(all.row(order[k]), all.row(order[k]) + all.p, all.row(order[k - 1]))) continue;
        keep.push_back(order[k]);
    }
    std::sort(keep.begin(), keep.end());
    TwoNNResult out;
    out.points = static_cast<int64_t>(keep.size());
    out.duplicates = all.n - out.points;
    if (out.points < 3) throw std::invalid_argument("TwoNNDimension: x needs at least 3 distinct rows");

    const int64_t n = out.points;
    std::vector<double> mu(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        double r1 = std::numeric_limits<double>::infinity(), r2 = r1;
        for (int64_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double d = SquaredDistance(all.row(keep[static_cast<std::size_t>(i)]), all.row(keep[static_cast<std::size_t>(j)]), all.p);
            if (d < r1) {
                r2 = r1;
                r1 = d;
            } else if (d < r2) {
                r2 = d;
            }
        }
        mu[static_cast<std::size_t>(i)] = std::sqrt(r2 / r1);
    }

    double sum_log = 0;
    for (double m : mu) sum_log += std::log(m);
    out.mle_dimension = static_cast<double>(n - 1) / sum_log;

    // Fit through the origin on the smallest mu: x = log mu, y = -log(1 - F), F = i / N for
    // i = 1 .. kept, as scikit-dimension and DADApy do.
    std::sort(mu.begin(), mu.end());
    // At most n - 1: F = 1 would make y infinite.
    const auto kept = std::min(n - 1, static_cast<int64_t>(static_cast<double>(n) * (1.0 - discard_fraction)));
    if (kept < 2) throw std::invalid_argument("TwoNNDimension: discard_fraction leaves fewer than 2 points to fit");
    double xy = 0, xx = 0;
    for (int64_t i = 0; i < kept; ++i) {
        const double lx = std::log(mu[static_cast<std::size_t>(i)]);
        const double ly = -std::log(1.0 - static_cast<double>(i + 1) / static_cast<double>(n));
        xy += lx * ly;
        xx += lx * lx;
    }
    if (xx == 0) throw std::invalid_argument("TwoNNDimension: every point's two nearest neighbors are equally far, so mu says nothing");
    out.dimension = xy / xx;
    return out;
}

}  // namespace pulsatrix
