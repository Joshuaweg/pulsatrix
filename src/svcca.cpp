#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/matrix_decompositions.hpp"
#include "pulsatrix/representation_similarity.hpp"

namespace pulsatrix {
namespace {

// A centered representation's singular directions in input space: the left singular vectors,
// rows of `u` (n values each), with the squared singular values, largest first.
struct Directions {
    int64_t n = 0;
    int64_t rank = 0;
    std::vector<double> lambda;  // every squared singular value, rounding-error ones included
    std::vector<double> u;       // `rank` rows of n
};

Directions SingularDirections(const Tensor& t, const char* function, const char* what) {
    if (t.rank() < 2) throw std::invalid_argument(std::string(function) + ": " + what + " must have shape (n, ...), a row per input");
    const int64_t n = t.shape().dim(0);
    const int64_t p = n == 0 ? 0 : static_cast<int64_t>(t.numel()) / n;
    if (n < 2 || p < 1) throw std::invalid_argument(std::string(function) + ": " + what + " needs at least 2 rows and 1 column");
    const std::vector<float> host = t.to_host_vector();
    std::vector<double> x(host.begin(), host.end());
    for (double v : x) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(function) + ": " + what + " has a NaN or infinite value");
    }
    for (int64_t a = 0; a < p; ++a) {
        double mean = 0;
        for (int64_t i = 0; i < n; ++i) mean += x[static_cast<std::size_t>(i * p + a)];
        mean /= static_cast<double>(n);
        for (int64_t i = 0; i < n; ++i) x[static_cast<std::size_t>(i * p + a)] -= mean;
    }
    // x x^T (n x n) has the left singular vectors as eigenvectors; x^T x (p x p) has the right
    // ones, w, which give the left ones as x w / s. Whichever is smaller.
    const bool by_rows = n <= p;
    const int64_t m = by_rows ? n : p;
    std::vector<double> gram(static_cast<std::size_t>(m * m), 0.0);
    if (by_rows) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s = 0;
                for (int64_t a = 0; a < p; ++a) s += x[static_cast<std::size_t>(i * p + a)] * x[static_cast<std::size_t>(j * p + a)];
                gram[static_cast<std::size_t>(i * m + j)] = gram[static_cast<std::size_t>(j * m + i)] = s;
            }
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const double* row = &x[static_cast<std::size_t>(i * p)];
            for (int64_t a = 0; a < p; ++a) {
                for (int64_t b = 0; b <= a; ++b) gram[static_cast<std::size_t>(a * m + b)] += row[a] * row[b];
            }
        }
        for (int64_t a = 0; a < m; ++a) {
            for (int64_t b = 0; b < a; ++b) gram[static_cast<std::size_t>(b * m + a)] = gram[static_cast<std::size_t>(a * m + b)];
        }
    }
    const DoubleEigenResult eig = SymmetricEigenInDouble(std::move(gram), m);
    Directions d;
    d.n = n;
    d.lambda = eig.values;
    // A singular value the size of rounding error is no direction at all.
    const double cutoff = std::max(eig.values.front(), 0.0) * static_cast<double>(std::max(n, p)) * std::numeric_limits<double>::epsilon();
    while (d.rank < m && eig.values[static_cast<std::size_t>(d.rank)] > cutoff) ++d.rank;
    if (d.rank == 0) throw std::invalid_argument(std::string(function) + ": " + what + " is constant over the inputs");
    d.u.assign(static_cast<std::size_t>(d.rank * n), 0.0);
    for (int64_t c = 0; c < d.rank; ++c) {
        const double* e = &eig.vectors[static_cast<std::size_t>(c * m)];
        double* u = &d.u[static_cast<std::size_t>(c * n)];
        if (by_rows) {
            std::copy(e, e + n, u);
        } else {
            const double s = std::sqrt(eig.values[static_cast<std::size_t>(c)]);
            for (int64_t i = 0; i < n; ++i) {
                double dot = 0;
                for (int64_t a = 0; a < p; ++a) dot += x[static_cast<std::size_t>(i * p + a)] * e[a];
                u[i] = dot / s;
            }
        }
    }
    return d;
}

// The canonical correlations between the spans of x's first kx directions and y's first ky: the
// singular values of U_x U_y^T, both bases orthonormal.
std::vector<double> Correlations(const Directions& x, int64_t kx, const Directions& y, int64_t ky, const char* function) {
    if (x.n != y.n) {
        throw std::invalid_argument(std::string(function) + ": the representations have " + std::to_string(x.n) + " and " +
                                    std::to_string(y.n) + " rows; they must describe the same inputs");
    }
    if (kx + ky >= x.n) {
        throw std::invalid_argument(std::string(function) + ": " + std::to_string(kx) + " + " + std::to_string(ky) +
                                    " directions need more than " + std::to_string(x.n) +
                                    " inputs; with this few, some canonical correlations are 1 whatever the data");
    }
    const int64_t n = x.n;
    // The smaller side's Gram matrix of M = U_x U_y^T has the squared correlations as eigenvalues.
    const bool x_side = kx <= ky;
    const Directions& small = x_side ? x : y;
    const Directions& large = x_side ? y : x;
    const int64_t ks = x_side ? kx : ky;
    const int64_t kl = x_side ? ky : kx;
    std::vector<double> mm(static_cast<std::size_t>(ks * kl), 0.0);
    for (int64_t a = 0; a < ks; ++a) {
        for (int64_t b = 0; b < kl; ++b) {
            double dot = 0;
            for (int64_t i = 0; i < n; ++i) dot += small.u[static_cast<std::size_t>(a * n + i)] * large.u[static_cast<std::size_t>(b * n + i)];
            mm[static_cast<std::size_t>(a * kl + b)] = dot;
        }
    }
    std::vector<double> g(static_cast<std::size_t>(ks * ks), 0.0);
    for (int64_t a = 0; a < ks; ++a) {
        for (int64_t b = 0; b <= a; ++b) {
            double dot = 0;
            for (int64_t c = 0; c < kl; ++c) dot += mm[static_cast<std::size_t>(a * kl + c)] * mm[static_cast<std::size_t>(b * kl + c)];
            g[static_cast<std::size_t>(a * ks + b)] = g[static_cast<std::size_t>(b * ks + a)] = dot;
        }
    }
    std::vector<double> rho = SymmetricEigenvalues(std::move(g), ks);
    for (double& r : rho) r = std::sqrt(std::clamp(r, 0.0, 1.0));
    return rho;
}

}  // namespace

std::vector<double> CanonicalCorrelations(const Tensor& x, const Tensor& y) {
    const Directions dx = SingularDirections(x, "CanonicalCorrelations", "x");
    const Directions dy = SingularDirections(y, "CanonicalCorrelations", "y");
    return Correlations(dx, dx.rank, dy, dy.rank, "CanonicalCorrelations");
}

SvccaResult Svcca(const Tensor& x, const Tensor& y, const SvccaOptions& options) {
    if (!(options.variance_kept > 0 && options.variance_kept <= 1)) throw std::invalid_argument("Svcca: variance_kept must be in (0, 1]");
    if (options.directions < 0) throw std::invalid_argument("Svcca: directions can't be negative");
    const Directions dx = SingularDirections(x, "Svcca", "x");
    const Directions dy = SingularDirections(y, "Svcca", "y");
    auto kept = [&](const Directions& d) {
        if (options.directions > 0) return std::min(options.directions, d.rank);
        double total = 0;
        for (double l : d.lambda) total += std::max(l, 0.0);
        // scikit-learn's rule: one more than the number of cumulative shares at or below the cut.
        int64_t k = 0;
        double share = 0;
        for (double l : d.lambda) {
            share += std::max(l, 0.0) / total;
            if (share > options.variance_kept) break;
            ++k;
        }
        return std::min(k + 1, d.rank);
    };
    SvccaResult r;
    r.x_directions = kept(dx);
    r.y_directions = kept(dy);
    r.correlations = Correlations(dx, r.x_directions, dy, r.y_directions, "Svcca");
    double sum = 0;
    for (double c : r.correlations) sum += c;
    r.similarity = sum / static_cast<double>(r.correlations.size());
    return r;
}

}  // namespace pulsatrix
