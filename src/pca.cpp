#include "pulsatrix/pca.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"
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

Rows ToRows(const Tensor& t, const char* function, const char* what) {
    if (t.rank() < 2) {
        throw std::invalid_argument(std::string(function) + ": " + what + " must have shape (n, ...), a row per input");
    }
    Rows r;
    r.n = t.shape().dim(0);
    r.p = r.n == 0 ? 0 : static_cast<int64_t>(t.numel()) / r.n;
    const std::vector<float> host = t.to_host_vector();
    r.x.reserve(host.size());
    for (float v : host) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(function) + ": " + what + " has a NaN or infinite value");
        r.x.push_back(v);
    }
    return r;
}

Pca Fit(Rows r, int64_t k, bool center) {
    if (r.n < 2) throw std::invalid_argument("FitPca: x needs at least 2 rows");
    if (k < 1 || k > std::min(r.n, r.p)) {
        throw std::invalid_argument("FitPca: components must be between 1 and min(rows, width) = " + std::to_string(std::min(r.n, r.p)));
    }
    Pca pca;
    pca.width = r.p;
    pca.samples = r.n;
    pca.mean.assign(static_cast<std::size_t>(r.p), 0.0);
    if (center) {
        for (int64_t i = 0; i < r.n; ++i) {
            for (int64_t j = 0; j < r.p; ++j) pca.mean[static_cast<std::size_t>(j)] += r.row(i)[j];
        }
        for (double& m : pca.mean) m /= static_cast<double>(r.n);
        for (int64_t i = 0; i < r.n; ++i) {
            for (int64_t j = 0; j < r.p; ++j) r.row(i)[j] -= pca.mean[static_cast<std::size_t>(j)];
        }
    }
    const double dof = static_cast<double>(r.n - 1);
    double trace = 0;
    for (double v : r.x) trace += v * v;
    pca.total_variance = trace / dof;

    // The same eigenvalues come from x^T x (width x width, its eigenvectors the components) and
    // x x^T (n x n, whose eigenvectors u give the components as x^T u / sqrt(lambda)).
    const bool by_rows = r.n < r.p;
    const int64_t m = by_rows ? r.n : r.p;
    std::vector<double> gram(static_cast<std::size_t>(m * m), 0.0);
    if (by_rows) {
        for (int64_t i = 0; i < r.n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s = 0;
                for (int64_t c = 0; c < r.p; ++c) s += r.row(i)[c] * r.row(j)[c];
                gram[static_cast<std::size_t>(i * m + j)] = gram[static_cast<std::size_t>(j * m + i)] = s;
            }
        }
    } else {
        for (int64_t i = 0; i < r.n; ++i) {
            const double* row = r.row(i);
            for (int64_t a = 0; a < r.p; ++a) {
                for (int64_t b = 0; b <= a; ++b) gram[static_cast<std::size_t>(a * m + b)] += row[a] * row[b];
            }
        }
        for (int64_t a = 0; a < m; ++a) {
            for (int64_t b = 0; b < a; ++b) gram[static_cast<std::size_t>(b * m + a)] = gram[static_cast<std::size_t>(a * m + b)];
        }
    }
    const DoubleEigenResult eig = SymmetricEigenInDouble(std::move(gram), m);
    // An eigenvalue the size of rounding error is no variance at all.
    const double cutoff = std::max(eig.values.front(), 0.0) * static_cast<double>(std::max(r.n, r.p)) * std::numeric_limits<double>::epsilon();
    for (int64_t c = 0; c < k; ++c) {
        const double lambda = eig.values[static_cast<std::size_t>(c)];
        if (!(lambda > cutoff)) {
            throw std::invalid_argument("FitPca: x has only " + std::to_string(c) + " directions with variance, fewer than the " +
                                        std::to_string(k) + " components asked for");
        }
        std::vector<double> v(static_cast<std::size_t>(r.p), 0.0);
        const double* e = &eig.vectors[static_cast<std::size_t>(c * m)];
        if (by_rows) {
            for (int64_t i = 0; i < r.n; ++i) {
                for (int64_t j = 0; j < r.p; ++j) v[static_cast<std::size_t>(j)] += e[i] * r.row(i)[j];
            }
            double norm = 0;
            for (double t : v) norm += t * t;
            norm = std::sqrt(norm);
            int64_t best = 0;
            for (int64_t j = 1; j < r.p; ++j) {
                if (std::fabs(v[static_cast<std::size_t>(j)]) > std::fabs(v[static_cast<std::size_t>(best)])) best = j;
            }
            const double scale = (v[static_cast<std::size_t>(best)] < 0 ? -1.0 : 1.0) / norm;
            for (double& t : v) t *= scale;
        } else {
            std::copy(e, e + r.p, v.begin());
        }
        pca.components.insert(pca.components.end(), v.begin(), v.end());
        pca.explained_variance.push_back(lambda / dof);
        pca.explained_variance_ratio.push_back(lambda / trace);
    }
    return pca;
}

}  // namespace

Pca FitPca(const Tensor& x, int64_t components, bool center) {
    return Fit(ToRows(x, "FitPca", "x"), components, center);
}

Tensor PcaProject(const Pca& pca, const Tensor& x, bool whiten) {
    const Rows r = ToRows(x, "PcaProject", "x");
    if (r.p != pca.width) {
        throw std::invalid_argument("PcaProject: x's rows are " + std::to_string(r.p) + " wide, the PCA's " + std::to_string(pca.width));
    }
    const int64_t k = pca.num_components();
    std::vector<float> out(static_cast<std::size_t>(r.n * k));
    std::vector<double> centered(static_cast<std::size_t>(r.p));
    for (int64_t i = 0; i < r.n; ++i) {
        for (int64_t j = 0; j < r.p; ++j) centered[static_cast<std::size_t>(j)] = r.row(i)[j] - pca.mean[static_cast<std::size_t>(j)];
        for (int64_t c = 0; c < k; ++c) {
            const double* v = &pca.components[static_cast<std::size_t>(c * r.p)];
            double s = 0;
            for (int64_t j = 0; j < r.p; ++j) s += centered[static_cast<std::size_t>(j)] * v[j];
            if (whiten) s /= std::sqrt(pca.explained_variance[static_cast<std::size_t>(c)]);
            out[static_cast<std::size_t>(i * k + c)] = static_cast<float>(s);
        }
    }
    return Tensor(Shape({r.n, k}), x.backend(), out, x.device());
}

Tensor PcaReconstruct(const Pca& pca, const Tensor& scores) {
    const int64_t k = pca.num_components();
    if (scores.rank() != 2 || scores.shape().dim(1) != k) {
        throw std::invalid_argument("PcaReconstruct: scores must be (n, " + std::to_string(k) + "), one column per component");
    }
    const Rows r = ToRows(scores, "PcaReconstruct", "scores");
    std::vector<float> out(static_cast<std::size_t>(r.n * pca.width));
    for (int64_t i = 0; i < r.n; ++i) {
        for (int64_t j = 0; j < pca.width; ++j) {
            double s = pca.mean[static_cast<std::size_t>(j)];
            for (int64_t c = 0; c < k; ++c) s += r.row(i)[c] * pca.components[static_cast<std::size_t>(c * pca.width + j)];
            out[static_cast<std::size_t>(i * pca.width + j)] = static_cast<float>(s);
        }
    }
    return Tensor(Shape({r.n, pca.width}), scores.backend(), out, scores.device());
}

std::vector<float> PcaComponent(const Pca& pca, int64_t index) {
    if (index < 0 || index >= pca.num_components()) throw std::invalid_argument("PcaComponent: index is out of range");
    const auto first = pca.components.begin() + index * pca.width;
    return std::vector<float>(first, first + pca.width);
}

double SubspaceOverlap(const Pca& a, const Pca& b, int64_t k) {
    if (a.width != b.width) throw std::invalid_argument("SubspaceOverlap: the two PCAs have different widths");
    if (k < 1 || k > a.num_components() || k > b.num_components()) {
        throw std::invalid_argument("SubspaceOverlap: k must be between 1 and both PCAs' number of components");
    }
    double sum = 0;
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            double dot = 0;
            for (int64_t c = 0; c < a.width; ++c) {
                dot += a.components[static_cast<std::size_t>(i * a.width + c)] * b.components[static_cast<std::size_t>(j * b.width + c)];
            }
            sum += dot * dot;
        }
    }
    return sum / static_cast<double>(k);
}

double PcaSplitHalfOverlap(const Tensor& x, int64_t k, uint64_t seed) {
    const Rows all = ToRows(x, "PcaSplitHalfOverlap", "x");
    if (all.n < 4) throw std::invalid_argument("PcaSplitHalfOverlap: x needs at least 4 rows");
    std::vector<int64_t> order(static_cast<std::size_t>(all.n));
    std::iota(order.begin(), order.end(), int64_t{0});
    PortableRng rng{seed};
    for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[static_cast<std::size_t>(rng.next() % i)]);
    Rows halves[2];
    for (int h = 0; h < 2; ++h) {
        halves[h].p = all.p;
        const std::size_t from = h == 0 ? 0 : order.size() / 2, to = h == 0 ? order.size() / 2 : order.size();
        for (std::size_t i = from; i < to; ++i) halves[h].x.insert(halves[h].x.end(), all.row(order[i]), all.row(order[i]) + all.p);
        halves[h].n = static_cast<int64_t>(to - from);
    }
    return SubspaceOverlap(Fit(std::move(halves[0]), k, true), Fit(std::move(halves[1]), k, true), k);
}

}  // namespace pulsatrix
