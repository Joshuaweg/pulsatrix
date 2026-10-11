#include "pulsatrix/representation_similarity.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsatrix {

namespace {

// A representation with its columns centered: n rows of p values, row-major, in double.
struct Centered {
    int64_t n = 0;
    int64_t p = 0;
    std::vector<double> x;
};

Centered Center(const Tensor& t, const std::string& what) {
    if (t.rank() < 2) {
        throw std::invalid_argument("LinearCka: " + what + " must have shape (n, ...), a row per input, not rank " +
                                    std::to_string(t.rank()));
    }
    Centered c;
    c.n = t.shape().dim(0);
    c.p = c.n == 0 ? 0 : static_cast<int64_t>(t.numel()) / c.n;
    const std::vector<float> host = t.to_host_vector();
    c.x.assign(host.begin(), host.end());
    std::vector<double> mean(static_cast<std::size_t>(c.p), 0.0);
    for (int64_t i = 0; i < c.n; ++i) {
        const double* row = &c.x[static_cast<std::size_t>(i * c.p)];
        for (int64_t a = 0; a < c.p; ++a) {
            mean[static_cast<std::size_t>(a)] += row[a];
        }
    }
    for (double& m : mean) {
        m /= static_cast<double>(c.n);
    }
    for (int64_t i = 0; i < c.n; ++i) {
        double* row = &c.x[static_cast<std::size_t>(i * c.p)];
        for (int64_t a = 0; a < c.p; ++a) {
            row[a] -= mean[static_cast<std::size_t>(a)];
        }
    }
    return c;
}

void CheckInputs(int64_t n_x, int64_t n_y, CkaEstimator estimator) {
    if (n_x != n_y) {
        throw std::invalid_argument("LinearCka: the representations have " + std::to_string(n_x) + " and " +
                                    std::to_string(n_y) + " rows; they must describe the same inputs");
    }
    const int64_t min_n = estimator == CkaEstimator::Debiased ? 4 : 2;
    if (n_x < min_n) {
        throw std::invalid_argument("LinearCka: " + std::to_string(n_x) + " inputs; the " +
                                    (estimator == CkaEstimator::Debiased ? "debiased" : "biased") +
                                    " estimator needs at least " + std::to_string(min_n));
    }
}

double Ratio(double hsic_xy, double hsic_xx, double hsic_yy) {
    if (!(hsic_xx > 0) || !(hsic_yy > 0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return hsic_xy / std::sqrt(hsic_xx * hsic_yy);
}

// ---- through n × n Gram matrices: the cheaper way when there are fewer inputs than features ----

struct Gram {
    int64_t n = 0;
    std::vector<double> k;        // K = X Xᵀ of the centered features, n × n
    std::vector<double> row_sum;  // K's row sums (0 up to rounding, since X is centered)
    std::vector<double> diag;     // K's diagonal, each row's squared norm
    double total = 0;             // the sum of K
};

Gram MakeGram(const Centered& c) {
    Gram g;
    g.n = c.n;
    const auto n = static_cast<std::size_t>(c.n);
    g.k.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double* xi = &c.x[i * static_cast<std::size_t>(c.p)];
        for (std::size_t j = i; j < n; ++j) {
            const double* xj = &c.x[j * static_cast<std::size_t>(c.p)];
            double dot = 0;
            for (int64_t a = 0; a < c.p; ++a) {
                dot += xi[a] * xj[a];
            }
            g.k[i * n + j] = dot;
            g.k[j * n + i] = dot;
        }
    }
    g.row_sum.assign(n, 0.0);
    g.diag.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            g.row_sum[i] += g.k[i * n + j];
        }
        g.diag[i] = g.k[i * n + i];
        g.total += g.row_sum[i];
    }
    return g;
}

// HSIC between two Gram matrices, up to a factor that cancels in CKA. Biased: tr(K L) of the
// centered matrices. Debiased: Song et al.'s unbiased estimator, on K and L with their diagonals
// set to 0 (K̃, L̃): tr(K̃ L̃) + (1ᵀK̃1)(1ᵀL̃1) / ((n-1)(n-2)) - 2/(n-2) (K̃1)ᵀ(L̃1).
double Hsic(const Gram& a, const Gram& b, CkaEstimator estimator) {
    const auto n = static_cast<std::size_t>(a.n);
    double trace = 0;
    for (std::size_t i = 0; i < n * n; ++i) {
        trace += a.k[i] * b.k[i];
    }
    if (estimator == CkaEstimator::Biased) {
        return trace;
    }
    double diag_product = 0;
    double sum_a = a.total;
    double sum_b = b.total;
    double cross = 0;
    for (std::size_t i = 0; i < n; ++i) {
        diag_product += a.diag[i] * b.diag[i];
        sum_a -= a.diag[i];
        sum_b -= b.diag[i];
        cross += (a.row_sum[i] - a.diag[i]) * (b.row_sum[i] - b.diag[i]);
    }
    const double m = static_cast<double>(a.n);
    return (trace - diag_product) + sum_a * sum_b / ((m - 1) * (m - 2)) - 2.0 / (m - 2) * cross;
}

double CkaFromGrams(const Gram& x, const Gram& y, double hsic_xx, double hsic_yy, CkaEstimator estimator) {
    return Ratio(Hsic(x, y, estimator), hsic_xx, hsic_yy);
}

// ---- through the features: the cheaper way when there are more inputs than features ----------

// ||Aᵀ B||² for centered A (n × p) and B (n × q).
double CrossNormSquared(const Centered& a, const Centered& b) {
    const auto p = static_cast<std::size_t>(a.p);
    const auto q = static_cast<std::size_t>(b.p);
    std::vector<double> m(p * q, 0.0);
    for (int64_t i = 0; i < a.n; ++i) {
        const double* ai = &a.x[static_cast<std::size_t>(i) * p];
        const double* bi = &b.x[static_cast<std::size_t>(i) * q];
        for (std::size_t r = 0; r < p; ++r) {
            const double v = ai[r];
            double* row = &m[r * q];
            for (std::size_t c = 0; c < q; ++c) {
                row[c] += v * bi[c];
            }
        }
    }
    double s = 0;
    for (double v : m) {
        s += v * v;
    }
    return s;
}

std::vector<double> RowNormsSquared(const Centered& c) {
    std::vector<double> out(static_cast<std::size_t>(c.n), 0.0);
    for (int64_t i = 0; i < c.n; ++i) {
        const double* row = &c.x[static_cast<std::size_t>(i * c.p)];
        double s = 0;
        for (int64_t a = 0; a < c.p; ++a) {
            s += row[a] * row[a];
        }
        out[static_cast<std::size_t>(i)] = s;
    }
    return out;
}

// The debiased HSIC from feature-space quantities (Kornblith et al.'s reference code): the same
// estimator as Hsic(), since for centered features K̃1 = -(row norms²) and 1ᵀK̃1 = -Σ row norms².
double DebiasFeatureHsic(double cross, const std::vector<double>& rows_a, const std::vector<double>& rows_b) {
    const double n = static_cast<double>(rows_a.size());
    double dot = 0;
    double sum_a = 0;
    double sum_b = 0;
    for (std::size_t i = 0; i < rows_a.size(); ++i) {
        dot += rows_a[i] * rows_b[i];
        sum_a += rows_a[i];
        sum_b += rows_b[i];
    }
    return cross - n / (n - 2) * dot + sum_a * sum_b / ((n - 1) * (n - 2));
}

double FeatureCka(const Centered& x, const Centered& y, CkaEstimator estimator) {
    double xy = CrossNormSquared(x, y);
    double xx = CrossNormSquared(x, x);
    double yy = CrossNormSquared(y, y);
    if (estimator == CkaEstimator::Debiased) {
        const std::vector<double> rx = RowNormsSquared(x);
        const std::vector<double> ry = RowNormsSquared(y);
        xy = DebiasFeatureHsic(xy, rx, ry);
        xx = DebiasFeatureHsic(xx, rx, rx);
        yy = DebiasFeatureHsic(yy, ry, ry);
    }
    return Ratio(xy, xx, yy);
}

// Multiply-adds for each way: n²(p + q) through Gram matrices, n(pq + p² + q²) through features.
bool PreferGrams(int64_t n, int64_t p, int64_t q) {
    const double dn = static_cast<double>(n);
    const double dp = static_cast<double>(p);
    const double dq = static_cast<double>(q);
    return dn * dn * (dp + dq) <= dn * (dp * dq + dp * dp + dq * dq);
}

double CkaOfCentered(const Centered& x, const Centered& y, CkaEstimator estimator) {
    if (PreferGrams(x.n, x.p, y.p)) {
        const Gram gx = MakeGram(x);
        const Gram gy = MakeGram(y);
        return CkaFromGrams(gx, gy, Hsic(gx, gx, estimator), Hsic(gy, gy, estimator), estimator);
    }
    return FeatureCka(x, y, estimator);
}

}  // namespace

double LinearCka(const Tensor& x, const Tensor& y, CkaEstimator estimator) {
    const Centered cx = Center(x, "x");
    const Centered cy = Center(y, "y");
    CheckInputs(cx.n, cy.n, estimator);
    return CkaOfCentered(cx, cy, estimator);
}

double CkaMatrix::at(int64_t row, int64_t col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("CkaMatrix::at: (" + std::to_string(row) + ", " + std::to_string(col) +
                                ") is outside " + std::to_string(rows) + " × " + std::to_string(cols));
    }
    return values[static_cast<std::size_t>(row * cols + col)];
}

CkaMatrix LinearCkaMatrix(const std::vector<Tensor>& a, const std::vector<Tensor>& b, CkaEstimator estimator,
                          std::size_t max_gram_bytes) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("LinearCkaMatrix: both lists need at least one representation");
    }
    std::vector<Centered> ca;
    std::vector<Centered> cb;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ca.push_back(Center(a[i], "a[" + std::to_string(i) + "]"));
    }
    for (std::size_t j = 0; j < b.size(); ++j) {
        cb.push_back(Center(b[j], "b[" + std::to_string(j) + "]"));
    }
    const int64_t n = ca.front().n;
    for (const Centered& c : ca) {
        CheckInputs(n, c.n, estimator);
    }
    for (const Centered& c : cb) {
        CheckInputs(n, c.n, estimator);
    }

    CkaMatrix out;
    out.rows = static_cast<int64_t>(a.size());
    out.cols = static_cast<int64_t>(b.size());
    out.values.resize(a.size() * b.size());

    const double gram_bytes = static_cast<double>(a.size() + b.size()) * static_cast<double>(n) *
                              static_cast<double>(n) * sizeof(double);
    if (gram_bytes > static_cast<double>(max_gram_bytes)) {
        for (std::size_t i = 0; i < ca.size(); ++i) {
            for (std::size_t j = 0; j < cb.size(); ++j) {
                out.values[i * cb.size() + j] = CkaOfCentered(ca[i], cb[j], estimator);
            }
        }
        return out;
    }

    std::vector<Gram> ga;
    std::vector<Gram> gb;
    std::vector<double> self_a;
    std::vector<double> self_b;
    for (Centered& c : ca) {
        ga.push_back(MakeGram(c));
        self_a.push_back(Hsic(ga.back(), ga.back(), estimator));
        c = Centered{};  // the Gram matrix replaces it
    }
    for (Centered& c : cb) {
        gb.push_back(MakeGram(c));
        self_b.push_back(Hsic(gb.back(), gb.back(), estimator));
        c = Centered{};
    }
    for (std::size_t i = 0; i < ga.size(); ++i) {
        for (std::size_t j = 0; j < gb.size(); ++j) {
            out.values[i * gb.size() + j] = CkaFromGrams(ga[i], gb[j], self_a[i], self_b[j], estimator);
        }
    }
    return out;
}

}  // namespace pulsatrix
