#include "pulsatrix/matrix_decompositions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {
namespace {

constexpr int kMaxSweeps = 100;

// Row-major dense matrix in double, the working form of every routine here.
struct Dense {
    int64_t rows;
    int64_t cols;
    std::vector<double> v;
    double& at(int64_t i, int64_t j) { return v[static_cast<size_t>(i * cols + j)]; }
    double at(int64_t i, int64_t j) const { return v[static_cast<size_t>(i * cols + j)]; }
};

Dense load(const Tensor& a, const char* who) {
    if (a.rank() != 2) {
        throw std::invalid_argument(std::string(who) + ": input must be rank 2");
    }
    if (a.device() != DeviceType::Cpu) {
        throw std::invalid_argument(std::string(who) + ": input must be a CPU tensor");
    }
    Dense d{a.shape().dim(0), a.shape().dim(1), std::vector<double>(static_cast<size_t>(a.numel()))};
    for (size_t i = 0; i < d.v.size(); ++i) {
        if (!std::isfinite(a.data()[i])) {
            throw std::invalid_argument(std::string(who) + ": input has a non-finite entry");
        }
        d.v[i] = a.data()[i];
    }
    return d;
}

Dense load_symmetric(const Tensor& a, const char* who) {
    Dense d = load(a, who);
    if (d.rows != d.cols) {
        throw std::invalid_argument(std::string(who) + ": input must be square");
    }
    double scale = 1.0;
    for (double x : d.v) {
        scale = std::max(scale, std::fabs(x));
    }
    for (int64_t i = 0; i < d.rows; ++i) {
        for (int64_t j = i + 1; j < d.cols; ++j) {
            if (std::fabs(d.at(i, j) - d.at(j, i)) > 1e-5 * scale) {
                throw std::invalid_argument(std::string(who) + ": input must be symmetric");
            }
        }
    }
    return d;
}

Dense identity(int64_t n) {
    Dense d{n, n, std::vector<double>(static_cast<size_t>(n * n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        d.at(i, i) = 1.0;
    }
    return d;
}

// t = tan of the rotation angle that zeroes the pair's coupling: the smaller root of
// t^2 + 2*theta*t - 1 = 0, which keeps the rotation under 45 degrees (stable).
double rotation_tangent(double theta) {
    const double sign = theta >= 0.0 ? 1.0 : -1.0;
    return sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
}

// Index of a column's largest-magnitude entry, the first on a tie.
int64_t dominant_row(const Dense& m, int64_t col) {
    int64_t best = 0;
    for (int64_t i = 1; i < m.rows; ++i) {
        if (std::fabs(m.at(i, col)) > std::fabs(m.at(best, col))) {
            best = i;
        }
    }
    return best;
}

// Order of `values`, largest first; std::stable_sort keeps equal values in index order.
std::vector<int64_t> descending_order(const std::vector<double>& values) {
    std::vector<int64_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) { return values[i] > values[j]; });
    return order;
}

Tensor to_tensor(const std::vector<double>& v, Shape shape, DeviceBackend* backend) {
    return Tensor(std::move(shape), backend, std::vector<float>(v.begin(), v.end()));
}

// Columns of `m` in `order`, as a (m.rows, order.size()) tensor.
Tensor columns_to_tensor(const Dense& m, const std::vector<int64_t>& order, DeviceBackend* backend) {
    const auto k = static_cast<int64_t>(order.size());
    std::vector<double> out(static_cast<size_t>(m.rows * k));
    for (int64_t i = 0; i < m.rows; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            out[static_cast<size_t>(i * k + j)] = m.at(i, order[static_cast<size_t>(j)]);
        }
    }
    return to_tensor(out, Shape({m.rows, k}), backend);
}

// Thin SVD of a tall or square matrix (rows >= cols) by one-sided Jacobi. Returns W's
// orthogonalized columns in `w` (their norms are the singular values) and V in `v`.
void one_sided_jacobi(Dense& w, Dense& v) {
    const int64_t n = w.cols;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        bool rotated = false;
        for (int64_t p = 0; p < n; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int64_t i = 0; i < w.rows; ++i) {
                    alpha += w.at(i, p) * w.at(i, p);
                    beta += w.at(i, q) * w.at(i, q);
                    gamma += w.at(i, p) * w.at(i, q);
                }
                if (std::fabs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) {
                    continue;
                }
                rotated = true;
                const double t = rotation_tangent((beta - alpha) / (2.0 * gamma));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                for (int64_t i = 0; i < w.rows; ++i) {
                    const double wp = w.at(i, p), wq = w.at(i, q);
                    w.at(i, p) = c * wp - s * wq;
                    w.at(i, q) = s * wp + c * wq;
                }
                for (int64_t i = 0; i < v.rows; ++i) {
                    const double vp = v.at(i, p), vq = v.at(i, q);
                    v.at(i, p) = c * vp - s * vq;
                    v.at(i, q) = s * vp + c * vq;
                }
            }
        }
        if (!rotated) {
            return;
        }
    }
}

// Removes from column j of u its components along columns `done`, twice for accuracy.
void orthogonalize_against(Dense& u, int64_t j, const std::vector<int64_t>& done) {
    for (int pass = 0; pass < 2; ++pass) {
        for (int64_t d : done) {
            double dot = 0.0;
            for (int64_t i = 0; i < u.rows; ++i) {
                dot += u.at(i, j) * u.at(i, d);
            }
            for (int64_t i = 0; i < u.rows; ++i) {
                u.at(i, j) -= dot * u.at(i, d);
            }
        }
    }
}

double column_norm(const Dense& m, int64_t j) {
    double sum = 0.0;
    for (int64_t i = 0; i < m.rows; ++i) {
        sum += m.at(i, j) * m.at(i, j);
    }
    return std::sqrt(sum);
}

struct DenseSVD {
    Dense u;
    std::vector<double> s;
    Dense v;
};

// SVD of a tall or square matrix, columns sorted by descending singular value, with U
// completed to orthonormal columns where a singular value is zero. Signs not yet canonical.
DenseSVD tall_svd(Dense w) {
    const int64_t n = w.cols;
    Dense v = identity(n);
    one_sided_jacobi(w, v);

    std::vector<double> sigma(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        sigma[static_cast<size_t>(j)] = column_norm(w, j);
    }
    const std::vector<int64_t> order = descending_order(sigma);
    const double sigma_max = sigma[static_cast<size_t>(order[0])];
    // Below this, a "singular value" is round-off from an exactly rank-deficient input.
    const double zero_below = static_cast<double>(std::max(w.rows, n)) * 1e-14 * sigma_max;

    DenseSVD out{Dense{w.rows, n, std::vector<double>(static_cast<size_t>(w.rows * n), 0.0)},
                 std::vector<double>(static_cast<size_t>(n)), Dense{n, n, std::vector<double>(static_cast<size_t>(n * n))}};
    std::vector<int64_t> deficient;
    std::vector<int64_t> done;
    for (int64_t j = 0; j < n; ++j) {
        const int64_t src = order[static_cast<size_t>(j)];
        for (int64_t i = 0; i < n; ++i) {
            out.v.at(i, j) = v.at(i, src);
        }
        const double s = sigma[static_cast<size_t>(src)];
        if (s > zero_below && s > 0.0) {
            out.s[static_cast<size_t>(j)] = s;
            for (int64_t i = 0; i < w.rows; ++i) {
                out.u.at(i, j) = w.at(i, src) / s;
            }
            done.push_back(j);
        } else {
            out.s[static_cast<size_t>(j)] = 0.0;
            deficient.push_back(j);
        }
    }
    // Complete U for zero singular values from the standard basis, skipping any basis vector
    // (nearly) inside the span already built.
    int64_t basis = 0;
    for (int64_t j : deficient) {
        for (; basis < w.rows; ++basis) {
            for (int64_t i = 0; i < w.rows; ++i) {
                out.u.at(i, j) = i == basis ? 1.0 : 0.0;
            }
            orthogonalize_against(out.u, j, done);
            const double norm = column_norm(out.u, j);
            if (norm > 0.5) {
                for (int64_t i = 0; i < w.rows; ++i) {
                    out.u.at(i, j) /= norm;
                }
                ++basis;
                break;
            }
        }
        done.push_back(j);
    }
    return out;
}

Dense transpose(const Dense& m) {
    Dense t{m.cols, m.rows, std::vector<double>(m.v.size())};
    for (int64_t i = 0; i < m.rows; ++i) {
        for (int64_t j = 0; j < m.cols; ++j) {
            t.at(j, i) = m.at(i, j);
        }
    }
    return t;
}

// Householder reduction of symmetric `a` to tridiagonal form (Numerical Recipes' tred2,
// 0-indexed): on return d holds the diagonal, e[1..n-1] the subdiagonal, and `a` the
// orthogonal matrix Q with Q^T A Q tridiagonal.
void householder_tridiagonalize(Dense& a, std::vector<double>& d, std::vector<double>& e) {
    const int64_t n = a.rows;
    for (int64_t i = n - 1; i > 0; --i) {
        const int64_t l = i - 1;
        double h = 0.0;
        if (l > 0) {
            double scale = 0.0;
            for (int64_t k = 0; k <= l; ++k) {
                scale += std::fabs(a.at(i, k));
            }
            if (scale == 0.0) {
                e[i] = a.at(i, l);
            } else {
                for (int64_t k = 0; k <= l; ++k) {
                    a.at(i, k) /= scale;
                    h += a.at(i, k) * a.at(i, k);
                }
                double f = a.at(i, l);
                double g = f >= 0.0 ? -std::sqrt(h) : std::sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                a.at(i, l) = f - g;
                f = 0.0;
                for (int64_t j = 0; j <= l; ++j) {
                    a.at(j, i) = a.at(i, j) / h;
                    g = 0.0;
                    for (int64_t k = 0; k <= j; ++k) {
                        g += a.at(j, k) * a.at(i, k);
                    }
                    for (int64_t k = j + 1; k <= l; ++k) {
                        g += a.at(k, j) * a.at(i, k);
                    }
                    e[j] = g / h;
                    f += e[j] * a.at(i, j);
                }
                const double hh = f / (h + h);
                for (int64_t j = 0; j <= l; ++j) {
                    f = a.at(i, j);
                    e[j] = g = e[j] - hh * f;
                    for (int64_t k = 0; k <= j; ++k) {
                        a.at(j, k) -= f * e[k] + g * a.at(i, k);
                    }
                }
            }
        } else {
            e[i] = a.at(i, l);
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    // Accumulate the transformations into Q.
    for (int64_t i = 0; i < n; ++i) {
        const int64_t l = i - 1;
        if (d[i] != 0.0) {
            for (int64_t j = 0; j <= l; ++j) {
                double g = 0.0;
                for (int64_t k = 0; k <= l; ++k) {
                    g += a.at(i, k) * a.at(k, j);
                }
                for (int64_t k = 0; k <= l; ++k) {
                    a.at(k, j) -= g * a.at(k, i);
                }
            }
        }
        d[i] = a.at(i, i);
        a.at(i, i) = 1.0;
        for (int64_t j = 0; j <= l; ++j) {
            a.at(j, i) = 0.0;
            a.at(i, j) = 0.0;
        }
    }
}

// Eigenvalues of the tridiagonal (d, e) by QL with implicit shifts (Numerical Recipes' tqli,
// 0-indexed), rotating the rows of zt alongside: on return d holds the eigenvalues and row j
// of zt the eigenvector for d[j].
void tridiagonal_ql(std::vector<double>& d, std::vector<double>& e, Dense& zt) {
    const auto n = static_cast<int64_t>(d.size());
    const double eps = std::numeric_limits<double>::epsilon();
    for (int64_t i = 1; i < n; ++i) {
        e[i - 1] = e[i];
    }
    if (n > 0) {
        e[n - 1] = 0.0;
    }
    for (int64_t l = 0; l < n; ++l) {
        int iterations = 0;
        int64_t m = l;
        do {
            for (m = l; m < n - 1; ++m) {
                if (std::fabs(e[m]) <= eps * (std::fabs(d[m]) + std::fabs(d[m + 1]))) {
                    break;
                }
            }
            if (m == l) {
                break;
            }
            if (iterations++ == 60) {
                throw std::runtime_error("SymmetricEigen: QL iteration did not converge");
            }
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
            double s = 1.0, c = 1.0, p = 0.0;
            int64_t i = m - 1;
            for (; i >= l; --i) {
                double f = s * e[i];
                const double b = c * e[i];
                e[i + 1] = r = std::hypot(f, g);
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                d[i + 1] = g + (p = s * r);
                g = c * r - b;
                double* z_i = &zt.at(i, 0);
                double* z_next = &zt.at(i + 1, 0);
                for (int64_t k = 0; k < zt.cols; ++k) {
                    f = z_next[k];
                    z_next[k] = s * z_i[k] + c * f;
                    z_i[k] = c * z_i[k] - s * f;
                }
            }
            if (r == 0.0 && i >= l) {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        } while (m != l);
    }
}

}  // namespace

EigenResult SymmetricEigen(const Tensor& input) {
    Dense a = load_symmetric(input, "SymmetricEigen");
    const int64_t n = a.rows;
    std::vector<double> d(static_cast<size_t>(n)), e(static_cast<size_t>(n));
    householder_tridiagonalize(a, d, e);
    // zt row j = column j of the accumulated transform: QL then rotates contiguous rows.
    Dense zt = transpose(a);
    tridiagonal_ql(d, e, zt);
    Dense v = transpose(zt);
    const std::vector<int64_t> order = descending_order(d);
    for (int64_t j = 0; j < n; ++j) {
        if (v.at(dominant_row(v, j), j) < 0.0) {
            for (int64_t i = 0; i < n; ++i) {
                v.at(i, j) = -v.at(i, j);
            }
        }
    }
    std::vector<double> sorted(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        sorted[static_cast<size_t>(j)] = d[static_cast<size_t>(order[static_cast<size_t>(j)])];
    }
    return {to_tensor(sorted, Shape({n}), input.backend()), columns_to_tensor(v, order, input.backend())};
}

DominantEigenResult PowerIteration(const Tensor& input, int64_t max_iterations, float tolerance, uint64_t seed) {
    Dense a = load_symmetric(input, "PowerIteration");
    if (max_iterations < 1) {
        throw std::invalid_argument("PowerIteration: max_iterations must be >= 1");
    }
    if (!(tolerance > 0.0f)) {
        throw std::invalid_argument("PowerIteration: tolerance must be positive");
    }
    const int64_t n = a.rows;
    auto normalize = [](std::vector<double>& x) {
        double norm = 0.0;
        for (double e : x) {
            norm += e * e;
        }
        norm = std::sqrt(norm);
        for (double& e : x) {
            e /= norm;
        }
        return norm;
    };

    // Starting vector from the Numerical Recipes LCG (CONTRIBUTING.md's seeded-randomness rule),
    // uniform in [-1, 1): almost surely not orthogonal to the dominant eigenvector.
    std::vector<double> x(static_cast<size_t>(n));
    uint64_t state = seed;
    for (double& e : x) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        e = static_cast<double>(state >> 11) / static_cast<double>(1ULL << 53) * 2.0 - 1.0;
    }
    normalize(x);

    DominantEigenResult result{0.0f, Tensor(Shape({n}), input.backend()), max_iterations, false};
    std::vector<double> y(static_cast<size_t>(n));
    for (int64_t it = 1; it <= max_iterations; ++it) {
        for (int64_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                sum += a.at(i, j) * x[static_cast<size_t>(j)];
            }
            y[static_cast<size_t>(i)] = sum;
        }
        double norm_sq = 0.0;
        for (double e : y) {
            norm_sq += e * e;
        }
        if (norm_sq == 0.0) {
            // x lies in the null space: A x = 0 x, an exact (if not dominant-by-magnitude
            // unless A = 0) eigenpair, and no further progress is possible.
            result.iterations = it;
            result.converged = true;
            break;
        }
        normalize(y);
        double diff_minus = 0.0, diff_plus = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            diff_minus += (y[i] - x[i]) * (y[i] - x[i]);
            diff_plus += (y[i] + x[i]) * (y[i] + x[i]);
        }
        x.swap(y);
        if (std::sqrt(std::min(diff_minus, diff_plus)) <= tolerance) {
            result.iterations = it;
            result.converged = true;
            break;
        }
    }

    double rayleigh = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            ax += a.at(i, j) * x[static_cast<size_t>(j)];
        }
        rayleigh += x[static_cast<size_t>(i)] * ax;
    }
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) {
        if (std::fabs(x[static_cast<size_t>(i)]) > std::fabs(x[static_cast<size_t>(best)])) {
            best = i;
        }
    }
    if (x[static_cast<size_t>(best)] < 0.0) {
        for (double& e : x) {
            e = -e;
        }
    }
    result.value = static_cast<float>(rayleigh);
    result.vector = to_tensor(x, Shape({n}), input.backend());
    return result;
}

QRResult QR(const Tensor& input) {
    Dense a = load(input, "QR");
    const int64_t m = a.rows, n = a.cols;
    if (m < n) {
        throw std::invalid_argument("QR: input must be tall or square (rows >= cols)");
    }
    // Householder vectors, v_j acting on rows j..m-1; an empty vector means H_j = I.
    std::vector<std::vector<double>> reflectors(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        double norm = 0.0;
        for (int64_t i = j; i < m; ++i) {
            norm += a.at(i, j) * a.at(i, j);
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) {
            continue;  // already zero below and on the diagonal: nothing to reflect
        }
        // Reflect x onto alpha*e1 with alpha opposite in sign to x0, avoiding cancellation.
        const double alpha = a.at(j, j) >= 0.0 ? -norm : norm;
        std::vector<double> v(static_cast<size_t>(m - j));
        for (int64_t i = j; i < m; ++i) {
            v[static_cast<size_t>(i - j)] = a.at(i, j);
        }
        v[0] -= alpha;
        double v_norm = 0.0;
        for (double e : v) {
            v_norm += e * e;
        }
        v_norm = std::sqrt(v_norm);
        for (double& e : v) {
            e /= v_norm;
        }
        for (int64_t c = j; c < n; ++c) {
            double dot = 0.0;
            for (int64_t i = j; i < m; ++i) {
                dot += v[static_cast<size_t>(i - j)] * a.at(i, c);
            }
            for (int64_t i = j; i < m; ++i) {
                a.at(i, c) -= 2.0 * dot * v[static_cast<size_t>(i - j)];
            }
        }
        reflectors[static_cast<size_t>(j)] = std::move(v);
    }

    // Q = H_0 H_1 ... H_{n-1} applied to the first n columns of the identity.
    Dense q{m, n, std::vector<double>(static_cast<size_t>(m * n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        q.at(i, i) = 1.0;
    }
    for (int64_t j = n - 1; j >= 0; --j) {
        const std::vector<double>& v = reflectors[static_cast<size_t>(j)];
        if (v.empty()) {
            continue;
        }
        for (int64_t c = 0; c < n; ++c) {
            double dot = 0.0;
            for (int64_t i = j; i < m; ++i) {
                dot += v[static_cast<size_t>(i - j)] * q.at(i, c);
            }
            for (int64_t i = j; i < m; ++i) {
                q.at(i, c) -= 2.0 * dot * v[static_cast<size_t>(i - j)];
            }
        }
    }

    Dense r{n, n, std::vector<double>(static_cast<size_t>(n * n), 0.0)};
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            r.at(i, j) = a.at(i, j);
        }
    }
    // Make the factorization unique: R's diagonal non-negative.
    for (int64_t i = 0; i < n; ++i) {
        if (r.at(i, i) < 0.0) {
            for (int64_t j = i; j < n; ++j) {
                r.at(i, j) = -r.at(i, j);
            }
            for (int64_t k = 0; k < m; ++k) {
                q.at(k, i) = -q.at(k, i);
            }
        }
    }
    return {to_tensor(q.v, Shape({m, n}), input.backend()), to_tensor(r.v, Shape({n, n}), input.backend())};
}

SVDResult SVD(const Tensor& input) {
    Dense a = load(input, "SVD");
    const bool wide = a.rows < a.cols;
    // A wide A = (A^T)^T: decompose the tall transpose and swap the roles of U and V.
    DenseSVD d = tall_svd(wide ? transpose(a) : a);
    Dense& u = wide ? d.v : d.u;
    Dense& v = wide ? d.u : d.v;
    const int64_t k = static_cast<int64_t>(d.s.size());
    for (int64_t j = 0; j < k; ++j) {
        if (v.at(dominant_row(v, j), j) < 0.0) {
            for (int64_t i = 0; i < u.rows; ++i) {
                u.at(i, j) = -u.at(i, j);
            }
            for (int64_t i = 0; i < v.rows; ++i) {
                v.at(i, j) = -v.at(i, j);
            }
        }
    }
    return {to_tensor(u.v, Shape({u.rows, k}), input.backend()), to_tensor(d.s, Shape({k}), input.backend()),
            to_tensor(v.v, Shape({v.rows, k}), input.backend())};
}

}  // namespace pulsatrix
