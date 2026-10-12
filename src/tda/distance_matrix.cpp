#include "pulsatrix/tda/distance_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("PairwiseDistances: " + message);
}

constexpr double kFloatUnit = 0x1p-24;   // float32 unit roundoff
constexpr double kDoubleUnit = 0x1p-53;  // double unit roundoff

// Higham's gamma_k = k u / (1 - k u): the relative error bound of a k-term floating-point sum or
// dot product, in any order.
double Gamma(double k, double unit) {
    const double ku = k * unit;
    if (ku >= 0.5) Fail("the points' dimension is too large for a meaningful float32 error bound");
    return ku / (1.0 - ku);
}

// A hair over 1, to cover the rounding of the bound's own double-precision arithmetic.
constexpr double kSlack = 1.0 + 0x1p-48;

// The points as float32 rows, with how far each row moved when centering or normalizing in double
// was rounded back to float32 (0 when neither was asked for).
struct Prepared {
    int64_t n = 0;
    int64_t d = 0;
    std::vector<float> rows;
    std::vector<double> moved;
};

Prepared Prepare(const Tensor& points, const DistanceOptions& options) {
    if (points.rank() < 2) Fail("points must have shape (n, ...), a row per point");
    if (options.block_rows < 0) Fail("block_rows must be 0 (automatic) or positive");
    Prepared p;
    p.n = points.shape().dim(0);
    if (p.n < 2) Fail("needs at least 2 points");
    p.d = points.numel() / p.n;
    if (p.d < 1) Fail("points have no coordinates");
    p.rows = points.to_host_vector();
    for (float v : p.rows) {
        if (!std::isfinite(v)) Fail("points have a NaN or infinite value");
    }
    p.moved.assign(static_cast<std::size_t>(p.n), 0.0);
    if (!options.center && !options.normalize) return p;

    const std::size_t d = static_cast<std::size_t>(p.d);
    std::vector<double> x(p.rows.begin(), p.rows.end());
    if (options.center) {
        std::vector<double> mean(d, 0.0);
        for (int64_t i = 0; i < p.n; ++i) {
            for (std::size_t k = 0; k < d; ++k) mean[k] += x[static_cast<std::size_t>(i) * d + k];
        }
        for (double& m : mean) m /= static_cast<double>(p.n);
        for (int64_t i = 0; i < p.n; ++i) {
            for (std::size_t k = 0; k < d; ++k) x[static_cast<std::size_t>(i) * d + k] -= mean[k];
        }
    }
    for (int64_t i = 0; i < p.n; ++i) {
        double* row = &x[static_cast<std::size_t>(i) * d];
        double norm = 0;
        for (std::size_t k = 0; k < d; ++k) norm += row[k] * row[k];
        norm = std::sqrt(norm);
        if (options.normalize) {
            if (norm == 0) Fail("point " + std::to_string(i) + " is zero, so it can't be normalized");
            for (std::size_t k = 0; k < d; ++k) row[k] /= norm;
            norm = 1;
        }
        // Rounding to float32 moves the row by |rounded - row|; the double-precision centering and
        // normalizing that produced it are off from exact arithmetic by a few units of
        // (d + 8) 2^-53 of its norm.
        double moved = 0;
        for (std::size_t k = 0; k < d; ++k) {
            const float r = static_cast<float>(row[k]);
            p.rows[static_cast<std::size_t>(i) * d + k] = r;
            moved += (r - row[k]) * (r - row[k]);
        }
        p.moved[static_cast<std::size_t>(i)] =
            (std::sqrt(moved) + Gamma(static_cast<double>(d) + 8, kDoubleUnit) * 4 * norm) * kSlack;
    }
    return p;
}

// Folds one pair into the matrix: squared is the computed |x - y|^2, within squared_error of the
// exact value for the float32 rows; moved_i + moved_j covers the rows' own rounding.
struct Accumulator {
    DistanceMatrix* out;
    const Prepared* p;

    void Add(int64_t i, int64_t j, double squared, double squared_error) const {
        const double lo = std::sqrt(std::max(0.0, squared - squared_error) / kSlack);
        const double hi = std::sqrt((squared + squared_error) * kSlack);
        const float stored = static_cast<float>(std::sqrt(std::max(0.0, squared)));
        const double s = stored;
        const double error = std::max({hi - s, s - lo, 0.0}) * kSlack + p->moved[static_cast<std::size_t>(i)] +
                             p->moved[static_cast<std::size_t>(j)];
        out->values[out->index(i, j)] = stored;
        out->error_bound = std::max(out->error_bound, error);
        out->max_distance = std::max(out->max_distance, stored);
    }
};

void ExactDistances(const Prepared& p, const Accumulator& acc) {
    const std::size_t d = static_cast<std::size_t>(p.d);
    // Differences of floats are exact to within 2^-53 relative in double, and the d-term sum adds
    // gamma_d: |computed - exact| <= gamma_{d+3} exact <= 2 gamma_{d+3} computed.
    const double relative = 2 * Gamma(static_cast<double>(d) + 3, kDoubleUnit);
    for (int64_t i = 0; i < p.n; ++i) {
        const float* a = &p.rows[static_cast<std::size_t>(i) * d];
        for (int64_t j = i + 1; j < p.n; ++j) {
            const float* b = &p.rows[static_cast<std::size_t>(j) * d];
            double s = 0;
            for (std::size_t k = 0; k < d; ++k) {
                const double diff = static_cast<double>(a[k]) - static_cast<double>(b[k]);
                s += diff * diff;
            }
            acc.Add(i, j, s, relative * s);
        }
    }
}

void GemmDistances(const Prepared& p, DeviceBackend* backend, int64_t block_rows, const Accumulator& acc) {
    const std::size_t d = static_cast<std::size_t>(p.d);
    // Squared norms in double: products of floats are exact, the sum is within gamma_d.
    std::vector<double> norm2(static_cast<std::size_t>(p.n), 0.0);
    for (int64_t i = 0; i < p.n; ++i) {
        const float* a = &p.rows[static_cast<std::size_t>(i) * d];
        double s = 0;
        for (std::size_t k = 0; k < d; ++k) s += static_cast<double>(a[k]) * a[k];
        norm2[static_cast<std::size_t>(i)] = s;
    }
    const double gamma_float = Gamma(static_cast<double>(d), kFloatUnit);
    const double gamma_norm = Gamma(static_cast<double>(d), kDoubleUnit);
    const double gamma_combine = Gamma(3, kDoubleUnit);

    if (block_rows == 0) block_rows = std::max<int64_t>(1, (int64_t{1} << 24) / p.n);
    block_rows = std::min(block_rows, p.n);

    const Tensor points(Shape({p.n, p.d}), backend, p.rows);
    for (int64_t r0 = 0; r0 < p.n - 1; r0 += block_rows) {
        const int64_t rows = std::min(block_rows, p.n - r0);
        const int64_t cols = p.n - r0;  // every point from r0 on: the block's upper triangle
        Tensor block(Shape({rows, cols}), backend);
        const float* first = points.data() + static_cast<std::size_t>(r0) * d;
        backend->gemm_ex(first, false, first, true, block.data(), static_cast<std::size_t>(rows), d,
                         static_cast<std::size_t>(cols), 0.0f);
        const std::vector<float> dots = block.to_host_vector();
        for (int64_t bi = 0; bi < rows; ++bi) {
            const int64_t i = r0 + bi;
            const double ni = norm2[static_cast<std::size_t>(i)];
            for (int64_t j = i + 1; j < p.n; ++j) {
                const double nj = norm2[static_cast<std::size_t>(j)];
                const double dot = dots[static_cast<std::size_t>(bi * cols + (j - r0))];
                const double squared = ni + nj - 2 * dot;
                // The float32 dot product is within gamma_d sum |x_k y_k| <= gamma_d |x| |y| of the
                // exact one (Cauchy-Schwarz); the norms within gamma_d of theirs, from double
                // sums; and the three-term combination adds gamma_3 of its terms' magnitudes.
                const double error = 2 * gamma_float * std::sqrt(ni * nj) * kSlack +
                                     gamma_norm * (ni + nj) * (1 + gamma_norm) +
                                     gamma_combine * (ni + nj + 2 * std::abs(dot));
                acc.Add(i, j, squared, error * kSlack);
            }
        }
    }
}

DistanceMatrix Compute(const Tensor& points, DeviceBackend* backend, const DistanceOptions& options) {
    const Prepared p = Prepare(points, options);
    DistanceMatrix out;
    out.size = p.n;
    out.values.assign(static_cast<std::size_t>(p.n) * static_cast<std::size_t>(p.n - 1) / 2, 0.0f);
    const Accumulator acc{&out, &p};
    switch (options.method) {
        case DistanceMethod::Exact:
            ExactDistances(p, acc);
            break;
        case DistanceMethod::Gemm:
            GemmDistances(p, backend, options.block_rows, acc);
            break;
        default:
            Fail("unknown method");
    }
    return out;
}

}  // namespace

std::size_t DistanceMatrix::index(int64_t i, int64_t j) const {
    if (i == j || i < 0 || j < 0 || i >= size || j >= size) {
        throw std::out_of_range("DistanceMatrix: no pair (" + std::to_string(i) + ", " + std::to_string(j) + ")");
    }
    if (i > j) std::swap(i, j);
    // Rows 0 .. i-1 hold n-1, n-2, ..., n-i pairs.
    const auto n = static_cast<std::size_t>(size);
    const auto a = static_cast<std::size_t>(i);
    return a * n - a * (a + 1) / 2 + static_cast<std::size_t>(j - i - 1);
}

float DistanceMatrix::at(int64_t i, int64_t j) const {
    if (i == j && i >= 0 && i < size) return 0.0f;
    return values[index(i, j)];
}

DistanceMatrix PairwiseDistances(const Tensor& points, const DistanceOptions& options) {
    return Compute(points, points.backend(), options);
}

DistanceMatrix PairwiseDistances(const PointCloud& cloud, DeviceBackend* backend, const DistanceOptions& options) {
    return Compute(cloud.ToTensor(backend), backend, options);
}

}  // namespace pulsatrix
