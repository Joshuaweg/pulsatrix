// Internal to the TDA sources: the floating-point error analysis behind certified distances,
// shared by PairwiseDistances() (TDA-2) and the landmark selection (TDA-7). Not installed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix::tda_detail {

constexpr double kFloatUnit = 0x1p-24;   // float32 unit roundoff
constexpr double kDoubleUnit = 0x1p-53;  // double unit roundoff

// A hair over 1, to cover the rounding of a bound's own double-precision arithmetic.
constexpr double kSlack = 1.0 + 0x1p-48;

// Higham's gamma_k = k u / (1 - k u): the relative error bound of a k-term floating-point sum or
// dot product, in any order.
inline double Gamma(double k, double unit, const std::string& function) {
    const double ku = k * unit;
    if (ku >= 0.5) {
        throw std::invalid_argument(function + ": the points' dimension is too large for a meaningful float32 error bound");
    }
    return ku / (1.0 - ku);
}

// Points as float32 rows, and how far each row moved when centering or normalizing in double was
// rounded back to float32 (0 when neither was asked for).
struct PreparedPoints {
    int64_t n = 0;
    int64_t d = 0;
    std::vector<float> rows;
    std::vector<double> moved;
};

inline PreparedPoints PreparePoints(const Tensor& points, bool center, bool normalize, int64_t min_rows,
                                    const std::string& function) {
    auto fail = [&](const std::string& message) { throw std::invalid_argument(function + ": " + message); };
    if (points.rank() < 2) fail("points must have shape (n, ...), a row per point");
    PreparedPoints p;
    p.n = points.shape().dim(0);
    if (p.n < min_rows) fail("needs at least " + std::to_string(min_rows) + " points");
    p.d = points.numel() / p.n;
    if (p.d < 1) fail("points have no coordinates");
    p.rows = points.to_host_vector();
    for (float v : p.rows) {
        if (!std::isfinite(v)) fail("points have a NaN or infinite value");
    }
    p.moved.assign(static_cast<std::size_t>(p.n), 0.0);
    if (!center && !normalize) return p;

    const std::size_t d = static_cast<std::size_t>(p.d);
    std::vector<double> x(p.rows.begin(), p.rows.end());
    if (center) {
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
        if (normalize) {
            if (norm == 0) fail("point " + std::to_string(i) + " is zero, so it can't be normalized");
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
            (std::sqrt(moved) + Gamma(static_cast<double>(d) + 8, kDoubleUnit, function) * 4 * norm) * kSlack;
    }
    return p;
}

// Squared norms of the float32 rows, in double: products of floats are exact, and the d-term sum
// is within gamma_d.
inline std::vector<double> SquaredNorms(const PreparedPoints& p) {
    const std::size_t d = static_cast<std::size_t>(p.d);
    std::vector<double> norm2(static_cast<std::size_t>(p.n), 0.0);
    for (int64_t i = 0; i < p.n; ++i) {
        const float* a = &p.rows[static_cast<std::size_t>(i) * d];
        double s = 0;
        for (std::size_t k = 0; k < d; ++k) s += static_cast<double>(a[k]) * a[k];
        norm2[static_cast<std::size_t>(i)] = s;
    }
    return norm2;
}

// The error of |x|^2 + |y|^2 - 2 x.y, combined in double from squared norms ni and nj (double sums)
// and a float32 GEMM dot product. The dot product is within gamma_d sum |x_k y_k| <= gamma_d |x| |y|
// of the exact one (Cauchy-Schwarz); the norms within gamma_d of theirs; and the three-term
// combination adds gamma_3 of its terms' magnitudes.
struct GemmSquaredError {
    double gamma_float;
    double gamma_norm;
    double gamma_combine;

    GemmSquaredError(int64_t d, const std::string& function)
        : gamma_float(Gamma(static_cast<double>(d), kFloatUnit, function)),
          gamma_norm(Gamma(static_cast<double>(d), kDoubleUnit, function)),
          gamma_combine(Gamma(3, kDoubleUnit, function)) {}

    [[nodiscard]] double operator()(double ni, double nj, double dot) const {
        return (2 * gamma_float * std::sqrt(ni * nj) * kSlack + gamma_norm * (ni + nj) * (1 + gamma_norm) +
                gamma_combine * (ni + nj + 2 * std::abs(dot))) *
               kSlack;
    }
};

// How far `value` can be from the exact distance, given the computed squared distance and a bound
// on its error: the exact squared distance lies in [squared - error, squared + error] (and is at
// least 0), so the distance lies between the square roots of the ends.
inline double DistanceError(double squared, double squared_error, double value) {
    const double lo = std::sqrt(std::max(0.0, squared - squared_error) / kSlack);
    const double hi = std::sqrt((squared + squared_error) * kSlack);
    return std::max({hi - value, value - lo, 0.0}) * kSlack;
}

}  // namespace pulsatrix::tda_detail
