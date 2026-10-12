#include "pulsatrix/tda/distance_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "pulsatrix/device_backend.hpp"
#include "float_error.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("PairwiseDistances: " + message);
}

using tda_detail::DistanceError;
using tda_detail::Gamma;
using tda_detail::kDoubleUnit;
using Prepared = tda_detail::PreparedPoints;

Prepared Prepare(const Tensor& points, const DistanceOptions& options) {
    if (options.block_rows < 0) Fail("block_rows must be 0 (automatic) or positive");
    return tda_detail::PreparePoints(points, options.center, options.normalize, 2, "PairwiseDistances");
}

// Folds one pair into the matrix: squared is the computed |x - y|^2, within squared_error of the
// exact value for the float32 rows; moved_i + moved_j covers the rows' own rounding.
struct Accumulator {
    DistanceMatrix* out;
    const Prepared* p;

    void Add(int64_t i, int64_t j, double squared, double squared_error) const {
        const float stored = static_cast<float>(std::sqrt(std::max(0.0, squared)));
        const double error = DistanceError(squared, squared_error, stored) + p->moved[static_cast<std::size_t>(i)] +
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
    const double relative = 2 * Gamma(static_cast<double>(d) + 3, kDoubleUnit, "PairwiseDistances");
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
    const std::vector<double> norm2 = tda_detail::SquaredNorms(p);
    const tda_detail::GemmSquaredError squared_error(p.d, "PairwiseDistances");

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
                acc.Add(i, j, ni + nj - 2 * dot, squared_error(ni, nj, dot));
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
