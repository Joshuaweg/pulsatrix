#include "pulsatrix/tda/rtd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "pulsatrix/nearest_neighbors.hpp"
#include "pulsatrix/representation_similarity.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"

namespace pulsatrix {
namespace {

// An index in [0, n) from raw mt19937 output by multiply-shift: the same on every platform.
int64_t Index(std::mt19937& rng, int64_t n) {
    return static_cast<int64_t>((static_cast<uint64_t>(rng()) * static_cast<uint64_t>(n)) >> 32);
}

// The q-quantile of the pairwise distances, interpolated linearly as numpy's default does.
double Quantile(const DistanceMatrix& d, double q) {
    std::vector<float> v = d.values;
    std::sort(v.begin(), v.end());
    const double position = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (position - static_cast<double>(lo)) * (static_cast<double>(v[hi]) - v[lo]);
}

// Divides every distance by the matrix's 0.9 quantile, so that scale doesn't count.
DistanceMatrix Normalized(DistanceMatrix d, const char* which) {
    const double q = Quantile(d, 0.9);
    if (!(q > 0)) {
        throw std::invalid_argument(std::string("Rtd: ") + which + "'s distances are 0 at the 0.9 quantile, so they can't be normalized");
    }
    for (float& v : d.values) v = static_cast<float>(v / q);
    d.max_distance = static_cast<float>(d.max_distance / q);
    d.error_bound /= q;
    return d;
}

double TotalFiniteLength(const PersistenceDiagram& d) {
    double total = 0;
    for (const PersistencePair& p : d.pairs) {
        if (!p.essential()) total += p.persistence();
    }
    return total;
}

// The rows of points at the given indices, as a tensor on its backend.
Tensor Rows(const std::vector<float>& host, int64_t width, const std::vector<int64_t>& indices, DeviceBackend* backend) {
    std::vector<float> rows;
    rows.reserve(indices.size() * static_cast<std::size_t>(width));
    for (int64_t i : indices) {
        const auto begin = host.begin() + static_cast<std::ptrdiff_t>(i * width);
        rows.insert(rows.end(), begin, begin + static_cast<std::ptrdiff_t>(width));
    }
    return Tensor(Shape({static_cast<int64_t>(indices.size()), width}), backend, rows);
}

}  // namespace

PersistenceDiagram CrossBarcode(const DistanceMatrix& w, const DistanceMatrix& w_tilde, int dimension) {
    if (w.size != w_tilde.size) throw std::invalid_argument("CrossBarcode: the two matrices are for different numbers of points");
    if (w.size < 2) throw std::invalid_argument("CrossBarcode: needs at least 2 points");
    if (dimension < 1 || dimension > 7) throw std::invalid_argument("CrossBarcode: dimension must be in [1, 7]");
    const int64_t n = w.size;
    constexpr float kNever = std::numeric_limits<float>::infinity();

    // Vertices 0..n-1 are A_1..A_n, and n..2n-1 their copies A'_1..A'_n.
    DistanceMatrix m;
    m.size = 2 * n;
    m.values.resize(static_cast<std::size_t>(m.size) * static_cast<std::size_t>(m.size - 1) / 2);
    m.error_bound = std::max(w.error_bound, w_tilde.error_bound);
    float largest = 0;
    for (int64_t p = 0; p < m.size; ++p) {
        for (int64_t q = p + 1; q < m.size; ++q) {
            float d = 0;
            if (p >= n) {
                d = std::min(w.at(p - n, q - n), w_tilde.at(p - n, q - n));  // A'_i to A'_j
            } else if (q >= n) {
                const int64_t a = p, b = q - n;  // A_a to A'_b
                d = a < b ? w.at(a, b) : (a == b ? 0.0f : kNever);
            }
            m.values[m.index(p, q)] = d;
            if (d != kNever) largest = std::max(largest, d);
        }
    }
    m.max_distance = largest;
    RipsOptions o;
    o.max_dimension = dimension;
    o.threshold = largest;  // every finite edge; the infinite ones never appear
    return ComputeRipsPersistence(m, o).diagrams[static_cast<std::size_t>(dimension)];
}

RtdResult Rtd(const Tensor& a, const Tensor& b, const RtdOptions& options) {
    if (a.rank() < 2 || b.rank() < 2) throw std::invalid_argument("Rtd: a and b must have shape (n, ...)");
    const int64_t n = a.shape().dim(0);
    if (b.shape().dim(0) != n) throw std::invalid_argument("Rtd: a and b have different numbers of rows");
    if (n < 4) throw std::invalid_argument("Rtd: needs at least 4 rows");
    if (options.sample_size < 4 || options.trials < 1) {
        throw std::invalid_argument("Rtd: sample_size must be at least 4 and trials at least 1");
    }
    const int64_t width_a = a.numel() / n, width_b = b.numel() / n;
    const std::vector<float> host_a = a.to_host_vector(), host_b = b.to_host_vector();
    const bool all = n <= options.sample_size;
    const int64_t size = all ? n : options.sample_size;
    const int64_t trials = all ? 1 : options.trials;

    std::mt19937 rng(options.seed);
    std::vector<int64_t> order(static_cast<std::size_t>(n));
    RtdResult out;
    out.trials = trials;
    for (int64_t t = 0; t < trials; ++t) {
        std::iota(order.begin(), order.end(), int64_t{0});
        if (!all) {
            // A partial Fisher-Yates shuffle: the first `size` entries are a uniform random subset.
            for (int64_t i = 0; i < size; ++i) std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(i + Index(rng, n - i))]);
        }
        const std::vector<int64_t> batch(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(size));
        const DistanceMatrix wa = Normalized(PairwiseDistances(Rows(host_a, width_a, batch, a.backend())), "a");
        const DistanceMatrix wb = Normalized(PairwiseDistances(Rows(host_b, width_b, batch, b.backend())), "b");
        out.rtd_ab += TotalFiniteLength(CrossBarcode(wa, wb, 1));
        out.rtd_ba += TotalFiniteLength(CrossBarcode(wb, wa, 1));
    }
    out.rtd_ab /= static_cast<double>(trials);
    out.rtd_ba /= static_cast<double>(trials);
    out.rtd = (out.rtd_ab + out.rtd_ba) / 2;
    return out;
}

RepresentationComparison CompareRepresentations(const Tensor& a, const Tensor& b, const RepresentationComparisonOptions& options) {
    RepresentationComparison out;
    out.cka = LinearCka(a, b);
    NeighborOptions neighbors;
    neighbors.k = options.k;
    out.mutual_knn = MutualKnnAlignment(a, b, neighbors);
    out.rtd = Rtd(a, b, options.rtd);
    return out;
}

}  // namespace pulsatrix
