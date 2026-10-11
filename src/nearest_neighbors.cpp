#include "pulsatrix/nearest_neighbors.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
        throw std::invalid_argument(std::string(function) + ": " + what + " must have shape (n, ...), a row per point");
    }
    Rows r;
    r.n = t.shape().dim(0);
    r.p = r.n == 0 ? 0 : static_cast<int64_t>(t.numel()) / r.n;
    const std::vector<float> host = t.to_host_vector();
    r.x.reserve(host.size());
    for (float v : host) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string(function) + ": " + what + " has a NaN or infinite value");
        }
        r.x.push_back(v);
    }
    return r;
}

std::vector<double> Mean(const std::vector<const Rows*>& sets) {
    std::vector<double> mean(static_cast<std::size_t>(sets.front()->p), 0.0);
    int64_t count = 0;
    for (const Rows* s : sets) {
        for (int64_t i = 0; i < s->n; ++i) {
            const double* r = s->row(i);
            for (int64_t a = 0; a < s->p; ++a) {
                mean[static_cast<std::size_t>(a)] += r[a];
            }
        }
        count += s->n;
    }
    for (double& m : mean) {
        m /= static_cast<double>(count);
    }
    return mean;
}

void Subtract(Rows& r, const std::vector<double>& mean) {
    for (int64_t i = 0; i < r.n; ++i) {
        double* row = r.row(i);
        for (int64_t a = 0; a < r.p; ++a) {
            row[a] -= mean[static_cast<std::size_t>(a)];
        }
    }
}

// Scales rows to unit length, so a dot product is the cosine. Zero rows stay zero.
void Normalize(Rows& r) {
    for (int64_t i = 0; i < r.n; ++i) {
        double* row = r.row(i);
        double s = 0;
        for (int64_t a = 0; a < r.p; ++a) {
            s += row[a] * row[a];
        }
        if (s > 0) {
            const double inv = 1.0 / std::sqrt(s);
            for (int64_t a = 0; a < r.p; ++a) {
                row[a] *= inv;
            }
        }
    }
}

void Prepare(std::vector<Rows*> sets, const std::vector<const Rows*>& mean_of, const NeighborOptions& options) {
    if (options.center) {
        const std::vector<double> mean = Mean(mean_of);
        for (Rows* s : sets) {
            Subtract(*s, mean);
        }
    }
    if (options.metric == NeighborMetric::Cosine) {
        for (Rows* s : sets) {
            Normalize(*s);
        }
    }
}

// The k nearest points to each query. With `same`, queries and points are one set and a point
// is never its own neighbor.
Neighbors Search(const Rows& queries, const Rows& points, bool same, const NeighborOptions& options) {
    Neighbors out;
    out.rows = queries.n;
    out.k = options.k;
    out.metric = options.metric;
    out.indices.resize(static_cast<std::size_t>(queries.n * options.k));
    out.scores.resize(out.indices.size());
    const bool cosine = options.metric == NeighborMetric::Cosine;

    std::vector<double> score(static_cast<std::size_t>(points.n));
    std::vector<int64_t> order;
    order.reserve(static_cast<std::size_t>(points.n));
    for (int64_t q = 0; q < queries.n; ++q) {
        const double* x = queries.row(q);
        order.clear();
        for (int64_t j = 0; j < points.n; ++j) {
            if (same && j == q) {
                continue;
            }
            const double* y = points.row(j);
            double s = 0;
            if (cosine) {
                for (int64_t a = 0; a < points.p; ++a) {
                    s += x[a] * y[a];
                }
            } else {
                for (int64_t a = 0; a < points.p; ++a) {
                    const double d = x[a] - y[a];
                    s += d * d;
                }
            }
            score[static_cast<std::size_t>(j)] = s;
            order.push_back(j);
        }
        // Nearest first; ties keep the lower index first.
        const auto nearer = [&](int64_t i, int64_t j) {
            const double si = score[static_cast<std::size_t>(i)];
            const double sj = score[static_cast<std::size_t>(j)];
            if (si != sj) {
                return cosine ? si > sj : si < sj;
            }
            return i < j;
        };
        std::partial_sort(order.begin(), order.begin() + options.k, order.end(), nearer);
        for (int64_t r = 0; r < options.k; ++r) {
            const int64_t j = order[static_cast<std::size_t>(r)];
            const double s = score[static_cast<std::size_t>(j)];
            out.indices[static_cast<std::size_t>(q * options.k + r)] = j;
            out.scores[static_cast<std::size_t>(q * options.k + r)] = cosine ? s : std::sqrt(s);
        }
    }
    return out;
}

void CheckK(const char* function, int64_t k, int64_t available) {
    if (k < 1 || k > available) {
        throw std::invalid_argument(std::string(function) + ": k is " + std::to_string(k) + "; it must be between 1 and " +
                                    std::to_string(available) + ", the number of candidate neighbors");
    }
}

void CheckWidths(const char* function, const Rows& a, const Rows& b) {
    if (a.p != b.p) {
        throw std::invalid_argument(std::string(function) + ": the two sets have widths " + std::to_string(a.p) + " and " +
                                    std::to_string(b.p) + "; they must be in the same space");
    }
}

}  // namespace

int64_t Neighbors::index(int64_t row, int64_t rank) const {
    if (row < 0 || row >= rows || rank < 0 || rank >= k) {
        throw std::out_of_range("Neighbors::index: (" + std::to_string(row) + ", " + std::to_string(rank) + ") is outside " +
                                std::to_string(rows) + " × " + std::to_string(k));
    }
    return indices[static_cast<std::size_t>(row * k + rank)];
}

double Neighbors::score(int64_t row, int64_t rank) const {
    if (row < 0 || row >= rows || rank < 0 || rank >= k) {
        throw std::out_of_range("Neighbors::score: (" + std::to_string(row) + ", " + std::to_string(rank) + ") is outside " +
                                std::to_string(rows) + " × " + std::to_string(k));
    }
    return scores[static_cast<std::size_t>(row * k + rank)];
}

Neighbors NearestNeighbors(const Tensor& points, const NeighborOptions& options) {
    Rows p = ToRows(points, "NearestNeighbors", "points");
    CheckK("NearestNeighbors", options.k, p.n - 1);
    Prepare({&p}, {&p}, options);
    return Search(p, p, /*same=*/true, options);
}

Neighbors NearestNeighbors(const Tensor& queries, const Tensor& points, const NeighborOptions& options) {
    Rows q = ToRows(queries, "NearestNeighbors", "queries");
    Rows p = ToRows(points, "NearestNeighbors", "points");
    CheckWidths("NearestNeighbors", q, p);
    CheckK("NearestNeighbors", options.k, p.n);
    Prepare({&q, &p}, {&p}, options);
    return Search(q, p, /*same=*/false, options);
}

std::vector<std::pair<int64_t, int64_t>> MutualNearestNeighbors(const Tensor& a, const Tensor& b,
                                                                const NeighborOptions& options) {
    Rows ra = ToRows(a, "MutualNearestNeighbors", "a");
    Rows rb = ToRows(b, "MutualNearestNeighbors", "b");
    CheckWidths("MutualNearestNeighbors", ra, rb);
    CheckK("MutualNearestNeighbors", options.k, std::min(ra.n, rb.n));
    Prepare({&ra, &rb}, {&ra, &rb}, options);
    const Neighbors a_in_b = Search(ra, rb, false, options);
    const Neighbors b_in_a = Search(rb, ra, false, options);

    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (int64_t i = 0; i < ra.n; ++i) {
        std::vector<int64_t> js(a_in_b.indices.begin() + i * options.k, a_in_b.indices.begin() + (i + 1) * options.k);
        std::sort(js.begin(), js.end());
        for (int64_t j : js) {
            const auto begin = b_in_a.indices.begin() + j * options.k;
            if (std::find(begin, begin + options.k, i) != begin + options.k) {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
}

double MutualKnnAlignment(const Tensor& a, const Tensor& b, const NeighborOptions& options) {
    Rows ra = ToRows(a, "MutualKnnAlignment", "a");
    Rows rb = ToRows(b, "MutualKnnAlignment", "b");
    if (ra.n != rb.n) {
        throw std::invalid_argument("MutualKnnAlignment: the representations have " + std::to_string(ra.n) + " and " +
                                    std::to_string(rb.n) + " rows; they must describe the same inputs");
    }
    CheckK("MutualKnnAlignment", options.k, ra.n - 1);
    Prepare({&ra}, {&ra}, options);
    Prepare({&rb}, {&rb}, options);
    const Neighbors na = Search(ra, ra, true, options);
    const Neighbors nb = Search(rb, rb, true, options);

    const auto k = static_cast<std::size_t>(options.k);
    double total = 0;
    std::vector<int64_t> x(k);
    std::vector<int64_t> y(k);
    for (int64_t i = 0; i < ra.n; ++i) {
        std::copy_n(na.indices.begin() + i * options.k, k, x.begin());
        std::copy_n(nb.indices.begin() + i * options.k, k, y.begin());
        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());
        std::vector<int64_t> shared;
        std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(shared));
        total += static_cast<double>(shared.size()) / static_cast<double>(k);
    }
    return total / static_cast<double>(ra.n);
}

Hubness MeasureHubness(const Neighbors& neighbors, int64_t num_points) {
    if (num_points < 1) {
        throw std::invalid_argument("MeasureHubness: num_points must be at least 1");
    }
    Hubness h;
    h.occurrences.assign(static_cast<std::size_t>(num_points), 0);
    for (int64_t j : neighbors.indices) {
        if (j < 0 || j >= num_points) {
            throw std::invalid_argument("MeasureHubness: neighbor index " + std::to_string(j) + " is outside 0 to " +
                                        std::to_string(num_points - 1));
        }
        ++h.occurrences[static_cast<std::size_t>(j)];
    }
    const double n = static_cast<double>(num_points);
    double mean = 0;
    for (int64_t c : h.occurrences) {
        mean += static_cast<double>(c);
    }
    mean /= n;
    double m2 = 0;
    double m3 = 0;
    int64_t never = 0;
    for (int64_t c : h.occurrences) {
        const double d = static_cast<double>(c) - mean;
        m2 += d * d;
        m3 += d * d * d;
        never += c == 0 ? 1 : 0;
        h.max_occurrence = std::max(h.max_occurrence, c);
    }
    m2 /= n;
    m3 /= n;
    h.skewness = m2 > 0 ? m3 / std::pow(m2, 1.5) : 0.0;
    h.max_fraction = neighbors.rows > 0 ? static_cast<double>(h.max_occurrence) / static_cast<double>(neighbors.rows) : 0.0;
    h.never_a_neighbor = static_cast<double>(never) / n;
    return h;
}

}  // namespace pulsatrix
