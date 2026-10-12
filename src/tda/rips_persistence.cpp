#include "pulsatrix/tda/rips_persistence.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace pulsatrix {
namespace {

using Index = int64_t;

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("ComputeRipsPersistence: " + message);
}

// A simplex (by its index in the combinatorial number system) and its diameter, or a cochain entry
// with a coefficient.
struct Entry {
    float diameter = 0;
    Index index = 0;
    int coefficient = 1;
};

// The filtration order: by diameter, ties by index. Any order that refines the diameter gives the
// same diagram; this one makes critical edges and cocycles reproducible.
bool Before(const Entry& a, const Entry& b) {
    return a.diameter < b.diameter || (a.diameter == b.diameter && a.index < b.index);
}

// For std::priority_queue: the top is the earliest entry in the filtration.
struct LaterFirst {
    bool operator()(const Entry& a, const Entry& b) const { return Before(b, a); }
};
using CoboundaryHeap = std::priority_queue<Entry, std::vector<Entry>, LaterFirst>;

bool IsPrime(int p) {
    if (p < 2) return false;
    for (int d = 2; d * d <= p; ++d) {
        if (p % d == 0) return false;
    }
    return true;
}

// The Vietoris-Rips complex below a threshold, with simplices named by their index: the simplex
// with vertices u_0 > u_1 > ... > u_m has index sum_i C(u_i, m + 1 - i).
class RipsComplex {
public:
    RipsComplex(const DistanceMatrix& distances, double threshold, int max_simplex_dim, int p)
        : d_(distances), n_(distances.size), threshold_(threshold), p_(p),
          binomial_(static_cast<std::size_t>(max_simplex_dim + 2),
                    std::vector<Index>(static_cast<std::size_t>(n_ + 1), 0)) {
        // binomial_[k][v] = C(v, k), by Pascal's rule, saturating where it overflows.
        constexpr Index kMax = std::numeric_limits<Index>::max();
        for (std::size_t v = 0; v <= static_cast<std::size_t>(n_); ++v) {
            binomial_[0][v] = 1;
            for (std::size_t k = 1; k < binomial_.size(); ++k) {
                if (v == 0) continue;
                const Index a = binomial_[k - 1][v - 1], b = binomial_[k][v - 1];
                binomial_[k][v] = (a > kMax - b) ? kMax : a + b;
            }
        }
        if (binomial_.back()[static_cast<std::size_t>(n_)] == kMax) {
            Fail("too many points to index simplices of dimension " + std::to_string(max_simplex_dim) +
                 " in 63 bits");
        }
        row_offset_.resize(static_cast<std::size_t>(n_));
        for (int64_t i = 0; i < n_; ++i) {
            // pdist order: pair (i, j), i < j, is at i n - i (i + 1) / 2 + (j - i - 1).
            row_offset_[static_cast<std::size_t>(i)] = i * n_ - i * (i + 1) / 2 - i - 1;
        }
        inverse_.assign(static_cast<std::size_t>(p), 0);
        for (int a = 1; a < p; ++a) {
            for (int b = 1; b < p; ++b) {
                if ((a * b) % p == 1) {
                    inverse_[static_cast<std::size_t>(a)] = b;
                    break;
                }
            }
        }
    }

    [[nodiscard]] int64_t size() const { return n_; }
    [[nodiscard]] int p() const { return p_; }
    [[nodiscard]] int inverse(int a) const { return inverse_[static_cast<std::size_t>(a)]; }
    [[nodiscard]] bool within(float diameter) const { return static_cast<double>(diameter) <= threshold_; }
    [[nodiscard]] Index binomial(int64_t v, int k) const {
        return binomial_[static_cast<std::size_t>(k)][static_cast<std::size_t>(v)];
    }
    [[nodiscard]] float distance(int64_t i, int64_t j) const {
        if (i == j) return 0.0f;
        if (i > j) std::swap(i, j);
        return d_.values[static_cast<std::size_t>(row_offset_[static_cast<std::size_t>(i)] + j)];
    }

    // The index of the simplex with the given vertices, largest first.
    [[nodiscard]] Index IndexOf(const std::vector<int64_t>& v) const {
        Index index = 0;
        const int top = static_cast<int>(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) index += binomial(v[i], top - static_cast<int>(i));
        return index;
    }

    [[nodiscard]] float Diameter(const std::vector<int64_t>& v) const {
        float diameter = 0;
        for (std::size_t a = 0; a < v.size(); ++a) {
            for (std::size_t b = a + 1; b < v.size(); ++b) diameter = std::max(diameter, distance(v[a], v[b]));
        }
        return diameter;
    }

    // The cofacet of s (vertices v, largest first) that adds w, if it is within the threshold.
    bool Cofacet(const Entry& s, int dim, const std::vector<int64_t>& v, int64_t w, Entry& out) const {
        float diameter = s.diameter;
        for (int64_t u : v) diameter = std::max(diameter, distance(w, u));
        if (!within(diameter)) return false;
        std::size_t k = 0;  // v[0..k) are above w
        while (k < v.size() && v[k] > w) ++k;
        Index index = 0;
        for (std::size_t i = 0; i < k; ++i) index += binomial(v[i], dim + 2 - static_cast<int>(i));
        index += binomial(w, dim + 2 - static_cast<int>(k));
        for (std::size_t i = k; i < v.size(); ++i) index += binomial(v[i], dim + 1 - static_cast<int>(i));
        out = Entry{diameter, index, (k % 2 == 0) ? s.coefficient : (p_ - s.coefficient) % p_};
        return true;
    }

    // The oldest cofacet of s. Cofacet indices grow with the added vertex, and no cofacet is
    // shorter than s, so the first one of s's own diameter is the oldest and ends the search.
    bool OldestCofacet(const Entry& s, int dim, const std::vector<int64_t>& v, Entry& out) const {
        bool any = false;
        Entry c;
        for (int64_t w = 0; w < n_; ++w) {
            if (std::find(v.begin(), v.end(), w) != v.end() || !Cofacet(s, dim, v, w, c)) continue;
            if (!any || Before(c, out)) out = c;
            any = true;
            if (c.diameter == s.diameter) break;
        }
        return any;
    }

    // The youngest facet of tau (vertices u, largest first) of tau's own diameter. Removing a
    // smaller vertex gives a larger index, so the first such facet from the smallest is it.
    bool YoungestEqualFacet(const Entry& tau, const std::vector<int64_t>& u, Entry& out,
                            std::vector<int64_t>& facet) const {
        for (std::size_t r = u.size(); r-- > 0;) {
            facet.assign(u.begin(), u.end());
            facet.erase(facet.begin() + static_cast<std::ptrdiff_t>(r));
            const float diameter = Diameter(facet);
            if (diameter == tau.diameter) {
                out = Entry{diameter, IndexOf(facet), 1};
                return true;
            }
        }
        return false;
    }

    // Whether s and its oldest cofacet form a zero-persistence apparent pair (Bauer 2021): the
    // cofacet has s's diameter and s is its youngest facet. Such pairs need no reduction.
    bool ZeroApparentCofacet(const Entry& s, int dim, const std::vector<int64_t>& v, Entry& tau) const {
        if (!OldestCofacet(s, dim, v, tau) || tau.diameter != s.diameter) return false;
        std::vector<int64_t> u, facet;
        Vertices(tau.index, dim + 1, u);
        Entry youngest;
        return YoungestEqualFacet(tau, u, youngest, facet) && youngest.index == s.index;
    }

    // Whether tau, a (dim)-simplex with dim >= 2, is the cofacet of a zero-persistence apparent
    // pair; if so, sigma is the facet, with the coefficient of tau in its coboundary.
    bool ZeroApparentFacet(const Entry& tau, int dim, Entry& sigma) const {
        std::vector<int64_t> u, facet;
        Vertices(tau.index, dim, u);
        if (!YoungestEqualFacet(tau, u, sigma, facet)) return false;
        Entry oldest;
        if (!OldestCofacet(sigma, dim - 1, facet, oldest) || oldest.index != tau.index) return false;
        sigma.coefficient = oldest.coefficient;  // with sigma's coefficient 1
        return true;
    }

    // The dim + 1 vertices of a simplex, largest first.
    void Vertices(Index index, int dim, std::vector<int64_t>& out) const {
        out.clear();
        int64_t upper = n_ - 1;
        for (int k = dim + 1; k >= 1; --k) {
            // The largest v <= upper with C(v, k) <= index; C(k - 1, k) = 0 always qualifies.
            int64_t lo = k - 1, hi = upper;
            while (lo < hi) {
                const int64_t mid = lo + (hi - lo + 1) / 2;
                if (binomial(mid, k) <= index) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            out.push_back(lo);
            index -= binomial(lo, k);
            upper = lo - 1;
        }
    }

    // The longest edge among the given vertices, endpoints in increasing order; the first found
    // wins a tie.
    [[nodiscard]] std::array<int64_t, 2> LongestEdge(const std::vector<int64_t>& v) const {
        std::array<int64_t, 2> best{{-1, -1}};
        float length = -1;
        for (std::size_t a = 0; a < v.size(); ++a) {
            for (std::size_t b = a + 1; b < v.size(); ++b) {
                const float l = distance(v[a], v[b]);
                if (l > length) {
                    length = l;
                    best = {{std::min(v[a], v[b]), std::max(v[a], v[b])}};
                }
            }
        }
        return best;
    }

    // Calls f(cofacet) for every cofacet within the threshold of the dim-simplex s (vertices v,
    // largest first), with coefficient (-1)^k s.coefficient, k the number of s's vertices above the
    // added one.
    template <typename F>
    void ForEachCofacet(const Entry& s, int dim, const std::vector<int64_t>& v, F&& f) const {
        Entry c;
        std::size_t k = v.size();  // v[k..) are below w; v is largest first
        for (int64_t w = 0; w < n_; ++w) {
            while (k > 0 && v[k - 1] < w) --k;
            if (k > 0 && v[k - 1] == w) continue;
            if (Cofacet(s, dim, v, w, c)) f(c);
        }
    }

private:
    const DistanceMatrix& d_;
    int64_t n_;
    double threshold_;
    int p_;
    std::vector<std::vector<Index>> binomial_;
    std::vector<int64_t> row_offset_;
    std::vector<int> inverse_;
};

// Sorts features as PersistenceDiagram lists pairs: finite by death, then birth; essential last.
void SortFeatures(std::vector<RipsFeature>& features) {
    std::stable_sort(features.begin(), features.end(), [](const RipsFeature& a, const RipsFeature& b) {
        if (a.pair.essential() != b.pair.essential()) return !a.pair.essential();
        if (a.pair.death != b.pair.death) return a.pair.death < b.pair.death;
        return a.pair.birth < b.pair.birth;
    });
}

// Adds up a cochain's repeated entries mod p, dropping zeros, in index order.
std::vector<Entry> Compact(std::vector<Entry> entries, int p) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.index < b.index; });
    std::vector<Entry> out;
    for (const Entry& e : entries) {
        if (!out.empty() && out.back().index == e.index) {
            out.back().coefficient = (out.back().coefficient + e.coefficient) % p;
        } else {
            out.push_back(e);
        }
        if (out.back().coefficient == 0) out.pop_back();
    }
    return out;
}

// The heap's earliest entry with a nonzero total coefficient, left on the heap; false if the
// cochain is zero.
bool PopPivot(CoboundaryHeap& heap, int p, Entry& pivot) {
    while (!heap.empty()) {
        Entry top = heap.top();
        heap.pop();
        int sum = top.coefficient;
        while (!heap.empty() && heap.top().index == top.index) {
            sum = (sum + heap.top().coefficient) % p;
            heap.pop();
        }
        if (sum != 0) {
            top.coefficient = sum;
            heap.push(top);
            pivot = top;
            return true;
        }
    }
    return false;
}

struct DimensionResult {
    std::vector<RipsFeature> features;
    std::unordered_set<Index> pivots;  // (dim + 1)-simplices that kill a class: cleared next
};

// H0 by union-find over the edges in filtration order. Returns the features; the edges that merge
// components are the pivots that clear dimension 1.
DimensionResult ReduceDimensionZero(const RipsComplex& complex, std::vector<Entry> edges) {
    std::sort(edges.begin(), edges.end(), Before);
    const int64_t n = complex.size();
    std::vector<int64_t> root(static_cast<std::size_t>(n));
    std::iota(root.begin(), root.end(), int64_t{0});
    auto find = [&](int64_t x) {
        while (root[static_cast<std::size_t>(x)] != x) {
            root[static_cast<std::size_t>(x)] = root[static_cast<std::size_t>(root[static_cast<std::size_t>(x)])];
            x = root[static_cast<std::size_t>(x)];
        }
        return x;
    };
    DimensionResult out;
    std::vector<int64_t> v;
    for (const Entry& e : edges) {
        complex.Vertices(e.index, 1, v);
        const int64_t a = find(v[0]), b = find(v[1]);
        if (a == b) continue;
        root[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
        out.pivots.insert(e.index);
        if (e.diameter > 0) {
            RipsFeature f;
            f.pair = {0.0, static_cast<double>(e.diameter)};
            f.death_edge = {{v[1], v[0]}};
            out.features.push_back(f);
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        if (find(i) == i) out.features.push_back(RipsFeature{});  // (0, infinity)
    }
    return out;
}

// Persistent cohomology in one dimension: reduces the coboundary columns of the given simplices,
// from the latest in the filtration to the earliest.
DimensionResult ReduceDimension(const RipsComplex& complex, int dim, std::vector<Entry> columns, bool cocycles) {
    std::sort(columns.begin(), columns.end(), [](const Entry& a, const Entry& b) { return Before(b, a); });
    const int p = complex.p();
    struct Reduced {
        std::vector<Entry> column;  // the reduction matrix's column: a cochain of dim-simplices
        int pivot_coefficient;
    };
    std::vector<Reduced> reduced;
    std::unordered_map<Index, std::size_t> by_pivot;
    DimensionResult out;
    std::vector<int64_t> v, w;

    auto record = [&](const Entry& sigma, const Entry* tau, const std::vector<Entry>& column) {
        complex.Vertices(sigma.index, dim, v);
        if (tau != nullptr && tau->diameter == sigma.diameter) return;  // zero persistence
        RipsFeature f;
        f.pair.birth = sigma.diameter;
        f.birth_edge = complex.LongestEdge(v);
        if (tau != nullptr) {
            f.pair.death = tau->diameter;
            complex.Vertices(tau->index, dim + 1, w);
            f.death_edge = complex.LongestEdge(w);
        }
        if (cocycles) {
            for (const Entry& e : Compact(column, p)) {
                CocycleEntry c;
                complex.Vertices(e.index, dim, c.vertices);
                std::reverse(c.vertices.begin(), c.vertices.end());
                c.coefficient = e.coefficient;
                f.cocycle.push_back(std::move(c));
            }
        }
        out.features.push_back(std::move(f));
    };

    for (const Entry& sigma : columns) {
        complex.Vertices(sigma.index, dim, v);

        // Most columns are already reduced: their oldest cofacet is nobody's pivot yet, neither a
        // reduced column's nor that of an apparent pair (which is never reduced).
        Entry first, apparent;
        if (!complex.OldestCofacet(sigma, dim, v, first)) {
            record(sigma, nullptr, {sigma});
            continue;
        }
        if (by_pivot.find(first.index) == by_pivot.end() && !complex.ZeroApparentFacet(first, dim + 1, apparent)) {
            by_pivot.emplace(first.index, reduced.size());
            reduced.push_back({{sigma}, first.coefficient});
            out.pivots.insert(first.index);
            record(sigma, &first, {sigma});
            continue;
        }

        // Otherwise add earlier columns until the pivot is new, or the column vanishes.
        CoboundaryHeap heap;
        std::vector<Entry> column{sigma};
        complex.ForEachCofacet(sigma, dim, v, [&](const Entry& c) { heap.push(c); });
        Entry pivot;
        while (true) {
            if (!PopPivot(heap, p, pivot)) {
                record(sigma, nullptr, column);  // a cocycle that lives to the end
                break;
            }
            const auto it = by_pivot.find(pivot.index);
            Entry apparent;
            if (it == by_pivot.end() && complex.ZeroApparentFacet(pivot, dim + 1, apparent)) {
                // The pivot belongs to an apparent pair: add its facet's coboundary, which starts
                // at the pivot with coefficient apparent.coefficient.
                const int factor = ((p - pivot.coefficient) % p) * complex.inverse(apparent.coefficient) % p;
                apparent.coefficient = factor;
                column.push_back(apparent);
                complex.Vertices(apparent.index, dim, w);
                complex.ForEachCofacet(apparent, dim, w, [&](const Entry& c) { heap.push(c); });
                continue;
            }
            if (it == by_pivot.end()) {
                std::vector<Entry> compacted = Compact(column, p);
                by_pivot.emplace(pivot.index, reduced.size());
                reduced.push_back({compacted, pivot.coefficient});
                out.pivots.insert(pivot.index);
                record(sigma, &pivot, compacted);
                break;
            }
            const Reduced& other = reduced[it->second];
            const int factor = ((p - pivot.coefficient) % p) * complex.inverse(other.pivot_coefficient) % p;
            for (const Entry& e : other.column) {
                Entry scaled = e;
                scaled.coefficient = (e.coefficient * factor) % p;
                column.push_back(scaled);
                complex.Vertices(e.index, dim, w);
                complex.ForEachCofacet(scaled, dim, w, [&](const Entry& c) { heap.push(c); });
            }
        }
    }
    return out;
}

double EnclosingRadius(const DistanceMatrix& d) {
    double radius = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < d.size; ++i) {
        double farthest = 0;
        for (int64_t j = 0; j < d.size; ++j) farthest = std::max(farthest, static_cast<double>(d.at(i, j)));
        radius = std::min(radius, farthest);
    }
    return radius;
}

}  // namespace

RipsResult ComputeRipsPersistence(const DistanceMatrix& distances, const RipsOptions& options) {
    if (distances.size < 2) Fail("needs at least 2 points");
    if (options.max_dimension < 0 || options.max_dimension > 8) Fail("max_dimension must be in [0, 8]");
    if (options.coefficient >= (1 << 15) || !IsPrime(options.coefficient)) {
        Fail("coefficient must be a prime below 2^15, not " + std::to_string(options.coefficient));
    }
    RipsResult result;
    result.threshold = options.threshold < 0 ? EnclosingRadius(distances) : options.threshold;
    const RipsComplex complex(distances, result.threshold, options.max_dimension + 1, options.coefficient);
    const int64_t n = distances.size;

    // The edges within the threshold.
    std::vector<Entry> simplices;
    for (int64_t i = 1; i < n; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            const float d = distances.values[distances.index(i, j)];
            if (complex.within(d)) simplices.push_back({d, complex.binomial(i, 2) + j, 1});
        }
    }

    DimensionResult previous = ReduceDimensionZero(complex, simplices);
    result.features.push_back(std::move(previous.features));
    std::vector<int64_t> v;
    for (int dim = 1; dim <= options.max_dimension; ++dim) {
        // The columns: the dim-simplices that neither killed a class one dimension down (cleared)
        // nor belong to a zero-persistence apparent pair, which pairs without reduction.
        std::vector<Entry> columns;
        Entry apparent;
        auto consider = [&](const Entry& s, const std::vector<int64_t>& vertices) {
            if (previous.pivots.count(s.index) != 0) return;
            if (complex.ZeroApparentCofacet(s, dim, vertices, apparent)) return;
            if (dim >= 2 && complex.ZeroApparentFacet(s, dim, apparent)) return;
            columns.push_back(s);
        };
        if (dim == 1) {
            for (const Entry& s : simplices) {
                complex.Vertices(s.index, 1, v);
                consider(s, v);
            }
        } else {
            // The dim-simplices: each (dim - 1)-simplex with one vertex above its largest. They are
            // kept for the next dimension only if there is one.
            const bool keep = dim < options.max_dimension;
            std::vector<Entry> next;
            std::vector<int64_t> cofacet;
            for (const Entry& s : simplices) {
                complex.Vertices(s.index, dim - 1, v);
                for (int64_t top = v[0] + 1; top < n; ++top) {
                    float diameter = s.diameter;
                    for (int64_t u : v) diameter = std::max(diameter, complex.distance(top, u));
                    if (!complex.within(diameter)) continue;
                    const Entry c{diameter, complex.binomial(top, dim + 1) + s.index, 1};
                    cofacet.assign(1, top);
                    cofacet.insert(cofacet.end(), v.begin(), v.end());
                    consider(c, cofacet);
                    if (keep) next.push_back(c);
                }
            }
            simplices = std::move(next);
        }
        previous = ReduceDimension(complex, dim, std::move(columns), options.cocycles);
        result.features.push_back(std::move(previous.features));
    }

    for (std::size_t dim = 0; dim < result.features.size(); ++dim) {
        SortFeatures(result.features[dim]);
        PersistenceDiagram diagram;
        diagram.dimension = static_cast<int>(dim);
        diagram.error_bound = distances.error_bound;
        for (const RipsFeature& f : result.features[dim]) diagram.pairs.push_back(f.pair);
        result.diagrams.push_back(std::move(diagram));
    }
    return result;
}

}  // namespace pulsatrix
