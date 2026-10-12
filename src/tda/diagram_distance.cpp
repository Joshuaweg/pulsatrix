#include "pulsatrix/tda/diagram_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// A diagram split into its finite points and its essential features' births, sorted.
struct Split {
    std::vector<PersistencePair> finite;
    std::vector<double> essential;
};

Split SplitDiagram(const PersistenceDiagram& d) {
    Split s;
    for (const PersistencePair& p : d.pairs) {
        if (p.essential()) {
            s.essential.push_back(p.birth);
        } else {
            s.finite.push_back(p);
        }
    }
    std::sort(s.essential.begin(), s.essential.end());
    return s;
}

void CheckDimensions(const PersistenceDiagram& a, const PersistenceDiagram& b, const char* function) {
    if (a.dimension != b.dimension) {
        throw std::invalid_argument(std::string(function) + ": the diagrams are of dimensions " +
                                    std::to_string(a.dimension) + " and " + std::to_string(b.dimension));
    }
}

double LInf(const PersistencePair& p, const PersistencePair& q) {
    return std::max(std::abs(p.birth - q.birth), std::abs(p.death - q.death));
}

double ToDiagonal(const PersistencePair& p) { return (p.death - p.birth) / 2; }

// Whether a bipartite graph with n left and n right vertices has a perfect matching (Hopcroft-Karp).
bool HasPerfectMatching(const std::vector<std::vector<int>>& adjacency, int n) {
    std::vector<int> match_left(static_cast<std::size_t>(n), -1), match_right(static_cast<std::size_t>(n), -1);
    std::vector<int> level(static_cast<std::size_t>(n));
    int matched = 0;
    while (true) {
        // Layer the free left vertices' alternating paths.
        std::queue<int> queue;
        for (int u = 0; u < n; ++u) {
            level[static_cast<std::size_t>(u)] = match_left[static_cast<std::size_t>(u)] < 0 ? 0 : -1;
            if (level[static_cast<std::size_t>(u)] == 0) queue.push(u);
        }
        bool reachable = false;
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop();
            for (int v : adjacency[static_cast<std::size_t>(u)]) {
                const int w = match_right[static_cast<std::size_t>(v)];
                if (w < 0) {
                    reachable = true;
                } else if (level[static_cast<std::size_t>(w)] < 0) {
                    level[static_cast<std::size_t>(w)] = level[static_cast<std::size_t>(u)] + 1;
                    queue.push(w);
                }
            }
        }
        if (!reachable) break;
        // Augment along vertex-disjoint shortest paths, depth first without recursion.
        std::vector<std::size_t> next(static_cast<std::size_t>(n), 0);
        for (int start = 0; start < n; ++start) {
            if (match_left[static_cast<std::size_t>(start)] >= 0) continue;
            std::vector<int> path{start};
            while (!path.empty()) {
                const int u = path.back();
                const auto& edges = adjacency[static_cast<std::size_t>(u)];
                std::size_t& k = next[static_cast<std::size_t>(u)];
                bool advanced = false;
                for (; k < edges.size(); ++k) {
                    const int v = edges[k];
                    const int w = match_right[static_cast<std::size_t>(v)];
                    if (w < 0) {
                        // Free right vertex: flip the path.
                        int right = v;
                        for (std::size_t i = path.size(); i-- > 0;) {
                            const int left = path[i];
                            const int previous = match_left[static_cast<std::size_t>(left)];
                            match_left[static_cast<std::size_t>(left)] = right;
                            match_right[static_cast<std::size_t>(right)] = left;
                            right = previous;
                        }
                        ++matched;
                        path.clear();
                        advanced = true;
                        break;
                    }
                    if (level[static_cast<std::size_t>(w)] == level[static_cast<std::size_t>(u)] + 1) {
                        ++k;
                        path.push_back(w);
                        advanced = true;
                        break;
                    }
                }
                if (!advanced) {
                    level[static_cast<std::size_t>(u)] = -1;  // dead end this phase
                    path.pop_back();
                }
            }
        }
    }
    return matched == n;
}

// The bottleneck distance between two sets of finite points.
double FiniteBottleneck(const std::vector<PersistencePair>& a, const std::vector<PersistencePair>& b) {
    const int n1 = static_cast<int>(a.size()), n2 = static_cast<int>(b.size());
    if (n1 + n2 == 0) return 0;
    std::vector<double> candidates{0.0};
    for (const PersistencePair& p : a) candidates.push_back(ToDiagonal(p));
    for (const PersistencePair& q : b) candidates.push_back(ToDiagonal(q));
    for (const PersistencePair& p : a) {
        for (const PersistencePair& q : b) candidates.push_back(LInf(p, q));
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    // Left: a's points, then b's points' diagonal copies. Right: b's points, then a's copies.
    const int n = n1 + n2;
    auto feasible = [&](double delta) {
        std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(n));
        for (int i = 0; i < n1; ++i) {
            auto& edges = adjacency[static_cast<std::size_t>(i)];
            for (int j = 0; j < n2; ++j) {
                if (LInf(a[static_cast<std::size_t>(i)], b[static_cast<std::size_t>(j)]) <= delta) edges.push_back(j);
            }
            if (ToDiagonal(a[static_cast<std::size_t>(i)]) <= delta) edges.push_back(n2 + i);
        }
        for (int j = 0; j < n2; ++j) {
            auto& edges = adjacency[static_cast<std::size_t>(n1 + j)];
            if (ToDiagonal(b[static_cast<std::size_t>(j)]) <= delta) edges.push_back(j);
            for (int i = 0; i < n1; ++i) edges.push_back(n2 + i);  // diagonal to diagonal is free
        }
        return HasPerfectMatching(adjacency, n);
    };
    // The largest candidate is feasible: every point goes to the diagonal.
    std::size_t lo = 0, hi = candidates.size() - 1;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (feasible(candidates[mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return candidates[lo];
}

// The minimum cost of a perfect assignment (Hungarian algorithm with potentials, O(n^3)).
double MinimumAssignment(const std::vector<std::vector<double>>& cost) {
    const std::size_t n = cost.size();
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0), shortest(n + 1);
    std::vector<std::size_t> row_of(n + 1, 0), way(n + 1, 0);
    std::vector<char> used(n + 1);
    for (std::size_t i = 1; i <= n; ++i) {
        row_of[0] = i;
        std::size_t j0 = 0;
        std::fill(shortest.begin(), shortest.end(), kInfinity);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            const std::size_t i0 = row_of[j0];
            double delta = kInfinity;
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double reduced = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (reduced < shortest[j]) {
                    shortest[j] = reduced;
                    way[j] = j0;
                }
                if (shortest[j] < delta) {
                    delta = shortest[j];
                    j1 = j;
                }
            }
            for (std::size_t j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    shortest[j] -= delta;
                }
            }
            j0 = j1;
        } while (row_of[j0] != 0);
        do {
            const std::size_t j1 = way[j0];
            row_of[j0] = row_of[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    double total = 0;
    for (std::size_t j = 1; j <= n; ++j) total += cost[row_of[j] - 1][j - 1];
    return total;
}

}  // namespace

Certified<double> BottleneckDistance(const PersistenceDiagram& a, const PersistenceDiagram& b) {
    CheckDimensions(a, b, "BottleneckDistance");
    const Split sa = SplitDiagram(a), sb = SplitDiagram(b);
    Certified<double> out;
    out.terms = {{"first diagram: its error bound", a.error_bound}, {"second diagram: its error bound", b.error_bound}};
    if (sa.essential.size() != sb.essential.size()) {
        out.value = kInfinity;
        return out;
    }
    double essential = 0;
    for (std::size_t k = 0; k < sa.essential.size(); ++k) {
        essential = std::max(essential, std::abs(sa.essential[k] - sb.essential[k]));
    }
    out.value = std::max(essential, FiniteBottleneck(sa.finite, sb.finite));
    return out;
}

double WassersteinDistance(const PersistenceDiagram& a, const PersistenceDiagram& b, double order) {
    CheckDimensions(a, b, "WassersteinDistance");
    if (std::isnan(order) || order < 1) {
        throw std::invalid_argument("WassersteinDistance: order must be at least 1, not " + std::to_string(order));
    }
    if (std::isinf(order)) return BottleneckDistance(a, b).value;
    const Split sa = SplitDiagram(a), sb = SplitDiagram(b);
    if (sa.essential.size() != sb.essential.size()) return kInfinity;
    double total = 0;
    for (std::size_t k = 0; k < sa.essential.size(); ++k) total += std::pow(std::abs(sa.essential[k] - sb.essential[k]), order);

    // Rows: a's points, then b's points' diagonal copies. Columns: b's points, then a's copies. A
    // point may go to any diagonal copy at its distance to the diagonal; copies match each other
    // for free.
    const std::size_t n1 = sa.finite.size(), n2 = sb.finite.size(), n = n1 + n2;
    if (n > 0) {
        std::vector<std::vector<double>> cost(n, std::vector<double>(n, 0.0));
        for (std::size_t i = 0; i < n1; ++i) {
            const double to_diagonal = std::pow(ToDiagonal(sa.finite[i]), order);
            for (std::size_t j = 0; j < n2; ++j) cost[i][j] = std::pow(LInf(sa.finite[i], sb.finite[j]), order);
            for (std::size_t j = n2; j < n; ++j) cost[i][j] = to_diagonal;
        }
        for (std::size_t j = 0; j < n2; ++j) {
            const double to_diagonal = std::pow(ToDiagonal(sb.finite[j]), order);
            for (std::size_t i = n1; i < n; ++i) cost[i][j] = to_diagonal;
        }
        total += MinimumAssignment(cost);
    }
    return std::pow(total, 1.0 / order);
}

}  // namespace pulsatrix
