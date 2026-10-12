#include "pulsatrix/tda/h0.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pulsatrix {

H0Result ComputeH0(const DistanceMatrix& distances) {
    const int64_t n = distances.size;
    if (n < 2) throw std::invalid_argument("ComputeH0: needs at least 2 points");

    // Prim's algorithm: grow the tree from point 0, each step adding the closest outside point.
    const std::size_t count = static_cast<std::size_t>(n);
    std::vector<char> in_tree(count, 0);
    std::vector<float> best(count, std::numeric_limits<float>::infinity());
    std::vector<int64_t> parent(count, -1);
    H0Result result;
    result.edges.reserve(count - 1);
    int64_t added = 0;
    for (int64_t step = 0; step < n; ++step) {
        // The closest point not yet in the tree; the first one wins a tie.
        int64_t next = -1;
        for (int64_t j = 0; j < n; ++j) {
            if (!in_tree[static_cast<std::size_t>(j)] &&
                (next < 0 || best[static_cast<std::size_t>(j)] < best[static_cast<std::size_t>(next)])) {
                next = j;
            }
        }
        in_tree[static_cast<std::size_t>(next)] = 1;
        if (step > 0) {
            const int64_t p = parent[static_cast<std::size_t>(next)];
            result.edges.push_back({std::min(p, next), std::max(p, next), best[static_cast<std::size_t>(next)]});
        }
        added = next;
        for (int64_t j = 0; j < n; ++j) {
            if (in_tree[static_cast<std::size_t>(j)]) continue;
            const float d = distances.values[distances.index(added, j)];
            // Strictly shorter, so a tie keeps the earlier (lower-index) parent.
            if (d < best[static_cast<std::size_t>(j)] ||
                (d == best[static_cast<std::size_t>(j)] && added < parent[static_cast<std::size_t>(j)])) {
                best[static_cast<std::size_t>(j)] = d;
                parent[static_cast<std::size_t>(j)] = added;
            }
        }
    }
    std::sort(result.edges.begin(), result.edges.end(), [](const MstEdge& x, const MstEdge& y) {
        return std::tie(x.length, x.a, x.b) < std::tie(y.length, y.a, y.b);
    });

    result.diagram.dimension = 0;
    result.diagram.error_bound = distances.error_bound;
    for (const MstEdge& e : result.edges) {
        if (e.length > 0) result.diagram.pairs.push_back({0.0, static_cast<double>(e.length)});
    }
    result.diagram.pairs.push_back({0.0, std::numeric_limits<double>::infinity()});
    return result;
}

std::vector<int64_t> SingleLinkageClusters(const std::vector<MstEdge>& edges, int64_t points, double threshold) {
    if (points < 0) throw std::invalid_argument("SingleLinkageClusters: points must be non-negative");
    std::vector<int64_t> root(static_cast<std::size_t>(points));
    std::iota(root.begin(), root.end(), int64_t{0});
    auto find = [&](int64_t x) {
        while (root[static_cast<std::size_t>(x)] != x) {
            root[static_cast<std::size_t>(x)] = root[static_cast<std::size_t>(root[static_cast<std::size_t>(x)])];
            x = root[static_cast<std::size_t>(x)];
        }
        return x;
    };
    for (const MstEdge& e : edges) {
        if (e.a < 0 || e.b < 0 || e.a >= points || e.b >= points) {
            throw std::invalid_argument("SingleLinkageClusters: edge (" + std::to_string(e.a) + ", " +
                                        std::to_string(e.b) + ") has a point outside [0, " +
                                        std::to_string(points) + ")");
        }
        if (static_cast<double>(e.length) > threshold) continue;
        const int64_t ra = find(e.a), rb = find(e.b);
        if (ra != rb) root[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);  // the lower point is the root
    }
    std::vector<int64_t> label(static_cast<std::size_t>(points), -1);
    std::vector<int64_t> label_of_root(static_cast<std::size_t>(points), -1);
    int64_t next = 0;
    for (int64_t i = 0; i < points; ++i) {
        const int64_t r = find(i);
        if (label_of_root[static_cast<std::size_t>(r)] < 0) label_of_root[static_cast<std::size_t>(r)] = next++;
        label[static_cast<std::size_t>(i)] = label_of_root[static_cast<std::size_t>(r)];
    }
    return label;
}

}  // namespace pulsatrix
