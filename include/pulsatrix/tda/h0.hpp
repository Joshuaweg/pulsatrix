/** @file h0.hpp
 *  @brief H0 persistence, the clusters of a point cloud across scales, from its minimum spanning
 *         tree.
 *  @ingroup tda
 *
 * In the Vietoris-Rips filtration every point is a component at scale 0, and two components merge
 * when the shortest edge between them appears. That is single-linkage clustering, and the merge
 * scales are the edge lengths of a minimum spanning tree (Carlsson and Memoli, JMLR 2010, show
 * single linkage is the one stable clustering of this kind). So H0 needs no boundary matrices:
 * Prim's algorithm on the distance matrix gives it exactly, in O(n^2) time and O(n) extra memory.
 *
 * H0 only sees the tree. Research notes for this epic warn that many recent "LLM topology" results
 * use H0 alone; check such a result against plain nearest-neighbor statistics
 * (nearest_neighbors.hpp), which see much of the same thing.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"

namespace pulsatrix {

/** @brief One edge of a minimum spanning tree. */
struct MstEdge {
    /** @brief Its endpoints, a < b. */
    int64_t a = 0;
    int64_t b = 0;
    /** @brief Its length: the scale at which it merges two components. */
    float length = 0;
};

/** @brief ComputeH0()'s result. */
struct H0Result {
    /**
     * @brief The H0 diagram: one pair (0, length) for each tree edge of positive length, and one
     *        essential pair (0, infinity) for the component that never dies. Its error_bound is
     *        the distances' error bound.
     */
    PersistenceDiagram diagram;
    /** @brief The n - 1 edges of a minimum spanning tree, by increasing length (ties by a, then
     *         b). Edge k is the k-th merge of single-linkage clustering. */
    std::vector<MstEdge> edges;
};

/**
 * @brief H0 persistence of the Vietoris-Rips filtration on the given distances.
 * @note Prim's algorithm over the dense matrix: O(n^2) reads, the same order as reading the
 *       distances once, and far cheaper than computing them (O(n^2 d)). Equal lengths are broken
 *       by the lower point index, so the result is the same on every platform.
 * @throws std::invalid_argument if distances has fewer than 2 points.
 */
[[nodiscard]] H0Result ComputeH0(const DistanceMatrix& distances);

/**
 * @brief The clusters of single linkage cut at a scale: points joined by tree edges no longer
 *        than threshold share a label.
 * @param points The number of points the edges span.
 * @return One label per point, numbered 0, 1, ... in order of each cluster's lowest point.
 * @throws std::invalid_argument if an edge has an endpoint outside [0, points).
 */
[[nodiscard]] std::vector<int64_t> SingleLinkageClusters(const std::vector<MstEdge>& edges, int64_t points,
                                                         double threshold);

}  // namespace pulsatrix
