/** @file nearest_neighbors.hpp
 *  @brief Nearest neighbors over embeddings or activations, mutual nearest neighbors, and
 *         mutual-kNN alignment between two representations.
 *  @ingroup mech_interp
 *
 * The questions these answer: which inputs does a layer treat as alike, which items of two sets
 * pair up, and do two models agree about which inputs are alike (Huh et al., "The Platonic
 * Representation Hypothesis", ICML 2024)? Everything is exact (no approximate index) and computed
 * in double precision on the host, at about n × m × width multiply-adds for n queries and m
 * points.
 *
 * In high dimensions a few points can turn up in almost everyone's neighbor list ("hubs"), which
 * makes neighbor-based scores mean less. MeasureHubness() says how much that is happening.
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How nearness is measured. */
enum class NeighborMetric {
    /** @brief Cosine similarity: the angle between vectors, ignoring their lengths. Higher is
     *         nearer. A zero vector has similarity 0 with everything. */
    Cosine,
    /** @brief Euclidean distance. Lower is nearer. */
    Euclidean,
};

/** @brief Settings for the neighbor searches. */
struct NeighborOptions {
    /** @brief How many neighbors to find for each query. */
    int64_t k = 10;
    NeighborMetric metric = NeighborMetric::Cosine;
    /**
     * @brief Subtract the points' mean from every vector first. Activations often share a large
     *        common component, which makes every pair's cosine similarity high; centering removes
     *        it. It doesn't change Euclidean distances.
     */
    bool center = false;
};

/** @brief Each query's k nearest points, nearest first. */
struct Neighbors {
    int64_t rows = 0;  ///< the number of queries
    int64_t k = 0;
    NeighborMetric metric = NeighborMetric::Cosine;
    /** @brief Row-major rows × k: `indices[q * k + r]` is query q's r-th nearest point. */
    std::vector<int64_t> indices;
    /** @brief The same layout: cosine similarity, or Euclidean distance. */
    std::vector<double> scores;

    [[nodiscard]] int64_t index(int64_t row, int64_t rank) const;
    [[nodiscard]] double score(int64_t row, int64_t rank) const;
};

/**
 * @brief Each point's k nearest other points (a point is never its own neighbor).
 * @param points shape (n, ...): a row per point. Dimensions after the first are flattened.
 * @note Ties keep the lower index first, as top_k() does, so results are the same everywhere.
 * @throws std::invalid_argument if points has rank below 2 or a NaN or infinity, or k is not
 *         between 1 and n - 1.
 */
[[nodiscard]] Neighbors NearestNeighbors(const Tensor& points, const NeighborOptions& options = {});

/**
 * @brief Each query's k nearest points from another set of the same width, for example new inputs
 *        against a reference set. With `center`, both are centered on the points' mean.
 * @throws std::invalid_argument as the one-set version does, if the widths differ, or if k is
 *         not between 1 and the number of points.
 */
[[nodiscard]] Neighbors NearestNeighbors(const Tensor& queries, const Tensor& points,
                                         const NeighborOptions& options = {});

/**
 * @brief The pairs (i, j), a[i] from @p a and b[j] from @p b, where b[j] is among a[i]'s k nearest
 *        in b and a[i] is among b[j]'s k nearest in a. Used to match items across two sets in the
 *        same space, such as the same concepts in two batches.
 * @return The pairs sorted by i, then j. With `center`, both sets are centered on the mean of
 *         the two together.
 * @throws std::invalid_argument as NearestNeighbors() does, or if the widths differ.
 */
[[nodiscard]] std::vector<std::pair<int64_t, int64_t>> MutualNearestNeighbors(const Tensor& a, const Tensor& b,
                                                                              const NeighborOptions& options = {});

/**
 * @brief Mutual-kNN alignment (Huh et al. 2024): how far two representations of the same inputs
 *        agree on which inputs are alike. For each input, the fraction of its k nearest neighbors
 *        under @p a that are also among its k nearest under @p b, averaged over the inputs.
 * @param a,b the same n inputs in the same order, of any widths.
 * @return Between 0 and 1. Two unrelated representations score about k / (n - 1) by chance.
 * @note With the default options (cosine, no centering) this is Huh et al.'s `mutual_knn`.
 * @throws std::invalid_argument as NearestNeighbors() does, or if the row counts differ.
 */
[[nodiscard]] double MutualKnnAlignment(const Tensor& a, const Tensor& b, const NeighborOptions& options = {});

/** @brief How unevenly points appear in each other's neighbor lists. */
struct Hubness {
    /** @brief How many neighbor lists each point appears in (its "k-occurrence"). The mean is k. */
    std::vector<int64_t> occurrences;
    /** @brief The skewness of occurrences. Near 0 when points appear about equally often; well
     *         above 1 means a few hubs dominate (Radovanović et al., JMLR 2010). */
    double skewness = 0;
    /** @brief The most lists any one point is in, and its fraction of all queries. */
    int64_t max_occurrence = 0;
    double max_fraction = 0;
    /** @brief The fraction of points in nobody's list. */
    double never_a_neighbor = 0;
};

/**
 * @brief Measures hubness in @p neighbors, which name points 0 to @p num_points - 1.
 * @throws std::invalid_argument if num_points < 1 or an index is outside it.
 */
[[nodiscard]] Hubness MeasureHubness(const Neighbors& neighbors, int64_t num_points);

}  // namespace pulsatrix
