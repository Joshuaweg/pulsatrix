/** @file distance_matrix.hpp
 *  @brief Pairwise Euclidean distances of a point cloud, computed with one GEMM on any backend,
 *         with a certified bound on their floating-point error.
 *  @ingroup tda
 *
 * Distances are the only stage of persistent homology that touches the points' dimension, so they
 * run as a matrix product on the device: |x - y|^2 = |x|^2 + |y|^2 - 2 x.y. In float32 that formula
 * cancels badly for points close together (the research notes' warning for TDA-2), so the result
 * carries an error bound instead of hiding it. Stability (Cohen-Steiner, Edelsbrunner and Harer,
 * 2007) turns the bound into one on everything computed later: a persistence diagram built from
 * these distances is within error_bound, in bottleneck distance, of the exact one.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pulsatrix/tda/point_cloud.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How PairwiseDistances() computes the distances. */
enum class DistanceMethod {
    /** @brief |x|^2 + |y|^2 - 2 x.y, with the dot products from the backend's float32 GEMM and the
     *         rest in double on the host. O(n^2 d) on the device. */
    Gemm,
    /** @brief The sum of squared differences, in double on the host. Slower, with an error bound
     *         near double precision: a reference, or for clouds whose points nearly coincide. */
    Exact,
};

/** @brief PairwiseDistances()'s settings. */
struct DistanceOptions {
    /** @brief How to compute the distances. */
    DistanceMethod method = DistanceMethod::Gemm;
    /** @brief Subtract the mean point first. It doesn't change Euclidean distances, but it
     *         shrinks the norms, and with them the Gemm method's error, when the points share a
     *         large offset (an anisotropic layer). */
    bool center = false;
    /** @brief Scale every point to unit length (after centering, if centering). Distances are then
     *         sqrt(2 - 2 cos), a metric that orders pairs as cosine similarity does. */
    bool normalize = false;
    /** @brief Rows per GEMM block; 0 picks one so a block's output holds at most 2^24 values. The
     *         device holds the points and one block at a time. */
    int64_t block_rows = 0;
};

/**
 * @brief The distances between every pair of n points, condensed.
 *
 * Pairs are stored as scipy's `pdist` stores them, (0, 1), (0, 2), ..., (0, n-1), (1, 2), ..., so
 * a matrix can go to or come from Python unchanged, as RSA's Rdm does.
 */
struct DistanceMatrix {
    /** @brief The number of points, n. */
    int64_t size = 0;
    /** @brief The n (n - 1) / 2 distances, in pdist order, as float32. */
    std::vector<float> values;
    /**
     * @brief A bound on the error of every stored distance: |values[k] - exact| <= error_bound for
     *        every pair. "Exact" means the distance between the points as given, or, with
     *        center or normalize, as centered and normalized in exact arithmetic.
     * @note It is rigorous, not an estimate, under the standard model of floating-point
     *       arithmetic with float32 accumulation in the GEMM (no TF32 or other reduced-precision
     *       modes). It covers the dot products, the double-precision arithmetic, the rounding of
     *       centered or normalized points to float32, and the final rounding to float32.
     */
    double error_bound = 0;
    /** @brief The largest stored distance. */
    float max_distance = 0;

    /** @brief The index in values of the pair (i, j), i != j, in either order. */
    [[nodiscard]] std::size_t index(int64_t i, int64_t j) const;
    /** @brief The distance between points i and j; 0 when i == j. */
    [[nodiscard]] float at(int64_t i, int64_t j) const;
};

/**
 * @brief Euclidean distances between every pair of rows.
 * @param points shape (n, ...): a row per point, dimensions after the first flattened. With
 *        DistanceMethod::Gemm the products run on points' own backend.
 * @throws std::invalid_argument if points has rank below 2, fewer than 2 rows, a NaN or infinity,
 *         block_rows below 0, or (with normalize) a row of zeros after centering.
 */
[[nodiscard]] DistanceMatrix PairwiseDistances(const Tensor& points, const DistanceOptions& options = {});

/**
 * @brief Euclidean distances between every pair of a cloud's points, on the given backend.
 * @throws std::invalid_argument as above.
 */
[[nodiscard]] DistanceMatrix PairwiseDistances(const PointCloud& cloud, DeviceBackend* backend,
                                               const DistanceOptions& options = {});

}  // namespace pulsatrix
