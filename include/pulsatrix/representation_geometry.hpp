/** @file representation_geometry.hpp
 *  @brief One number per layer for the shape of its representations: anisotropy, effective rank
 *         and TwoNN intrinsic dimension.
 *  @ingroup mech_interp
 *
 * Language models' representations are anisotropic (Ethayarajh, EMNLP 2019): they share a few
 * large common directions, so the cosine similarity of two unrelated inputs is high, often 0.5 or
 * more, and raw cosine similarity says little about meaning. Anisotropy() measures that, and
 * shows how much centering removes. Effective rank says how many directions a layer spreads its
 * inputs along, and TwoNN how many dimensions the inputs actually vary in locally, which can be far
 * fewer than the width.
 *
 * Everything is computed in double precision on the host.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The mean cosine similarity over all pairs of different rows: about 0 for directions
 *        spread evenly, near 1 when every row points the same way.
 * @param x shape (n, ...): a row per input. Dimensions after the first are flattened.
 * @param center Subtract the mean row first. Centering removes the common direction, so the
 *        result drops to about -1/(n - 1) when nothing but that direction was shared.
 * @note Exact over all n(n - 1) pairs, at the cost of one pass: the sum of unit vectors u gives
 *       `(|sum u|^2 - n) / (n (n - 1))`.
 * @throws std::invalid_argument if x has rank below 2, fewer than 2 rows, a NaN or infinity, or a
 *         row of zeros (after centering, if centering).
 */
[[nodiscard]] double Anisotropy(const Tensor& x, bool center = false);

/** @brief MeasureEffectiveRank()'s result. */
struct EffectiveRank {
    /** @brief Roy and Vetterli's effective rank: exp of the entropy of the singular values divided
     *         by their sum. Between 1 and min(n, width). */
    double effective_rank = 0;
    /** @brief The participation ratio of the variances, (sum s^2)^2 / sum s^4: a coarser count
     *         that the largest directions dominate more. */
    double participation_ratio = 0;
    /** @brief The singular values, largest first: min(n, width) of them. */
    std::vector<double> singular_values;
};

/**
 * @brief How many directions the rows spread along, from the singular values of x.
 * @param center Subtract the mean row first (the default). Without it, a large common direction,
 *        the anisotropy, takes up a share of the spectrum and lowers the effective rank.
 * @note The singular values come from the smaller Gram matrix, x x^T or x^T x, in double precision.
 * @throws std::invalid_argument if x has rank below 2, fewer than 2 rows, a NaN or infinity, or is
 *         all zeros (after centering, if centering).
 */
[[nodiscard]] EffectiveRank MeasureEffectiveRank(const Tensor& x, bool center = true);

/** @brief TwoNNDimension()'s result. */
struct TwoNNResult {
    /** @brief Facco et al.'s estimate: the slope through the origin of -log(1 - F(mu)) against
     *         log(mu), over the points left after discarding the largest ratios. scikit-dimension's
     *         TwoNN and DADApy's compute_id_2NN compute the same. */
    double dimension = 0;
    /** @brief The maximum-likelihood estimate over all points, (N - 1) / sum log(mu), as DADApy's
     *         `algorithm="ml"` gives it. */
    double mle_dimension = 0;
    /** @brief Distinct points used. */
    int64_t points = 0;
    /** @brief Rows dropped because they repeat an earlier row exactly. */
    int64_t duplicates = 0;
};

/**
 * @brief TwoNN intrinsic dimension (Facco, d'Errico, Rodriguez and Laio, Scientific Reports 2017):
 *        from each point's distances to its two nearest neighbors, mu = r2 / r1, whose distribution
 *        depends only on the dimension the points vary in locally.
 * @param x shape (n, ...): a row per point, under Euclidean distance.
 * @param discard_fraction The share of the largest mu values the fit leaves out, 0.1 in the paper.
 * @note Repeated rows are removed first, as DADApy's remove_identical_points() does: a zero
 *       distance makes mu infinite. The search is exact, at n^2 x width multiply-adds.
 * @note The estimate is biased low when the dimension is high for the number of points (tens of
 *       dimensions need thousands of points), and it measures the scale of the nearest neighbors
 *       only: noise on a low-dimensional manifold raises it.
 * @throws std::invalid_argument if x has rank below 2, a NaN or infinity, fewer than 3 distinct
 *         rows, or discard_fraction isn't in (0, 1), or leaves fewer than 2 points to fit.
 */
[[nodiscard]] TwoNNResult TwoNNDimension(const Tensor& x, double discard_fraction = 0.1);

}  // namespace pulsatrix
