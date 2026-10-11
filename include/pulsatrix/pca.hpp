/** @file pca.hpp
 *  @brief Principal component analysis of a layer's representations (INT-5): 2D and 3D views,
 *         directions to steer along, and fewer dimensions for what comes after.
 *  @ingroup mech_interp
 *
 * FitPca() finds the directions along which a set of inputs' representations vary most.
 * PcaProject() places inputs on them, for a scatter plot (viz/projection.hpp) or as input to
 * methods that want fewer dimensions. PcaComponent() gives a direction as a steering vector.
 *
 * It matches scikit-learn's `PCA` (the full solver): the data are centered, variances divide by
 * n - 1, and each component's sign makes its largest-magnitude entry positive, so the same data
 * always give the same picture. Everything is in double precision on the host, through the
 * covariance (width x width) or the Gram matrix (n x n), whichever is smaller.
 *
 * Two cautions. Components whose variances are close can swap or rotate between two samples of
 * the same data, so a 2D picture isn't always stable: PcaSplitHalfOverlap() checks. And the
 * largest directions of variance aren't necessarily the ones the model uses: a direction found
 * here is a hypothesis to test, for example by steering along it.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief A fitted PCA. */
struct Pca {
    /** @brief The width of the rows it was fitted on. */
    int64_t width = 0;
    /** @brief How many rows it was fitted on. */
    int64_t samples = 0;
    /** @brief The rows' mean, subtracted before projecting. All zeros when fitted uncentered. */
    std::vector<double> mean;
    /** @brief Row-major `(components, width)`: unit directions, most variance first. */
    std::vector<double> components;
    /** @brief Each component's variance: the variance of the projections, dividing by n - 1. */
    std::vector<double> explained_variance;
    /** @brief Each component's share of the total variance. */
    std::vector<double> explained_variance_ratio;
    /** @brief The variance over every direction, the sum of the per-column variances. */
    double total_variance = 0;

    [[nodiscard]] int64_t num_components() const { return static_cast<int64_t>(explained_variance.size()); }
};

/**
 * @brief Fits the @p components directions of most variance.
 * @param x shape (n, ...): a row per input. Dimensions after the first are flattened.
 * @param center Subtract the mean row first, as PCA does. Without it the first component tends to
 *        point at the mean, and the "variances" are mean squares.
 * @throws std::invalid_argument if x has rank below 2, fewer than 2 rows, or a NaN or infinity;
 *         if components is below 1 or above min(n, width); or if a requested component has no
 *         variance, so has no direction.
 */
[[nodiscard]] Pca FitPca(const Tensor& x, int64_t components, bool center = true);

/**
 * @brief Each row's coordinates on the components: `(x - mean) components^T`, shape
 *        (n, components), on x's backend.
 * @param whiten Divide each coordinate by its component's standard deviation, so every axis has
 *        variance 1 on the fitted data.
 * @throws std::invalid_argument if x's rows aren't pca.width wide, or have a NaN or infinity.
 */
[[nodiscard]] Tensor PcaProject(const Pca& pca, const Tensor& x, bool whiten = false);

/**
 * @brief The rows that coordinates map back to: `mean + scores components`, shape (n, width).
 *        Projecting and reconstructing keeps only what the components hold.
 * @throws std::invalid_argument if scores isn't (n, pca.num_components()), or has a NaN or
 *         infinity.
 */
[[nodiscard]] Tensor PcaReconstruct(const Pca& pca, const Tensor& scores);

/**
 * @brief Component @p index as a unit vector of floats, for SteeringHook(). One standard deviation
 *        along it is `sqrt(explained_variance[index])`, a natural unit for the coefficient.
 * @throws std::invalid_argument if index is out of range.
 */
[[nodiscard]] std::vector<float> PcaComponent(const Pca& pca, int64_t index);

/**
 * @brief How much two PCAs' first @p k components span the same space: the mean squared cosine of
 *        the principal angles between them, `|A_k B_k^T|_F^2 / k`. 1 when the spans are the same
 *        (whatever the order or rotation inside them), 0 when they are orthogonal, and about
 *        k / width for unrelated random subspaces.
 * @throws std::invalid_argument if the widths differ or k is not between 1 and both PCAs'
 *         num_components().
 */
[[nodiscard]] double SubspaceOverlap(const Pca& a, const Pca& b, int64_t k);

/**
 * @brief Fits @p k components on each half of the rows, split at random, and returns their
 *        SubspaceOverlap(). Near 1 means the k-dimensional picture is a property of the data; well
 *        below 1 means it depends on which inputs happened to be drawn.
 * @throws std::invalid_argument if x has fewer than 4 rows, or as FitPca() does for each half.
 */
[[nodiscard]] double PcaSplitHalfOverlap(const Tensor& x, int64_t k, uint64_t seed = 0);

}  // namespace pulsatrix
