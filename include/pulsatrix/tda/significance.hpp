/** @file significance.hpp
 *  @brief Is a loop real? Significance tests for persistence diagrams: the universal null
 *         distribution, bootstrap confidence bands, and baseline comparisons against Gaussian
 *         clouds or a random network.
 *  @ingroup tda
 *
 * A persistence diagram of noise is full of short-lived features, and a few of them always live
 * longer than the rest. Every reported feature needs a test that says how surprising it is:
 *
 * - UniversalNullTest(): Bobrowski and Skraba (Sci. Rep. 2023) found that, suitably normalized,
 *   the death/birth ratios of random point clouds follow one distribution whatever the cloud came
 *   from. It gives every feature a p-value from one diagram, with no resampling. It is a conjecture
 *   backed by experiments, and assumes most of the diagram is noise.
 * - BootstrapBand(): Fasy et al. (Ann. Statist. 2014). Resamples the cloud and measures how far
 *   the diagram moves; features farther from the diagonal than that are significant.
 * - Baselines: the same statistic on clouds with nothing to find, a Gaussian with the same mean and
 *   covariance (GaussianBaseline()) or a randomly initialized network's activations
 *   (NullModelBaseline), and a rank p-value (BaselinePValue()).
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One feature's result in UniversalNullTest(). */
struct FeatureSignificance {
    /** @brief The feature's index in the diagram's pairs. */
    int64_t index = 0;
    /** @brief death / birth, the multiplicative persistence. */
    double ratio = 0;
    /** @brief The normalized statistic: A log log(ratio) + B. */
    double ell = 0;
    /** @brief exp(-e^ell): the chance of an ell this large under the left-skewed Gumbel null. */
    double p_value = 1;
    /** @brief p_value < alpha / (number of features tested), Bonferroni's correction. */
    bool significant = false;
};

/** @brief UniversalNullTest()'s result. */
struct UniversalNullResult {
    /** @brief One entry per tested feature (finite, birth above 0), in diagram order. */
    std::vector<FeatureSignificance> features;
    /** @brief The mean of log log(ratio) over the tested features, L-bar. */
    double mean_log_log = 0;
    /** @brief alpha / features.size(): the corrected threshold. */
    double threshold = 0;
};

/**
 * @brief Bobrowski and Skraba's universal null test for every feature of a Vietoris-Rips diagram
 *        in dimension 1 or above: ell = log log(death / birth) - L-bar - (Euler-Mascheroni's
 *        constant), compared with the left-skewed Gumbel, whose survival function is exp(-e^ell).
 * @param alpha The family-wise significance level, split over the features by Bonferroni.
 * @note A = 1 and B = -gamma - L-bar are the paper's values for Vietoris-Rips filtrations. The test
 *       assumes the diagram is mostly noise: with only a handful of features, L-bar is set by the
 *       signal itself and the test loses its power. Use diagrams of hundreds of points or more.
 *       Point sets spaced too evenly (lattices, repulsive processes) don't follow the null.
 * @throws std::invalid_argument if the diagram is H0 (all births are 0), has fewer than 2 finite
 *         features with birth above 0, or alpha is outside (0, 1).
 */
[[nodiscard]] UniversalNullResult UniversalNullTest(const PersistenceDiagram& diagram, double alpha = 0.05);

/** @brief BootstrapBand()'s settings. */
struct BootstrapOptions {
    /** @brief How many resamples. */
    int64_t resamples = 100;
    /** @brief The band is the (1 - alpha) quantile of the bootstrap distances. */
    double alpha = 0.05;
    /** @brief The random seed: the same seed gives the same band on every platform. */
    uint32_t seed = 0;
    /** @brief The Rips settings for every resample's diagram (the threshold is reused as given;
     *         with the default, each resample uses its own enclosing radius). */
    RipsOptions rips;
};

/** @brief BootstrapBand()'s result. */
struct BootstrapBandResult {
    /** @brief The (1 - alpha) quantile c of the bottleneck distances: the diagram is within c of
     *         the population's with probability about 1 - alpha. */
    double band = 0;
    /** @brief Every resample's bottleneck distance to the original diagram, sorted. */
    std::vector<double> distances;
    /** @brief The original diagram in the tested dimension. */
    PersistenceDiagram diagram;
    /** @brief Indices of its features with persistence above 2 * band: farther from the diagonal
     *         than the band reaches. */
    std::vector<int64_t> significant;
};

/**
 * @brief A bootstrap confidence band for a Rips diagram (after Fasy, Lecci, Rinaldo, Wasserman,
 *        Balakrishnan and Singh, Ann. Statist. 2014): resample the points with replacement, compute
 *        each resample's diagram, and take the (1 - alpha) quantile of their bottleneck distances
 *        to the original.
 * @param distances The whole cloud's distances; resamples reuse them, so nothing is recomputed.
 * @param dimension The homology dimension to test.
 * @note Fasy et al. prove coverage for distance-function filtrations; for Rips diagrams the band is
 *       the standard heuristic. A resample repeats some points, which only adds zero-length edges.
 * @throws std::invalid_argument if resamples < 1, alpha is outside (0, 1), or dimension is above
 *         options.rips.max_dimension.
 */
[[nodiscard]] BootstrapBandResult BootstrapBand(const DistanceMatrix& distances, int dimension,
                                                const BootstrapOptions& options = {});

/**
 * @brief count points from a Gaussian with the mean and sample covariance of the rows of points:
 *        mu + Z^T (X - mu) / sqrt(n - 1) with Z standard normal (one GEMM on points' backend), so no
 *        decomposition of the covariance is needed and its rank is kept.
 * @param points shape (n, ...): a row per point, n >= 2.
 * @return shape (count, width), on points' backend.
 * @throws std::invalid_argument if points has rank below 2, fewer than 2 rows, a NaN or infinity,
 *         or count < 1.
 */
[[nodiscard]] Tensor GaussianBaseline(const Tensor& points, int64_t count, uint32_t seed = 0);

/**
 * @brief The rank p-value of an observed statistic against the same statistic on baselines:
 *        (1 + #{baseline >= observed}) / (1 + baselines.size()). Small means the observed value is
 *        larger than chance gives; the smallest possible is 1 / (1 + baselines.size()).
 * @throws std::invalid_argument if baselines is empty or a value is NaN.
 */
[[nodiscard]] double BaselinePValue(double observed, const std::vector<double>& baselines);

/** @brief The largest finite persistence in a diagram, 0 if it has none: a common statistic for
 *         BaselinePValue(). */
[[nodiscard]] double MaxPersistence(const PersistenceDiagram& diagram);

/** @brief A sub-matrix of a distance matrix: the distances among the given points (repeats
 *         allowed), in the given order. */
[[nodiscard]] DistanceMatrix SubDistanceMatrix(const DistanceMatrix& distances, const std::vector<int64_t>& indices);

}  // namespace pulsatrix
