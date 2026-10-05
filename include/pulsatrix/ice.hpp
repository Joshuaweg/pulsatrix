/** @file ice.hpp
 *  @brief Individual conditional expectation (ICE), its centered and derivative forms,
 *         two-feature partial dependence (CFS-1) and accumulated local effects (CFS-2).
 *  @ingroup interpretability_agnostic
 */
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief An evenly spaced grid for one feature, built the way scikit-learn's
 *        `partial_dependence` builds it: the feature's distinct values when there are fewer
 *        than @p grid_size of them, otherwise @p grid_size points from the lower to the upper
 *        percentile of the feature's values (`scipy.stats.mstats.mquantiles` with its default
 *        plotting positions).
 * @param background Instances to read the feature from; all the same shape.
 * @param feature_index Flat index of the feature within an instance.
 * @param grid_size Number of points, >= 2.
 * @param lower_percentile,upper_percentile In [0, 1], lower < upper. Defaults are scikit-learn's.
 * @throws std::invalid_argument if background is empty, the feature index is out of range, the
 *         percentiles are invalid, grid_size < 2, or the two percentiles coincide (a feature
 *         that is nearly constant; pass a grid explicitly instead).
 */
[[nodiscard]] std::vector<float> FeatureGrid(const std::vector<Tensor>& background, int64_t feature_index,
                                             int64_t grid_size = 100, float lower_percentile = 0.05f,
                                             float upper_percentile = 0.95f);

/**
 * @brief ICE curves for one feature: `curves[i][k] = f(x_i with feature j set to grid[k])[target]`.
 *
 * Goldstein et al. (2015). The partial dependence curve is their mean, so ICE shows what PDP
 * averages away: instances whose response to the feature differs, which is the signature of an
 * interaction.
 */
struct IceResult {
    int64_t feature_index = 0;
    int64_t target_index = 0;
    std::vector<float> grid;
    int64_t num_instances = 0;
    /** @brief Row-major (num_instances, grid.size()). */
    std::vector<float> curves;
    /** @brief Each instance's own value of the feature, where its prediction sits on its curve. */
    std::vector<float> feature_values;

    /** @brief The partial dependence curve: the mean of the curves at each grid point. */
    [[nodiscard]] std::vector<float> partial_dependence() const;
    /**
     * @brief Centered ICE (c-ICE): each curve minus its value at grid[@p anchor], so curves that
     *        differ only by an offset coincide and differing shapes stand out.
     * @throws std::out_of_range if anchor is not a grid index.
     */
    [[nodiscard]] std::vector<float> centered(int64_t anchor = 0) const;
    /**
     * @brief Derivative ICE (d-ICE): the slope of each curve, by central differences inside the
     *        grid and one-sided differences at its ends (numpy.gradient). Where the slopes vary
     *        across instances at the same grid point, the feature interacts with others there.
     * @throws std::logic_error if the grid has fewer than 2 points.
     */
    [[nodiscard]] std::vector<float> derivative() const;
};

/**
 * @brief Computes ICE curves for every instance.
 * @param predict Forward pass for one instance: input in, output out.
 * @param instances The instances to draw curves for, all the same shape. Their mean is the
 *        partial dependence over them.
 * @param feature_index Flat index of the feature to vary.
 * @param target_index Flat index of the output element to read.
 * @param grid Values to set the feature to (see FeatureGrid). Non-empty.
 * @note Calls predict instances.size() * grid.size() times. Tensors are built beside each
 *       instance (same backend and device); only the target element is read back.
 * @throws std::invalid_argument if instances or grid is empty, the instances differ in shape, or
 *         an index is out of range.
 */
[[nodiscard]] IceResult ComputeIce(const std::function<Tensor(const Tensor&)>& predict,
                                   const std::vector<Tensor>& instances, int64_t feature_index,
                                   int64_t target_index, const std::vector<float>& grid);

/** @brief Partial dependence on two features at once, over a grid of value pairs. */
struct PartialDependence2D {
    int64_t feature_x = 0;
    int64_t feature_y = 0;
    int64_t target_index = 0;
    std::vector<float> grid_x;
    std::vector<float> grid_y;
    /** @brief Row-major (grid_y.size(), grid_x.size()): `values[r][c]` is the mean prediction with
     *         feature_y = grid_y[r] and feature_x = grid_x[c], so it draws as an image with y down
     *         the rows. */
    std::vector<float> values;
};

/**
 * @brief Two-feature partial dependence: for every pair (grid_x[c], grid_y[r]), the mean over
 *        the background of the prediction with both features set. Where it isn't the sum of the
 *        two one-feature curves, the features interact.
 * @throws std::invalid_argument for the same inputs ComputeIce rejects, or feature_x == feature_y.
 */
[[nodiscard]] PartialDependence2D ComputePartialDependence2D(const std::function<Tensor(const Tensor&)>& predict,
                                                             const std::vector<Tensor>& background,
                                                             int64_t feature_x, int64_t feature_y,
                                                             int64_t target_index, const std::vector<float>& grid_x,
                                                             const std::vector<float>& grid_y);

/**
 * @brief First-order accumulated local effects (Apley and Zhu 2020) of one feature.
 *
 * The feature's range is cut at quantiles into bins holding about equal numbers of instances.
 * Within each bin, every instance in it is moved to the bin's lower and upper edge and the change
 * in the output is averaged; the averages are summed from the left and centered so their
 * count-weighted mean is zero. Because each instance only moves within its own bin, ALE never
 * reads the model at combinations of feature values the data doesn't contain, which is where PDP
 * misleads when features are correlated.
 */
struct AleResult {
    int64_t feature_index = 0;
    int64_t target_index = 0;
    /** @brief Bin edges, strictly increasing: the minimum, then the upper edge of each bin. */
    std::vector<float> edges;
    /** @brief The centered accumulated effect at each edge. */
    std::vector<float> effects;
    /** @brief Instances in each bin (edges.size() - 1 of them). */
    std::vector<int64_t> counts;
    /** @brief Each instance's own value of the feature, for a rug. */
    std::vector<float> feature_values;
};

/**
 * @brief Computes first-order ALE the way PyALE 1.2 (and R's ALEPlot) does: edges at the
 *        feature's type-1 quantiles (`num_bins` + 1 of them, duplicates dropped), bins closed on
 *        the right with the first also holding the minimum, and centering by the count-weighted
 *        mean of each bin's midpoint effect.
 * @param num_bins Requested number of bins, >= 1; ties in the feature can merge bins.
 * @note 2 * instances.size() model calls. An empty bin contributes no effect.
 * @throws std::invalid_argument for the same inputs ComputeIce rejects, num_bins < 1, or a
 *         feature that takes only one value.
 */
[[nodiscard]] AleResult ComputeAle(const std::function<Tensor(const Tensor&)>& predict,
                                   const std::vector<Tensor>& instances, int64_t feature_index, int64_t target_index,
                                   int64_t num_bins = 20);

}  // namespace pulsatrix
