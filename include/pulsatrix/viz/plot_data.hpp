/** @file plot_data.hpp
 *  @brief Pure Attribution/Dataset -> plot-ready-array transforms (no ImGui/ImPlot include).
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/circuit_graph.hpp"
#include "pulsatrix/dataset.hpp"

namespace pulsatrix {

/** @brief One feature-importance bar: a label and a signed value (sign carries direction). */
struct BarSeries {
    std::vector<std::string> labels;
    std::vector<float> values;
};

/**
 * @brief Converts an Attribution's per-feature values into labeled, magnitude-sorted bars
 *        for a horizontal bar chart -- the canonical XAI feature-importance encoding
 *        (hc_information_visualization.md SS5: position on a shared axis, Cleveland-McGill
 *        rank 1).
 * @param attr A rank-1 Attribution (one value per feature), or a rank-2 Attribution whose
 *        leading (batch) dimension is 1 -- every native/surrogate explainer in this
 *        codebase is always-batched post campaign_exai_dl_library_batch_dimension_support,
 *        so a single-prediction explanation is shape (1, num_features), not (num_features,).
 * @param top_k Number of highest-|value| features to keep, in descending |value| order.
 * @return Bars sorted by descending absolute value, truncated to top_k (or fewer, if attr
 *         has fewer features than top_k). Labels come from attr.metadata["feature_names"]
 *         (comma-separated) when present, otherwise "feature_<index>".
 * @throws std::invalid_argument if attr.values is not rank 1 or batch-1 rank 2, or if
 *         top_k <= 0.
 */
[[nodiscard]] BarSeries ToFeatureImportanceBars(const Attribution& attr, int top_k);

/** @brief One waterfall step: a labeled delta and the running cumulative value after it. */
struct WaterfallStep {
    std::string label;
    float delta;
    float cumulative;
};

/**
 * @brief Converts an Attribution's per-feature values into a cascading waterfall from a
 *        real baseline to the final prediction (hc_information_visualization.md SS5/SS6:
 *        the one case where a bar chart legitimately does not start at zero, since the
 *        baseline itself is the meaningful reference point -- Tufte's lie-factor rule is
 *        satisfied relative to that baseline, not to zero).
 * @param attr A rank-1 Attribution (one value per feature), or a rank-2 Attribution whose
 *        leading (batch) dimension is 1 (see ToFeatureImportanceBars's note on always-batched
 *        explainer output), in the explainer's own feature order -- deliberately NOT sorted
 *        by magnitude, since a waterfall's narrative is the accumulation path, not a ranking.
 * @param baseline_value The starting reference value (e.g. Integrated Gradients' baseline
 *        prediction, or the model's mean output).
 * @throws std::invalid_argument if attr.values is not rank 1 or batch-1 rank 2.
 */
[[nodiscard]] std::vector<WaterfallStep> ToWaterfallSteps(const Attribution& attr, float baseline_value);

/** @brief One floating waterfall bar: spans [bottom, top] on the value axis. */
struct WaterfallBar {
    float bottom;
    float top;
    /** @brief True for a non-negative delta (drawn in the "increase" color). */
    bool increase;
};

/**
 * @brief Converts waterfall steps into floating bars, each spanning from the previous running
 *        total to its own running total -- i.e. bar i covers [min(c_{i-1}, c_i), max(c_{i-1},
 *        c_i)] with c_{-1} = baseline_value. Correct for any sign of baseline or running total:
 *        a cascade that starts below zero, crosses zero, or stays negative (e.g. explaining a
 *        negative logit) floats exactly where the running total is, rather than being anchored
 *        at zero the way stacked bar segments are.
 * @param steps ToWaterfallSteps's output.
 * @param baseline_value The same baseline passed to ToWaterfallSteps.
 */
[[nodiscard]] std::vector<WaterfallBar> ToWaterfallBars(const std::vector<WaterfallStep>& steps, float baseline_value);

/** @brief A row-major 2D grid of unsigned magnitude values, ready for a heatmap plot. */
struct HeatmapGrid {
    std::vector<float> values;
    int64_t rows;
    int64_t cols;
};

/**
 * @brief Reshapes an Attribution's values into a 2D grid for a saliency overlay heatmap.
 * @param attr A rank-2 Attribution, a rank-3 Attribution whose leading (channel or batch)
 *        dimension is 1 (single-channel saliency map, or Grad-CAM's batch-1 (1, H, W) map,
 *        squeezed), or a rank-4 Attribution of shape (1, 1, H, W) -- the shape every
 *        input-space image explainer (Saliency, IntegratedGradients, LRP, LIME, KernelSHAP)
 *        returns for a single batched single-channel image such as an MNIST digit.
 * @throws std::invalid_argument if attr.values is not reshapable to a 2D grid by the rules
 *         above (e.g. rank 1, rank 3 with more than one channel, or rank 4 with a batch or
 *         channel dimension other than 1).
 */
[[nodiscard]] HeatmapGrid ToSaliencyHeatmap(const Attribution& attr);

/** @brief The color-scale range and colormap family a heatmap's values call for. */
struct HeatmapColorScale {
    /** @brief Value mapped to the colormap's low end. */
    float scale_min;
    /** @brief Value mapped to the colormap's high end. */
    float scale_max;
    /** @brief True iff the grid has any negative value: the caller must then use a diverging
     *         colormap (DivergingColormap) centred on zero, since scale_min == -scale_max. When
     *         false, values are unsigned magnitudes and a sequential colormap (Viridis) over
     *         [0, scale_max] is correct. */
    bool is_signed;
};

/**
 * @brief Chooses a heatmap's color scale from its values (hc_information_visualization.md SS4:
 *        sequential maps for unsigned magnitude, diverging maps centred on the meaningful
 *        midpoint -- zero -- for signed quantities).
 * @return For an all-non-negative grid, [0, max] with is_signed == false. For a grid with any
 *         negative value, the symmetric range [-max|v|, +max|v|] with is_signed == true, so
 *         zero always lands on the diverging map's neutral midpoint and equal magnitudes of
 *         opposite sign get equal color intensity. A degenerate all-zero (or empty) grid
 *         returns [0, 1] rather than a zero-width range.
 */
[[nodiscard]] HeatmapColorScale ComputeHeatmapColorScale(const HeatmapGrid& grid);

/** @brief One beeswarm point: the attribution value (x) and a collision-avoidance vertical offset (y). */
struct BeeswarmPoint {
    float x;
    float y;
};

/**
 * @brief Converts one feature's attribution value across many repeated/independent runs
 *        into jittered (x, y) points for a beeswarm plot -- the global-explanation
 *        distribution view (hc_information_visualization.md SS5: "SHAP beeswarm plot...
 *        each point is one prediction; x-position is the SHAP value"). Deterministic,
 *        density-based jitter: points are binned along x, then colliding points within a
 *        bin are alternately offset above/below y=0 so they read as spread rather than
 *        overlapping -- not a random jitter, so two calls with the same input always
 *        produce the same layout.
 * @param runs Attribution results across many samples/predictions (not necessarily
 *        repeated runs on one input -- unlike ComputeAttributionStability, this is the
 *        global view across many different inputs).
 * @param feature_index Which feature (element position) to plot.
 * @throws std::invalid_argument if runs is empty.
 * @throws std::out_of_range if feature_index is out of range for any run's Attribution.
 */
[[nodiscard]] std::vector<BeeswarmPoint> ToBeeswarmPoints(const std::vector<Attribution>& runs, int64_t feature_index);

/**
 * @brief Human-readable label for a CircuitGraph node: the node's own label when it has one,
 *        otherwise its operation type and id (e.g. "Conv #1", "Activation #2") -- so an
 *        unlabeled ComputationGraph still renders as a readable layer diagram rather than a
 *        row of anonymous "node_<id>" markers.
 */
[[nodiscard]] std::string CircuitNodeDisplayLabel(const CircuitNode& node);

/** @brief Equal-width histogram bins: rows.size() == counts.size() + 1 edges. */
struct HistogramBins {
    std::vector<float> bin_edges;
    std::vector<int64_t> counts;
};

/**
 * @brief Bins one Sample field's values (flattened across every sample's Tensor at that
 *        field position, mirroring DatasetValidator's aggregation convention) into
 *        num_bins equal-width bins, for a per-field distribution histogram
 *        (hc_information_visualization.md's data-preview touchpoint).
 * @param dataset The dataset to scan. Must be non-empty.
 * @param field_index Which Sample field position to bin.
 * @param num_bins Number of equal-width bins, must be positive.
 * @throws std::invalid_argument if dataset is empty or num_bins <= 0.
 * @throws std::out_of_range if field_index is out of range for the dataset's samples.
 * @note A constant field (max == min) places every value in the first bin rather than
 *       dividing by a zero-width bin range.
 */
[[nodiscard]] HistogramBins ToFieldHistogramBins(const Dataset& dataset, int64_t field_index, int num_bins);

/** @brief An interleaved-RGB, row-major byte buffer ready for a texture upload. */
struct RgbImageBuffer {
    std::vector<unsigned char> pixels;
    int64_t height;
    int64_t width;
};

/**
 * @brief Converts a decoded image Tensor into an interleaved-RGB byte buffer for GPU texture
 *        upload (the data-preview "image grid" touchpoint's pure half -- the actual
 *        glTexImage2D call lives in TextureCache, which is GL-dependent and untestable here).
 * @param image_chw A rank-3 (C, H, W) tensor, or a rank-4 tensor with a leading batch
 *        dimension of 1 ((1, C, H, W) -- ImageFolderDataset::get()'s convention). C must be
 *        1 (replicated across R/G/B) or 3.
 * @note Values are assumed already normalized to [0, 1] and clamped before scaling to
 *       [0, 255] -- out-of-range input (e.g. an un-normalized decoder output) is clamped
 *       rather than wrapped, since a wrapped byte would silently invert bright/dark regions.
 * @throws std::invalid_argument if image_chw's rank/channel count don't match the above.
 */
[[nodiscard]] RgbImageBuffer ToRgbImageBuffer(const Tensor& image_chw);

}  // namespace pulsatrix
