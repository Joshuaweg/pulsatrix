/** @file svg.hpp
 *  @brief Dependency-free SVG figures from viz documents: bar, waterfall, heatmap, token strip,
 *         beeswarm, partial dependence, tornado, counterfactual, Morris and Sobol charts.
 *  @ingroup visualization
 *
 *  Each function returns a complete standalone SVG file as a string. No GPU, display, font
 *  library or ImGui is involved, so figures can be made in CI and on headless machines. Colors
 *  come from colormap.hpp, the same as the ImGui widgets: DivergingColormap for signed values
 *  (blue negative, white zero, red positive) and ViridisColormap for unsigned magnitudes.
 *
 *  Output is deterministic: the same input gives the same bytes on every platform. Text widths
 *  are estimated (0.6 em per character), since there is no font library; labels are sized to
 *  leave room for that estimate.
 *
 *  `tools/render/pulsatrix_svg.cpp` wraps these as a command-line tool that turns a JSON
 *  document into an SVG file.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/viz/document.hpp"

namespace pulsatrix {

/** @brief Options shared by every SVG chart. */
struct SvgOptions {
    /** @brief Figure width in pixels (a heatmap narrower than this ends at its color bar). The
     *         height follows from the content. */
    int width = 640;
    /** @brief Drawn above the chart when not empty. */
    std::string title;
    /** @brief Base font size in pixels. */
    int font_size = 12;
};

/**
 * @brief Horizontal bars of the @p top_k features with the largest |attribution|, largest first,
 *        on an axis through zero, each colored by its signed value and labeled with it. Labels
 *        come from metadata["feature_names"] (comma-separated) when present.
 * @throws std::invalid_argument for the same inputs ToFeatureImportanceBars rejects, a non-finite
 *         value, or options with width < 200 or font_size < 1.
 */
[[nodiscard]] std::string RenderBarChartSvg(const AttributionDocument& doc, int top_k = 10, const SvgOptions& options = {});

/**
 * @brief A waterfall from @p baseline_value through each feature's contribution, in the
 *        document's feature order, to the final value: red columns raise the running total,
 *        blue columns lower it, with the baseline and the final value marked.
 * @throws std::invalid_argument for the same inputs ToWaterfallSteps rejects, a non-finite value
 *         or baseline, or invalid options.
 */
[[nodiscard]] std::string RenderWaterfallSvg(const AttributionDocument& doc, float baseline_value,
                                             const SvgOptions& options = {});

/**
 * @brief The heatmap's grid with square cells and a color bar. Signed grids use the diverging
 *        map on a range symmetric about zero, unsigned grids Viridis on [0, max]
 *        (ComputeHeatmapColorScale's rule). Non-finite cells are gray and left out of the
 *        scale. Row and column labels are drawn when the document has them.
 * @note Grids up to 4096 cells are drawn as one rectangle per cell. Larger grids, such as a
 *       224 x 224 saliency map, are embedded as a lossless PNG image drawn with sharp pixel
 *       edges, which keeps the file small.
 * @throws std::invalid_argument if the document is inconsistent (see ToJson) or empty, or for
 *         invalid options.
 */
[[nodiscard]] std::string RenderHeatmapSvg(const HeatmapDocument& doc, const SvgOptions& options = {});

/**
 * @brief The text's tokens in reading order, wrapped across lines, each on a background colored
 *        by its relevance (diverging, scaled to the largest |relevance|). Tokens are drawn in a
 *        monospace font, so their widths are exact, with whitespace kept. Non-finite relevance
 *        is drawn gray. The target, when set, is shown after the text.
 * @throws std::invalid_argument if tokens and relevance differ in length, there are no tokens,
 *         or for invalid options.
 */
[[nodiscard]] std::string RenderTokenStripSvg(const TokenRelevanceDocument& doc, const SvgOptions& options = {});

/**
 * @brief One beeswarm row per feature in @p feature_indices: each point is one document's value
 *        for that feature, placed by value on a shared axis through zero, spread vertically so
 *        points don't overlap (ToBeeswarmPoints's deterministic layout), and colored by value.
 *        Labels come from the first document's metadata["feature_names"] when present.
 * @param docs Attributions of the same shape for different inputs.
 * @throws std::invalid_argument if @p docs or @p feature_indices is empty, a feature index is
 *         out of range for any document, a value is non-finite, or for invalid options.
 */
[[nodiscard]] std::string RenderBeeswarmSvg(const std::vector<AttributionDocument>& docs,
                                            const std::vector<int64_t>& feature_indices,
                                            const SvgOptions& options = {});

/** @brief Which form of the ICE curves RenderPartialDependenceSvg draws (CFS-1). */
enum class IceStyle {
    /** The predictions themselves. */
    Raw,
    /** Each curve minus its value at the first grid point (c-ICE). */
    Centered,
    /** Each curve's slope (d-ICE). */
    Derivative,
};

/** @brief Options for RenderPartialDependenceSvg. */
struct PartialDependenceSvgOptions {
    IceStyle style = IceStyle::Raw;
    /** @brief At most this many ICE curves are drawn, evenly spaced through the instances; the
     *         average still uses them all. 0 draws the average only. */
    int max_curves = 100;
};

/**
 * @brief A partial dependence plot: thin ICE curves under the bold average, over the feature's
 *        grid, with a rug of the instances' own feature values along the bottom. For
 *        IceStyle::Centered and IceStyle::Derivative the average is recomputed from the
 *        transformed curves, and a dashed line marks zero.
 * @throws std::invalid_argument for a document ToJson rejects, a non-finite value, a style other
 *         than Raw for a document without ICE curves, negative max_curves, or unusable options.
 */
[[nodiscard]] std::string RenderPartialDependenceSvg(const PartialDependenceDocument& doc,
                                                     const PartialDependenceSvgOptions& pd = {},
                                                     const SvgOptions& options = {});

/**
 * @brief A tornado chart: one row per feature, the @p top_k with the largest swing first. Each row
 *        has two bars from the unchanged output, one to the output at the feature's high value
 *        (red) and one to its low value (blue), labeled with those values; a dashed line marks
 *        the unchanged output.
 * @throws std::invalid_argument for an empty document, a non-finite value, top_k < 1, or unusable
 *         options.
 */
[[nodiscard]] std::string RenderTornadoSvg(const SensitivityDocument& doc, int top_k = 10,
                                           const SvgOptions& options = {});

/**
 * @brief What a counterfactual changed: a verdict line (whether it reaches the target, and the
 *        output before and after), then one row per changed feature, costliest first, with its
 *        old and new value and a bar of its change in units of its scale, colored by direction.
 *        A feature counts as changed when it moved by more than 1e-3 of its scale.
 * @param max_rows At most this many changed features are listed; a line says how many more.
 * @throws std::invalid_argument for a non-finite value, max_rows < 1, or unusable options.
 */
[[nodiscard]] std::string RenderCounterfactualSvg(const CounterfactualDocument& doc, int max_rows = 12,
                                                  const SvgOptions& options = {});

/**
 * @brief Morris screening as a scatter of mu* (importance) against sigma (how much the effect
 *        varies), one labeled point per feature with a horizontal bar for mu*'s confidence
 *        interval. Points above the dashed line sigma = mu* have effects that vary more than
 *        their average: nonlinear, or interacting with other features.
 * @throws std::invalid_argument for an empty document, a non-finite or negative mu*, sigma or
 *         confidence half-width, or unusable options.
 */
[[nodiscard]] std::string RenderMorrisSvg(const MorrisDocument& doc, const SvgOptions& options = {});

/**
 * @brief Sobol indices: one row per feature, the @p top_k with the largest total order first,
 *        with a bar for the first-order index, a lighter bar for the total-order index behind it,
 *        and whiskers for both confidence intervals. The gap between the two bars is the share
 *        of variance the feature explains only through interactions.
 * @throws std::invalid_argument for an empty document, a non-finite value, top_k < 1, or
 *         unusable options.
 */
[[nodiscard]] std::string RenderSobolSvg(const SobolDocument& doc, int top_k = 10, const SvgOptions& options = {});

/**
 * @brief Several counterfactuals of one input side by side (CFS-7): a column for the input and one
 *        per counterfactual (headed by whether it reaches the target), and a row for every feature
 *        that at least one of them changes. Changed values are on a background colored by the
 *        change's direction and size in scale units; unchanged ones are blank.
 * @throws std::invalid_argument if @p docs is empty, the documents describe different inputs
 *         (feature names or original values differ), a value is non-finite, or the options are
 *         unusable.
 */
[[nodiscard]] std::string RenderCounterfactualSetSvg(const std::vector<CounterfactualDocument>& docs,
                                                     const SvgOptions& options = {});

}  // namespace pulsatrix
