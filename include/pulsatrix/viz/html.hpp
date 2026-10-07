/** @file html.hpp
 *  @brief Self-contained HTML views of the viz documents (VIZ-3): interactive Vega-Lite charts
 *         with tooltips, zoom and PNG/SVG export, and plain-HTML text views, in any browser or
 *         notebook.
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/svg.hpp"  // IceStyle, PartialDependenceSvgOptions

namespace pulsatrix {

/** @brief The versions of Vega, Vega-Lite and vega-embed the pages load (BSD-3-Clause). */
inline constexpr const char* kVegaVersion = "6.4.0";
inline constexpr const char* kVegaLiteVersion = "6.4.3";
inline constexpr const char* kVegaEmbedVersion = "7.3.0";

/** @brief How a page gets the Vega libraries. */
enum class HtmlScripts {
    /** @brief From the jsDelivr CDN, pinned to the versions above and checked with Subresource
     *         Integrity hashes. The page is small, but viewing it needs a network connection. */
    Cdn,
    /** @brief Copied into the page from `script_dir`, so it works offline and can be mailed as
     *         one file (about 830 KB more). `tools/render/fetch_vega.sh` downloads the files. */
    Inline,
};

/** @brief Options for every HTML view. */
struct HtmlOptions {
    HtmlScripts scripts = HtmlScripts::Cdn;
    /** @brief With HtmlScripts::Inline: a directory holding vega.min.js, vega-lite.min.js and
     *         vega-embed.min.js at the pinned versions. */
    std::string script_dir;
    /** @brief The chart's width in pixels. */
    int width = 640;
    /** @brief Replaces the view's own title when not empty. */
    std::string title;
};

// ---- Vega-Lite specifications ---------------------------------------------------------------
// The charts as Vega-Lite specs, for embedding in your own page or notebook. Each throws
// std::invalid_argument for the same problems as its SVG counterpart (svg.hpp).

/** @brief Horizontal bars of the @p top_k features with the largest |attribution|, colored by
 *         sign on the SVG renderer's diverging scale. */
[[nodiscard]] JsonValue ToVegaLiteBarChart(const AttributionDocument& doc, int top_k = 10, const HtmlOptions& options = {});
/** @brief A waterfall from @p baseline_value through the features' contributions to the
 *         prediction (ToWaterfallSteps). */
[[nodiscard]] JsonValue ToVegaLiteWaterfall(const AttributionDocument& doc, float baseline_value,
                                            const HtmlOptions& options = {});
/** @brief A heatmap: diverging around zero when any value is negative, Viridis otherwise.
 *         Non-finite cells are gray. */
[[nodiscard]] JsonValue ToVegaLiteHeatmap(const HeatmapDocument& doc, const HtmlOptions& options = {});
/** @brief One row per feature in @p feature_indices: every document's value for it, spread so
 *         points don't overlap (ToBeeswarmPoints). */
[[nodiscard]] JsonValue ToVegaLiteBeeswarm(const std::vector<AttributionDocument>& docs,
                                           const std::vector<int64_t>& feature_indices, const HtmlOptions& options = {});
/** @brief A partial dependence plot: ICE curves under the average, and a rug of the instances'
 *         feature values. Zoom with the mouse wheel; drag to pan. */
[[nodiscard]] JsonValue ToVegaLitePartialDependence(const PartialDependenceDocument& doc,
                                                    const PartialDependenceSvgOptions& pd = {},
                                                    const HtmlOptions& options = {});
/** @brief A tornado chart of the @p top_k features with the largest swing. */
[[nodiscard]] JsonValue ToVegaLiteTornado(const SensitivityDocument& doc, int top_k = 10, const HtmlOptions& options = {});
/** @brief The changed features of a counterfactual, costliest first, as bars of their change in
 *         scale units; the tooltip has the old and new values. */
[[nodiscard]] JsonValue ToVegaLiteCounterfactual(const CounterfactualDocument& doc, int max_rows = 12,
                                                 const HtmlOptions& options = {});
/** @brief Several counterfactuals of one input: a grid of features (rows) by counterfactuals
 *         (columns), each cell colored by the change in scale units and labeled with the new value. */
[[nodiscard]] JsonValue ToVegaLiteCounterfactualSet(const std::vector<CounterfactualDocument>& docs,
                                                    const HtmlOptions& options = {});
/** @brief Morris screening: mu* against sigma per feature, with mu*'s confidence interval and the
 *         sigma = mu* line. */
[[nodiscard]] JsonValue ToVegaLiteMorris(const MorrisDocument& doc, const HtmlOptions& options = {});
/** @brief Sobol indices: first-order bars in front of total-order ones, with confidence whiskers. */
[[nodiscard]] JsonValue ToVegaLiteSobol(const SobolDocument& doc, int top_k = 10, const HtmlOptions& options = {});
/** @brief A training log: one zoomable line chart per scalar tag, and one histogram per
 *         histogram tag. */
[[nodiscard]] JsonValue ToVegaLiteTrainingLog(const TrainingLogDocument& doc, const HtmlOptions& options = {});
/** @brief A feature's activation histogram (the dashboard's chart; RenderFeatureDashboardHtml adds
 *         the top examples). */
[[nodiscard]] JsonValue ToVegaLiteFeatureHistogram(const FeatureDashboardDocument& doc, const HtmlOptions& options = {});

// ---- Pages ------------------------------------------------------------------------------------

/** @brief A complete page that draws @p spec with vega-embed, which adds the PNG/SVG export menu.
 *  @param heading Shown above the chart. @param extra_html Appended after it, as is.
 *  @throws std::runtime_error if inline scripts are asked for and can't be read. */
[[nodiscard]] std::string VegaLitePage(const JsonValue& spec, const std::string& heading, const HtmlOptions& options = {},
                                       const std::string& extra_html = "");

[[nodiscard]] std::string RenderBarChartHtml(const AttributionDocument& doc, int top_k = 10, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderWaterfallHtml(const AttributionDocument& doc, float baseline_value,
                                              const HtmlOptions& options = {});
[[nodiscard]] std::string RenderHeatmapHtml(const HeatmapDocument& doc, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderBeeswarmHtml(const std::vector<AttributionDocument>& docs,
                                             const std::vector<int64_t>& feature_indices, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderPartialDependenceHtml(const PartialDependenceDocument& doc,
                                                      const PartialDependenceSvgOptions& pd = {},
                                                      const HtmlOptions& options = {});
[[nodiscard]] std::string RenderTornadoHtml(const SensitivityDocument& doc, int top_k = 10, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderCounterfactualHtml(const CounterfactualDocument& doc, int max_rows = 12,
                                                   const HtmlOptions& options = {});
[[nodiscard]] std::string RenderCounterfactualSetHtml(const std::vector<CounterfactualDocument>& docs,
                                                      const HtmlOptions& options = {});
[[nodiscard]] std::string RenderMorrisHtml(const MorrisDocument& doc, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderSobolHtml(const SobolDocument& doc, int top_k = 10, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderTrainingLogHtml(const TrainingLogDocument& doc, const HtmlOptions& options = {});
/** @brief The feature's statistics, its activation histogram, and its top examples with each
 *         token shaded by its activation. */
[[nodiscard]] std::string RenderFeatureDashboardHtml(const FeatureDashboardDocument& doc, const HtmlOptions& options = {});

/**
 * @brief The text with each piece (token or word) on a background colored by its relevance, as
 *        plain HTML with no scripts: the browser lays out and shapes the text, so every script reads
 *        correctly (right-to-left, joined Arabic, Indic conjuncts), and wraps it. Hovering a piece
 *        shows its score; unscored pieces are plain text. Scripts option is ignored.
 * @throws std::invalid_argument for the problems RenderTokenStripSvg rejects.
 */
[[nodiscard]] std::string RenderTokenRelevanceHtml(const TokenRelevanceDocument& doc, const HtmlOptions& options = {});

}  // namespace pulsatrix
