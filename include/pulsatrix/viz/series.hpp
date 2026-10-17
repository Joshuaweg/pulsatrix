/** @file series.hpp
 *  @brief The series document (VIZ-9): measured points with optional intervals, for sweeps and
 *         trade-off curves, and its SVG view. The Vega-Lite view is in html.hpp.
 *  @ingroup visualization
 *  A series is one line of (x, y) points, each with an optional interval [lo, hi] (a confidence
 *  interval, a range over seeds). A panel holds several series on shared axes, and a document
 *  holds one or more panels with the same axis labels and scales: small multiples. Either axis
 *  can be linear or logarithmic.
 *
 *  It follows the rules of every viz document (document.hpp): a `"schema"` of
 *  `pulsatrix.series.v1`, non-finite numbers as null plus a `"nonfinite"` entry, and readers
 *  that name the JSON Pointer of the first problem.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {

/** @brief One line of points. */
struct Series {
    /** @brief Names the series in the legend. A series keeps its color in every panel. */
    std::string name;
    std::vector<double> x;
    std::vector<double> y;
    /** @brief Empty, or one per point: the interval drawn around y. lo <= hi. */
    std::vector<double> lo;
    std::vector<double> hi;

    [[nodiscard]] size_t size() const { return x.size(); }
    [[nodiscard]] bool has_intervals() const { return !lo.empty(); }
};

/** @brief Several series on shared axes. */
struct SeriesPanel {
    /** @brief Drawn above the panel when not empty. */
    std::string title;
    std::vector<Series> series;
};

/** @brief How an axis maps values to positions. */
enum class AxisScale { Linear, Log };

[[nodiscard]] const char* AxisScaleName(AxisScale scale);

struct SeriesDocument {
    std::string title;
    std::string x_label;
    std::string y_label;
    AxisScale x_scale = AxisScale::Linear;
    AxisScale y_scale = AxisScale::Linear;
    /** @brief What lo and hi are, for example "95% Wilson interval over 5 seeds". Shown under
     *         the chart when not empty. */
    std::string interval;
    /** @brief At least one, each with at least one series. */
    std::vector<SeriesPanel> panels;
};

/**
 * @throws std::invalid_argument for a document that breaks the rules above: no panels, a panel
 *         with no series, a series with no points or no name, two series with the same name in a
 *         panel, y (or lo, hi) of a different length from x, or lo > hi.
 */
[[nodiscard]] std::string ToJson(const SeriesDocument& doc);
/** @throws std::invalid_argument naming the JSON Pointer of the first problem, including every
 *          rule ToJson() checks. */
[[nodiscard]] SeriesDocument ParseSeriesDocument(std::string_view json);

/** @brief Options for RenderSeriesSvg(). */
struct SeriesViewOptions {
    /** @brief Give every panel the same y range, so their heights compare directly. */
    bool shared_y = false;
};

/**
 * @brief The panels stacked top to bottom, each a line per series with a marker per point and a
 *        whisker per interval; one legend for all panels. Every marker has a tooltip with its
 *        values.
 * @note On a log axis every value must be positive, except that a lower interval bound at or
 *       below zero runs the whisker to the bottom of the axis.
 * @throws std::invalid_argument for an invalid document (see ToJson()), a non-finite value, a
 *         value that can't go on a log axis, or options with width < 200 or font_size < 1.
 */
[[nodiscard]] std::string RenderSeriesSvg(const SeriesDocument& doc, const SeriesViewOptions& view = {},
                                          const SvgOptions& options = {});

}  // namespace pulsatrix
