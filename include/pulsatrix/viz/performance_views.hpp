/** @file performance_views.hpp
 *  @brief Figures for the Performance question (KS-3): a reliability diagram of a classifier's
 *         calibration, and the error on each slice of the data with its interval and test.
 *  @ingroup visualization
 *
 *  Both read the documents performance.hpp writes, `pulsatrix.classification_report.v1` and
 *  `pulsatrix.slice_error.v1`, so `pulsatrix_svg` draws them too. The figures follow svg.hpp's
 *  rules: standalone files, deterministic bytes, no font library, and a tooltip (`<title>`) on
 *  every mark.
 *
 *  Conformal coverage by slice is a slice-error figure of the misses EvaluateCoverage() reports,
 *  with the target miss rate alpha as the reference line.
 */
#pragma once

#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/performance.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {

/**
 * @brief Reads a `pulsatrix.classification_report.v1` document, as ToJson(ClassificationReport)
 *        writes it.
 * @throws std::invalid_argument if it isn't one, or doesn't hold together: a score outside its
 *         range, bins out of order or whose counts don't sum to n, or a confusion matrix or
 *         per-class list that isn't classes long.
 */
[[nodiscard]] ClassificationReport ParseClassificationReport(std::string_view json);

/**
 * @brief Reads a `pulsatrix.slice_error.v1` document, as ToJson(SliceErrorReport) writes it.
 * @throws std::invalid_argument if it isn't one, or doesn't hold together: no slices, counts that
 *         don't sum to n, an error outside its interval, a p-value outside [0, 1] or alpha
 *         outside (0, 1).
 */
[[nodiscard]] SliceErrorReport ParseSliceErrorReport(std::string_view json);

/** @brief One classifier on a reliability diagram. */
struct ReliabilitySeries {
    /** @brief Its name in the legend, such as "one network" or "ensemble of 5". */
    std::string name;
    ClassificationReport report;
};

/**
 * @brief A reliability diagram (Guo et al., ICML 2017): each confidence bin's accuracy against its
 *        mean confidence, one line per classifier. A calibrated classifier lies on the diagonal;
 *        points below it are overconfident, above it underconfident. The marks grow with the
 *        bin's share of the inputs, and a strip below shows that share, since a bin of three
 *        inputs can sit far from the diagonal by chance. The legend gives each one's ECE.
 * @throws std::invalid_argument if there are no series, more than 10, a report has no
 *         reliability bins, or the options are unusable.
 */
[[nodiscard]] std::string RenderReliabilityDiagramSvg(const std::vector<ReliabilitySeries>& series,
                                                      const SvgOptions& options = {});

/** @brief The diagram of one classifier. */
[[nodiscard]] std::string RenderReliabilityDiagramSvg(const ClassificationReport& report, const SvgOptions& options = {});

/** @brief RenderSliceErrorSvg()'s settings. */
struct SliceErrorViewOptions {
    /** @brief What the error is, for the axis: "error rate", "miss rate", "squared error". */
    std::string metric = "error";
    /** @brief A value to draw as a reference line, such as conformal prediction's target miss rate
     *         alpha; NaN for none. */
    double reference = std::numeric_limits<double>::quiet_NaN();
    /** @brief The reference line's label. */
    std::string reference_label = "target";
    /** @brief Draws at most this many slices, the worst first, and counts the rest. */
    int max_slices = 30;
};

/**
 * @brief The error on each slice, worst first: a point with its interval, the error of the whole
 *        dataset as a dashed line, and each slice's p-value. Slices worse than the rest, after
 *        Bonferroni's correction, are red.
 * @throws std::invalid_argument if the report has no slices, max_slices < 1, the reference is
 *         infinite or negative, or the options are unusable.
 */
[[nodiscard]] std::string RenderSliceErrorSvg(const SliceErrorReport& report, const SliceErrorViewOptions& view = {},
                                              const SvgOptions& options = {});

}  // namespace pulsatrix
