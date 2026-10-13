/** @file topology_views.hpp
 *  @brief Three views of a persistence diagram document (TDA-9): the persistence diagram, the
 *         barcode and the Betti curve.
 *  @ingroup visualization
 *
 *  The figures follow svg.hpp's rules: standalone files, deterministic bytes, no font library.
 *  Each dimension has its own color. Every point and bar carries a tooltip (`<title>`) with its
 *  birth, death and, when tested, p-value.
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/topology_documents.hpp"

namespace pulsatrix {

/** @brief Settings shared by the three views. */
struct PersistenceViewOptions {
    /** @brief The dimensions to draw; empty for all of them. */
    std::vector<int> dimensions;
    /** @brief The barcode draws at most this many bars per dimension, the most persistent first,
     *         and counts the rest. */
    int max_bars = 40;
    /** @brief Scales at which the Betti curve is evaluated. */
    int grid_points = 200;
};

/**
 * @brief Birth against death, one point per feature. Points far above the diagonal persist long.
 *        A shaded band along the diagonal marks persistence within twice the certified error
 *        bound, which the distances' own error could explain. Essential features sit on a dashed
 *        line marked ∞. When tests are attached, significant features are filled and the others
 *        hollow; a bootstrap band is drawn as a dashed line parallel to the diagonal.
 * @throws std::invalid_argument for a document ToJson() rejects, a dimension it doesn't hold, or
 *         unusable options.
 */
[[nodiscard]] std::string RenderPersistenceDiagramSvg(const PersistenceDiagramDocument& doc,
                                                      const PersistenceViewOptions& view = {},
                                                      const SvgOptions& options = {});

/**
 * @brief One bar per feature, from its birth to its death, grouped by dimension and longest first.
 *        Each end has a whisker of ±the certified error bound; essential bars end in an arrow.
 *        Features that failed an attached test are drawn faint.
 * @throws std::invalid_argument as RenderPersistenceDiagramSvg() does, or if max_bars < 1.
 */
[[nodiscard]] std::string RenderBarcodeSvg(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view = {},
                                           const SvgOptions& options = {});

/**
 * @brief The Betti curve of each dimension: how many features are alive at each scale (BettiCurve()).
 * @note The curve isn't covered by the error bound: features near the diagonal can appear or
 *       vanish under a small change in the distances, and the figure says so.
 * @throws std::invalid_argument as RenderPersistenceDiagramSvg() does, or if grid_points < 2.
 */
[[nodiscard]] std::string RenderBettiCurveSvg(const PersistenceDiagramDocument& doc, const PersistenceViewOptions& view = {},
                                              const SvgOptions& options = {});

}  // namespace pulsatrix
