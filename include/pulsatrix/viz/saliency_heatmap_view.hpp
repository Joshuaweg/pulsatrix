/** @file saliency_heatmap_view.hpp
 *  @brief Saliency/attribution heatmap using a perceptually uniform colormap.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief Draws a 2D saliency heatmap plus a colormap scale bar. Unsigned magnitudes (e.g.
 *        Grad-CAM) use Viridis (perceptually uniform, colorblind-safe --
 *        hc_information_visualization.md SS4) over [0, max]; signed attributions (any
 *        negative value -- gradients, IG, LRP, LIME, SHAP) use the blue-white-red
 *        DivergingColormap over the symmetric range [-max|v|, +max|v|], so zero is always the
 *        neutral midpoint. The choice is ComputeHeatmapColorScale's (plot_data.hpp, unit-tested).
 *        Row 0 of the grid is drawn at the top (image convention) with square cells.
 * @note v1 scope: the standalone heatmap only. Overlaying it atop a source image (the
 *       design doc's "overlay heatmap... sufficient figure/ground contrast" guidance)
 *       needs an OpenGL texture upload path -- deferred to Phase C's TextureCache
 *       infrastructure (see plans/okay-we-have-now-buzzing-moth.md), not built here.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note. All real logic
 *       (reshaping) lives in plot_data.hpp's ToSaliencyHeatmap, which is.
 */
class SaliencyHeatmapView {
public:
    /**
     * @param title ImPlot plot title/id.
     * @param attr A rank-2, single-channel rank-3, or (1, 1, H, W) rank-4 Attribution --
     *        see ToSaliencyHeatmap. Fills the available content region (minus the scale bar).
     */
    static void Draw(const char* title, const Attribution& attr);
};

}  // namespace pulsatrix
