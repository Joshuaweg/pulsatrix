/** @file saliency_heatmap_view.hpp
 *  @brief Saliency/attribution heatmap using a perceptually uniform colormap.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief Draws a 2D saliency heatmap with Viridis (perceptually uniform, colorblind-safe --
 *        hc_information_visualization.md SS4) for unsigned magnitude.
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
     * @param attr A rank-2 (or single-channel rank-3) Attribution.
     */
    static void Draw(const char* title, const Attribution& attr);
};

}  // namespace pulsatrix
