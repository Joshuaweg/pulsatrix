/** @file attribution_bar_chart.hpp
 *  @brief Horizontal feature-importance bar chart for an Attribution.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief Draws a horizontal bar chart of an Attribution's top-k features
 *        (hc_information_visualization.md SS1/SS5: position-on-shared-axis is the most
 *        accurately perceived encoding for magnitude -- Cleveland-McGill rank 1 -- and
 *        horizontal bars read naturally with long feature names). Bar length encodes
 *        magnitude; hue (DivergingColormap) encodes sign, never magnitude. Bars start at
 *        zero (Tufte lie-factor = 1).
 * @note Must be called between ImPlot::BeginPlot()/EndPlot() -- no, this owns its own
 *       BeginPlot/EndPlot pair; call directly from an ImGui::Begin/End window body.
 *       Deliberately not unit-tested (GoogleTest cannot meaningfully exercise ImPlot draw
 *       calls) -- see plans/okay-we-have-now-buzzing-moth.md Testing Strategy. All real
 *       logic (sorting, normalization) lives in plot_data.hpp/colormap.hpp, which are.
 */
class AttributionBarChart {
public:
    /**
     * @param title ImPlot plot title/id.
     * @param attr A rank-1 Attribution.
     * @param top_k Number of highest-|value| features to display.
     * @param shared_max_abs When > 0, the color/scale normalization denominator to use
     *        instead of this chart's own max |value| -- lets a caller showing several
     *        ExplanationScoreCards at once (small multiples) keep every card's color scale
     *        comparable (hc_information_visualization.md SS6: "small multiples... using
     *        shared axes"), rather than each auto-scaling independently and silently making
     *        cross-card comparison invalid.
     */
    static void Draw(const char* title, const Attribution& attr, int top_k = 10, float shared_max_abs = -1.0f);
};

}  // namespace pulsatrix
