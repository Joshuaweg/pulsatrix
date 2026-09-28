/** @file attribution_waterfall.hpp
 *  @brief Cascading waterfall chart from a baseline to a prediction, for local attribution.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief Draws a waterfall chart cascading from baseline_value to the final prediction
 *        (hc_information_visualization.md SS5: "correct for local explanation with
 *        directional attribution... bars cascade from base rate to prediction; color
 *        encodes direction; length encodes magnitude"). This is the one bar chart in this
 *        module that legitimately does not start at zero -- Tufte's lie-factor rule is
 *        satisfied relative to baseline_value, the meaningful reference point, not zero.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note. All real logic
 *       (cumulative computation) lives in plot_data.hpp's ToWaterfallSteps, which is.
 */
class AttributionWaterfallChart {
public:
    /**
     * @param title ImPlot plot title/id.
     * @param attr A rank-1 Attribution, in the explainer's own feature order.
     * @param baseline_value The starting reference value (e.g. IG's baseline prediction).
     */
    static void Draw(const char* title, const Attribution& attr, float baseline_value);
};

}  // namespace pulsatrix
