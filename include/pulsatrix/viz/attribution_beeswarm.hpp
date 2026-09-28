/** @file attribution_beeswarm.hpp
 *  @brief Beeswarm plot of one feature's attribution distribution across many samples.
 *  @ingroup visualization
 */
#pragma once

#include <vector>

#include "pulsatrix/attribution.hpp"

namespace pulsatrix {

/**
 * @brief Draws a beeswarm plot for one feature's attribution values across many
 *        samples/predictions -- the global "how does the model behave overall" view
 *        (hc_information_visualization.md SS5), complementing AttributionBarChart's
 *        single-prediction "why this one" view.
 * @note ImPlot has no native beeswarm primitive -- the jitter/collision-avoidance layout is
 *       computed by plot_data.hpp's ToBeeswarmPoints (unit-tested); this draw call only
 *       plots already-jittered points. Colors by the attribution value's own sign/magnitude
 *       (DivergingColormap) rather than the original feature value, since Attribution does
 *       not carry the input feature value alongside its attribution -- a real limitation
 *       relative to a canonical SHAP beeswarm's color encoding, not an oversight.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note.
 */
class AttributionBeeswarmView {
public:
    /**
     * @param title ImPlot plot title/id.
     * @param attributions_across_samples One Attribution per sample/prediction.
     * @param feature_index Which feature (element position) to plot.
     */
    static void Draw(const char* title, const std::vector<Attribution>& attributions_across_samples,
                      int64_t feature_index);
};

}  // namespace pulsatrix
