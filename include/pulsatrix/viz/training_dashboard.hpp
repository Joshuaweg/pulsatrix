/** @file training_dashboard.hpp
 *  @brief Live training dashboard: per-tag run summary plus ImPlotMetricsSink's charts.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/viz/implot_metrics_sink.hpp"

namespace pulsatrix {

/**
 * @brief Draws a live training dashboard: a data-ink-minimal per-tag summary line
 *        (last/min/max value, current step -- Tufte SS6: no chartjunk, just the numbers
 *        that matter) followed by ImPlotMetricsSink's line charts and histogram snapshots.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note.
 */
class TrainingDashboard {
public:
    static void Draw(const ImPlotMetricsSink& sink);
};

}  // namespace pulsatrix
