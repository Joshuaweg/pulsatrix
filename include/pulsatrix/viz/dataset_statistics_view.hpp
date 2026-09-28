/** @file dataset_statistics_view.hpp
 *  @brief Per-field distribution histograms + data-quality issue panel for a Dataset.
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/dataset.hpp"

namespace pulsatrix {

/**
 * @brief Draws one histogram per Sample field (small multiples,
 *        hc_information_visualization.md SS6) plus a flagged-issue count, using
 *        DatasetValidator's existing production API (ComputeStatistics/DetectIssues) --
 *        no new core code needed, this is purely a consumer of data that already exists.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note. Binning itself is
 *       plot_data.hpp's ToFieldHistogramBins, which is unit-tested.
 */
class DatasetStatisticsView {
public:
    /**
     * @param title A unique ImGui child id for this view.
     * @param dataset The dataset to summarize. Must be non-empty.
     * @param num_bins Histogram bin count per field.
     */
    static void Draw(const char* title, const Dataset& dataset, int num_bins = 20);
};

}  // namespace pulsatrix
