#include <vector>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/implot_metrics_sink.hpp"

namespace pulsatrix {

void ImPlotMetricsSink::Draw() const {
    for (const auto& [tag, series] : scalar_series_) {
        if (series.values.empty()) {
            continue;
        }
        if (ImPlot::BeginPlot(tag.c_str(), ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Step", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            std::vector<double> steps_as_double(series.steps.begin(), series.steps.end());
            ImPlot::PlotLine(tag.c_str(), steps_as_double.data(), series.values.data(),
                              static_cast<int>(series.values.size()));
            ImPlot::EndPlot();
        }
    }

    for (const auto& [tag, values] : latest_histograms_) {
        if (values.empty()) {
            continue;
        }
        std::string title = tag + " (histogram)";
        if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Value", "Count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotHistogram(tag.c_str(), values.data(), static_cast<int>(values.size()));
            ImPlot::EndPlot();
        }
    }
}

}  // namespace pulsatrix
