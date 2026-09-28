#include "pulsatrix/viz/saliency_heatmap_view.hpp"

#include <algorithm>
#include <cmath>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void SaliencyHeatmapView::Draw(const char* title, const Attribution& attr) {
    HeatmapGrid grid = ToSaliencyHeatmap(attr);
    if (grid.values.empty()) {
        return;
    }

    float max_abs = 0.0f;
    for (float v : grid.values) {
        max_abs = std::max(max_abs, std::abs(v));
    }
    // A constant-zero attribution (max_abs == 0) would otherwise divide by zero inside
    // ImPlot's internal color-scale normalization.
    double scale_max = max_abs > 0.0f ? static_cast<double>(max_abs) : 1.0;

    ImPlot::PushColormap(ImPlotColormap_Viridis);
    if (ImPlot::BeginPlot(title, ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations,
                           ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);
        ImPlot::PlotHeatmap("##saliency", grid.values.data(), static_cast<int>(grid.rows),
                             static_cast<int>(grid.cols), 0.0, scale_max, nullptr);
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}

}  // namespace pulsatrix
