#include "pulsatrix/viz/attribution_beeswarm.hpp"

#include <algorithm>
#include <string>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void AttributionBeeswarmView::Draw(const char* title, const std::vector<Attribution>& attributions_across_samples,
                                    int64_t feature_index) {
    std::vector<BeeswarmPoint> points = ToBeeswarmPoints(attributions_across_samples, feature_index);
    if (points.empty()) {
        return;
    }

    float max_abs = 0.0f;
    for (const BeeswarmPoint& p : points) {
        max_abs = std::max(max_abs, std::abs(p.x));
    }

    if (ImPlot::BeginPlot(title, ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("Attribution", nullptr, ImPlotAxisFlags_AutoFit,
                           ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoDecorations);
        for (size_t i = 0; i < points.size(); ++i) {
            RgbColor color = DivergingColormap(NormalizeSigned(points[i].x, max_abs));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, ImVec4(color.r, color.g, color.b, 0.85f));
            double x = static_cast<double>(points[i].x);
            double y = static_cast<double>(points[i].y);
            std::string series_id = "##point" + std::to_string(i);
            ImPlot::PlotScatter(series_id.c_str(), &x, &y, 1);
        }
        ImPlot::EndPlot();
    }
}

}  // namespace pulsatrix
