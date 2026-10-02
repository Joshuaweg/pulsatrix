#include "pulsatrix/viz/attribution_beeswarm.hpp"

#include <algorithm>
#include <cmath>
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
    float min_x = points.front().x;
    float max_x = points.front().x;
    float max_abs_y = 0.0f;
    for (const BeeswarmPoint& p : points) {
        max_abs = std::max(max_abs, std::abs(p.x));
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        max_abs_y = std::max(max_abs_y, std::abs(p.y));
    }
    // Explicit, padded limits instead of AutoFit: AutoFit puts the extreme points exactly on
    // the plot border, clipping their markers in half.
    double x_pad = std::max(0.05 * static_cast<double>(max_x - min_x), 1e-6 + 0.05 * static_cast<double>(max_abs));
    double y_extent = static_cast<double>(max_abs_y) + 0.3;

    if (ImPlot::BeginPlot(title, ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("Attribution", nullptr, 0, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(static_cast<double>(min_x) - x_pad, static_cast<double>(max_x) + x_pad, -y_extent,
                                y_extent, ImPlotCond_Always);
        for (size_t i = 0; i < points.size(); ++i) {
            RgbColor color = DivergingColormap(NormalizeSigned(points[i].x, max_abs));
            // Outline set explicitly too -- otherwise ImPlot draws each one-point series' outline
            // in the next auto colormap color, so points read as arbitrary categorical hues.
            ImVec4 fill(color.r, color.g, color.b, 0.9f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4.0f, fill, 1.0f, fill);
            double x = static_cast<double>(points[i].x);
            double y = static_cast<double>(points[i].y);
            std::string series_id = "##point" + std::to_string(i);
            ImPlot::PlotScatter(series_id.c_str(), &x, &y, 1);
        }
        ImPlot::EndPlot();
    }
}

}  // namespace pulsatrix
