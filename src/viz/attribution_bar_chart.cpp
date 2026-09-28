#include "pulsatrix/viz/attribution_bar_chart.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void AttributionBarChart::Draw(const char* title, const Attribution& attr, int top_k, float shared_max_abs) {
    BarSeries bars = ToFeatureImportanceBars(attr, top_k);
    if (bars.values.empty()) {
        return;
    }

    float max_abs = shared_max_abs;
    if (max_abs <= 0.0f) {
        max_abs = 0.0f;
        for (float v : bars.values) {
            max_abs = std::max(max_abs, std::abs(v));
        }
    }

    int n = static_cast<int>(bars.values.size());
    std::vector<double> positions(static_cast<size_t>(n));
    std::vector<const char*> label_ptrs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<double>(i);
        label_ptrs[static_cast<size_t>(i)] = bars.labels[static_cast<size_t>(i)].c_str();
    }

    if (ImPlot::BeginPlot(title, ImVec2(-1, 0))) {
        ImPlot::SetupAxes("Attribution", nullptr, ImPlotAxisFlags_AutoFit,
                           ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Invert);
        ImPlot::SetupAxisTicks(ImAxis_Y1, positions.data(), n, label_ptrs.data());

        for (int i = 0; i < n; ++i) {
            RgbColor color = DivergingColormap(NormalizeSigned(bars.values[static_cast<size_t>(i)], max_abs));
            ImPlot::SetNextFillStyle(ImVec4(color.r, color.g, color.b, 1.0f));
            float value = bars.values[static_cast<size_t>(i)];
            std::string series_id = "##bar" + std::to_string(i);
            ImPlot::PlotBars(series_id.c_str(), &value, 1, 0.6, static_cast<double>(i), ImPlotBarsFlags_Horizontal);
        }
        ImPlot::EndPlot();
    }
}

}  // namespace pulsatrix
