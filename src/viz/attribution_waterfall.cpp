#include "pulsatrix/viz/attribution_waterfall.hpp"

#include <algorithm>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {
namespace {

// Registered once (function-local static, thread-safe init) the first time a waterfall is
// drawn -- requires an active ImPlot context, which VizWindow guarantees by the time any
// Draw() call happens. base is fully transparent (a floating bar's invisible "floor"); the
// other two use the same diverging-colormap endpoints as AttributionBarChart, so a positive
// or negative contribution reads identically across every chart type in this module.
ImPlotColormap WaterfallColormap() {
    static const ImPlotColormap kColormap = [] {
        RgbColor pos = DivergingColormap(1.0f);
        RgbColor neg = DivergingColormap(-1.0f);
        ImVec4 cols[3] = {
            ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
            ImVec4(pos.r, pos.g, pos.b, 1.0f),
            ImVec4(neg.r, neg.g, neg.b, 1.0f),
        };
        return ImPlot::AddColormap("pulsatrix_waterfall", cols, 3, true);
    }();
    return kColormap;
}

}  // namespace

void AttributionWaterfallChart::Draw(const char* title, const Attribution& attr, float baseline_value) {
    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, baseline_value);
    if (steps.empty()) {
        return;
    }

    int n = static_cast<int>(steps.size());
    std::vector<float> base(static_cast<size_t>(n));
    std::vector<float> pos_delta(static_cast<size_t>(n));
    std::vector<float> neg_delta(static_cast<size_t>(n));
    std::vector<double> positions(static_cast<size_t>(n));
    std::vector<const char*> label_ptrs(static_cast<size_t>(n));

    float cumulative_prev = baseline_value;
    for (int i = 0; i < n; ++i) {
        size_t idx = static_cast<size_t>(i);
        float delta = steps[idx].delta;
        float cumulative = steps[idx].cumulative;
        base[idx] = std::min(cumulative_prev, cumulative);
        if (delta >= 0.0f) {
            pos_delta[idx] = delta;
            neg_delta[idx] = 0.0f;
        } else {
            neg_delta[idx] = -delta;
            pos_delta[idx] = 0.0f;
        }
        positions[idx] = static_cast<double>(i);
        label_ptrs[idx] = steps[idx].label.c_str();
        cumulative_prev = cumulative;
    }

    std::vector<float> stacked_values;
    stacked_values.reserve(static_cast<size_t>(n) * 3);
    stacked_values.insert(stacked_values.end(), base.begin(), base.end());
    stacked_values.insert(stacked_values.end(), pos_delta.begin(), pos_delta.end());
    stacked_values.insert(stacked_values.end(), neg_delta.begin(), neg_delta.end());

    static const char* kItemLabels[3] = {"##base", "Increase", "Decrease"};

    ImPlot::PushColormap(WaterfallColormap());
    if (ImPlot::BeginPlot(title, ImVec2(-1, 0))) {
        ImPlot::SetupAxes(nullptr, "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, label_ptrs.data());
        ImPlot::PlotBarGroups(kItemLabels, stacked_values.data(), 3, n, 0.67, 0, ImPlotBarGroupsFlags_Stacked);
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}

}  // namespace pulsatrix
