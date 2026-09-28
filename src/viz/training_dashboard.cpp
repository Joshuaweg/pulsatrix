#include "pulsatrix/viz/training_dashboard.hpp"

#include <algorithm>

#include <imgui.h>

namespace pulsatrix {

void TrainingDashboard::Draw(const ImPlotMetricsSink& sink) {
    for (const auto& [tag, series] : sink.scalar_series()) {
        if (series.values.empty()) {
            continue;
        }
        double last = series.values.back();
        double min_value = *std::min_element(series.values.begin(), series.values.end());
        double max_value = *std::max_element(series.values.begin(), series.values.end());
        ImGui::Text("%s -- step %d, last %.4f, min %.4f, max %.4f", tag.c_str(), series.steps.back(), last,
                    min_value, max_value);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    sink.Draw();
}

}  // namespace pulsatrix
