#include "pulsatrix/viz/explanation_score_card.hpp"

#include <algorithm>
#include <cmath>

#include <imgui.h>

#include "pulsatrix/viz/attribution_bar_chart.hpp"
#include "pulsatrix/viz/confidence_meter.hpp"

namespace pulsatrix {

void ExplanationScoreCard::Draw(const char* title, const Input& input, ScoreCardScaleContext* shared_scale) {
    ImGui::PushID(title);
    ImGui::BeginChild(title, ImVec2(0, 380), true);
    ImGui::TextUnformatted(title);
    ImGui::Separator();

    float shared_max_abs = -1.0f;
    if (shared_scale != nullptr) {
        const float* data = input.attribution.values.data();
        int64_t n = input.attribution.values.numel();
        for (int64_t i = 0; i < n; ++i) {
            shared_scale->max_abs_attribution = std::max(shared_scale->max_abs_attribution, std::abs(data[i]));
        }
        shared_max_abs = shared_scale->max_abs_attribution;
    }

    // Top row: what was shown, how confident was the model.
    ImGui::Columns(2, nullptr, false);
    ImGui::TextUnformatted(input.input_label.c_str());
    ConfidenceMeter::Draw("Confidence", input.confidence);
    ImGui::NextColumn();
    AttributionBarChart::Draw("##bars", input.attribution, 10, shared_max_abs);
    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Bottom row: is this explanation trustworthy -- a distinct semantic group from the
    // prediction-focused top row (Gestalt proximity), separated by the rule above.
    ImGui::Columns(2, nullptr, false);
    ImGui::Text("Conservation delta: %.4f", static_cast<double>(input.conservation.delta()));
    ImGui::NextColumn();
    if (input.has_stability) {
        ImGui::Text("Stability variance: %.4f", static_cast<double>(input.stability.mean_variance));
    } else {
        ImGui::TextUnformatted("Stability: 0.0 (deterministic)");
    }
    ImGui::Columns(1);

    ImGui::EndChild();
    ImGui::PopID();
}

}  // namespace pulsatrix
