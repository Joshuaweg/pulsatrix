#include "pulsatrix/viz/confidence_meter.hpp"

#include <algorithm>
#include <cstdio>

#include <imgui.h>

namespace pulsatrix {

void ConfidenceMeter::Draw(const char* label, float confidence_0_to_1) {
    float clamped = std::clamp(confidence_0_to_1, 0.0f, 1.0f);
    char overlay[32];
    std::snprintf(overlay, sizeof(overlay), "%.1f%%", static_cast<double>(clamped) * 100.0);

    ImGui::TextUnformatted(label);
    ImGui::ProgressBar(clamped, ImVec2(-1, 0), overlay);
}

}  // namespace pulsatrix
