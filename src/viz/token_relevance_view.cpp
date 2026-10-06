#include "pulsatrix/viz/token_relevance_view.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

#include <imgui.h>

#include "pulsatrix/viz/colormap.hpp"

namespace pulsatrix {

namespace {

ImU32 ToImColor(const RgbColor& c, float alpha = 1.0f) { return ImGui::ColorConvertFloat4ToU32(ImVec4(c.r, c.g, c.b, alpha)); }

/** @brief Black or white, whichever reads better on @p c. */
ImU32 TextOn(const RgbColor& c) {
    const float luminance = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
    return luminance > 0.5f ? IM_COL32(0, 0, 0, 255) : IM_COL32(255, 255, 255, 255);
}

}  // namespace

void TokenRelevanceView::Draw(const char* id, const TokenRelevanceDocument& doc, float shared_max_abs) {
    if (doc.tokens.size() != doc.relevance.size()) {
        ImGui::TextUnformatted("token relevance: one score per piece is needed");
        return;
    }
    float max_abs = shared_max_abs;
    if (max_abs <= 0.0f) {
        max_abs = 0.0f;
        for (size_t i = 0; i < doc.relevance.size(); ++i) {
            if (doc.is_scored(i) && std::isfinite(doc.relevance[i])) max_abs = std::max(max_abs, std::abs(doc.relevance[i]));
        }
    }
    ImGui::PushID(id);
    ImGui::Text("%s relevance (%s)", doc.granularity == "word" ? "Word" : "Token", doc.method.c_str());
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float left = ImGui::GetCursorScreenPos().x;
    const float right = left + ImGui::GetContentRegionAvail().x;
    const float line_h = ImGui::GetTextLineHeightWithSpacing() + 2.0f;
    ImVec2 pos = ImGui::GetCursorScreenPos();
    for (size_t i = 0; i < doc.tokens.size(); ++i) {
        std::string_view rest = doc.tokens[i];
        while (true) {
            const size_t nl = rest.find('\n');
            const std::string part(rest.substr(0, nl));
            const ImVec2 size = ImGui::CalcTextSize(part.c_str());
            if (pos.x + size.x > right && pos.x > left) {
                pos = ImVec2(left, pos.y + line_h);
            }
            if (!part.empty()) {
                const ImVec2 end(pos.x + size.x, pos.y + ImGui::GetTextLineHeight());
                ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
                if (doc.is_scored(i)) {
                    const float r = doc.relevance[i];
                    const RgbColor c = std::isfinite(r) ? DivergingColormap(NormalizeSigned(r, max_abs)) : RgbColor{0.74f, 0.74f, 0.74f};
                    draw->AddRectFilled(pos, end, ToImColor(c));
                    text = TextOn(c);
                }
                draw->AddText(pos, text, part.c_str());
                if (ImGui::IsMouseHoveringRect(pos, end)) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(doc.tokens[i].c_str());
                    if (doc.is_scored(i)) {
                        ImGui::Text("relevance %.4g", static_cast<double>(doc.relevance[i]));
                    } else {
                        ImGui::TextUnformatted("(context, not scored)");
                    }
                    ImGui::EndTooltip();
                }
                pos.x += size.x;
            }
            if (nl == std::string_view::npos) break;
            pos = ImVec2(left, pos.y + line_h);
            rest.remove_prefix(nl + 1);
        }
    }
    // Reserve the space the text took, then the target and unassigned lines.
    ImGui::SetCursorScreenPos(ImVec2(left, pos.y + line_h));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    if (!doc.target.empty()) ImGui::Text("explaining \"%s\"", doc.target.c_str());
    if (doc.unassigned && std::isfinite(*doc.unassigned)) {
        ImGui::Text("relevance outside the pieces shown: %.4g", static_cast<double>(*doc.unassigned));
    }
    ImGui::PopID();
}

}  // namespace pulsatrix
