#include "pulsatrix/viz/image_grid_view.hpp"

#include <algorithm>
#include <cstdint>

#include <imgui.h>

namespace pulsatrix {

void ImageGridView::Draw(const char* title, const Dataset& dataset, TextureCache& cache, int64_t start_index,
                          int64_t count, int columns, float thumbnail_size,
                          const std::vector<std::string>* captions) {
    int64_t size = dataset.size();
    if (size == 0 || start_index >= size) {
        return;
    }
    int64_t end_index = std::min(start_index + count, size);

    ImGui::PushID(title);
    for (int64_t i = start_index; i < end_index; ++i) {
        Sample sample = dataset.get(i);
        if (sample.fields.empty()) {
            continue;
        }
        TextureId texture = cache.GetOrUpload(i, sample.fields[0]);
        ImGui::BeginGroup();
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(texture)),
                     ImVec2(thumbnail_size, thumbnail_size));
        size_t caption_index = static_cast<size_t>(i - start_index);
        if (captions != nullptr && caption_index < captions->size()) {
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + thumbnail_size);
            ImGui::TextUnformatted((*captions)[caption_index].c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndGroup();

        bool last_in_row = ((i - start_index + 1) % columns) == 0;
        bool last_overall = (i + 1) == end_index;
        if (!last_in_row && !last_overall) {
            ImGui::SameLine();
        }
    }
    ImGui::PopID();
}

}  // namespace pulsatrix
