#include "pulsatrix/viz/saliency_heatmap_view.hpp"

#include <algorithm>
#include <array>
#include <string>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {
namespace {

// ImPlot has no built-in blue-white-red map matching DivergingColormap, so register one
// (sampled from the same anchors, interpolated -- qual=false) once per ImPlot context.
ImPlotColormap SignedHeatmapColormap() {
    ImPlotColormap existing = ImPlot::GetColormapIndex("pulsatrix_diverging");
    if (existing != -1) {
        return existing;
    }
    constexpr int kSamples = 11;
    std::array<ImVec4, kSamples> cols{};
    for (int i = 0; i < kSamples; ++i) {
        float t = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(kSamples - 1);
        RgbColor c = DivergingColormap(t);
        cols[static_cast<size_t>(i)] = ImVec4(c.r, c.g, c.b, 1.0f);
    }
    return ImPlot::AddColormap("pulsatrix_diverging", cols.data(), kSamples, false);
}

void DrawGrid(const char* title, const HeatmapGrid& grid) {
    if (grid.values.empty()) {
        return;
    }

    // Signed attributions (gradients, IG, LRP, LIME, SHAP) use the diverging map centred on
    // zero; unsigned magnitudes (Grad-CAM after its ReLU) use Viridis over [0, max] -- see
    // ComputeHeatmapColorScale.
    HeatmapColorScale scale = ComputeHeatmapColorScale(grid);
    ImPlotColormap cmap = scale.is_signed ? SignedHeatmapColormap() : ImPlotColormap_Viridis;

    constexpr float kScaleBarWidth = 70.0f;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float plot_width = std::max(avail.x - kScaleBarWidth - ImGui::GetStyle().ItemSpacing.x, 50.0f);
    float plot_height = std::max(avail.y, 50.0f);
    // Size the plot to the grid's aspect ratio (square cells) so the frame hugs the image
    // instead of leaving empty bands around it.
    const float aspect = static_cast<float>(grid.rows) / static_cast<float>(grid.cols);
    if (plot_width * aspect <= plot_height) {
        plot_height = plot_width * aspect;
    } else {
        plot_width = plot_height / aspect;
    }

    ImPlot::PushColormap(cmap);
    // No ImPlotAxisFlags_Invert: PlotHeatmap already draws row 0 at the top of a default
    // (non-inverted) y axis, so inverting it would render the image upside down.
    if (ImPlot::BeginPlot(title, ImVec2(plot_width, plot_height),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_Equal)) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_AutoFit);
        // Bounds in grid units (cols x rows) so ImPlotFlags_Equal keeps each cell square.
        ImPlot::PlotHeatmap("##saliency", grid.values.data(), static_cast<int>(grid.rows),
                            static_cast<int>(grid.cols), static_cast<double>(scale.scale_min),
                            static_cast<double>(scale.scale_max), nullptr, ImPlotPoint(0, 0),
                            ImPlotPoint(static_cast<double>(grid.cols), static_cast<double>(grid.rows)));
        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    std::string scale_id = std::string("##scale_") + title;
    ImPlot::ColormapScale(scale_id.c_str(), static_cast<double>(scale.scale_min),
                          static_cast<double>(scale.scale_max), ImVec2(kScaleBarWidth, plot_height), "%.2g");
    ImPlot::PopColormap();
}

}  // namespace

void SaliencyHeatmapView::Draw(const char* title, const Attribution& attr) { DrawGrid(title, ToSaliencyHeatmap(attr)); }

void SaliencyHeatmapView::Draw(const char* title, const HeatmapDocument& doc) { DrawGrid(title, ToHeatmapGrid(doc)); }

}  // namespace pulsatrix
