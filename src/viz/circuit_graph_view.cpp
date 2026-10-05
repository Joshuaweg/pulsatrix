#include "pulsatrix/viz/circuit_graph_view.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void CircuitGraphView::Draw(const char* title, const CircuitGraph& graph) {
    const std::vector<CircuitNode>& nodes = graph.nodes();
    const std::vector<CircuitEdge>& edges = graph.edges();
    if (nodes.empty()) {
        return;
    }

    std::unordered_map<NodeId, int> index_of;
    for (size_t i = 0; i < nodes.size(); ++i) {
        index_of[nodes[i].id] = static_cast<int>(i);
    }

    float max_ablation = 0.0f;
    for (const CircuitNode& node : nodes) {
        max_ablation = std::max(max_ablation, node.ablation_effect);
    }
    float max_weight = 0.0f;
    for (const CircuitEdge& edge : edges) {
        max_weight = std::max(max_weight, edge.weight);
    }

    if (ImPlot::BeginPlot(title, ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        // Fixed limits: every node sits at y = 0, and AutoFit on a constant series pins that
        // row to the plot border (markers clipped in half); a node at the end of the row would
        // likewise sit on the x border.
        ImPlot::SetupAxes("Depth", nullptr, 0, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxesLimits(-0.5, static_cast<double>(nodes.size()) - 0.5, -1.0, 1.0, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0.0, static_cast<double>(nodes.size()) - 1.0, static_cast<int>(nodes.size()));

        // Edges first so node markers draw on top of them.
        for (const CircuitEdge& edge : edges) {
            auto from_it = index_of.find(edge.from);
            auto to_it = index_of.find(edge.to);
            if (from_it == index_of.end() || to_it == index_of.end()) {
                continue;
            }
            double xs[2] = {static_cast<double>(from_it->second), static_cast<double>(to_it->second)};
            double ys[2] = {0.0, 0.0};
            float thickness = 1.0f + 4.0f * NormalizeUnsigned(edge.weight, max_weight);
            ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), thickness);
            ImPlot::PlotLine("##edge", xs, ys, 2);
        }

        for (size_t i = 0; i < nodes.size(); ++i) {
            const CircuitNode& node = nodes[i];
            float normalized = NormalizeUnsigned(node.ablation_effect, max_ablation);
            RgbColor color = ViridisColormap(normalized);
            float marker_size = 6.0f + 14.0f * normalized;
            double x = static_cast<double>(i);
            double y = 0.0;

            // Outline set explicitly: otherwise ImPlot outlines each one-point series in the next
            // auto colormap color, adding a meaningless categorical hue around every node.
            ImVec4 fill(color.r, color.g, color.b, 1.0f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, marker_size, fill, 1.0f, fill);
            std::string series_id = "##node" + std::to_string(i);
            ImPlot::PlotScatter(series_id.c_str(), &x, &y, 1);

            std::string label = CircuitNodeDisplayLabel(node);
            ImPlot::PlotText(label.c_str(), x, y, ImVec2(0, -(marker_size + 12.0f)));
            char effect[32];
            std::snprintf(effect, sizeof(effect), "%.3g", static_cast<double>(node.ablation_effect));
            ImPlot::PlotText(effect, x, y, ImVec2(0, marker_size + 12.0f));
        }
        ImPlot::EndPlot();
    }
}

void CircuitGraphView::Draw(const char* title, const CircuitGraphDocument& doc) { Draw(title, ToCircuitGraph(doc)); }

}  // namespace pulsatrix
