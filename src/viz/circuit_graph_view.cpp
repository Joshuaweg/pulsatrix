#include "pulsatrix/viz/circuit_graph_view.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/viz/colormap.hpp"

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
        ImPlot::SetupAxes("Depth", nullptr, ImPlotAxisFlags_AutoFit,
                           ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoDecorations);

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

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, marker_size, ImVec4(color.r, color.g, color.b, 1.0f),
                                        1.0f);
            std::string series_id = "##node" + std::to_string(i);
            ImPlot::PlotScatter(series_id.c_str(), &x, &y, 1);

            std::string label = node.label.value_or("node_" + std::to_string(node.id));
            ImPlot::PlotText(label.c_str(), x, y, ImVec2(0, -20));
        }
        ImPlot::EndPlot();
    }
}

}  // namespace pulsatrix
