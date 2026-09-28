#include "pulsatrix/viz/dataset_statistics_view.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/dataset_validator.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void DatasetStatisticsView::Draw(const char* title, const Dataset& dataset, int num_bins) {
    if (dataset.size() == 0) {
        return;
    }

    DatasetStatistics stats = DatasetValidator::ComputeStatistics(dataset);
    std::vector<ValidationIssue> issues = DatasetValidator::DetectIssues(dataset, stats);

    ImGui::PushID(title);
    ImGui::Text("%zu samples, %zu field(s), %zu flagged issue(s)", static_cast<size_t>(dataset.size()),
                stats.fields.size(), issues.size());

    for (size_t field_index = 0; field_index < stats.fields.size(); ++field_index) {
        const FieldStatistics& field = stats.fields[field_index];
        ImGui::Separator();
        ImGui::Text("Field %zu -- mean %.4f, std %.4f, min %.4f, max %.4f, n=%lld", field_index,
                    static_cast<double>(field.mean), static_cast<double>(field.std_dev),
                    static_cast<double>(field.min_value), static_cast<double>(field.max_value),
                    static_cast<long long>(field.count));

        HistogramBins bins = ToFieldHistogramBins(dataset, static_cast<int64_t>(field_index), num_bins);
        std::vector<double> centers(bins.counts.size());
        std::vector<double> counts_as_double(bins.counts.size());
        for (size_t i = 0; i < bins.counts.size(); ++i) {
            centers[i] = (bins.bin_edges[i] + bins.bin_edges[i + 1]) / 2.0;
            counts_as_double[i] = static_cast<double>(bins.counts[i]);
        }
        double bin_width = bins.bin_edges.size() > 1 ? (bins.bin_edges[1] - bins.bin_edges[0]) : 1.0;

        std::string plot_id = "##field" + std::to_string(field_index);
        if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, 150))) {
            ImPlot::SetupAxes("Value", "Count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotBars("##hist", centers.data(), counts_as_double.data(), static_cast<int>(centers.size()),
                              bin_width * 0.9);
            ImPlot::EndPlot();
        }
    }
    ImGui::PopID();
}

}  // namespace pulsatrix
