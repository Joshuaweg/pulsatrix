#include "pulsatrix/viz/implot_metrics_sink.hpp"

namespace pulsatrix {

void ImPlotMetricsSink::log_scalar(const std::string& tag, double value, int step) {
    ScalarSeries& series = scalar_series_[tag];
    series.steps.push_back(step);
    series.values.push_back(value);
}

void ImPlotMetricsSink::log_histogram(const std::string& tag, const Tensor& values, int /*step*/) {
    latest_histograms_[tag] = std::vector<float>(values.data(), values.data() + values.numel());
}

}  // namespace pulsatrix
