/** @file implot_metrics_sink.hpp
 *  @brief Concrete MetricsSink that buffers training scalars/histograms for live plotting.
 *  @ingroup visualization
 */
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One scalar tag's logged (step, value) pairs, in log order. */
struct ScalarSeries {
    std::vector<int> steps;
    std::vector<double> values;
};

/**
 * @brief A concrete MetricsSink (metrics_sink.hpp's own doc comment names this the expected
 *        extension point: "Concrete writers... implement this") that buffers every logged
 *        scalar into a per-tag time series, and keeps the latest histogram snapshot per tag,
 *        for live rendering by a training dashboard. Zero core changes needed -- MetricsSink*
 *        is already threaded through every training loop in this codebase
 *        (XorNetwork::train_step, MnistConvNet::train_step).
 * @note This header has no ImGui/ImPlot include -- log_scalar()/log_histogram() and the
 *       accessors below are implemented in src/implot_metrics_sink.cpp (unconditional, part
 *       of pulsatrix_core) and are unit-tested directly. Only Draw() (implemented in
 *       src/viz/implot_metrics_sink_draw.cpp, part of the gated pulsatrix_viz target) calls
 *       into ImPlot, and is deliberately not unit-tested -- see
 *       plans/okay-we-have-now-buzzing-moth.md Testing Strategy.
 */
class ImPlotMetricsSink : public MetricsSink {
public:
    void log_scalar(const std::string& tag, double value, int step) override;
    void log_histogram(const std::string& tag, const Tensor& values, int step) override;

    /** @brief Every scalar tag's full logged series so far. */
    [[nodiscard]] const std::unordered_map<std::string, ScalarSeries>& scalar_series() const {
        return scalar_series_;
    }

    /** @brief Every histogram tag's most recently logged snapshot (not a running history --
     *         only the latest values per tag are kept). */
    [[nodiscard]] const std::unordered_map<std::string, std::vector<float>>& latest_histograms() const {
        return latest_histograms_;
    }

    /** @brief Renders one line chart per scalar tag (small multiples) plus the latest
     *         histogram snapshot per tag. Not unit-tested -- see the class-level note. */
    void Draw() const;

private:
    std::unordered_map<std::string, ScalarSeries> scalar_series_;
    std::unordered_map<std::string, std::vector<float>> latest_histograms_;
};

}  // namespace pulsatrix
