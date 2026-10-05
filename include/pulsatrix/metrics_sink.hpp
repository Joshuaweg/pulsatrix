/** @file metrics_sink.hpp
 *  @brief Keeps monitoring/visualization tools out of the training core -- same OCP/DIP
 *         pattern as DeviceBackend/ExplainerContext.
 *  @ingroup dl_modules
 */
#pragma once

#include <string>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Interface the training loop logs scalars/histograms through. Concrete writers
 *        (TensorBoard event format, W&B, CSV, ...) implement this; the training loop and
 *        Phase 4 validation harness only ever see MetricsSink.
 * @note Concrete sinks: NoOpMetricsSink below, and ImPlotMetricsSink
 *       (viz/implot_metrics_sink.hpp), whose log can be saved as a `pulsatrix.training_log.v1`
 *       JSON document and replayed into any sink (viz/document.hpp).
 */
class MetricsSink {
public:
    virtual ~MetricsSink() = default;

    /**
     * @brief Logs a scalar value (e.g. loss, accuracy).
     * @param tag Metric name.
     * @param value Metric value.
     * @param step Training step this value corresponds to.
     */
    virtual void log_scalar(const std::string& tag, double value, int step) = 0;

    /**
     * @brief Logs a distribution of values (e.g. a weight tensor's values).
     * @param tag Metric name.
     * @param values Values to log.
     * @param step Training step this snapshot corresponds to.
     */
    virtual void log_histogram(const std::string& tag, const Tensor& values, int step) = 0;
};

/** @brief Does nothing. The charter's stated minimum viable MetricsSink implementation. */
class NoOpMetricsSink : public MetricsSink {
public:
    void log_scalar(const std::string&, double, int) override {}
    void log_histogram(const std::string&, const Tensor&, int) override {}
};

}  // namespace pulsatrix
