/** @file trial.hpp
 *  @brief A single hyperparameter-optimization trial: the configuration tried, plus every
 *         metric value recorded against it.
 *  @ingroup hyperparameter_optimization
 */
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/search_space.hpp"

namespace pulsatrix {

/** @brief One recorded metric value: a named tag, its value, and the training step it was
 *         logged at (mirrors MetricsSink::log_scalar's own (tag, value, step) shape). */
struct MetricRecord {
    std::string tag;
    double value;
    int step;
};

/**
 * @brief A configuration paired with the metric history observed while evaluating it
 *        (instantiate a network under this configuration, train it, record whatever metrics
 *        the training loop reports). Deliberately decoupled from SearchSpace -- a Trial
 *        records what actually happened, a SearchSpace describes what could be proposed;
 *        neither needs to reference the other directly.
 */
class Trial {
public:
    explicit Trial(Configuration configuration) : configuration_(std::move(configuration)) {}

    /** @brief The hyperparameter configuration this trial evaluates. */
    [[nodiscard]] const Configuration& configuration() const { return configuration_; }

    /** @brief Records one metric observation. */
    void RecordMetric(std::string tag, double value, int step) {
        metrics_.push_back(MetricRecord{std::move(tag), value, step});
    }

    /** @brief Every metric recorded, in recording order. */
    [[nodiscard]] const std::vector<MetricRecord>& metrics() const { return metrics_; }

    /**
     * @brief The most recently recorded value for tag (by recording order, not by step
     *        number -- a caller recording out of step order gets the last call's value).
     * @return std::nullopt if tag was never recorded.
     */
    [[nodiscard]] std::optional<double> LatestMetric(const std::string& tag) const {
        for (auto it = metrics_.rbegin(); it != metrics_.rend(); ++it) {
            if (it->tag == tag) {
                return it->value;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief The best value recorded for tag.
     * @param maximize If true, "best" means largest; if false, smallest (e.g. a loss metric
     *        should pass maximize = false).
     * @return std::nullopt if tag was never recorded.
     */
    [[nodiscard]] std::optional<double> BestMetric(const std::string& tag, bool maximize) const {
        std::optional<double> best;
        for (const auto& record : metrics_) {
            if (record.tag != tag) {
                continue;
            }
            if (!best.has_value() || (maximize ? record.value > *best : record.value < *best)) {
                best = record.value;
            }
        }
        return best;
    }

private:
    Configuration configuration_;
    std::vector<MetricRecord> metrics_;
};

}  // namespace pulsatrix
