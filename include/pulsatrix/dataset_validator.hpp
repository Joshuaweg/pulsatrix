/** @file dataset_validator.hpp
 *  @brief Generic per-field descriptive statistics + missingness/outlier detection over any Dataset.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"

namespace pulsatrix {

/** @brief Descriptive statistics for one Sample field position, aggregated across a whole Dataset. */
struct FieldStatistics {
    int64_t count = 0;     ///< Total elements aggregated (across every sample's Tensor for this field).
    float mean = 0.0f;
    float std_dev = 0.0f;  ///< Population standard deviation.
    float min_value = 0.0f;
    float max_value = 0.0f;
};

/** @brief One Dataset's per-field descriptive statistics. */
struct DatasetStatistics {
    std::vector<FieldStatistics> fields;
};

/** @brief One detected data-quality issue: a missing (NaN) value or a statistical outlier. */
struct ValidationIssue {
    int64_t sample_index;
    int64_t field_index;
    std::string description;
};

/**
 * @brief Generic data validation over any Dataset implementation -- descriptive statistics
 *        and missingness/outlier detection only (campaign_exai_dl_library_data_pipeline
 *        Decision Point 5: distributional drift detection and bias/fairness metrics are
 *        explicitly out of scope, named follow-ups for a future campaign).
 * @note Operates purely against the Dataset interface -- no modality-specific code. Field
 *       statistics aggregate every element of every sample's Tensor for that field
 *       position (flattened, not per-position), since field shapes may legitimately vary
 *       in size across samples (e.g. text sequence length).
 */
class DatasetValidator {
public:
    /**
     * @brief Computes per-field descriptive statistics across every sample in dataset.
     * @throws std::invalid_argument if dataset is empty.
     * @throws std::runtime_error if samples have inconsistent field counts (a schema
     *         violation, detected while scanning) -- external boundary: dataset content is
     *         caller-supplied, not an internal invariant.
     */
    [[nodiscard]] static DatasetStatistics ComputeStatistics(const Dataset& dataset);

    /**
     * @brief Scans dataset for missing (NaN) values and statistical outliers.
     * @param stats Per-field baseline statistics (from ComputeStatistics) used for
     *        z-score outlier detection.
     * @param z_score_threshold Elements with |value - mean| / std_dev exceeding this
     *        threshold are flagged as outliers. A field with std_dev == 0 (constant data)
     *        is skipped for outlier detection (z-score is undefined), not flagged.
     * @return Every detected issue, in sample-then-field-then-element scan order.
     */
    [[nodiscard]] static std::vector<ValidationIssue> DetectIssues(const Dataset& dataset,
                                                                    const DatasetStatistics& stats,
                                                                    float z_score_threshold = 3.0f);
};

}  // namespace pulsatrix
