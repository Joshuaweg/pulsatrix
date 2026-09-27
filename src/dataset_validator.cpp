#include "pulsatrix/dataset_validator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pulsatrix {

namespace {
void AccumulateSample(const Sample& sample, size_t field_count, std::vector<int64_t>& counts,
                       std::vector<double>& sums, std::vector<double>& sum_squares, std::vector<float>& mins,
                       std::vector<float>& maxs) {
    if (sample.fields.size() != field_count) {
        throw std::runtime_error(
            "DatasetValidator::ComputeStatistics: inconsistent field count across samples (schema violation)");
    }
    for (size_t f = 0; f < field_count; ++f) {
        const Tensor& tensor = sample.fields[f];
        int64_t n = tensor.numel();
        for (int64_t i = 0; i < n; ++i) {
            float value = tensor.data()[i];
            counts[f] += 1;
            sums[f] += value;
            sum_squares[f] += static_cast<double>(value) * static_cast<double>(value);
            mins[f] = std::min(mins[f], value);
            maxs[f] = std::max(maxs[f], value);
        }
    }
}
}  // namespace

DatasetStatistics DatasetValidator::ComputeStatistics(const Dataset& dataset) {
    int64_t size = dataset.size();
    if (size <= 0) {
        throw std::invalid_argument("DatasetValidator::ComputeStatistics: dataset must not be empty");
    }

    Sample first = dataset.get(0);
    size_t field_count = first.fields.size();

    std::vector<int64_t> counts(field_count, 0);
    std::vector<double> sums(field_count, 0.0);
    std::vector<double> sum_squares(field_count, 0.0);
    std::vector<float> mins(field_count, std::numeric_limits<float>::max());
    std::vector<float> maxs(field_count, std::numeric_limits<float>::lowest());

    AccumulateSample(first, field_count, counts, sums, sum_squares, mins, maxs);
    for (int64_t i = 1; i < size; ++i) {
        AccumulateSample(dataset.get(i), field_count, counts, sums, sum_squares, mins, maxs);
    }

    DatasetStatistics stats;
    stats.fields.resize(field_count);
    for (size_t f = 0; f < field_count; ++f) {
        FieldStatistics& field_stats = stats.fields[f];
        field_stats.count = counts[f];
        if (counts[f] > 0) {
            double mean = sums[f] / static_cast<double>(counts[f]);
            double variance = sum_squares[f] / static_cast<double>(counts[f]) - mean * mean;
            variance = std::max(variance, 0.0);  // guards tiny negative values from floating-point error
            field_stats.mean = static_cast<float>(mean);
            field_stats.std_dev = static_cast<float>(std::sqrt(variance));
            field_stats.min_value = mins[f];
            field_stats.max_value = maxs[f];
        }
    }
    return stats;
}

std::vector<ValidationIssue> DatasetValidator::DetectIssues(const Dataset& dataset, const DatasetStatistics& stats,
                                                             float z_score_threshold) {
    std::vector<ValidationIssue> issues;
    int64_t size = dataset.size();

    for (int64_t sample_index = 0; sample_index < size; ++sample_index) {
        Sample sample = dataset.get(sample_index);
        for (size_t f = 0; f < sample.fields.size(); ++f) {
            const Tensor& tensor = sample.fields[f];
            const FieldStatistics* field_stats = f < stats.fields.size() ? &stats.fields[f] : nullptr;

            for (int64_t i = 0; i < tensor.numel(); ++i) {
                float value = tensor.data()[i];
                if (std::isnan(value)) {
                    issues.push_back({sample_index, static_cast<int64_t>(f), "missing value (NaN)"});
                    continue;
                }
                if (field_stats != nullptr && field_stats->std_dev > 0.0f) {
                    float z_score = std::abs(value - field_stats->mean) / field_stats->std_dev;
                    if (z_score > z_score_threshold) {
                        issues.push_back({sample_index, static_cast<int64_t>(f),
                                           "outlier: z-score " + std::to_string(z_score) + " exceeds threshold " +
                                               std::to_string(z_score_threshold)});
                    }
                }
            }
        }
    }
    return issues;
}

}  // namespace pulsatrix
