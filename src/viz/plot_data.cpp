#include "pulsatrix/viz/plot_data.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace pulsatrix {
namespace {

std::vector<std::string> SplitCsv(const std::string& csv) {
    std::vector<std::string> parts;
    std::stringstream stream(csv);
    std::string item;
    while (std::getline(stream, item, ',')) {
        parts.push_back(item);
    }
    return parts;
}

// Every native/surrogate explainer in this codebase is always-batched (campaign_exai_dl_
// library_batch_dimension_support), so a single-prediction Attribution is shape
// (1, num_features), not (num_features,) -- accept both rank-1 and batch-1 rank-2, so
// callers don't have to squeeze the batch dimension themselves before calling into this
// module. A rank-2 tensor with batch != 1 (e.g. Grad-CAM's spatial (H, W) map) is
// deliberately still rejected -- it isn't a per-feature vector, batch-1 or otherwise.
void ValidateFeatureVectorShape(const Shape& shape, const char* caller) {
    bool is_rank1 = shape.rank() == 1;
    bool is_batch_one_rank2 = shape.rank() == 2 && shape.dim(0) == 1;
    if (!is_rank1 && !is_batch_one_rank2) {
        throw std::invalid_argument(std::string(caller) +
                                     ": attr.values must be rank 1, or rank 2 with a single leading batch dimension");
    }
}

std::vector<std::string> FeatureLabels(const Attribution& attr, int64_t count) {
    auto it = attr.metadata.find("feature_names");
    if (it != attr.metadata.end()) {
        return SplitCsv(it->second);
    }
    std::vector<std::string> labels;
    labels.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        labels.push_back("feature_" + std::to_string(i));
    }
    return labels;
}

}  // namespace

BarSeries ToFeatureImportanceBars(const Attribution& attr, int top_k) {
    ValidateFeatureVectorShape(attr.values.shape(), "ToFeatureImportanceBars");
    if (top_k <= 0) {
        throw std::invalid_argument("ToFeatureImportanceBars: top_k must be positive");
    }

    int64_t n = attr.values.numel();
    std::vector<std::string> labels = FeatureLabels(attr, n);
    const float* data = attr.values.data();

    std::vector<int64_t> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), int64_t{0});
    std::stable_sort(indices.begin(), indices.end(),
                      [&](int64_t a, int64_t b) { return std::abs(data[a]) > std::abs(data[b]); });

    int64_t count = std::min<int64_t>(top_k, n);
    BarSeries bars;
    bars.labels.reserve(static_cast<size_t>(count));
    bars.values.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        int64_t idx = indices[static_cast<size_t>(i)];
        bars.labels.push_back(idx < static_cast<int64_t>(labels.size()) ? labels[static_cast<size_t>(idx)]
                                                                         : ("feature_" + std::to_string(idx)));
        bars.values.push_back(data[idx]);
    }
    return bars;
}

std::vector<WaterfallStep> ToWaterfallSteps(const Attribution& attr, float baseline_value) {
    ValidateFeatureVectorShape(attr.values.shape(), "ToWaterfallSteps");

    int64_t n = attr.values.numel();
    std::vector<std::string> labels = FeatureLabels(attr, n);
    const float* data = attr.values.data();

    std::vector<WaterfallStep> steps;
    steps.reserve(static_cast<size_t>(n));
    float cumulative = baseline_value;
    for (int64_t i = 0; i < n; ++i) {
        cumulative += data[i];
        steps.push_back(WaterfallStep{labels[static_cast<size_t>(i)], data[i], cumulative});
    }
    return steps;
}

HeatmapGrid ToSaliencyHeatmap(const Attribution& attr) {
    const Shape& shape = attr.values.shape();
    int64_t rows;
    int64_t cols;
    if (shape.rank() == 2) {
        rows = shape.dim(0);
        cols = shape.dim(1);
    } else if (shape.rank() == 3 && shape.dim(0) == 1) {
        rows = shape.dim(1);
        cols = shape.dim(2);
    } else {
        throw std::invalid_argument(
            "ToSaliencyHeatmap: attr.values must be rank 2, or rank 3 with a single leading channel");
    }

    HeatmapGrid grid;
    grid.rows = rows;
    grid.cols = cols;
    grid.values.assign(attr.values.data(), attr.values.data() + attr.values.numel());
    return grid;
}

RgbImageBuffer ToRgbImageBuffer(const Tensor& image_chw) {
    const Shape& shape = image_chw.shape();
    int64_t channels;
    int64_t height;
    int64_t width;
    if (shape.rank() == 3) {
        channels = shape.dim(0);
        height = shape.dim(1);
        width = shape.dim(2);
    } else if (shape.rank() == 4 && shape.dim(0) == 1) {
        channels = shape.dim(1);
        height = shape.dim(2);
        width = shape.dim(3);
    } else {
        throw std::invalid_argument(
            "ToRgbImageBuffer: image_chw must be rank 3 (C,H,W), or rank 4 with a single leading batch dimension");
    }
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("ToRgbImageBuffer: image_chw must have 1 or 3 channels");
    }

    RgbImageBuffer buffer;
    buffer.height = height;
    buffer.width = width;
    buffer.pixels.resize(static_cast<size_t>(height * width * 3));

    const float* data = image_chw.data();
    int64_t plane_size = height * width;
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            int64_t pixel_offset = y * width + x;
            for (int64_t out_channel = 0; out_channel < 3; ++out_channel) {
                int64_t src_channel = (channels == 1) ? 0 : out_channel;
                float value = std::clamp(data[src_channel * plane_size + pixel_offset], 0.0f, 1.0f);
                buffer.pixels[static_cast<size_t>(pixel_offset * 3 + out_channel)] =
                    static_cast<unsigned char>(value * 255.0f);
            }
        }
    }
    return buffer;
}

std::vector<BeeswarmPoint> ToBeeswarmPoints(const std::vector<Attribution>& runs, int64_t feature_index) {
    if (runs.empty()) {
        throw std::invalid_argument("ToBeeswarmPoints: runs must be non-empty");
    }

    std::vector<float> xs;
    xs.reserve(runs.size());
    for (const Attribution& run : runs) {
        if (feature_index < 0 || feature_index >= run.values.numel()) {
            throw std::out_of_range("ToBeeswarmPoints: feature_index out of range for a run's Attribution");
        }
        xs.push_back(run.values.data()[feature_index]);
    }

    float min_x = xs.front();
    float max_x = xs.front();
    for (float v : xs) {
        min_x = std::min(min_x, v);
        max_x = std::max(max_x, v);
    }
    float range = max_x - min_x;

    int num_bins = std::max<int>(1, static_cast<int>(std::sqrt(static_cast<double>(xs.size()))));
    std::vector<std::vector<size_t>> bins(static_cast<size_t>(num_bins));
    for (size_t i = 0; i < xs.size(); ++i) {
        int bin = range > 0.0f ? static_cast<int>((xs[i] - min_x) / range * static_cast<float>(num_bins)) : 0;
        bin = std::clamp(bin, 0, num_bins - 1);
        bins[static_cast<size_t>(bin)].push_back(i);
    }

    constexpr float kJitterUnit = 0.15f;
    std::vector<BeeswarmPoint> points(xs.size());
    for (const std::vector<size_t>& bin_indices : bins) {
        for (size_t rank = 0; rank < bin_indices.size(); ++rank) {
            size_t idx = bin_indices[rank];
            // Alternating stack: rank 0 at y=0, then -1, +1, -2, +2, ... units -- spreads
            // colliding points symmetrically above/below the axis rather than in one
            // direction, matching a real beeswarm's visual balance.
            int offset_index = static_cast<int>((rank + 1) / 2);
            float sign = (rank % 2 == 0) ? 1.0f : -1.0f;
            float y = (rank == 0) ? 0.0f : sign * static_cast<float>(offset_index) * kJitterUnit;
            points[idx] = BeeswarmPoint{xs[idx], y};
        }
    }
    return points;
}

HistogramBins ToFieldHistogramBins(const Dataset& dataset, int64_t field_index, int num_bins) {
    if (dataset.size() == 0) {
        throw std::invalid_argument("ToFieldHistogramBins: dataset must be non-empty");
    }
    if (num_bins <= 0) {
        throw std::invalid_argument("ToFieldHistogramBins: num_bins must be positive");
    }

    std::vector<float> values;
    for (int64_t i = 0; i < dataset.size(); ++i) {
        Sample sample = dataset.get(i);
        if (field_index < 0 || static_cast<size_t>(field_index) >= sample.fields.size()) {
            throw std::out_of_range("ToFieldHistogramBins: field_index out of range for sample");
        }
        const Tensor& field = sample.fields[static_cast<size_t>(field_index)];
        const float* data = field.data();
        for (int64_t j = 0; j < field.numel(); ++j) {
            values.push_back(data[j]);
        }
    }

    float min_value = values.front();
    float max_value = values.front();
    for (float v : values) {
        min_value = std::min(min_value, v);
        max_value = std::max(max_value, v);
    }
    float range = max_value - min_value;

    HistogramBins bins;
    bins.counts.assign(static_cast<size_t>(num_bins), 0);
    bins.bin_edges.reserve(static_cast<size_t>(num_bins) + 1);
    for (int i = 0; i <= num_bins; ++i) {
        float fraction = static_cast<float>(i) / static_cast<float>(num_bins);
        bins.bin_edges.push_back(range > 0.0f ? min_value + range * fraction : min_value);
    }

    if (range <= 0.0f) {
        // Constant field: every value belongs in the same (only meaningful) bin rather than
        // dividing by a zero-width range.
        bins.counts[0] = static_cast<int64_t>(values.size());
        return bins;
    }

    for (float v : values) {
        int bin = static_cast<int>((v - min_value) / range * static_cast<float>(num_bins));
        bin = std::clamp(bin, 0, num_bins - 1);
        bins.counts[static_cast<size_t>(bin)]++;
    }
    return bins;
}

}  // namespace pulsatrix
