#include "pulsatrix/ice.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

void CheckInstances(const std::vector<Tensor>& instances, const char* where) {
    if (instances.empty()) {
        throw std::invalid_argument(std::string(where) + ": no instances");
    }
    for (const Tensor& t : instances) {
        if (t.shape() != instances.front().shape()) {
            throw std::invalid_argument(std::string(where) + ": instances differ in shape");
        }
    }
}

void CheckFeature(const std::vector<Tensor>& instances, int64_t feature, const char* where) {
    if (feature < 0 || feature >= instances.front().numel()) {
        throw std::invalid_argument(std::string(where) + ": feature index " + std::to_string(feature) +
                                    " is out of range");
    }
}

// The target element of predict(instance with each feature in `features` set to its value).
float PredictWith(const std::function<Tensor(const Tensor&)>& predict, const Tensor& like, std::vector<float> values,
                  const std::vector<std::pair<int64_t, float>>& features, int64_t target, const char* where) {
    for (const auto& [index, value] : features) {
        values[static_cast<size_t>(index)] = value;
    }
    Tensor out = predict(Tensor(like.shape(), like.backend(), values, like.device()));
    if (target < 0 || target >= out.numel()) {
        throw std::invalid_argument(std::string(where) + ": target index " + std::to_string(target) +
                                    " is out of range for an output of " + std::to_string(out.numel()));
    }
    return out.read_element(target);
}

// scipy.stats.mstats.mquantiles with its defaults (alphap = betap = 0.4) on sorted data.
double MQuantile(const std::vector<double>& sorted, double p) {
    const double n = static_cast<double>(sorted.size());
    const double m = 0.4 + p * (1.0 - 0.4 - 0.4);
    const double aleph = n * p + m;
    const double k = std::floor(std::clamp(aleph, 1.0, n - 1.0));
    const double gamma = std::clamp(aleph - k, 0.0, 1.0);
    const auto i = static_cast<size_t>(k);
    return (1.0 - gamma) * sorted[i - 1] + gamma * sorted[i];
}

}  // namespace

std::vector<float> FeatureGrid(const std::vector<Tensor>& background, int64_t feature_index, int64_t grid_size,
                               float lower_percentile, float upper_percentile) {
    CheckInstances(background, "FeatureGrid");
    CheckFeature(background, feature_index, "FeatureGrid");
    if (grid_size < 2) {
        throw std::invalid_argument("FeatureGrid: grid_size must be at least 2");
    }
    if (!(lower_percentile >= 0.0f && upper_percentile <= 1.0f && lower_percentile < upper_percentile)) {
        throw std::invalid_argument("FeatureGrid: percentiles must satisfy 0 <= lower < upper <= 1");
    }
    std::vector<double> values;
    values.reserve(background.size());
    for (const Tensor& t : background) {
        values.push_back(t.read_element(feature_index));
    }
    std::sort(values.begin(), values.end());
    std::vector<double> distinct = values;
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    if (static_cast<int64_t>(distinct.size()) < grid_size) {
        return {distinct.begin(), distinct.end()};
    }
    // mquantiles needs at least two values; distinct.size() >= grid_size >= 2 guarantees it.
    const double lo = MQuantile(values, lower_percentile);
    const double hi = MQuantile(values, upper_percentile);
    if (std::fabs(hi - lo) <= 1e-8 + 1e-5 * std::fabs(lo)) {  // numpy.allclose
        throw std::invalid_argument("FeatureGrid: the percentiles are too close together to build a grid");
    }
    std::vector<float> grid(static_cast<size_t>(grid_size));
    const double step = (hi - lo) / static_cast<double>(grid_size - 1);
    for (int64_t k = 0; k < grid_size; ++k) {
        grid[static_cast<size_t>(k)] = static_cast<float>(k == grid_size - 1 ? hi : lo + step * static_cast<double>(k));
    }
    return grid;
}

std::vector<float> IceResult::partial_dependence() const {
    const size_t g = grid.size();
    std::vector<double> sum(g, 0.0);
    for (int64_t i = 0; i < num_instances; ++i) {
        for (size_t k = 0; k < g; ++k) {
            sum[k] += curves[static_cast<size_t>(i) * g + k];
        }
    }
    std::vector<float> mean(g);
    for (size_t k = 0; k < g; ++k) {
        mean[k] = static_cast<float>(sum[k] / static_cast<double>(num_instances));
    }
    return mean;
}

std::vector<float> IceResult::centered(int64_t anchor) const {
    const size_t g = grid.size();
    if (anchor < 0 || static_cast<size_t>(anchor) >= g) {
        throw std::out_of_range("IceResult::centered: anchor is not a grid index");
    }
    std::vector<float> out(curves.size());
    for (int64_t i = 0; i < num_instances; ++i) {
        const size_t row = static_cast<size_t>(i) * g;
        for (size_t k = 0; k < g; ++k) {
            out[row + k] = curves[row + k] - curves[row + static_cast<size_t>(anchor)];
        }
    }
    return out;
}

std::vector<float> IceResult::derivative() const {
    const size_t g = grid.size();
    if (g < 2) {
        throw std::logic_error("IceResult::derivative: the grid needs at least 2 points");
    }
    std::vector<float> out(curves.size());
    for (int64_t i = 0; i < num_instances; ++i) {
        const float* c = curves.data() + static_cast<size_t>(i) * g;
        float* d = out.data() + static_cast<size_t>(i) * g;
        d[0] = (c[1] - c[0]) / (grid[1] - grid[0]);
        d[g - 1] = (c[g - 1] - c[g - 2]) / (grid[g - 1] - grid[g - 2]);
        for (size_t k = 1; k + 1 < g; ++k) {
            // numpy.gradient's second-order formula, which handles an uneven grid.
            const double h0 = grid[k] - grid[k - 1];
            const double h1 = grid[k + 1] - grid[k];
            d[k] = static_cast<float>((h0 * h0 * c[k + 1] - h1 * h1 * c[k - 1] + (h1 * h1 - h0 * h0) * c[k]) /
                                      (h0 * h1 * (h0 + h1)));
        }
    }
    return out;
}

IceResult ComputeIce(const std::function<Tensor(const Tensor&)>& predict, const std::vector<Tensor>& instances,
                     int64_t feature_index, int64_t target_index, const std::vector<float>& grid) {
    CheckInstances(instances, "ComputeIce");
    CheckFeature(instances, feature_index, "ComputeIce");
    if (grid.empty()) {
        throw std::invalid_argument("ComputeIce: the grid is empty");
    }
    IceResult result;
    result.feature_index = feature_index;
    result.target_index = target_index;
    result.grid = grid;
    result.num_instances = static_cast<int64_t>(instances.size());
    result.curves.reserve(instances.size() * grid.size());
    for (const Tensor& instance : instances) {
        const std::vector<float> values = instance.to_host_vector();
        result.feature_values.push_back(values[static_cast<size_t>(feature_index)]);
        for (float v : grid) {
            result.curves.push_back(
                PredictWith(predict, instance, values, {{feature_index, v}}, target_index, "ComputeIce"));
        }
    }
    return result;
}

PartialDependence2D ComputePartialDependence2D(const std::function<Tensor(const Tensor&)>& predict,
                                               const std::vector<Tensor>& background, int64_t feature_x,
                                               int64_t feature_y, int64_t target_index,
                                               const std::vector<float>& grid_x, const std::vector<float>& grid_y) {
    CheckInstances(background, "ComputePartialDependence2D");
    CheckFeature(background, feature_x, "ComputePartialDependence2D");
    CheckFeature(background, feature_y, "ComputePartialDependence2D");
    if (feature_x == feature_y) {
        throw std::invalid_argument("ComputePartialDependence2D: the two features must differ");
    }
    if (grid_x.empty() || grid_y.empty()) {
        throw std::invalid_argument("ComputePartialDependence2D: a grid is empty");
    }
    std::vector<std::vector<float>> values;
    values.reserve(background.size());
    for (const Tensor& t : background) {
        values.push_back(t.to_host_vector());
    }
    PartialDependence2D result{feature_x, feature_y, target_index, grid_x, grid_y, {}};
    result.values.reserve(grid_x.size() * grid_y.size());
    for (float vy : grid_y) {
        for (float vx : grid_x) {
            double sum = 0.0;
            for (size_t b = 0; b < background.size(); ++b) {
                sum += PredictWith(predict, background[b], values[b], {{feature_x, vx}, {feature_y, vy}}, target_index,
                                   "ComputePartialDependence2D");
            }
            result.values.push_back(static_cast<float>(sum / static_cast<double>(background.size())));
        }
    }
    return result;
}

namespace {

// PyALE's quantile_ied (its "type 1" quantile) on sorted data, for q in (0, 1).
double QuantileIed(const std::vector<double>& sorted, double q) {
    const double n = static_cast<double>(sorted.size() - 1);
    const double pos = n * q;
    const auto j = static_cast<size_t>(pos);
    if (pos - static_cast<double>(j) != 0.0) {
        return sorted[j];
    }
    return j == 0 ? 0.0 : sorted[j - 1];
}

}  // namespace

AleResult ComputeAle(const std::function<Tensor(const Tensor&)>& predict, const std::vector<Tensor>& instances,
                     int64_t feature_index, int64_t target_index, int64_t num_bins) {
    CheckInstances(instances, "ComputeAle");
    CheckFeature(instances, feature_index, "ComputeAle");
    if (num_bins < 1) {
        throw std::invalid_argument("ComputeAle: num_bins must be at least 1");
    }
    const auto f = static_cast<size_t>(feature_index);
    std::vector<std::vector<float>> values;
    std::vector<double> sorted;
    for (const Tensor& t : instances) {
        values.push_back(t.to_host_vector());
        sorted.push_back(values.back()[f]);
    }
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> edges{sorted.front()};
    const double step = 1.0 / static_cast<double>(num_bins);
    for (int64_t k = 0; k <= num_bins; ++k) {
        const double q = static_cast<double>(k) * step;  // numpy.linspace(0, 1, num_bins + 1)
        edges.push_back(k == 0 ? sorted.front() : k == num_bins ? sorted.back() : QuantileIed(sorted, q));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) {
        throw std::invalid_argument("ComputeAle: the feature takes only one value");
    }
    const size_t bins = edges.size() - 1;

    std::vector<double> sum(bins, 0.0);
    std::vector<int64_t> count(bins, 0);
    AleResult r;
    r.feature_index = feature_index;
    r.target_index = target_index;
    for (size_t i = 0; i < instances.size(); ++i) {
        const double v = values[i][f];
        r.feature_values.push_back(static_cast<float>(v));
        // Bin k holds (edges[k], edges[k + 1]]; the first also holds edges[0].
        size_t k = static_cast<size_t>(std::lower_bound(edges.begin() + 1, edges.end(), v) - (edges.begin() + 1));
        k = std::min(k, bins - 1);
        const float lo = PredictWith(predict, instances[i], values[i], {{feature_index, static_cast<float>(edges[k])}},
                                     target_index, "ComputeAle");
        const float hi = PredictWith(predict, instances[i], values[i],
                                     {{feature_index, static_cast<float>(edges[k + 1])}}, target_index, "ComputeAle");
        sum[k] += static_cast<double>(hi) - static_cast<double>(lo);
        ++count[k];
    }
    std::vector<double> accumulated(bins + 1, 0.0);
    for (size_t k = 0; k < bins; ++k) {
        accumulated[k + 1] = accumulated[k] + (count[k] > 0 ? sum[k] / static_cast<double>(count[k]) : 0.0);
    }
    double center = 0.0;
    for (size_t k = 0; k < bins; ++k) {
        center += (accumulated[k + 1] + accumulated[k]) / 2.0 * static_cast<double>(count[k]);
    }
    center /= static_cast<double>(instances.size());
    for (size_t k = 0; k <= bins; ++k) {
        r.edges.push_back(static_cast<float>(edges[k]));
        r.effects.push_back(static_cast<float>(accumulated[k] - center));
    }
    r.counts = count;
    return r;
}

}  // namespace pulsatrix
