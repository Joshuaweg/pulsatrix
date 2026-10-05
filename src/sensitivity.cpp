#include "pulsatrix/sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

float Target(const std::function<Tensor(const Tensor&)>& predict, const Tensor& like, const std::vector<float>& values,
             int64_t target, const char* where) {
    Tensor out = predict(Tensor(like.shape(), like.backend(), values, like.device()));
    if (target < 0 || target >= out.numel()) {
        throw std::invalid_argument(std::string(where) + ": target index " + std::to_string(target) +
                                    " is out of range for an output of " + std::to_string(out.numel()));
    }
    return out.read_element(target);
}

std::vector<std::vector<float>> HostValues(const std::vector<Tensor>& background, const Shape& shape,
                                           const char* where) {
    if (background.empty()) {
        throw std::invalid_argument(std::string(where) + ": background is empty");
    }
    std::vector<std::vector<float>> values;
    for (const Tensor& t : background) {
        if (t.shape() != shape) {
            throw std::invalid_argument(std::string(where) + ": background instances differ in shape");
        }
        values.push_back(t.to_host_vector());
    }
    return values;
}

// numpy.percentile's default (linear) interpolation on sorted data.
double Percentile(const std::vector<double>& sorted, double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

}  // namespace

SensitivityBounds DeltaBounds(const Tensor& input, float delta) {
    if (!(delta > 0.0f)) {
        throw std::invalid_argument("DeltaBounds: delta must be positive");
    }
    SensitivityBounds b{input.to_host_vector(), {}};
    b.high = b.low;
    for (size_t i = 0; i < b.low.size(); ++i) {
        b.low[i] -= delta;
        b.high[i] += delta;
    }
    return b;
}

SensitivityBounds ScaledDeltaBounds(const Tensor& input, const std::vector<Tensor>& background, float num_std) {
    if (!(num_std > 0.0f)) {
        throw std::invalid_argument("ScaledDeltaBounds: num_std must be positive");
    }
    const auto values = HostValues(background, input.shape(), "ScaledDeltaBounds");
    SensitivityBounds b{input.to_host_vector(), {}};
    b.high = b.low;
    const double n = static_cast<double>(values.size());
    for (size_t j = 0; j < b.low.size(); ++j) {
        double mean = 0.0;
        for (const auto& v : values) mean += v[j];
        mean /= n;
        double var = 0.0;
        for (const auto& v : values) var += (v[j] - mean) * (v[j] - mean);
        const auto step = static_cast<float>(num_std * std::sqrt(var / n));  // population std, as numpy.std
        b.low[j] -= step;
        b.high[j] += step;
    }
    return b;
}

SensitivityBounds RangeBounds(const std::vector<Tensor>& background, float lower_percentile, float upper_percentile) {
    if (!(lower_percentile >= 0.0f && upper_percentile <= 1.0f && lower_percentile < upper_percentile)) {
        throw std::invalid_argument("RangeBounds: percentiles must satisfy 0 <= lower < upper <= 1");
    }
    if (background.empty()) {
        throw std::invalid_argument("RangeBounds: background is empty");
    }
    const auto values = HostValues(background, background.front().shape(), "RangeBounds");
    SensitivityBounds b;
    for (size_t j = 0; j < values.front().size(); ++j) {
        std::vector<double> column;
        for (const auto& v : values) column.push_back(v[j]);
        std::sort(column.begin(), column.end());
        b.low.push_back(static_cast<float>(Percentile(column, lower_percentile)));
        b.high.push_back(static_cast<float>(Percentile(column, upper_percentile)));
    }
    return b;
}

float FeatureSensitivity::swing() const { return std::fabs(output_high - output_low); }

float FeatureSensitivity::slope() const { return high == low ? 0.0f : (output_high - output_low) / (high - low); }

LocalSensitivityResult ComputeLocalSensitivity(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                               int64_t target_index, const SensitivityBounds& bounds,
                                               const std::vector<int64_t>& features) {
    const int64_t n = input.numel();
    if (static_cast<int64_t>(bounds.low.size()) != n || static_cast<int64_t>(bounds.high.size()) != n) {
        throw std::invalid_argument("ComputeLocalSensitivity: bounds need one low and one high value per feature");
    }
    std::vector<int64_t> order = features;
    if (order.empty()) {
        for (int64_t j = 0; j < n; ++j) order.push_back(j);
    }
    for (int64_t j : order) {
        if (j < 0 || j >= n) {
            throw std::invalid_argument("ComputeLocalSensitivity: feature index " + std::to_string(j) +
                                        " is out of range");
        }
    }
    const std::vector<float> x = input.to_host_vector();
    LocalSensitivityResult result;
    result.target_index = target_index;
    result.output = Target(predict, input, x, target_index, "ComputeLocalSensitivity");
    for (int64_t j : order) {
        const auto k = static_cast<size_t>(j);
        FeatureSensitivity s;
        s.feature_index = j;
        s.value = x[k];
        s.low = bounds.low[k];
        s.high = bounds.high[k];
        std::vector<float> moved = x;
        moved[k] = s.low;
        s.output_low = Target(predict, input, moved, target_index, "ComputeLocalSensitivity");
        moved[k] = s.high;
        s.output_high = Target(predict, input, moved, target_index, "ComputeLocalSensitivity");
        result.features.push_back(s);
    }
    return result;
}

Attribution Occlusion(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input, int64_t target_index,
                      const std::vector<int64_t>& window, const std::vector<int64_t>& strides, float baseline) {
    const auto rank = static_cast<size_t>(input.rank());
    if (window.size() != rank || strides.size() != rank) {
        throw std::invalid_argument("Occlusion: window and strides need one size per input dimension");
    }
    std::vector<int64_t> dims(rank), shifts(rank);
    int64_t windows = 1;
    for (size_t d = 0; d < rank; ++d) {
        dims[d] = input.shape().dim(d);
        if (window[d] < 1 || window[d] > dims[d]) {
            throw std::invalid_argument("Occlusion: each window size must be in [1, the input's size]");
        }
        if (strides[d] < 1 || (window[d] < dims[d] && strides[d] > window[d])) {
            throw std::invalid_argument("Occlusion: each stride must be >= 1 and at most the window size");
        }
        const int64_t room = dims[d] - window[d];
        shifts[d] = (room + strides[d] - 1) / strides[d] + 1;  // ceil(room / stride) + 1, as Captum
        windows *= shifts[d];
    }
    const std::vector<float> x = input.to_host_vector();
    const float base = Target(predict, input, x, target_index, "Occlusion");
    std::vector<double> total(x.size(), 0.0);
    std::vector<int64_t> count(x.size(), 0);
    std::vector<int64_t> pos(rank);
    for (int64_t w = 0; w < windows; ++w) {
        // Captum numbers windows with the first dimension changing fastest.
        int64_t rest = w;
        for (size_t d = 0; d < rank; ++d) {
            pos[d] = (rest % shifts[d]) * strides[d];
            rest /= shifts[d];
        }
        // Every flat index inside the window, cropped at the input's edge, in row-major order.
        std::vector<int64_t> extent(rank);
        int64_t cells = 1;
        for (size_t d = 0; d < rank; ++d) {
            extent[d] = std::min(pos[d] + window[d], dims[d]) - pos[d];
            cells *= extent[d];
        }
        std::vector<size_t> covered;
        covered.reserve(static_cast<size_t>(cells));
        for (int64_t c = 0; c < cells; ++c) {
            int64_t r = c;
            size_t flat = 0;
            size_t place = 1;
            for (size_t d = rank; d-- > 0;) {
                const int64_t offset = r % extent[d];
                r /= extent[d];
                flat += static_cast<size_t>(pos[d] + offset) * place;
                place *= static_cast<size_t>(dims[d]);
            }
            covered.push_back(flat);
        }
        std::vector<float> occluded = x;
        for (size_t i : covered) occluded[i] = baseline;
        const double drop = static_cast<double>(base) - Target(predict, input, occluded, target_index, "Occlusion");
        for (size_t i : covered) {
            total[i] += drop;
            ++count[i];
        }
    }
    std::vector<float> attr(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        attr[i] = count[i] > 0 ? static_cast<float>(total[i] / static_cast<double>(count[i])) : 0.0f;
    }
    std::string window_text, stride_text;
    for (size_t d = 0; d < rank; ++d) {
        window_text += (d ? "x" : "") + std::to_string(window[d]);
        stride_text += (d ? "x" : "") + std::to_string(strides[d]);
    }
    return Attribution{"occlusion",
                       Tensor(input.shape(), input.backend(), attr, input.device()),
                       {{"target_index", std::to_string(target_index)},
                        {"window", window_text},
                        {"strides", stride_text},
                        {"baseline", std::to_string(baseline)}}};
}

}  // namespace pulsatrix
