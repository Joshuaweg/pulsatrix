#include "pulsatrix/explanation_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "portable_random.hpp"
#include "pulsatrix/null_model_baseline.hpp"

namespace pulsatrix {
namespace {

std::vector<double> magnitudes(const Attribution& a) {
    std::vector<double> m;
    for (float v : a.values.to_host_vector()) {
        m.push_back(std::fabs(static_cast<double>(v)));
    }
    return m;
}

std::vector<double> ranks(const std::vector<float>& v) {
    std::vector<size_t> order(v.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size());
    for (size_t i = 0; i < order.size();) {
        size_t j = i;
        while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) {
            ++j;
        }
        const double average = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) {
            r[order[k]] = average;
        }
        i = j + 1;
    }
    return r;
}

float score_of(const PredictFn& predict, const Tensor& input, int64_t target) {
    const Tensor out = predict(input);
    if (target < 0 || target >= out.numel()) {
        throw std::invalid_argument("perturbation curve: target is outside the prediction");
    }
    return out.read_element(target);
}

// Feature indices from most to least relevant; equal values keep index order.
std::vector<size_t> relevance_order(const Tensor& input, const Attribution& attribution, const PerturbationOptions& o) {
    if (attribution.values.shape() != input.shape()) {
        throw std::invalid_argument("perturbation curve: the attribution must have the input's shape");
    }
    if (o.steps < 1) {
        throw std::invalid_argument("perturbation curve: steps must be >= 1");
    }
    const std::vector<float> a = attribution.values.to_host_vector();
    std::vector<size_t> order(a.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return a[x] > a[y]; });
    return order;
}

// changed[i] for the first `count` features in `order`.
std::vector<bool> first(const std::vector<size_t>& order, size_t count) {
    std::vector<bool> flags(order.size(), false);
    for (size_t k = 0; k < count; ++k) {
        flags[order[k]] = true;
    }
    return flags;
}

PerturbationCurve curve(const PredictFn& predict, const Tensor& input, const Attribution& attribution,
                        const PerturbationOptions& o, bool insertion) {
    const std::vector<size_t> order = relevance_order(input, attribution, o);
    const size_t n = order.size();
    PerturbationCurve c{{}, {}, 0.0f};
    for (int64_t step = 0; step <= o.steps; ++step) {
        const auto count = static_cast<size_t>(
            std::llround(static_cast<double>(step) * static_cast<double>(n) / static_cast<double>(o.steps)));
        std::vector<bool> removed = first(order, count);
        if (insertion) {
            removed.flip();  // restored features are the first `count`; the rest stay imputed
        }
        c.fractions.push_back(static_cast<float>(step) / static_cast<float>(o.steps));
        c.scores.push_back(score_of(predict, Impute(input, removed, o.imputation), o.target));
    }
    double auc = 0.0;
    for (size_t i = 1; i < c.scores.size(); ++i) {
        auc += (c.fractions[i] - c.fractions[i - 1]) * (static_cast<double>(c.scores[i]) + c.scores[i - 1]) / 2.0;
    }
    c.auc = static_cast<float>(auc);
    return c;
}

// ROAD's noisy linear imputation on one (H, W) plane, solved by Gauss-Seidel: each removed pixel
// equals the weighted mean of its in-bounds neighbors (1/6 direct, 1/12 diagonal, renormalized
// at borders).
void impute_plane(std::vector<double>& x, const std::vector<bool>& removed, size_t offset, int64_t h, int64_t w) {
    static const int kDi[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int kDj[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static const double kWeight[8] = {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12};
    bool any_kept = false;
    for (int64_t p = 0; p < h * w; ++p) {
        any_kept = any_kept || !removed[offset + static_cast<size_t>(p)];
    }
    if (!any_kept) {
        for (int64_t p = 0; p < h * w; ++p) {
            x[offset + static_cast<size_t>(p)] = 0.0;  // nothing to interpolate from
        }
        return;
    }
    for (int64_t p = 0; p < h * w; ++p) {
        if (removed[offset + static_cast<size_t>(p)]) {
            x[offset + static_cast<size_t>(p)] = 0.0;
        }
    }
    for (int iteration = 0; iteration < 10000; ++iteration) {
        double change = 0.0;
        for (int64_t i = 0; i < h; ++i) {
            for (int64_t j = 0; j < w; ++j) {
                const size_t p = offset + static_cast<size_t>(i * w + j);
                if (!removed[p]) {
                    continue;
                }
                double sum = 0.0, weight = 0.0;
                for (int k = 0; k < 8; ++k) {
                    const int64_t ni = i + kDi[k], nj = j + kDj[k];
                    if (ni >= 0 && ni < h && nj >= 0 && nj < w) {
                        sum += kWeight[k] * x[offset + static_cast<size_t>(ni * w + nj)];
                        weight += kWeight[k];
                    }
                }
                const double updated = weight > 0.0 ? sum / weight : 0.0;
                change = std::max(change, std::fabs(updated - x[p]));
                x[p] = updated;
            }
        }
        if (change < 1e-9) {
            break;
        }
    }
}

}  // namespace

float Sparseness(const Attribution& attribution) {
    std::vector<double> m = magnitudes(attribution);
    const double total = std::accumulate(m.begin(), m.end(), 0.0);
    if (total == 0.0 || m.empty()) {
        return 0.0f;
    }
    std::sort(m.begin(), m.end());
    const auto n = static_cast<double>(m.size());
    double g = 0.0;
    for (size_t i = 0; i < m.size(); ++i) {
        g += (2.0 * static_cast<double>(i + 1) - n - 1.0) * m[i];
    }
    return static_cast<float>(g / (n * total));
}

float Complexity(const Attribution& attribution) {
    const std::vector<double> m = magnitudes(attribution);
    const double total = std::accumulate(m.begin(), m.end(), 0.0);
    if (total == 0.0) {
        return 0.0f;
    }
    double h = 0.0;
    for (double v : m) {
        if (v > 0.0) {
            const double p = v / total;
            h -= p * std::log(p);
        }
    }
    return static_cast<float>(h);
}

float SpearmanRankCorrelation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument("SpearmanRankCorrelation: inputs must have the same size, at least 2");
    }
    const std::vector<double> rx = ranks(x), ry = ranks(y);
    const double mean = (static_cast<double>(x.size()) + 1.0) / 2.0;
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (size_t i = 0; i < rx.size(); ++i) {
        cov += (rx[i] - mean) * (ry[i] - mean);
        vx += (rx[i] - mean) * (rx[i] - mean);
        vy += (ry[i] - mean) * (ry[i] - mean);
    }
    if (vx == 0.0 || vy == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(cov / std::sqrt(vx * vy));
}

Tensor Impute(const Tensor& input, const std::vector<bool>& removed, const Imputation& imputation) {
    if (static_cast<int64_t>(removed.size()) != input.numel()) {
        throw std::invalid_argument("Impute: need one removed flag per input element");
    }
    const std::vector<float> original = input.to_host_vector();
    std::vector<double> x(original.begin(), original.end());
    if (imputation.kind == Imputation::Kind::Constant) {
        for (size_t i = 0; i < x.size(); ++i) {
            if (removed[i]) {
                x[i] = imputation.value;
            }
        }
    } else {
        if (input.rank() < 2) {
            throw std::invalid_argument("Impute: NoisyLinear needs an input of rank >= 2 (..., H, W)");
        }
        const int64_t h = input.shape().dim(input.rank() - 2), w = input.shape().dim(input.rank() - 1);
        for (size_t offset = 0; offset < x.size(); offset += static_cast<size_t>(h * w)) {
            impute_plane(x, removed, offset, h, w);
        }
        if (imputation.noise_std > 0.0f) {
            PortableRng rng{imputation.seed};
            for (size_t i = 0; i < x.size(); ++i) {
                if (removed[i]) {
                    x[i] += imputation.noise_std * rng.gaussian();
                }
            }
        }
    }
    return Tensor(input.shape(), input.backend(), std::vector<float>(x.begin(), x.end()), input.device());
}

PerturbationCurve DeletionCurve(const PredictFn& predict, const Tensor& input, const Attribution& attribution,
                                const PerturbationOptions& options) {
    return curve(predict, input, attribution, options, /*insertion=*/false);
}

PerturbationCurve InsertionCurve(const PredictFn& predict, const Tensor& input, const Attribution& attribution,
                                 const PerturbationOptions& options) {
    return curve(predict, input, attribution, options, /*insertion=*/true);
}

RandomizationResult ModelParameterRandomizationTest(Module& model, const ExplainFn& explain, const Tensor& input,
                                                    uint64_t seed) {
    // Layers: first name segment, in parameters order; randomized from the last (output) down.
    std::vector<std::string> layers;
    for (const NamedParamRef& p : model.named_parameters()) {
        const std::string layer = p.name.substr(0, p.name.find('.'));
        if (std::find(layers.begin(), layers.end(), layer) == layers.end()) {
            layers.push_back(layer);
        }
    }
    std::reverse(layers.begin(), layers.end());

    ParameterSnapshot saved(model);  // restores the model however this returns
    std::vector<float> original;
    for (double v : magnitudes(explain(input))) {
        original.push_back(static_cast<float>(v));
    }
    RandomizationResult result;
    PortableRng seeds{seed};
    for (const std::string& layer : layers) {
        ReinitializeParameters(model, seeds.next(), layer);
        std::vector<float> now;
        for (double v : magnitudes(explain(input))) {
            now.push_back(static_cast<float>(v));
        }
        result.layers.push_back(layer);
        result.similarity.push_back(SpearmanRankCorrelation(original, now));
    }
    return result;
}

}  // namespace pulsatrix
