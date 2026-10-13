// XAI-4: prototypes and contrasts by nearest neighbors, and ProtoDash (Gurumoorthy et al. 2019).

#include "pulsatrix/prototypes.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace pulsatrix {

namespace {

int64_t ArgMax(const std::vector<float>& v) {
    return static_cast<int64_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

double SquaredDistance(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return sum;
}

std::vector<float> Point(const Tensor& t, const EmbedFn& embed, const std::vector<float>& scale) {
    if (embed) return embed(t);
    std::vector<float> v = t.to_host_vector();
    for (size_t j = 0; j < scale.size(); ++j) v[j] /= scale[j];
    return v;
}

// Solves max_w w'mu - w'Kw/2 over w >= 0 by cyclic coordinate descent, from w.
void NonNegativeQuadratic(const std::vector<std::vector<double>>& k, const std::vector<double>& mu, std::vector<double>& w) {
    for (int sweep = 0; sweep < 100000; ++sweep) {
        double change = 0;
        for (size_t j = 0; j < w.size(); ++j) {
            double rest = mu[j];
            for (size_t l = 0; l < w.size(); ++l) {
                if (l != j) rest -= k[j][l] * w[l];
            }
            const double next = std::max(0.0, rest / k[j][j]);
            change = std::max(change, std::fabs(next - w[j]));
            w[j] = next;
        }
        if (change < 1e-13) break;
    }
}

}  // namespace

PrototypeResult FindPrototypes(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                               const std::vector<Tensor>& background, const PrototypeOptions& options) {
    if (background.empty()) throw std::invalid_argument("FindPrototypes: background is empty");
    if (options.k < 1) throw std::invalid_argument("FindPrototypes: k must be at least 1");
    for (const Tensor& b : background) {
        if (b.numel() != input.numel()) throw std::invalid_argument("FindPrototypes: a background input isn't the input's size");
    }
    if (!options.embed && !options.scale.empty()) {
        if (static_cast<int64_t>(options.scale.size()) != input.numel()) {
            throw std::invalid_argument("FindPrototypes: scale needs one value per feature");
        }
        for (float s : options.scale) {
            if (!(s > 0.0f)) throw std::invalid_argument("FindPrototypes: every scale must be positive");
        }
    }
    PrototypeResult result;
    result.predicted_class = ArgMax(predict(input).to_host_vector());
    const std::vector<float> query = Point(input, options.embed, options.scale);
    std::vector<Example> all;
    for (size_t i = 0; i < background.size(); ++i) {
        const std::vector<float> p = Point(background[i], options.embed, options.scale);
        if (p.size() != query.size()) throw std::invalid_argument("FindPrototypes: embeddings differ in size");
        all.push_back(Example{static_cast<int64_t>(i), std::sqrt(SquaredDistance(query, p)),
                              ArgMax(predict(background[i]).to_host_vector())});
    }
    std::stable_sort(all.begin(), all.end(), [](const Example& a, const Example& b) { return a.distance < b.distance; });
    for (const Example& e : all) {
        std::vector<Example>& into = e.predicted_class == result.predicted_class ? result.prototypes : result.contrasts;
        if (static_cast<int64_t>(into.size()) < options.k) into.push_back(e);
    }
    return result;
}

ProtoDashResult ProtoDash(const std::vector<Tensor>& source, const std::vector<Tensor>& target, const ProtoDashOptions& options) {
    if (source.empty() || target.empty()) throw std::invalid_argument("ProtoDash: source and target must not be empty");
    if (options.count < 1) throw std::invalid_argument("ProtoDash: count must be at least 1");
    if (!(options.sigma >= 0.0)) throw std::invalid_argument("ProtoDash: sigma must not be negative");
    std::vector<std::vector<float>> xs, zs;
    for (const Tensor& t : source) xs.push_back(Point(t, options.embed, {}));
    for (const Tensor& t : target) zs.push_back(Point(t, options.embed, {}));
    for (const auto& v : xs) {
        if (v.size() != xs.front().size()) throw std::invalid_argument("ProtoDash: points differ in size");
    }
    for (const auto& v : zs) {
        if (v.size() != xs.front().size()) throw std::invalid_argument("ProtoDash: points differ in size");
    }
    const size_t n = xs.size();

    double sigma = options.sigma;
    if (sigma == 0.0) {
        std::vector<double> d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) d.push_back(std::sqrt(SquaredDistance(xs[i], xs[j])));
        }
        if (!d.empty()) {
            std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
            sigma = d[d.size() / 2];
        }
        if (!(sigma > 0.0)) sigma = 1.0;  // one point, or all equal
    }
    const double gamma = 1.0 / (2.0 * sigma * sigma);
    const auto kernel = [gamma](const std::vector<float>& a, const std::vector<float>& b) {
        return std::exp(-gamma * SquaredDistance(a, b));
    };

    std::vector<double> mu(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        for (const auto& z : zs) mu[j] += kernel(xs[j], z);
        mu[j] /= static_cast<double>(zs.size());
    }
    // Kernel rows of the chosen points against every source point, filled as points are chosen.
    std::vector<std::vector<double>> rows;
    std::vector<size_t> chosen;
    std::vector<double> w;
    ProtoDashResult result;
    result.sigma = sigma;
    const auto limit = static_cast<size_t>(std::min<int64_t>(options.count, static_cast<int64_t>(n)));
    while (chosen.size() < limit) {
        // gradient_j = mu_j - sum_l K(j, chosen_l) w_l
        size_t best = n;
        double best_gradient = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (std::find(chosen.begin(), chosen.end(), j) != chosen.end()) continue;
            double g = mu[j];
            for (size_t l = 0; l < chosen.size(); ++l) g -= rows[l][j] * w[l];
            if (g > best_gradient) best_gradient = g, best = j;
        }
        if (best == n) break;  // no point would raise the objective
        chosen.push_back(best);
        std::vector<double> row(n);
        for (size_t j = 0; j < n; ++j) row[j] = kernel(xs[best], xs[j]);
        rows.push_back(std::move(row));
        w.push_back(0.0);

        std::vector<std::vector<double>> ks(chosen.size(), std::vector<double>(chosen.size()));
        std::vector<double> mus(chosen.size());
        for (size_t a = 0; a < chosen.size(); ++a) {
            mus[a] = mu[chosen[a]];
            for (size_t b = 0; b < chosen.size(); ++b) ks[a][b] = rows[a][chosen[b]];
        }
        NonNegativeQuadratic(ks, mus, w);
        double objective = 0;
        for (size_t a = 0; a < chosen.size(); ++a) {
            objective += w[a] * mus[a];
            for (size_t b = 0; b < chosen.size(); ++b) objective -= 0.5 * w[a] * ks[a][b] * w[b];
        }
        result.objective.push_back(objective);
    }
    const double total = std::accumulate(w.begin(), w.end(), 0.0);
    for (size_t a = 0; a < chosen.size(); ++a) {
        result.indices.push_back(static_cast<int64_t>(chosen[a]));
        result.weights.push_back(total > 0 ? w[a] / total : 0.0);
    }
    return result;
}

JsonValue ToJson(const PrototypeResult& result) {
    const auto examples = [](const std::vector<Example>& list) {
        JsonValue::Array out;
        for (const Example& e : list) {
            out.push_back(JsonValue(JsonValue::Object{
                {"index", JsonValue(e.index)},
                {"distance", JsonValue(e.distance)},
                {"predicted_class", JsonValue(e.predicted_class)},
            }));
        }
        return JsonValue(std::move(out));
    };
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.prototypes.v1"},
        {"predicted_class", JsonValue(result.predicted_class)},
        {"prototypes", examples(result.prototypes)},
        {"contrasts", examples(result.contrasts)},
    });
}

}  // namespace pulsatrix
