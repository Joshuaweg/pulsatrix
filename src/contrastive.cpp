// XAI-4: the contrastive explanation method (Dhurandhar et al. 2018), pertinent negatives and
// positives by FISTA with a binary search over c, as in the paper's code.

#include "pulsatrix/contrastive.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pulsatrix {

namespace {

enum class Kind { Negative, Positive };

int64_t ArgMax(const std::vector<float>& v) {
    return static_cast<int64_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

// The largest output other than t0, and where it is.
size_t Rival(const std::vector<float>& z, size_t t0) {
    size_t best = t0 == 0 ? 1 : 0;
    for (size_t i = 0; i < z.size(); ++i) {
        if (i != t0 && z[i] > z[best]) best = i;
    }
    return best;
}

ContrastiveResult Solve(ExplainerContext& ctx, const Tensor& input, const ContrastiveOptions& o, Kind kind, const char* who) {
    const std::vector<float> x = input.to_host_vector();
    const size_t n = x.size();
    if (!o.baseline.empty() && o.baseline.size() != n) throw std::invalid_argument(std::string(who) + ": baseline needs one value per feature");
    if (!o.upper.empty() && o.upper.size() != n) throw std::invalid_argument(std::string(who) + ": upper needs one value per feature");
    if (o.kappa < 0.0f || o.beta < 0.0f || !(o.c_init > 0.0f) || o.search_steps < 1 || o.steps < 1 || !(o.learning_rate > 0.0f)) {
        throw std::invalid_argument(std::string(who) + ": an option is out of range");
    }
    const std::vector<float> b = o.baseline.empty() ? std::vector<float>(n, 0.0f) : o.baseline;
    auto forward = [&](const std::vector<float>& v) {
        return ctx.forward_pass(Tensor(input.shape(), input.backend(), v, input.device()));
    };
    const std::vector<float> z0 = forward(x).to_host_vector();
    if (z0.size() < 2) throw std::invalid_argument(std::string(who) + ": the model needs at least two outputs");
    const auto t0 = static_cast<size_t>(ArgMax(z0));

    // The box δ lives in, per feature.
    std::vector<float> lo(n, 0.0f), hi(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
        if (kind == Kind::Negative) {
            if (x[j] == b[j]) hi[j] = o.upper.empty() ? std::numeric_limits<float>::infinity() : std::max(0.0f, o.upper[j] - x[j]);
        } else {
            lo[j] = std::min(0.0f, x[j] - b[j]);
            hi[j] = std::max(0.0f, x[j] - b[j]);
        }
    }
    const auto point = [&](const std::vector<float>& d) {
        std::vector<float> p(n);
        for (size_t j = 0; j < n; ++j) p[j] = (kind == Kind::Negative ? x[j] : b[j]) + d[j];
        return p;
    };
    // The hinge's argument: positive while the explanation hasn't reached its goal.
    const auto margin = [&](const std::vector<float>& z, size_t* rival) {
        const size_t r = Rival(z, t0);
        if (rival != nullptr) *rival = r;
        return kind == Kind::Negative ? z[t0] - z[r] : z[r] - z[t0];
    };
    const auto valid = [&](const std::vector<float>& z) {
        const bool kept = static_cast<size_t>(ArgMax(z)) == t0;
        return margin(z, nullptr) <= -o.kappa && (kind == Kind::Negative ? !kept : kept);
    };
    const auto elastic = [&](const std::vector<float>& d) {
        double l1 = 0, l2 = 0;
        for (float v : d) l1 += std::fabs(v), l2 += static_cast<double>(v) * v;
        return o.beta * l1 + l2;
    };

    ContrastiveResult result;
    result.original_class = static_cast<int64_t>(t0);
    std::vector<float> best;
    double best_elastic = std::numeric_limits<double>::infinity();
    float best_c = 0.0f;
    double c_lo = 0.0, c_hi = 1e10, c = o.c_init;
    for (int round = 0; round < o.search_steps; ++round) {
        std::vector<float> d(n, 0.0f), y(n, 0.0f);
        bool success = false;
        for (int k = 0; k < o.steps; ++k) {
            const Tensor out = forward(point(y));
            const std::vector<float> z = out.to_host_vector();
            size_t r = 0;
            std::vector<float> grad(n, 0.0f);
            if (margin(z, &r) > -o.kappa) {
                std::vector<float> seed(z.size(), 0.0f);
                const float sign = kind == Kind::Negative ? 1.0f : -1.0f;
                seed[t0] = sign * static_cast<float>(c);
                seed[r] = -sign * static_cast<float>(c);
                grad = ctx.backward_pass(Tensor(out.shape(), out.backend(), seed, out.device())).to_host_vector();
            }
            // Proximal step: the smooth part's gradient, then soft-thresholding (L1) and the box.
            std::vector<float> next(n);
            const float shrink = o.learning_rate * o.beta;
            for (size_t j = 0; j < n; ++j) {
                const float v = y[j] - o.learning_rate * (grad[j] + 2.0f * y[j]);
                const float s = v > shrink ? v - shrink : (v < -shrink ? v + shrink : 0.0f);
                next[j] = std::min(hi[j], std::max(lo[j], s));
            }
            const float momentum = static_cast<float>(k) / static_cast<float>(k + 3);
            for (size_t j = 0; j < n; ++j) y[j] = std::min(hi[j], std::max(lo[j], next[j] + momentum * (next[j] - d[j])));
            d = std::move(next);

            if (valid(forward(point(d)).to_host_vector())) {
                success = true;
                const double e = elastic(d);
                if (e < best_elastic) best_elastic = e, best = d, best_c = static_cast<float>(c);
            }
        }
        if (success) {
            c_hi = std::min(c_hi, c);
            c = (c_lo + c_hi) / 2.0;
        } else {
            c_lo = std::max(c_lo, c);
            c = c_hi < 1e9 ? (c_lo + c_hi) / 2.0 : c * 10.0;
        }
    }

    result.found = !best.empty();
    result.delta = result.found ? best : std::vector<float>(n, 0.0f);
    result.explanation = result.found ? point(result.delta) : x;
    result.explanation_class = ArgMax(forward(result.explanation).to_host_vector());
    for (size_t j = 0; j < n; ++j) {
        result.l1 += std::fabs(result.delta[j]);
        if (std::fabs(result.delta[j]) > 1e-6f) result.features.push_back(static_cast<int64_t>(j));
    }
    result.c = best_c;
    return result;
}

}  // namespace

ContrastiveResult FindPertinentNegative(ExplainerContext& ctx, const Tensor& input, const ContrastiveOptions& options) {
    return Solve(ctx, input, options, Kind::Negative, "FindPertinentNegative");
}

ContrastiveResult FindPertinentPositive(ExplainerContext& ctx, const Tensor& input, const ContrastiveOptions& options) {
    return Solve(ctx, input, options, Kind::Positive, "FindPertinentPositive");
}

JsonValue ToJson(const ContrastiveResult& result, const std::string& kind) {
    JsonValue::Array delta, explanation, features;
    for (float v : result.delta) delta.push_back(JsonValue(static_cast<double>(v)));
    for (float v : result.explanation) explanation.push_back(JsonValue(static_cast<double>(v)));
    for (int64_t f : result.features) features.push_back(JsonValue(f));
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.contrastive.v1"},
        {"kind", kind},
        {"found", result.found},
        {"original_class", JsonValue(result.original_class)},
        {"explanation_class", JsonValue(result.explanation_class)},
        {"delta", JsonValue(std::move(delta))},
        {"explanation", JsonValue(std::move(explanation))},
        {"features", JsonValue(std::move(features))},
        {"l1", JsonValue(result.l1)},
        {"c", JsonValue(static_cast<double>(result.c))},
    });
}

}  // namespace pulsatrix
