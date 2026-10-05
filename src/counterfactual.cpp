#include "pulsatrix/counterfactual.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

void CheckIndices(const std::vector<int64_t>& indices, int64_t n, const char* what) {
    for (int64_t j : indices) {
        if (j < 0 || j >= n) {
            throw std::invalid_argument(std::string("FindCounterfactual: ") + what + " index " + std::to_string(j) +
                                        " is out of range");
        }
    }
}

// How far the outputs are from the target (0 once reached), and d(loss)/d(outputs).
float TargetLoss(const std::vector<float>& z, const CounterfactualTarget& t, std::vector<float>* grad) {
    if (grad != nullptr) grad->assign(z.size(), 0.0f);
    const auto i = static_cast<size_t>(t.index);
    if (t.kind == CounterfactualTarget::Kind::Range) {
        if (z[i] < t.low) {
            if (grad != nullptr) (*grad)[i] = -1.0f;
            return t.low - z[i];
        }
        if (z[i] > t.high) {
            if (grad != nullptr) (*grad)[i] = 1.0f;
            return z[i] - t.high;
        }
        return 0.0f;
    }
    size_t best = i == 0 ? 1 : 0;
    for (size_t k = 0; k < z.size(); ++k) {
        if (k != i && z[k] > z[best]) best = k;
    }
    const float hinge = z[best] - z[i] + t.margin;
    if (hinge <= 0.0f) return 0.0f;
    if (grad != nullptr) {
        (*grad)[best] = 1.0f;
        (*grad)[i] = -1.0f;
    }
    return hinge;
}

}  // namespace

std::vector<float> MedianAbsoluteDeviation(const std::vector<Tensor>& background) {
    if (background.empty()) {
        throw std::invalid_argument("MedianAbsoluteDeviation: background is empty");
    }
    std::vector<std::vector<float>> values;
    for (const Tensor& t : background) {
        if (t.shape() != background.front().shape()) {
            throw std::invalid_argument("MedianAbsoluteDeviation: background instances differ in shape");
        }
        values.push_back(t.to_host_vector());
    }
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t m = v.size() / 2;
        return v.size() % 2 == 1 ? v[m] : (v[m - 1] + v[m]) / 2.0;
    };
    std::vector<float> mad(values.front().size());
    for (size_t j = 0; j < mad.size(); ++j) {
        std::vector<double> column;
        for (const auto& v : values) column.push_back(v[j]);
        const double med = median(column);
        for (double& c : column) c = std::fabs(c - med);
        const double d = median(column);
        mad[j] = d > 0.0 ? static_cast<float>(d) : 1.0f;
    }
    return mad;
}

CounterfactualResult FindCounterfactual(ExplainerContext& ctx, const Tensor& input, const CounterfactualTarget& target,
                                        const CounterfactualConstraints& constraints,
                                        const CounterfactualOptions& options) {
    const int64_t n = input.numel();
    const auto un = static_cast<size_t>(n);
    if (!(options.lambda > 0.0f) || !(options.lambda_growth >= 1.0f) || options.max_rounds < 1 ||
        options.steps_per_round < 1 || !(options.learning_rate > 0.0f) || !(options.change_tolerance >= 0.0f)) {
        throw std::invalid_argument("FindCounterfactual: invalid options");
    }
    if (target.kind == CounterfactualTarget::Kind::Range && !(target.low <= target.high)) {
        throw std::invalid_argument("FindCounterfactual: the target range needs low <= high");
    }
    auto check_length = [&](const std::vector<float>& v, const char* what) {
        if (!v.empty() && v.size() != un) {
            throw std::invalid_argument(std::string("FindCounterfactual: ") + what + " needs one value per feature");
        }
    };
    check_length(constraints.scale, "scale");
    check_length(constraints.lower, "lower");
    check_length(constraints.upper, "upper");
    CheckIndices(constraints.immutable, n, "immutable");
    CheckIndices(constraints.integer, n, "integer");
    for (const auto& g : constraints.one_hot_groups) CheckIndices(g, n, "one-hot");

    std::vector<float> scale = constraints.scale.empty() ? std::vector<float>(un, 1.0f) : constraints.scale;
    for (float s : scale) {
        if (!(s > 0.0f)) throw std::invalid_argument("FindCounterfactual: every scale must be positive");
    }
    std::vector<bool> frozen(un, false);
    for (int64_t j : constraints.immutable) frozen[static_cast<size_t>(j)] = true;

    const std::vector<float> x0 = input.to_host_vector();
    auto evaluate = [&](const std::vector<float>& x) {
        return ctx.forward_pass(Tensor(input.shape(), input.backend(), x, input.device()));
    };
    Tensor out0 = evaluate(x0);
    if (target.index < 0 || target.index >= out0.numel()) {
        throw std::invalid_argument("FindCounterfactual: target index is out of range for an output of " +
                                    std::to_string(out0.numel()));
    }
    if (target.kind == CounterfactualTarget::Kind::Class && out0.numel() < 2) {
        throw std::invalid_argument("FindCounterfactual: a class target needs at least 2 outputs");
    }
    auto distance = [&](const std::vector<float>& x) {
        double d = 0.0;
        for (size_t j = 0; j < un; ++j) d += std::fabs(x[j] - x0[j]) / scale[j];
        return d;
    };
    auto clip = [&](size_t j, float v) {
        if (!constraints.lower.empty()) v = std::max(v, constraints.lower[j]);
        if (!constraints.upper.empty()) v = std::min(v, constraints.upper[j]);
        return v;
    };

    CounterfactualResult result{input, false};
    result.output_before = out0.read_element(target.index);
    std::vector<float> x = x0;
    std::vector<float> best;
    double best_distance = std::numeric_limits<double>::infinity();
    float lambda = options.lambda;
    std::vector<float> seed;
    for (int round = 0; round < options.max_rounds; ++round) {
        result.rounds = round + 1;
        for (int step = 0; step < options.steps_per_round; ++step) {
            Tensor out = evaluate(x);
            const float loss = TargetLoss(out.to_host_vector(), target, &seed);
            if (loss == 0.0f) {
                const double d = distance(x);
                if (d < best_distance) {
                    best_distance = d;
                    best = x;
                }
            }
            std::vector<float> grad(un, 0.0f);
            if (loss > 0.0f) {
                grad = ctx.backward_pass(Tensor(out.shape(), out.backend(), seed, out.device())).to_host_vector();
            }
            // Proximal step: a gradient step on the prediction loss, then soft-thresholding toward
            // the input (the proximal operator of the L1 distance), then the limits.
            const float lr = options.learning_rate;
            for (size_t j = 0; j < un; ++j) {
                if (frozen[j]) continue;
                const float v = x[j] - lr * lambda * grad[j] - x0[j];
                const float shrink = lr / scale[j];
                const float moved = v > shrink ? v - shrink : (v < -shrink ? v + shrink : 0.0f);
                x[j] = clip(j, x0[j] + moved);
            }
        }
        if (!best.empty()) break;
        lambda *= options.lambda_growth;
    }
    result.final_lambda = lambda;

    // Rounding and one-hot projection, then the verdict on what is actually returned.
    std::vector<float> cf = best.empty() ? x : best;
    for (int64_t j : constraints.integer) {
        const auto k = static_cast<size_t>(j);
        cf[k] = clip(k, std::round(cf[k]));
    }
    for (const auto& group : constraints.one_hot_groups) {
        if (group.empty()) continue;
        int64_t arg = group.front();
        for (int64_t j : group) {
            if (cf[static_cast<size_t>(j)] > cf[static_cast<size_t>(arg)]) arg = j;
        }
        for (int64_t j : group) cf[static_cast<size_t>(j)] = j == arg ? 1.0f : 0.0f;
    }
    // Snapping a category back to one-hot can undo a change the search only made partway (a
    // category moved from 0 to 0.3 snaps back to 0). If that happens, try every category of
    // every group, one at a time, and keep the nearest valid result.
    auto reaches = [&](const std::vector<float>& v) {
        return TargetLoss(evaluate(v).to_host_vector(), target, nullptr) == 0.0f;
    };
    if (!reaches(cf)) {
        std::vector<float> nearest;
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (const auto& group : constraints.one_hot_groups) {
            if (std::any_of(group.begin(), group.end(), [&](int64_t j) { return frozen[static_cast<size_t>(j)]; })) {
                continue;
            }
            for (int64_t on : group) {
                std::vector<float> trial = cf;
                for (int64_t j : group) trial[static_cast<size_t>(j)] = j == on ? 1.0f : 0.0f;
                const double d = distance(trial);
                if (d < nearest_distance && reaches(trial)) {
                    nearest_distance = d;
                    nearest = trial;
                }
            }
        }
        if (!nearest.empty()) cf = nearest;
    }
    Tensor out = evaluate(cf);
    result.valid = TargetLoss(out.to_host_vector(), target, nullptr) == 0.0f;
    result.output_after = out.read_element(target.index);
    result.counterfactual = Tensor(input.shape(), input.backend(), cf, input.device());
    result.distance = static_cast<float>(distance(cf));
    double l2 = 0.0;
    for (size_t j = 0; j < un; ++j) {
        const double dj = cf[j] - x0[j];
        l2 += dj * dj;
        if (std::fabs(dj) > options.change_tolerance * scale[j]) ++result.num_changed;
    }
    result.l2 = static_cast<float>(std::sqrt(l2));
    return result;
}

}  // namespace pulsatrix
