#include "pulsatrix/counterfactual.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
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

namespace {

// Everything both searches share: checked constraints, the distance, the limits, the target test,
// and the finishing step (rounding, one-hot snapping, metrics).
class Search {
public:
    using Outputs = std::function<std::vector<float>(const std::vector<float>&)>;

    Search(const char* where, const Tensor& input, const CounterfactualTarget& target,
           const CounterfactualConstraints& constraints, float change_tolerance, Outputs outputs)
        : where_(where), input_(input), target_(target), c_(constraints), tolerance_(change_tolerance),
          outputs_(std::move(outputs)), x0_(input.to_host_vector()), n_(x0_.size()) {
        if (!(change_tolerance >= 0.0f)) fail("change_tolerance must not be negative");
        if (target.kind == CounterfactualTarget::Kind::Range && !(target.low <= target.high)) {
            fail("the target range needs low <= high");
        }
        auto check_length = [&](const std::vector<float>& v, const char* what) {
            if (!v.empty() && v.size() != n_) fail(std::string(what) + " needs one value per feature");
        };
        check_length(c_.scale, "scale");
        check_length(c_.lower, "lower");
        check_length(c_.upper, "upper");
        const auto n = static_cast<int64_t>(n_);
        CheckIndices(c_.immutable, n, "immutable");
        CheckIndices(c_.integer, n, "integer");
        for (const auto& g : c_.one_hot_groups) CheckIndices(g, n, "one-hot");
        scale_ = c_.scale.empty() ? std::vector<float>(n_, 1.0f) : c_.scale;
        for (float s : scale_) {
            if (!(s > 0.0f)) fail("every scale must be positive");
        }
        frozen_.assign(n_, false);
        for (int64_t j : c_.immutable) frozen_[static_cast<size_t>(j)] = true;
        const std::vector<float> z = outputs_(x0_);
        if (target.index < 0 || target.index >= static_cast<int64_t>(z.size())) {
            fail("target index is out of range for an output of " + std::to_string(z.size()));
        }
        if (target.kind == CounterfactualTarget::Kind::Class && z.size() < 2) {
            fail("a class target needs at least 2 outputs");
        }
        output_before_ = z[static_cast<size_t>(target.index)];
    }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument(std::string(where_) + ": " + what);
    }
    const std::vector<float>& x0() const { return x0_; }
    const std::vector<float>& scale() const { return scale_; }
    bool frozen(size_t j) const { return frozen_[j]; }
    size_t size() const { return n_; }
    std::vector<float> outputs(const std::vector<float>& x) const { return outputs_(x); }

    float clip(size_t j, float v) const {
        if (!c_.lower.empty()) v = std::max(v, c_.lower[j]);
        if (!c_.upper.empty()) v = std::min(v, c_.upper[j]);
        return v;
    }
    double distance(const std::vector<float>& x) const {
        double d = 0.0;
        for (size_t j = 0; j < n_; ++j) d += std::fabs(x[j] - x0_[j]) / scale_[j];
        return d;
    }
    bool reaches(const std::vector<float>& x) const { return TargetLoss(outputs_(x), target_, nullptr) == 0.0f; }

    // Rounds integer features and snaps one-hot groups (trying every category of every group if
    // snapping loses the target), then judges and measures what is returned.
    CounterfactualResult finish(std::vector<float> cf) const {
        for (int64_t j : c_.integer) {
            const auto k = static_cast<size_t>(j);
            cf[k] = clip(k, std::round(cf[k]));
        }
        for (const auto& group : c_.one_hot_groups) {
            if (group.empty()) continue;
            int64_t arg = group.front();
            for (int64_t j : group) {
                if (cf[static_cast<size_t>(j)] > cf[static_cast<size_t>(arg)]) arg = j;
            }
            for (int64_t j : group) cf[static_cast<size_t>(j)] = j == arg ? 1.0f : 0.0f;
        }
        // Snapping a category back to one-hot can undo a change made only partway (a category
        // moved from 0 to 0.3 snaps back to 0).
        if (!reaches(cf)) {
            std::vector<float> nearest;
            double nearest_distance = std::numeric_limits<double>::infinity();
            for (const auto& group : c_.one_hot_groups) {
                if (std::any_of(group.begin(), group.end(), [&](int64_t j) { return frozen_[static_cast<size_t>(j)]; })) {
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
        const std::vector<float> z = outputs_(cf);
        CounterfactualResult r{Tensor(input_.shape(), input_.backend(), cf, input_.device())};
        r.valid = TargetLoss(z, target_, nullptr) == 0.0f;
        r.output_before = output_before_;
        r.output_after = z[static_cast<size_t>(target_.index)];
        r.distance = static_cast<float>(distance(cf));
        double l2 = 0.0;
        for (size_t j = 0; j < n_; ++j) {
            const double dj = cf[j] - x0_[j];
            l2 += dj * dj;
            if (std::fabs(dj) > tolerance_ * scale_[j]) ++r.num_changed;
        }
        r.l2 = static_cast<float>(std::sqrt(l2));
        return r;
    }

private:
    const char* where_;
    const Tensor& input_;
    const CounterfactualTarget& target_;
    const CounterfactualConstraints& c_;
    float tolerance_;
    Outputs outputs_;
    std::vector<float> x0_;
    size_t n_;
    std::vector<float> scale_;
    std::vector<bool> frozen_;
    float output_before_ = 0.0f;
};

// A platform-independent generator (std::mt19937_64's output is fixed by the standard).
class Rng {
public:
    explicit Rng(uint64_t seed) : engine_(seed) {}
    double uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }
    double normal() {  // Box-Muller
        const double u1 = 1.0 - uniform();
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

private:
    std::mt19937_64 engine_;
};

}  // namespace

CounterfactualResult FindCounterfactual(ExplainerContext& ctx, const Tensor& input, const CounterfactualTarget& target,
                                        const CounterfactualConstraints& constraints,
                                        const CounterfactualOptions& options) {
    if (!(options.lambda > 0.0f) || !(options.lambda_growth >= 1.0f) || options.max_rounds < 1 ||
        options.steps_per_round < 1 || !(options.learning_rate > 0.0f)) {
        throw std::invalid_argument("FindCounterfactual: invalid options");
    }
    auto forward = [&](const std::vector<float>& x) {
        return ctx.forward_pass(Tensor(input.shape(), input.backend(), x, input.device()));
    };
    const Search search("FindCounterfactual", input, target, constraints, options.change_tolerance,
                        [&](const std::vector<float>& x) { return forward(x).to_host_vector(); });
    const std::vector<float>& x0 = search.x0();
    const size_t un = search.size();

    std::vector<float> x = x0;
    std::vector<float> best;
    double best_distance = std::numeric_limits<double>::infinity();
    float lambda = options.lambda;
    std::vector<float> seed;
    int rounds = 0;
    for (int round = 0; round < options.max_rounds; ++round) {
        rounds = round + 1;
        for (int step = 0; step < options.steps_per_round; ++step) {
            Tensor out = forward(x);
            const float loss = TargetLoss(out.to_host_vector(), target, &seed);
            if (loss == 0.0f) {
                const double d = search.distance(x);
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
                if (search.frozen(j)) continue;
                const float v = x[j] - lr * lambda * grad[j] - x0[j];
                const float shrink = lr / search.scale()[j];
                const float moved = v > shrink ? v - shrink : (v < -shrink ? v + shrink : 0.0f);
                x[j] = search.clip(j, x0[j] + moved);
            }
        }
        if (!best.empty()) break;
        lambda *= options.lambda_growth;
    }
    CounterfactualResult result = search.finish(best.empty() ? x : best);
    result.rounds = rounds;
    result.final_lambda = lambda;
    return result;
}

CounterfactualResult GrowingSpheresCounterfactual(const std::function<Tensor(const Tensor&)>& predict,
                                                  const Tensor& input, const CounterfactualTarget& target,
                                                  const CounterfactualConstraints& constraints,
                                                  const GrowingSpheresOptions& options) {
    if (options.samples_per_layer < 1 || !(options.initial_radius > 0.0f) || options.max_layers < 1) {
        throw std::invalid_argument("GrowingSpheresCounterfactual: invalid options");
    }
    const Search search("GrowingSpheresCounterfactual", input, target, constraints, options.change_tolerance,
                        [&](const std::vector<float>& x) {
                            return predict(Tensor(input.shape(), input.backend(), x, input.device())).to_host_vector();
                        });
    const std::vector<float>& x0 = search.x0();
    std::vector<size_t> free;
    for (size_t j = 0; j < search.size(); ++j) {
        if (!search.frozen(j)) free.push_back(j);
    }
    CounterfactualResult unchanged = search.finish(x0);
    if (free.empty() || unchanged.valid) {
        return unchanged;
    }
    const double dim = static_cast<double>(free.size());
    Rng rng(options.seed);

    // One point uniform in the shell a <= |z| <= b around the input, in scale units, then limited.
    auto sample = [&](double a, double b) {
        std::vector<double> dir(free.size());
        double norm = 0.0;
        for (double& v : dir) {
            v = rng.normal();
            norm += v * v;
        }
        norm = std::sqrt(norm);
        const double r = std::pow(std::pow(a, dim) + rng.uniform() * (std::pow(b, dim) - std::pow(a, dim)), 1.0 / dim);
        std::vector<float> x = x0;
        for (size_t i = 0; i < free.size(); ++i) {
            const size_t j = free[i];
            x[j] = search.clip(j, static_cast<float>(x0[j] + dir[i] / norm * r * search.scale()[j]));
        }
        return x;
    };
    // The nearest (in scaled L2) of a layer's points that reaches the target, or empty.
    auto nearest_enemy = [&](double a, double b) {
        std::vector<float> best;
        double best_d = std::numeric_limits<double>::infinity();
        for (int i = 0; i < options.samples_per_layer; ++i) {
            std::vector<float> x = sample(a, b);
            double d = 0.0;
            for (size_t j : free) d += std::pow((x[j] - x0[j]) / search.scale()[j], 2.0);
            if (d < best_d && search.reaches(x)) {
                best_d = d;
                best = std::move(x);
            }
        }
        return best;
    };

    // Shrink the first ball until it holds no point that reaches the target, then grow outward
    // in shells of the same width until one does (Laugel et al. 2018).
    double eta = options.initial_radius;
    int layers = 0;
    std::vector<float> enemy = nearest_enemy(0.0, eta);
    ++layers;
    while (!enemy.empty() && layers < options.max_layers) {
        eta /= 2.0;
        std::vector<float> closer = nearest_enemy(0.0, eta);
        ++layers;
        if (closer.empty()) break;
        enemy = std::move(closer);
    }
    for (double a = eta; enemy.empty() && layers < options.max_layers; a += eta) {
        enemy = nearest_enemy(a, a + eta);
        ++layers;
    }
    if (enemy.empty()) {
        CounterfactualResult r = search.finish(x0);
        r.rounds = layers;
        return r;
    }
    // Feature selection: put features back to their original value, smallest change first, as
    // long as the point still reaches the target.
    std::vector<size_t> by_change = free;
    std::sort(by_change.begin(), by_change.end(), [&](size_t a, size_t b) {
        return std::fabs(enemy[a] - x0[a]) / search.scale()[a] < std::fabs(enemy[b] - x0[b]) / search.scale()[b];
    });
    for (size_t j : by_change) {
        std::vector<float> trial = enemy;
        trial[j] = x0[j];
        if (search.reaches(trial)) enemy = std::move(trial);
    }
    CounterfactualResult r = search.finish(enemy);
    r.rounds = layers;
    return r;
}

DiverseCounterfactualResult FindDiverseCounterfactuals(ExplainerContext& ctx, const Tensor& input,
                                                       const CounterfactualTarget& target,
                                                       const CounterfactualConstraints& constraints,
                                                       const DiverseCounterfactualOptions& options) {
    if (options.count < 1 || !(options.proximity_weight >= 0.0f) || !(options.diversity_weight >= 0.0f) ||
        !(options.learning_rate > 0.0f) || options.max_steps < 1 || !(options.initial_spread >= 0.0f)) {
        throw std::invalid_argument("FindDiverseCounterfactuals: invalid options");
    }
    auto forward = [&](const std::vector<float>& x) {
        return ctx.forward_pass(Tensor(input.shape(), input.backend(), x, input.device()));
    };
    const Search search("FindDiverseCounterfactuals", input, target, constraints, options.change_tolerance,
                        [&](const std::vector<float>& x) { return forward(x).to_host_vector(); });
    const std::vector<float>& x0 = search.x0();
    const std::vector<float>& scale = search.scale();
    const size_t n = search.size();
    const auto k = static_cast<size_t>(options.count);

    Rng rng(options.seed);
    std::vector<std::vector<float>> cf(k, x0);
    for (auto& c : cf) {
        for (size_t j = 0; j < n; ++j) {
            if (!search.frozen(j)) {
                c[j] = search.clip(j, static_cast<float>(x0[j] + (2.0 * rng.uniform() - 1.0) * options.initial_spread * scale[j]));
            }
        }
    }
    auto pair_distance = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double d = 0.0;
        for (size_t j = 0; j < n; ++j) d += std::fabs(a[j] - b[j]) / scale[j];
        return d;
    };
    auto sign = [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); };

    std::vector<float> seed;
    for (int step = 0; step < options.max_steps; ++step) {
        std::vector<std::vector<double>> grad(k, std::vector<double>(n, 0.0));
        // Prediction hinge and proximity, per counterfactual.
        for (size_t i = 0; i < k; ++i) {
            Tensor out = forward(cf[i]);
            if (TargetLoss(out.to_host_vector(), target, &seed) > 0.0f) {
                const std::vector<float> g =
                    ctx.backward_pass(Tensor(out.shape(), out.backend(), seed, out.device())).to_host_vector();
                for (size_t j = 0; j < n; ++j) grad[i][j] += g[j];
            }
            for (size_t j = 0; j < n; ++j) {
                grad[i][j] += options.proximity_weight * sign(cf[i][j] - x0[j]) / scale[j] / static_cast<double>(k);
            }
        }
        // Diversity: d log det K / d K = K^-1 (K is symmetric), and K_ij = 1 / (1 + d_ij).
        if (k > 1 && options.diversity_weight > 0.0f) {
            std::vector<double> K(k * k), inv(k * k, 0.0);
            for (size_t a = 0; a < k; ++a) {
                for (size_t b = 0; b < k; ++b) K[a * k + b] = 1.0 / (1.0 + pair_distance(cf[a], cf[b]));
                inv[a * k + a] = 1.0;
            }
            // Gauss-Jordan with partial pivoting; a nearly singular K (two equal counterfactuals)
            // gets a small ridge so the step stays finite.
            std::vector<double> m = K;
            for (size_t a = 0; a < k; ++a) m[a * k + a] += 1e-6;
            bool ok = true;
            for (size_t col = 0; col < k && ok; ++col) {
                size_t piv = col;
                for (size_t r = col + 1; r < k; ++r) {
                    if (std::fabs(m[r * k + col]) > std::fabs(m[piv * k + col])) piv = r;
                }
                if (std::fabs(m[piv * k + col]) < 1e-12) {
                    ok = false;
                    break;
                }
                for (size_t c2 = 0; c2 < k; ++c2) {
                    std::swap(m[col * k + c2], m[piv * k + c2]);
                    std::swap(inv[col * k + c2], inv[piv * k + c2]);
                }
                const double p = m[col * k + col];
                for (size_t c2 = 0; c2 < k; ++c2) {
                    m[col * k + c2] /= p;
                    inv[col * k + c2] /= p;
                }
                for (size_t r = 0; r < k; ++r) {
                    if (r == col) continue;
                    const double f = m[r * k + col];
                    for (size_t c2 = 0; c2 < k; ++c2) {
                        m[r * k + c2] -= f * m[col * k + c2];
                        inv[r * k + c2] -= f * inv[col * k + c2];
                    }
                }
            }
            if (ok) {
                for (size_t a = 0; a < k; ++a) {
                    for (size_t b = 0; b < k; ++b) {
                        if (a == b) continue;
                        const double kab = K[a * k + b];
                        // d K_ab / d cf_a[j] = -K_ab^2 * sign(cf_a[j] - cf_b[j]) / scale_j; K_ab and
                        // K_ba both depend on it, hence the factor 2.
                        const double w = 2.0 * inv[a * k + b] * -(kab * kab);
                        for (size_t j = 0; j < n; ++j) {
                            // Minimizing -diversity_weight * log det K.
                            grad[a][j] -= options.diversity_weight * w * sign(cf[a][j] - cf[b][j]) / scale[j];
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (search.frozen(j)) continue;
                cf[i][j] = search.clip(j, static_cast<float>(cf[i][j] - options.learning_rate * grad[i][j]));
            }
        }
    }

    // Post-hoc sparsity: undo each counterfactual's smallest changes while it still reaches the target.
    DiverseCounterfactualResult result;
    for (auto& c : cf) {
        if (search.reaches(c)) {
            std::vector<size_t> order;
            for (size_t j = 0; j < n; ++j) {
                if (!search.frozen(j)) order.push_back(j);
            }
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return std::fabs(c[a] - x0[a]) / scale[a] < std::fabs(c[b] - x0[b]) / scale[b];
            });
            for (size_t j : order) {
                std::vector<float> trial = c;
                trial[j] = x0[j];
                if (search.reaches(trial)) c = std::move(trial);
            }
        }
        result.counterfactuals.push_back(search.finish(c));
        result.counterfactuals.back().rounds = 1;
    }
    std::vector<std::vector<float>> valid;
    for (const auto& r : result.counterfactuals) {
        if (r.valid) valid.push_back(r.counterfactual.to_host_vector());
    }
    result.validity = static_cast<float>(valid.size()) / static_cast<float>(k);
    if (valid.size() > 1) {
        double dsum = 0.0, csum = 0.0;
        size_t pairs = 0;
        for (size_t a = 0; a < valid.size(); ++a) {
            for (size_t b = a + 1; b < valid.size(); ++b) {
                dsum += pair_distance(valid[a], valid[b]);
                size_t differ = 0;
                for (size_t j = 0; j < n; ++j) {
                    if (std::fabs(valid[a][j] - valid[b][j]) > options.change_tolerance * scale[j]) ++differ;
                }
                csum += static_cast<double>(differ) / static_cast<double>(n);
                ++pairs;
            }
        }
        result.diversity = static_cast<float>(dsum / static_cast<double>(pairs));
        result.count_diversity = static_cast<float>(csum / static_cast<double>(pairs));
    }
    return result;
}

float Plausibility(const Tensor& counterfactual, const std::vector<Tensor>& background, const std::vector<float>& scale,
                   int k) {
    if (background.empty() || k < 1) {
        throw std::invalid_argument("Plausibility: needs a non-empty background and k >= 1");
    }
    const std::vector<float> x = counterfactual.to_host_vector();
    if (!scale.empty() && scale.size() != x.size()) {
        throw std::invalid_argument("Plausibility: scale needs one value per feature");
    }
    for (float s : scale) {
        if (!(s > 0.0f)) throw std::invalid_argument("Plausibility: every scale must be positive");
    }
    std::vector<double> d;
    for (const Tensor& t : background) {
        if (t.shape() != counterfactual.shape()) {
            throw std::invalid_argument("Plausibility: background instances must be shaped like the counterfactual");
        }
        const std::vector<float> b = t.to_host_vector();
        double s = 0.0;
        for (size_t j = 0; j < x.size(); ++j) s += std::fabs(x[j] - b[j]) / (scale.empty() ? 1.0f : scale[j]);
        d.push_back(s);
    }
    const size_t m = std::min(d.size(), static_cast<size_t>(k));
    std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(m), d.end());
    double sum = 0.0;
    for (size_t i = 0; i < m; ++i) sum += d[i];
    return static_cast<float>(sum / static_cast<double>(m));
}

}  // namespace pulsatrix
