#include "pulsatrix/tcav.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

// Rows of a tensor as doubles: n rows of width p, row-major.
struct Rows {
    int64_t n = 0;
    int64_t p = 0;
    std::vector<double> x;

    [[nodiscard]] const double* row(int64_t i) const { return &x[static_cast<std::size_t>(i * p)]; }
};

Rows ToRows(const Tensor& t, const char* function, const char* what) {
    if (t.rank() < 2) throw std::invalid_argument(std::string(function) + ": " + what + " must have shape (n, ...), a row per example");
    Rows r;
    r.n = t.shape().dim(0);
    r.p = r.n == 0 ? 0 : static_cast<int64_t>(t.numel()) / r.n;
    const std::vector<float> host = t.to_host_vector();
    r.x.reserve(host.size());
    for (float v : host) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(function) + ": " + what + " has a NaN or infinite value");
        r.x.push_back(v);
    }
    return r;
}

double Dot(const double* a, const double* b, int64_t p) {
    double s = 0;
    for (int64_t j = 0; j < p; ++j) s += a[j] * b[j];
    return s;
}

bool HeldOut(int64_t i, int64_t every) { return every > 0 && i % every == every - 1; }

// A linear classifier, score = w . x + b, its positive side the concept.
struct Linear {
    std::vector<double> w;
    double b = 0;
};

Linear FitLogistic(const std::vector<const double*>& rows, const std::vector<int>& labels, int64_t p, const CavOptions& o) {
    const auto n = static_cast<double>(rows.size());
    // Standardize by the training rows, fit there, and map the weights back.
    std::vector<double> mean(static_cast<std::size_t>(p), 0.0), sd(static_cast<std::size_t>(p), 0.0);
    for (const double* r : rows) {
        for (int64_t j = 0; j < p; ++j) mean[static_cast<std::size_t>(j)] += r[j];
    }
    for (double& m : mean) m /= n;
    for (const double* r : rows) {
        for (int64_t j = 0; j < p; ++j) sd[static_cast<std::size_t>(j)] += (r[j] - mean[static_cast<std::size_t>(j)]) * (r[j] - mean[static_cast<std::size_t>(j)]);
    }
    for (double& s : sd) s = s > 0 ? std::sqrt(s / n) : 1.0;
    std::vector<double> z(rows.size() * static_cast<std::size_t>(p));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (int64_t j = 0; j < p; ++j) {
            z[i * static_cast<std::size_t>(p) + static_cast<std::size_t>(j)] =
                (rows[i][j] - mean[static_cast<std::size_t>(j)]) / sd[static_cast<std::size_t>(j)];
        }
    }
    // Step size from the loss's curvature bound, 0.25 lambda_max(Z^T Z / n) + l2, with lambda_max
    // estimated by power iteration from a fixed start, so the fit is the same on every platform.
    std::vector<double> v(static_cast<std::size_t>(p), 1.0 / std::sqrt(static_cast<double>(p))), u(static_cast<std::size_t>(p));
    double lambda = 1.0;
    for (int it = 0; it < 30; ++it) {
        std::fill(u.begin(), u.end(), 0.0);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const double* zi = &z[i * static_cast<std::size_t>(p)];
            const double proj = Dot(zi, v.data(), p);
            for (int64_t j = 0; j < p; ++j) u[static_cast<std::size_t>(j)] += proj * zi[j] / n;
        }
        lambda = std::sqrt(Dot(u.data(), u.data(), p));
        if (!(lambda > 0)) break;
        for (int64_t j = 0; j < p; ++j) v[static_cast<std::size_t>(j)] = u[static_cast<std::size_t>(j)] / lambda;
    }
    // The power iteration approaches lambda_max from below; 1.1 leaves room.
    const double step = 1.0 / (0.25 * std::max(lambda, 1e-12) * 1.1 + o.l2);
    std::vector<double> w(static_cast<std::size_t>(p), 0.0), grad(static_cast<std::size_t>(p));
    double b = 0;
    for (int64_t it = 0; it < o.iterations; ++it) {
        std::fill(grad.begin(), grad.end(), 0.0);
        double grad_b = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const double* zi = &z[i * static_cast<std::size_t>(p)];
            const double s = Dot(zi, w.data(), p) + b;
            const double prob = s >= 0 ? 1.0 / (1.0 + std::exp(-s)) : std::exp(s) / (1.0 + std::exp(s));
            const double err = (prob - labels[i]) / n;
            for (int64_t j = 0; j < p; ++j) grad[static_cast<std::size_t>(j)] += err * zi[j];
            grad_b += err;
        }
        for (int64_t j = 0; j < p; ++j) w[static_cast<std::size_t>(j)] -= step * (grad[static_cast<std::size_t>(j)] + o.l2 * w[static_cast<std::size_t>(j)]);
        b -= step * grad_b;
    }
    Linear out;
    out.w.resize(static_cast<std::size_t>(p));
    out.b = b;
    for (int64_t j = 0; j < p; ++j) {
        out.w[static_cast<std::size_t>(j)] = w[static_cast<std::size_t>(j)] / sd[static_cast<std::size_t>(j)];
        out.b -= out.w[static_cast<std::size_t>(j)] * mean[static_cast<std::size_t>(j)];
    }
    return out;
}

// The regularized incomplete beta function I_x(a, b), by its continued fraction (Lentz's method),
// as in Numerical Recipes' betai.
double IncompleteBeta(double a, double b, double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x));
    // The fraction converges fast for x < (a + 1) / (a + b + 2); use the symmetry otherwise.
    if (x > (a + 1) / (a + b + 2)) return 1 - IncompleteBeta(b, a, 1 - x);
    constexpr double kTiny = 1e-300;
    double c = 1, d = 1 - (a + b) * x / (a + 1);
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1 / d;
    double f = d;
    for (int m = 1; m <= 300; ++m) {
        const double m2 = 2.0 * m;
        double num = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
        d = 1 + num * d;
        d = std::fabs(d) < kTiny ? 1 / kTiny : 1 / d;
        c = 1 + num / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        f *= d * c;
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
        d = 1 + num * d;
        d = std::fabs(d) < kTiny ? 1 / kTiny : 1 / d;
        c = 1 + num / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        const double delta = d * c;
        f *= delta;
        if (std::fabs(delta - 1) < 1e-15) break;
    }
    return front * f / a;
}

void CheckLayer(const ExplainerContext& ctx, NodeId layer, const char* function) {
    if (layer > ctx.modules().size()) {
        throw std::invalid_argument(std::string(function) + ": layer " + std::to_string(layer) + " isn't a node of the model, which has " +
                                    std::to_string(ctx.modules().size() + 1) + " (0 is the input, i + 1 module i's output)");
    }
}

Tensor AsRows(const Tensor& t) {
    const int64_t n = t.shape().dim(0);
    return Tensor(Shape({n, n == 0 ? 0 : static_cast<int64_t>(t.numel()) / n}), t.backend(), t.to_host_vector(), t.device());
}

}  // namespace

Cav TrainCav(const Tensor& concept_activations, const Tensor& random_activations, const CavOptions& options) {
    const Rows pos = ToRows(concept_activations, "TrainCav", "concept_activations");
    const Rows neg = ToRows(random_activations, "TrainCav", "random_activations");
    if (pos.p != neg.p || pos.p == 0) throw std::invalid_argument("TrainCav: the two sets of activations must have the same, nonzero width");
    if (options.holdout_every < 0 || options.holdout_every == 1) {
        throw std::invalid_argument("TrainCav: holdout_every must be 0 (none) or at least 2");
    }
    if (options.iterations < 1 || !(options.l2 >= 0)) throw std::invalid_argument("TrainCav: iterations must be positive and l2 not negative");
    const int64_t p = pos.p;
    std::vector<const double*> train, test;
    std::vector<int> train_labels, test_labels;
    for (const auto& [set, label] : {std::pair<const Rows*, int>{&pos, 1}, std::pair<const Rows*, int>{&neg, 0}}) {
        int64_t trained = 0, tested = 0;
        for (int64_t i = 0; i < set->n; ++i) {
            if (HeldOut(i, options.holdout_every)) {
                test.push_back(set->row(i));
                test_labels.push_back(label);
                ++tested;
            } else {
                train.push_back(set->row(i));
                train_labels.push_back(label);
                ++trained;
            }
        }
        if (trained < 2 || (options.holdout_every > 0 && tested < 1)) {
            throw std::invalid_argument("TrainCav: each set needs at least 2 training examples" +
                                        std::string(options.holdout_every > 0 ? " and 1 held out" : ""));
        }
    }

    Linear classifier;
    if (options.method == CavMethod::LogisticRegression) {
        classifier = FitLogistic(train, train_labels, p, options);
    } else {
        std::vector<double> mp(static_cast<std::size_t>(p), 0.0), mn(static_cast<std::size_t>(p), 0.0);
        double np = 0, nn = 0;
        for (std::size_t i = 0; i < train.size(); ++i) {
            std::vector<double>& m = train_labels[i] == 1 ? mp : mn;
            (train_labels[i] == 1 ? np : nn) += 1;
            for (int64_t j = 0; j < p; ++j) m[static_cast<std::size_t>(j)] += train[i][j];
        }
        classifier.w.resize(static_cast<std::size_t>(p));
        for (int64_t j = 0; j < p; ++j) {
            mp[static_cast<std::size_t>(j)] /= np;
            mn[static_cast<std::size_t>(j)] /= nn;
            classifier.w[static_cast<std::size_t>(j)] = mp[static_cast<std::size_t>(j)] - mn[static_cast<std::size_t>(j)];
            classifier.b -= classifier.w[static_cast<std::size_t>(j)] * (mp[static_cast<std::size_t>(j)] + mn[static_cast<std::size_t>(j)]) / 2;
        }
    }
    const double norm = std::sqrt(Dot(classifier.w.data(), classifier.w.data(), p));
    if (!(norm > 0)) throw std::invalid_argument("TrainCav: the concept and random activations can't be told apart at all");

    Cav cav;
    cav.direction.resize(static_cast<std::size_t>(p));
    for (int64_t j = 0; j < p; ++j) cav.direction[static_cast<std::size_t>(j)] = classifier.w[static_cast<std::size_t>(j)] / norm;
    const std::vector<const double*>& scored = test.empty() ? train : test;
    const std::vector<int>& truth = test.empty() ? train_labels : test_labels;
    int64_t correct = 0;
    for (std::size_t i = 0; i < scored.size(); ++i) {
        correct += ((Dot(scored[i], classifier.w.data(), p) + classifier.b > 0) == (truth[i] == 1)) ? 1 : 0;
    }
    cav.accuracy = static_cast<double>(correct) / static_cast<double>(scored.size());
    return cav;
}

std::vector<double> ConceptualSensitivities(const Tensor& gradients, const Cav& cav) {
    const Rows g = ToRows(gradients, "ConceptualSensitivities", "gradients");
    if (g.p != static_cast<int64_t>(cav.direction.size())) {
        throw std::invalid_argument("ConceptualSensitivities: the gradients are " + std::to_string(g.p) + " wide, the CAV " +
                                    std::to_string(cav.direction.size()));
    }
    std::vector<double> out(static_cast<std::size_t>(g.n));
    for (int64_t i = 0; i < g.n; ++i) out[static_cast<std::size_t>(i)] = Dot(g.row(i), cav.direction.data(), g.p);
    return out;
}

double TcavScore(const std::vector<double>& sensitivities) {
    if (sensitivities.empty()) throw std::invalid_argument("TcavScore: no sensitivities");
    const auto positive = std::count_if(sensitivities.begin(), sensitivities.end(), [](double s) { return s > 0; });
    return static_cast<double>(positive) / static_cast<double>(sensitivities.size());
}

TTestResult StudentTTest(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() < 2 || b.size() < 2) throw std::invalid_argument("StudentTTest: each sample needs at least 2 values");
    for (const auto* s : {&a, &b}) {
        for (double v : *s) {
            if (!std::isfinite(v)) throw std::invalid_argument("StudentTTest: a value isn't finite");
        }
    }
    auto moments = [](const std::vector<double>& s) {
        const double mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
        double ss = 0;
        for (double v : s) ss += (v - mean) * (v - mean);
        return std::pair<double, double>{mean, ss};
    };
    const auto [ma, ssa] = moments(a);
    const auto [mb, ssb] = moments(b);
    const auto na = static_cast<double>(a.size()), nb = static_cast<double>(b.size());
    TTestResult r;
    r.degrees_of_freedom = na + nb - 2;
    const double pooled = (ssa + ssb) / r.degrees_of_freedom;
    if (!(pooled > 0)) {
        r.t = ma == mb ? 0.0 : std::copysign(std::numeric_limits<double>::infinity(), ma - mb);
        r.p_value = ma == mb ? 1.0 : 0.0;
        return r;
    }
    r.t = (ma - mb) / std::sqrt(pooled * (1 / na + 1 / nb));
    const double df = r.degrees_of_freedom;
    r.p_value = IncompleteBeta(df / 2, 0.5, df / (df + r.t * r.t));
    return r;
}

Tensor LayerActivations(ExplainerContext& ctx, const Tensor& inputs, NodeId layer) {
    CheckLayer(ctx, layer, "LayerActivations");
    (void)ctx.forward_pass(inputs);
    return AsRows(ctx.activation(layer));
}

Tensor LayerGradients(ExplainerContext& ctx, const Tensor& inputs, NodeId layer, int64_t target, DeviceBackend* backend) {
    CheckLayer(ctx, layer, "LayerGradients");
    const Tensor output = ctx.forward_pass(inputs);
    if (output.rank() != 2) throw std::invalid_argument("LayerGradients: the model's output must be (n, classes)");
    const int64_t n = output.shape().dim(0), classes = output.shape().dim(1);
    if (target < 0 || target >= classes) throw std::invalid_argument("LayerGradients: target is out of range");
    std::vector<float> seed(static_cast<std::size_t>(n * classes), 0.0f);
    for (int64_t i = 0; i < n; ++i) seed[static_cast<std::size_t>(i * classes + target)] = 1.0f;
    (void)ctx.backward_pass(Tensor(output.shape(), backend, seed, output.device()));
    return AsRows(ctx.gradient(layer));
}

TcavResult RunTcav(ExplainerContext& ctx, NodeId layer, int64_t target, const Tensor& class_inputs, const Tensor& concept_inputs,
                   const std::vector<Tensor>& random_inputs, DeviceBackend* backend, const TcavOptions& options) {
    if (random_inputs.size() < 3) throw std::invalid_argument("RunTcav: needs at least 3 sets of random examples");
    if (!(options.alpha > 0 && options.alpha < 1)) throw std::invalid_argument("RunTcav: alpha must be in (0, 1)");
    const Tensor gradients = LayerGradients(ctx, class_inputs, layer, target, backend);
    const Tensor concept_acts = LayerActivations(ctx, concept_inputs, layer);
    std::vector<Tensor> random_acts;
    for (const Tensor& r : random_inputs) random_acts.push_back(LayerActivations(ctx, r, layer));

    TcavResult result;
    const std::size_t sets = random_acts.size();
    for (std::size_t i = 0; i < sets; ++i) {
        const Cav cav = TrainCav(concept_acts, random_acts[i], options.cav);
        result.scores.push_back(TcavScore(ConceptualSensitivities(gradients, cav)));
        result.cav_accuracies.push_back(cav.accuracy);
        const Cav null = TrainCav(random_acts[i], random_acts[(i + 1) % sets], options.cav);
        result.random_scores.push_back(TcavScore(ConceptualSensitivities(gradients, null)));
    }
    result.mean_score = std::accumulate(result.scores.begin(), result.scores.end(), 0.0) / static_cast<double>(sets);
    result.mean_random_score = std::accumulate(result.random_scores.begin(), result.random_scores.end(), 0.0) / static_cast<double>(sets);
    result.test = StudentTTest(result.scores, result.random_scores);
    result.significant = result.test.p_value < options.alpha;
    return result;
}

}  // namespace pulsatrix
