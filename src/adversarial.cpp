// FGSM and PGD (KS-4). The updates follow torchattacks' FGSM, PGD and PGDL2: the steps, the
// projection onto the epsilon-ball, the clip into the valid range and the random start.

#include "pulsatrix/adversarial.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/module.hpp"
#include "portable_random.hpp"

namespace pulsatrix {

DifferentiableClassifier ClassifierFrom(ExplainerContext& context) {
    return {[&context](const Tensor& x) { return context.forward_pass(x); },
            [&context](const Tensor& g) { return context.backward_pass(g); }};
}

DifferentiableClassifier ClassifierFrom(Module& model) {
    return {[&model](const Tensor& x) { return model.forward(x); }, [&model](const Tensor& g) { return model.backward(g); }};
}

namespace {

// The gradient norms get this added before dividing, as torchattacks does.
constexpr double kTiny = 1e-10;

void CheckOptions(const AttackOptions& o, const char* who) {
    if (!(std::isfinite(o.epsilon) && o.epsilon > 0)) throw std::invalid_argument(std::string(who) + ": epsilon must be finite and positive");
    if (!(o.clip_min < o.clip_max)) throw std::invalid_argument(std::string(who) + ": clip_min must be below clip_max");
}

// The classifier at one point: its logits, the loss of each input's label, and the loss's
// gradient with respect to the input.
struct Evaluation {
    std::vector<int64_t> predicted;
    std::vector<double> loss;
    std::vector<float> gradient;  // empty unless asked for
};

class Attack {
public:
    Attack(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels, DeviceBackend* backend,
           const char* who)
        : model_(model), input_(input), labels_(labels), backend_(explainer_detail::backend_beside(input, backend)), who_(who) {
        if (input.rank() < 1 || input.shape().dim(0) < 1) throw std::invalid_argument(std::string(who) + ": the input needs at least one row");
        n_ = input.shape().dim(0);
        if (static_cast<int64_t>(labels.size()) != n_) {
            throw std::invalid_argument(std::string(who) + ": needs one label per input, " + std::to_string(n_) + ", got " +
                                        std::to_string(labels.size()));
        }
        d_ = input.numel() / n_;
        x0_ = input.to_host_vector();
    }

    int64_t n() const { return n_; }
    int64_t d() const { return d_; }
    const std::vector<float>& x0() const { return x0_; }

    Tensor upload(const std::vector<float>& values) const { return Tensor(input_.shape(), backend_, values, input_.device()); }

    Evaluation evaluate(const std::vector<float>& x, bool want_gradient) const {
        const Tensor logits = model_.forward(upload(x));
        if (logits.rank() != 2 || logits.shape().dim(0) != n_ || logits.shape().dim(1) < 2) {
            throw std::invalid_argument(std::string(who_) + ": the model's output must be (N, C) logits with C >= 2");
        }
        const int64_t c = logits.shape().dim(1);
        const std::vector<float> z = logits.to_host_vector();
        Evaluation e;
        std::vector<float> seed(z.size());
        for (int64_t i = 0; i < n_; ++i) {
            const int64_t label = labels_[static_cast<size_t>(i)];
            if (label < 0 || label >= c) {
                throw std::invalid_argument(std::string(who_) + ": label " + std::to_string(label) + " is outside the " + std::to_string(c) +
                                            " classes");
            }
            const float* row = z.data() + i * c;
            const double top = *std::max_element(row, row + c);
            double sum = 0;
            for (int64_t k = 0; k < c; ++k) sum += std::exp(static_cast<double>(row[k]) - top);
            int64_t best = 0;
            for (int64_t k = 0; k < c; ++k) {
                if (row[k] > row[best]) best = k;
                const double p = std::exp(static_cast<double>(row[k]) - top) / sum;
                seed[static_cast<size_t>(i * c + k)] = static_cast<float>(p - (k == label ? 1.0 : 0.0));
            }
            e.predicted.push_back(best);
            e.loss.push_back(-(static_cast<double>(row[label]) - top - std::log(sum)));
        }
        if (want_gradient) {
            const Tensor g = model_.backward(Tensor(logits.shape(), explainer_detail::backend_beside(logits, backend_), seed, logits.device()));
            if (g.numel() != n_ * d_) throw std::invalid_argument(std::string(who_) + ": the input gradient doesn't have the input's size");
            e.gradient = g.to_host_vector();
        }
        return e;
    }

private:
    const DifferentiableClassifier& model_;
    const Tensor& input_;
    const std::vector<int64_t>& labels_;
    DeviceBackend* backend_;
    const char* who_;
    int64_t n_ = 0, d_ = 0;
    std::vector<float> x0_;
};

double RowNorm(const std::vector<float>& v, int64_t i, int64_t d) {
    double s = 0;
    for (int64_t j = 0; j < d; ++j) s += static_cast<double>(v[static_cast<size_t>(i * d + j)]) * v[static_cast<size_t>(i * d + j)];
    return std::sqrt(s);
}

// x += step along the gradient (against it when targeted): its sign under L∞, its direction under L2.
void StepAlong(std::vector<float>& x, const std::vector<float>& g, double step, AttackNorm norm, bool targeted, int64_t n, int64_t d) {
    const double direction = targeted ? -1.0 : 1.0;
    for (int64_t i = 0; i < n; ++i) {
        const double scale = norm == AttackNorm::L2 ? 1.0 / (RowNorm(g, i, d) + kTiny) : 1.0;
        for (int64_t j = 0; j < d; ++j) {
            const size_t k = static_cast<size_t>(i * d + j);
            const double gk = g[k];
            const double move = norm == AttackNorm::Linf ? (gk > 0 ? 1.0 : gk < 0 ? -1.0 : 0.0) : gk * scale;
            x[k] = static_cast<float>(x[k] + direction * step * move);
        }
    }
}

// Projects x onto the epsilon-ball around x0, then clips it into [lo, hi].
void Project(std::vector<float>& x, const std::vector<float>& x0, const AttackOptions& o, int64_t n, int64_t d) {
    std::vector<float> delta(x.size());
    for (size_t k = 0; k < x.size(); ++k) delta[k] = x[k] - x0[k];
    for (int64_t i = 0; i < n; ++i) {
        const double factor = o.norm == AttackNorm::L2 ? std::min(1.0, o.epsilon / RowNorm(delta, i, d)) : 1.0;
        for (int64_t j = 0; j < d; ++j) {
            const size_t k = static_cast<size_t>(i * d + j);
            double dk = delta[k];
            dk = o.norm == AttackNorm::Linf ? std::clamp(dk, -o.epsilon, o.epsilon) : dk * factor;
            x[k] = static_cast<float>(std::clamp(static_cast<double>(x0[k]) + dk, o.clip_min, o.clip_max));
        }
    }
}

void Clip(std::vector<float>& x, const AttackOptions& o) {
    for (float& v : x) v = static_cast<float>(std::clamp(static_cast<double>(v), o.clip_min, o.clip_max));
}

AttackResult Finish(const Attack& a, std::vector<float> x, const Evaluation& clean, const std::vector<int64_t>& labels,
                    const AttackOptions& o, bool targeted) {
    const Evaluation adv = a.evaluate(x, false);
    AttackResult r{a.upload(x), clean.predicted, adv.predicted, {}, clean.loss, adv.loss, {}, 0.0, 0.0, 0.0, targeted};
    int64_t successes = 0, robust = 0, correct = 0;
    for (int64_t i = 0; i < a.n(); ++i) {
        const size_t s = static_cast<size_t>(i);
        const bool ok = targeted ? adv.predicted[s] == labels[s] : adv.predicted[s] != labels[s];
        r.success.push_back(ok);
        successes += ok ? 1 : 0;
        robust += adv.predicted[s] == labels[s] ? 1 : 0;
        correct += clean.predicted[s] == labels[s] ? 1 : 0;
        double size = 0;
        for (int64_t j = 0; j < a.d(); ++j) {
            const size_t k = static_cast<size_t>(i * a.d() + j);
            const double dk = static_cast<double>(x[k]) - static_cast<double>(a.x0()[k]);
            size = o.norm == AttackNorm::L2 ? size + dk * dk : std::max(size, std::abs(dk));
        }
        r.perturbation.push_back(o.norm == AttackNorm::L2 ? std::sqrt(size) : size);
    }
    const double n = static_cast<double>(a.n());
    r.success_rate = static_cast<double>(successes) / n;
    if (!targeted) {
        r.robust_accuracy = static_cast<double>(robust) / n;
        r.clean_accuracy = static_cast<double>(correct) / n;
    }
    return r;
}

const char* NormName(AttackNorm norm) { return norm == AttackNorm::L2 ? "l2" : "linf"; }

}  // namespace

AttackResult Fgsm(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels, DeviceBackend* backend,
                  const AttackOptions& options, bool targeted) {
    CheckOptions(options, "Fgsm");
    const Attack a(model, input, labels, backend, "Fgsm");
    const Evaluation clean = a.evaluate(a.x0(), true);
    std::vector<float> x = a.x0();
    StepAlong(x, clean.gradient, options.epsilon, options.norm, targeted, a.n(), a.d());
    Clip(x, options);
    return Finish(a, std::move(x), clean, labels, options, targeted);
}

AttackResult Pgd(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels, DeviceBackend* backend,
                 const AttackOptions& options, bool targeted) {
    CheckOptions(options, "Pgd");
    if (options.steps < 1) throw std::invalid_argument("Pgd: steps must be at least 1");
    if (!(std::isfinite(options.step_size) && options.step_size >= 0)) throw std::invalid_argument("Pgd: step_size must be finite and not negative");
    const double step = options.step_size > 0 ? options.step_size : 2.5 * options.epsilon / static_cast<double>(options.steps);
    const Attack a(model, input, labels, backend, "Pgd");
    const Evaluation clean = a.evaluate(a.x0(), false);
    std::vector<float> x = a.x0();
    if (options.random_start) {
        PortableRng rng{options.seed};
        if (options.norm == AttackNorm::Linf) {
            for (float& v : x) v = static_cast<float>(v + (2 * rng.uniform() - 1) * options.epsilon);
        } else {
            // A Gaussian direction scaled to a radius uniform in [0, epsilon], as torchattacks' PGDL2.
            std::vector<float> delta(x.size());
            for (float& v : delta) v = static_cast<float>(rng.gaussian());
            for (int64_t i = 0; i < a.n(); ++i) {
                const double scale = rng.uniform() / RowNorm(delta, i, a.d()) * options.epsilon;
                for (int64_t j = 0; j < a.d(); ++j) {
                    const size_t k = static_cast<size_t>(i * a.d() + j);
                    x[k] = static_cast<float>(x[k] + delta[k] * scale);
                }
            }
        }
        Clip(x, options);
    }
    for (int64_t s = 0; s < options.steps; ++s) {
        const Evaluation e = a.evaluate(x, true);
        StepAlong(x, e.gradient, step, options.norm, targeted, a.n(), a.d());
        Project(x, a.x0(), options, a.n(), a.d());
    }
    return Finish(a, std::move(x), clean, labels, options, targeted);
}

RobustnessCurve RobustAccuracyCurve(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels,
                                    const std::vector<double>& epsilons, DeviceBackend* backend, const AttackOptions& options) {
    if (epsilons.empty()) throw std::invalid_argument("RobustAccuracyCurve: needs at least one epsilon");
    RobustnessCurve curve;
    curve.attack = "pgd";
    curve.norm = options.norm;
    curve.n = input.rank() >= 1 ? input.shape().dim(0) : 0;
    curve.steps = options.steps;
    for (double eps : epsilons) {
        if (!(std::isfinite(eps) && eps > 0)) throw std::invalid_argument("RobustAccuracyCurve: every epsilon must be finite and positive");
        AttackOptions o = options;
        o.epsilon = eps;
        o.step_size = options.step_size > 0 ? options.step_size * eps : 0.0;
        const AttackResult r = Pgd(model, input, labels, backend, o, false);
        curve.clean_accuracy = r.clean_accuracy;
        RobustnessPoint p;
        p.epsilon = eps;
        p.robust_accuracy = r.robust_accuracy;
        int64_t correct = 0, flipped = 0, successes = 0;
        double size = 0;
        for (size_t i = 0; i < labels.size(); ++i) {
            const bool was_right = r.clean_predicted[i] == labels[i];
            correct += was_right ? 1 : 0;
            flipped += was_right && r.success[i] ? 1 : 0;
            if (r.success[i]) size += r.perturbation[i], ++successes;
        }
        p.flip_rate = correct > 0 ? static_cast<double>(flipped) / static_cast<double>(correct) : 0.0;
        p.mean_perturbation = successes > 0 ? size / static_cast<double>(successes) : 0.0;
        curve.points.push_back(p);
    }
    return curve;
}

JsonValue ToJson(const AttackResult& r, const AttackOptions& o) {
    JsonValue::Array inputs;
    for (size_t i = 0; i < r.success.size(); ++i) {
        inputs.push_back(JsonValue(JsonValue::Object{
            {"clean_predicted", JsonValue(r.clean_predicted[i])},
            {"adversarial_predicted", JsonValue(r.adversarial_predicted[i])},
            {"success", JsonValue(static_cast<bool>(r.success[i]))},
            {"clean_loss", JsonValue(r.clean_loss[i])},
            {"adversarial_loss", JsonValue(r.adversarial_loss[i])},
            {"perturbation", JsonValue(r.perturbation[i])},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.attack.v1"},
        {"norm", NormName(o.norm)},
        {"epsilon", JsonValue(o.epsilon)},
        {"targeted", JsonValue(r.targeted)},
        {"n", JsonValue(static_cast<int64_t>(r.success.size()))},
        {"success_rate", JsonValue(r.success_rate)},
        {"clean_accuracy", JsonValue(r.clean_accuracy)},
        {"robust_accuracy", JsonValue(r.robust_accuracy)},
        {"inputs", JsonValue(std::move(inputs))},
    });
}

JsonValue ToJson(const RobustnessCurve& c) {
    JsonValue::Array points;
    for (const RobustnessPoint& p : c.points) {
        points.push_back(JsonValue(JsonValue::Object{
            {"epsilon", JsonValue(p.epsilon)},
            {"robust_accuracy", JsonValue(p.robust_accuracy)},
            {"flip_rate", JsonValue(p.flip_rate)},
            {"mean_perturbation", JsonValue(p.mean_perturbation)},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.robustness_curve.v1"},
        {"attack", c.attack},
        {"norm", NormName(c.norm)},
        {"n", JsonValue(c.n)},
        {"steps", JsonValue(c.steps)},
        {"clean_accuracy", JsonValue(c.clean_accuracy)},
        {"points", JsonValue(std::move(points))},
    });
}

}  // namespace pulsatrix
