// Explanation sensitivity (KS-4): Quantus's max- and average-sensitivity and local Lipschitz
// estimate over label-keeping perturbations, and the sensitivity at FGSM's point.

#include "pulsatrix/explanation_robustness.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "portable_random.hpp"

namespace pulsatrix {

namespace {

int64_t ArgMax(const Tensor& output) {
    const std::vector<float> v = output.to_host_vector();
    if (v.empty()) throw std::invalid_argument("ExplanationSensitivity: the model's output is empty");
    return static_cast<int64_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

double Distance(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        const double d = static_cast<double>(a[k]) - static_cast<double>(b[k]);
        s += d * d;
    }
    return std::sqrt(s);
}

}  // namespace

SensitivityReport ExplanationSensitivity(const Tensor& input, const ExplainFn& explain, const PredictFn& predict,
                                         const DifferentiableClassifier* classifier, DeviceBackend* backend,
                                         const SensitivityOptions& options) {
    if (input.rank() < 1 || input.shape().dim(0) != 1) throw std::invalid_argument("ExplanationSensitivity: the input must be (1, ...)");
    if (!(std::isfinite(options.radius) && options.radius > 0)) {
        throw std::invalid_argument("ExplanationSensitivity: radius must be finite and positive");
    }
    if (options.samples < 1) throw std::invalid_argument("ExplanationSensitivity: samples must be at least 1");
    if (options.max_halvings < 0) throw std::invalid_argument("ExplanationSensitivity: max_halvings must not be negative");

    const std::vector<float> x0 = input.to_host_vector();
    double mean = 0, largest = 0;
    for (float v : x0) mean += v, largest = std::max(largest, std::abs(static_cast<double>(v)));
    mean /= static_cast<double>(x0.size());
    double var = 0;
    for (float v : x0) var += (v - mean) * (v - mean);
    double scale = std::sqrt(var / static_cast<double>(x0.size()));
    if (!(scale > 0)) scale = largest > 0 ? largest : 1.0;

    SensitivityReport r;
    r.radius = options.radius * scale;
    r.samples = options.samples;
    DeviceBackend* beside = explainer_detail::backend_beside(input, backend);
    const std::vector<float> e0 = explain(input).values.to_host_vector();
    const double e0_norm = Distance(e0, std::vector<float>(e0.size(), 0.0f));
    const int64_t predicted = ArgMax(predict(input));
    auto relative = [&](const std::vector<float>& e) {
        if (e.size() != e0.size()) throw std::invalid_argument("ExplanationSensitivity: the explanation's size changed between calls");
        const double d = Distance(e, e0);
        return e0_norm > 0 ? d / e0_norm : d > 0 ? std::numeric_limits<double>::infinity() : 0.0;
    };

    PortableRng rng{options.seed};
    double sum = 0;
    for (int64_t s = 0; s < options.samples; ++s) {
        std::vector<float> x(x0.size());
        for (size_t k = 0; k < x.size(); ++k) x[k] = static_cast<float>(x0[k] + (2 * rng.uniform() - 1) * r.radius);
        const Tensor xt(input.shape(), beside, x, input.device());
        if (ArgMax(predict(xt)) != predicted) continue;
        const std::vector<float> e = explain(xt).values.to_host_vector();
        const double sensitivity = relative(e);
        const double moved = Distance(x, x0);
        r.max_sensitivity = std::max(r.max_sensitivity, sensitivity);
        sum += sensitivity;
        if (moved > 0) r.local_lipschitz = std::max(r.local_lipschitz, Distance(e, e0) / moved);
        ++r.kept;
    }
    if (r.kept > 0) r.avg_sensitivity = sum / static_cast<double>(r.kept);

    if (classifier != nullptr) {
        AttackOptions o;
        o.epsilon = r.radius;
        for (int64_t h = 0; h <= options.max_halvings; ++h, o.epsilon /= 2) {
            const AttackResult a = Fgsm(*classifier, input, {predicted}, backend, o);
            if (a.adversarial_predicted.front() != predicted) continue;
            r.adversarial_sensitivity = relative(explain(a.adversarial).values.to_host_vector());
            r.adversarial_epsilon = o.epsilon;
            break;
        }
    }
    return r;
}

JsonValue ToJson(const SensitivityReport& r) {
    JsonValue::Object o{
        {"schema", "pulsatrix.explanation_sensitivity.v1"},
        {"radius", JsonValue(r.radius)},
        {"samples", JsonValue(r.samples)},
        {"kept", JsonValue(r.kept)},
        {"max_sensitivity", JsonValue(r.max_sensitivity)},
        {"avg_sensitivity", JsonValue(r.avg_sensitivity)},
        {"local_lipschitz", JsonValue(r.local_lipschitz)},
    };
    if (r.adversarial_sensitivity) {
        o.emplace_back("adversarial_sensitivity", JsonValue(*r.adversarial_sensitivity));
        o.emplace_back("adversarial_epsilon", JsonValue(r.adversarial_epsilon));
    }
    return JsonValue(std::move(o));
}

}  // namespace pulsatrix
