// KS-3: Monte Carlo dropout (Gal and Ghahramani 2016), deep ensembles (Lakshminarayanan, Pritzel
// and Blundell 2017) and the entropy decomposition of their samples (Houlsby et al. 2011; Depeweg et
// al. 2018).

#include "pulsatrix/uncertainty.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"

namespace pulsatrix {
namespace {

double Entropy(const std::vector<double>& p) {
    double h = 0.0;
    for (double v : p) {
        if (v > 0.0) h -= v * std::log(v);
    }
    return h;
}

// Whether @p m drops anything in training mode.
bool Drops(Module& m) {
    if (auto* d = dynamic_cast<DropoutModule*>(&m)) return d->p() > 0.0f;
    if (auto* l = dynamic_cast<LinearModule*>(&m)) return l->lora() != nullptr && l->lora()->dropout() > 0.0f;
    return false;
}

// Restores every module's training mode, parents first so a container's cascade doesn't undo its
// children's.
class ModeGuard {
public:
    explicit ModeGuard(Module& root) {
        saved_.emplace_back(&root, root.is_training());
        for (const NamedModuleRef& m : NamedModules(root)) saved_.emplace_back(m.module, m.module->is_training());
    }
    ~ModeGuard() {
        for (auto& [module, training] : saved_) module->set_training(training);
    }
    ModeGuard(const ModeGuard&) = delete;
    ModeGuard& operator=(const ModeGuard&) = delete;

    [[nodiscard]] const std::vector<std::pair<Module*, bool>>& modules() const { return saved_; }

private:
    std::vector<std::pair<Module*, bool>> saved_;
};

}  // namespace

std::vector<Tensor> McDropoutSamples(Module& model, const std::function<Tensor(const Tensor&)>& predict,
                                     const Tensor& input, int64_t samples) {
    if (samples < 2) throw std::invalid_argument("McDropoutSamples: needs at least 2 samples");
    ModeGuard guard(model);
    std::vector<Module*> dropout;
    for (const auto& entry : guard.modules()) {
        if (Drops(*entry.first)) dropout.push_back(entry.first);
    }
    if (dropout.empty()) {
        throw std::invalid_argument("McDropoutSamples: the model has no dropout layer with p > 0, so every sample would be the same");
    }
    model.set_training(false);
    for (Module* m : dropout) m->set_training(true);
    std::vector<Tensor> out;
    out.reserve(static_cast<size_t>(samples));
    for (int64_t s = 0; s < samples; ++s) out.push_back(predict(input));
    return out;
}

std::vector<Tensor> EnsembleSamples(const std::vector<std::function<Tensor(const Tensor&)>>& members, const Tensor& input) {
    if (members.size() < 2) throw std::invalid_argument("EnsembleSamples: needs at least 2 members");
    std::vector<Tensor> out;
    out.reserve(members.size());
    for (const auto& member : members) out.push_back(member(input));
    return out;
}

UncertaintyDecomposition DecomposeUncertainty(const std::vector<Tensor>& samples) {
    if (samples.size() < 2) throw std::invalid_argument("DecomposeUncertainty: needs at least 2 samples");
    const Shape shape = samples.front().shape();
    if (shape.rank() != 2 || shape.dim(0) < 1 || shape.dim(1) < 2) {
        throw std::invalid_argument("DecomposeUncertainty: each sample must be (N, K) with N >= 1 and K >= 2");
    }
    const int64_t n = shape.dim(0), k = shape.dim(1);
    const double t = static_cast<double>(samples.size());
    std::vector<double> mean(static_cast<size_t>(n * k), 0.0), aleatoric(static_cast<size_t>(n), 0.0);
    std::vector<std::vector<int64_t>> votes(static_cast<size_t>(n));
    for (const Tensor& sample : samples) {
        if (sample.shape() != shape) throw std::invalid_argument("DecomposeUncertainty: the samples' shapes differ");
        const std::vector<float> p = sample.to_host_vector();
        for (int64_t i = 0; i < n; ++i) {
            std::vector<double> row(static_cast<size_t>(k));
            for (int64_t j = 0; j < k; ++j) {
                const double v = p[static_cast<size_t>(i * k + j)];
                if (!std::isfinite(v) || v < 0.0) {
                    throw std::invalid_argument("DecomposeUncertainty: a probability is negative or not finite");
                }
                row[static_cast<size_t>(j)] = v;
                mean[static_cast<size_t>(i * k + j)] += v / t;
            }
            aleatoric[static_cast<size_t>(i)] += Entropy(row) / t;
            votes[static_cast<size_t>(i)].push_back(std::max_element(row.begin(), row.end()) - row.begin());
        }
    }
    std::vector<int64_t> predicted;
    std::vector<double> total, epistemic, agreement;
    std::vector<float> mean_f(mean.size());
    for (int64_t i = 0; i < n; ++i) {
        const auto first = mean.begin() + i * k;
        const std::vector<double> row(first, first + k);
        const int64_t top = std::max_element(row.begin(), row.end()) - row.begin();
        const double h = Entropy(row);
        predicted.push_back(top);
        total.push_back(h);
        // Jensen: H[E p] >= E H[p]; the difference can come out a rounding error below 0.
        epistemic.push_back(std::max(0.0, h - aleatoric[static_cast<size_t>(i)]));
        const std::vector<int64_t>& v = votes[static_cast<size_t>(i)];
        agreement.push_back(static_cast<double>(std::count(v.begin(), v.end(), top)) / t);
        for (int64_t j = 0; j < k; ++j) mean_f[static_cast<size_t>(i * k + j)] = static_cast<float>(row[static_cast<size_t>(j)]);
    }
    return UncertaintyDecomposition{Tensor(shape, samples.front().backend(), mean_f, samples.front().device()),
                                    std::move(predicted),
                                    std::move(total),
                                    std::move(aleatoric),
                                    std::move(epistemic),
                                    std::move(agreement),
                                    static_cast<int64_t>(samples.size())};
}

GaussianMixturePrediction CombineGaussians(const std::vector<std::vector<double>>& means,
                                           const std::vector<std::vector<double>>& variances) {
    if (means.size() < 2) throw std::invalid_argument("CombineGaussians: needs at least 2 members");
    const size_t n = means.front().size();
    if (n == 0) throw std::invalid_argument("CombineGaussians: the members predict nothing");
    if (!variances.empty() && variances.size() != means.size()) {
        throw std::invalid_argument("CombineGaussians: needs a variance per member, or none");
    }
    for (size_t m = 0; m < means.size(); ++m) {
        if (means[m].size() != n || (!variances.empty() && variances[m].size() != n)) {
            throw std::invalid_argument("CombineGaussians: the members predict different numbers of inputs");
        }
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(means[m][i])) throw std::invalid_argument("CombineGaussians: a mean isn't finite");
            if (!variances.empty() && !(std::isfinite(variances[m][i]) && variances[m][i] >= 0.0)) {
                throw std::invalid_argument("CombineGaussians: a variance is negative or not finite");
            }
        }
    }
    const double count = static_cast<double>(means.size());
    GaussianMixturePrediction g;
    for (size_t i = 0; i < n; ++i) {
        double mu = 0.0, noise = 0.0;
        for (size_t m = 0; m < means.size(); ++m) {
            mu += means[m][i] / count;
            if (!variances.empty()) noise += variances[m][i] / count;
        }
        // The spread of the means, computed around their mean rather than as E[mu^2] - mu^2, which
        // cancels badly when the means are large and close.
        double spread = 0.0;
        for (size_t m = 0; m < means.size(); ++m) spread += (means[m][i] - mu) * (means[m][i] - mu) / count;
        g.mean.push_back(mu);
        g.aleatoric.push_back(noise);
        g.epistemic.push_back(spread);
        g.variance.push_back(noise + spread);
    }
    return g;
}

GaussianNLLLoss::GaussianNLLLoss(DeviceBackend* backend, double min_variance)
    : backend_(backend), min_variance_(min_variance), last_prediction_(Shape({0}), backend) {
    if (!(min_variance > 0.0) || !std::isfinite(min_variance)) {
        throw std::invalid_argument("GaussianNLLLoss: min_variance must be positive");
    }
}

double GaussianNLLLoss::variance(double s) const {
    // softplus(s) = log(1 + e^s), written so neither branch overflows.
    const double softplus = s > 0.0 ? s + std::log1p(std::exp(-s)) : std::log1p(std::exp(s));
    return softplus + min_variance_;
}

float GaussianNLLLoss::forward(const Tensor& prediction, const Tensor& target) {
    const Shape& p = prediction.shape();
    if (p.rank() != 2 || p.dim(1) != 2 || p.dim(0) < 1) {
        throw std::invalid_argument("GaussianNLLLoss: the prediction must be (N, 2), a mean and s for each input");
    }
    const int64_t n = p.dim(0);
    const Shape& t = target.shape();
    const bool column = t.rank() == 2 && t.dim(0) == n && t.dim(1) == 1;
    if (!column && !(t.rank() == 1 && t.dim(0) == n)) {
        throw std::invalid_argument("GaussianNLLLoss: the target must be (N, 1) or (N)");
    }
    const std::vector<float> x = prediction.to_host_vector();
    std::vector<float> y = target.to_host_vector();
    double loss = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double mu = x[static_cast<size_t>(2 * i)], s = x[static_cast<size_t>(2 * i + 1)], yi = y[static_cast<size_t>(i)];
        if (!std::isfinite(mu) || !std::isfinite(s) || !std::isfinite(yi)) {
            throw std::invalid_argument("GaussianNLLLoss: a prediction or target isn't finite");
        }
        const double var = variance(s);
        loss += 0.5 * (std::log(var) + (yi - mu) * (yi - mu) / var);
    }
    last_prediction_ = prediction;
    last_target_ = std::move(y);
    has_forwarded_ = true;
    return static_cast<float>(loss / static_cast<double>(n));
}

Tensor GaussianNLLLoss::backward() const {
    if (!has_forwarded_) throw std::logic_error("GaussianNLLLoss::backward: forward() hasn't been called");
    const std::vector<float> x = last_prediction_.to_host_vector();
    const int64_t n = last_prediction_.shape().dim(0);
    std::vector<float> grad(x.size());
    for (int64_t i = 0; i < n; ++i) {
        const double mu = x[static_cast<size_t>(2 * i)], s = x[static_cast<size_t>(2 * i + 1)];
        const double r = mu - last_target_[static_cast<size_t>(i)];
        const double var = variance(s);
        const double sigmoid = 1.0 / (1.0 + std::exp(-s));  // d softplus / ds
        grad[static_cast<size_t>(2 * i)] = static_cast<float>(r / var / static_cast<double>(n));
        grad[static_cast<size_t>(2 * i + 1)] =
            static_cast<float>(0.5 * (1.0 / var - r * r / (var * var)) * sigmoid / static_cast<double>(n));
    }
    return Tensor(last_prediction_.shape(), backend_, grad, last_prediction_.device());
}

JsonValue ToJson(const UncertaintyDecomposition& d) {
    const auto numbers = [](const std::vector<double>& v) {
        JsonValue::Array a;
        for (double x : v) a.emplace_back(x);
        return JsonValue(std::move(a));
    };
    JsonValue::Array predicted;
    for (int64_t c : d.predicted) predicted.emplace_back(c);
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.uncertainty.v1"},
        {"samples", JsonValue(d.samples)},
        {"units", "nats"},
        {"predicted", JsonValue(std::move(predicted))},
        {"total", numbers(d.total)},
        {"aleatoric", numbers(d.aleatoric)},
        {"epistemic", numbers(d.epistemic)},
        {"agreement", numbers(d.agreement)},
    });
}

}  // namespace pulsatrix
