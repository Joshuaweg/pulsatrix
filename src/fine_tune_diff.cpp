#include "pulsatrix/fine_tune_diff.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/ia3.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/matrix_decompositions.hpp"

namespace pulsatrix {
namespace {

// Where a sample's scores sit in a flat (N, C) or (N, L, C) output.
struct OutputLayout {
    int64_t samples = 0, classes = 0, stride = 0, offset = 0;
};

OutputLayout Layout(const Tensor& output, int64_t position) {
    OutputLayout l;
    if (output.rank() == 2) {
        l.samples = output.shape().dim(0);
        l.classes = output.shape().dim(1);
        l.stride = l.classes;
        return l;
    }
    if (output.rank() != 3) throw std::invalid_argument("DiffRelevance: the output must be (N, C) or (N, L, C)");
    l.samples = output.shape().dim(0);
    const int64_t len = output.shape().dim(1);
    l.classes = output.shape().dim(2);
    const int64_t p = position < 0 ? len + position : position;
    if (p < 0 || p >= len) throw std::invalid_argument("DiffRelevance: position " + std::to_string(position) + " is out of range");
    l.stride = len * l.classes;
    l.offset = p * l.classes;
    return l;
}

// One model's explanation of @p classes (chosen by its own top score when empty, then filled in).
Tensor Explain(Module& model, const Tensor& input, const DiffTarget& target, std::vector<int64_t>& classes,
               std::vector<double>& scores, const LRPRuleConfig& config, Shape& output_shape) {
    const Tensor output = model.forward(input);
    output_shape = output.shape();
    const OutputLayout l = Layout(output, target.position);
    const std::vector<float> values = output.to_host_vector();
    if (classes.empty()) {
        for (int64_t s = 0; s < l.samples; ++s) {
            const auto begin = values.begin() + s * l.stride + l.offset;
            classes.push_back(static_cast<int64_t>(std::max_element(begin, begin + l.classes) - begin));
        }
    }
    if (static_cast<int64_t>(classes.size()) != l.samples) throw std::invalid_argument("DiffRelevance: need one class per sample");
    std::vector<float> seed(values.size(), 0.0f);
    scores.clear();
    for (int64_t s = 0; s < l.samples; ++s) {
        const int64_t c = classes[static_cast<std::size_t>(s)];
        if (c < 0 || c >= l.classes) throw std::invalid_argument("DiffRelevance: class " + std::to_string(c) + " is out of range");
        const std::size_t at = static_cast<std::size_t>(s * l.stride + l.offset + c);
        seed[at] = values[at];
        scores.push_back(values[at]);
    }
    return model.propagate_relevance(Tensor(output.shape(), output.backend(), seed, output.device()), config);
}

Tensor Difference(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("DiffRelevance: the two models' relevance differs in shape");
    std::vector<float> d = a.to_host_vector();
    const std::vector<float> bv = b.to_host_vector();
    for (std::size_t i = 0; i < d.size(); ++i) d[i] -= bv[i];
    return Tensor(a.shape(), a.backend(), d, a.device());
}

// Turns a model's adapters off for its lifetime: the LoRA, IA3 and intervention switches that were on.
class AdaptersOff {
public:
    explicit AdaptersOff(Module& model) {
        // LoraLayers() and IA3Layers() walk the model's children; a bare LinearModule is its own layer.
        std::vector<std::pair<std::string, LinearModule*>> lora = LoraLayers(model), ia3 = IA3Layers(model);
        if (auto* root = dynamic_cast<LinearModule*>(&model)) {
            if (root->lora() != nullptr) lora.emplace_back("(the model)", root);
            if (root->ia3() != nullptr) ia3.emplace_back("(the model)", root);
        }
        for (const auto& [name, layer] : lora) {
            if (layer->lora()->merged()) throw std::invalid_argument("DiffAdapterRelevance: LoRA layer '" + name + "' is merged");
            if (layer->lora()->enabled()) lora_.push_back(layer);
        }
        for (const auto& [name, layer] : ia3) {
            if (layer->ia3()->merged()) throw std::invalid_argument("DiffAdapterRelevance: IA3 layer '" + name + "' is merged");
            if (layer->ia3()->enabled()) ia3_.push_back(layer);
        }
        if (auto* lm = dynamic_cast<CausalLM*>(&model); lm != nullptr && lm->interventions_enabled()) {
            for (int64_t i = 0; i < lm->num_layers(); ++i) {
                if (lm->intervention(i) != nullptr) {
                    lm_ = lm;
                    break;
                }
            }
        }
        if (lora_.empty() && ia3_.empty() && lm_ == nullptr) throw std::invalid_argument("DiffAdapterRelevance: the model has no enabled adapters");
        for (LinearModule* l : lora_) l->set_lora_enabled(false);
        for (LinearModule* l : ia3_) l->set_ia3_enabled(false);
        if (lm_ != nullptr) lm_->set_interventions_enabled(false);
    }
    ~AdaptersOff() {
        for (LinearModule* l : lora_) l->set_lora_enabled(true);
        for (LinearModule* l : ia3_) l->set_ia3_enabled(true);
        if (lm_ != nullptr) lm_->set_interventions_enabled(true);
    }
    AdaptersOff(const AdaptersOff&) = delete;
    AdaptersOff& operator=(const AdaptersOff&) = delete;

private:
    std::vector<LinearModule*> lora_, ia3_;
    CausalLM* lm_ = nullptr;
};

// The output-space singular vectors of a (in, out) weight, largest first: the eigenvectors of W^T W,
// row-major (out, out), and its eigenvalues.
DoubleEigenResult OutputSpace(const Tensor& w) {
    const int64_t in = w.shape().dim(0), out = w.shape().dim(1);
    const std::vector<float> v = w.to_host_vector();
    std::vector<double> gram(static_cast<std::size_t>(out * out), 0.0);
    for (int64_t i = 0; i < in; ++i) {
        const float* row = &v[static_cast<std::size_t>(i * out)];
        for (int64_t a = 0; a < out; ++a) {
            const double ra = row[a];
            if (ra == 0) continue;
            double* g = &gram[static_cast<std::size_t>(a * out)];
            for (int64_t b = a; b < out; ++b) g[b] += ra * row[b];
        }
    }
    for (int64_t a = 0; a < out; ++a)
        for (int64_t b = 0; b < a; ++b) gram[static_cast<std::size_t>(a * out + b)] = gram[static_cast<std::size_t>(b * out + a)];
    return SymmetricEigenInDouble(std::move(gram), out);
}

}  // namespace

RelevanceDiff DiffRelevance(Module& base, Module& tuned, const Tensor& input, const DiffTarget& target, const LRPRuleConfig& config) {
    RelevanceDiff out{Tensor(Shape({0}), input.backend(), input.device()), Tensor(Shape({0}), input.backend(), input.device()),
                      Tensor(Shape({0}), input.backend(), input.device()), target.classes, {}, {}};
    Shape tuned_shape({0}), base_shape({0});
    out.tuned = Explain(tuned, input, target, out.classes, out.tuned_scores, config, tuned_shape);
    out.base = Explain(base, input, target, out.classes, out.base_scores, config, base_shape);
    if (tuned_shape != base_shape) throw std::invalid_argument("DiffRelevance: the two models' outputs differ in shape");
    out.difference = Difference(out.tuned, out.base);
    return out;
}

RelevanceDiff DiffAdapterRelevance(Module& model, const Tensor& input, const DiffTarget& target, const LRPRuleConfig& config) {
    RelevanceDiff out{Tensor(Shape({0}), input.backend(), input.device()), Tensor(Shape({0}), input.backend(), input.device()),
                      Tensor(Shape({0}), input.backend(), input.device()), target.classes, {}, {}};
    Shape tuned_shape({0}), base_shape({0});
    {
        AdaptersOff check(model);  // validates before the tuned run, then restores at once
    }
    out.tuned = Explain(model, input, target, out.classes, out.tuned_scores, config, tuned_shape);
    {
        AdaptersOff off(model);
        out.base = Explain(model, input, target, out.classes, out.base_scores, config, base_shape);
    }
    out.difference = Difference(out.tuned, out.base);
    return out;
}

void TokenRelevanceShift::add(const Tensor& token_ids, const RelevanceDiff& diff) {
    if (token_ids.shape() != diff.tuned.shape() || token_ids.shape() != diff.base.shape() || token_ids.rank() != 2) {
        throw std::invalid_argument("TokenRelevanceShift::add: token ids and relevance must both be (N, L)");
    }
    const int64_t n = token_ids.shape().dim(0), len = token_ids.shape().dim(1);
    if (static_cast<int64_t>(diff.tuned_scores.size()) != n || static_cast<int64_t>(diff.base_scores.size()) != n) {
        throw std::invalid_argument("TokenRelevanceShift::add: the diff has the wrong number of scores");
    }
    const std::vector<float> ids = token_ids.to_host_vector(), t = diff.tuned.to_host_vector(), b = diff.base.to_host_vector();
    for (int64_t s = 0; s < n; ++s) {
        const double ts = diff.tuned_scores[static_cast<std::size_t>(s)], bs = diff.base_scores[static_cast<std::size_t>(s)];
        for (int64_t p = 0; p < len; ++p) {
            const std::size_t k = static_cast<std::size_t>(s * len + p);
            Sum& sum = sums_[static_cast<int64_t>(ids[k])];
            ++sum.count;
            sum.tuned += ts != 0 ? t[k] / ts : 0.0;
            sum.base += bs != 0 ? b[k] / bs : 0.0;
        }
    }
}

std::vector<TokenShift> TokenRelevanceShift::all() const {
    std::vector<TokenShift> out;
    for (const auto& [token, sum] : sums_) {
        const double c = static_cast<double>(sum.count);
        out.push_back({token, sum.count, sum.tuned / c, sum.base / c, (sum.tuned - sum.base) / c});
    }
    return out;
}

std::vector<TokenShift> TokenRelevanceShift::top(int64_t n, bool decrease, int64_t min_count) const {
    std::vector<TokenShift> out;
    for (const TokenShift& s : all()) {
        if (s.count >= min_count) out.push_back(s);
    }
    std::stable_sort(out.begin(), out.end(), [decrease](const TokenShift& a, const TokenShift& b) {
        return decrease ? a.difference < b.difference : a.difference > b.difference;
    });
    if (static_cast<int64_t>(out.size()) > n) out.resize(static_cast<std::size_t>(std::max<int64_t>(n, 0)));
    return out;
}

IntruderResult IntruderDimensions(const Tensor& base, const Tensor& tuned, int64_t k, double epsilon) {
    if (base.rank() != 2 || base.shape() != tuned.shape()) throw std::invalid_argument("IntruderDimensions: the weights must be matrices of one shape");
    if (k < 1) throw std::invalid_argument("IntruderDimensions: k must be at least 1");
    if (!(epsilon > 0 && epsilon <= 1)) throw std::invalid_argument("IntruderDimensions: epsilon must be in (0, 1]");
    const int64_t in = base.shape().dim(0), out = base.shape().dim(1);
    const int64_t rank = std::min(in, out);  // the thin SVD's vectors; the rest only complete a basis
    k = std::min(k, rank);
    const DoubleEigenResult b = OutputSpace(base), t = OutputSpace(tuned);
    IntruderResult r;
    for (int64_t j = 0; j < k; ++j) {
        const double* u = &t.vectors[static_cast<std::size_t>(j * out)];
        double best = 0;
        for (int64_t i = 0; i < rank; ++i) {
            const double* v = &b.vectors[static_cast<std::size_t>(i * out)];
            double dot = 0;
            for (int64_t a = 0; a < out; ++a) dot += u[a] * v[a];
            best = std::max(best, std::fabs(dot));
        }
        r.max_similarity.push_back(best);
        r.singular_values.push_back(std::sqrt(std::max(t.values[static_cast<std::size_t>(j)], 0.0)));
        if (best < epsilon) ++r.count;
    }
    return r;
}

std::vector<LayerIntruders> LoraIntruderDimensions(Module& model, int64_t k, double epsilon) {
    std::vector<LayerIntruders> out;
    for (const auto& [name, layer] : LoraLayers(model)) {
        if (layer->lora()->merged()) continue;
        out.push_back({name, IntruderDimensions(layer->weight(), layer->lora()->effective_weight(layer->weight()), k, epsilon)});
    }
    if (out.empty()) throw std::invalid_argument("LoraIntruderDimensions: the model has no unmerged LoRA adapters");
    return out;
}

std::vector<LayerIntruders> IntruderDimensions(Module& base, Module& tuned, int64_t k, double epsilon) {
    std::map<std::string, LinearModule*> base_layers;
    for (const NamedModuleRef& m : NamedModules(base)) {
        if (auto* l = dynamic_cast<LinearModule*>(m.module)) base_layers[m.name] = l;
    }
    std::vector<LayerIntruders> out;
    for (const NamedModuleRef& m : NamedModules(tuned)) {
        auto* l = dynamic_cast<LinearModule*>(m.module);
        if (l == nullptr) continue;
        const auto it = base_layers.find(m.name);
        if (it == base_layers.end() || it->second->weight().shape() != l->weight().shape()) continue;
        out.push_back({m.name, IntruderDimensions(it->second->weight(), l->weight(), k, epsilon)});
    }
    if (out.empty()) throw std::invalid_argument("IntruderDimensions: the models share no Linear layer");
    return out;
}

}  // namespace pulsatrix
