#include "pulsatrix/loreft.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <stdexcept>

#include "lrp_rules.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

// Notation: P = rotate_layer = R^T (d, r), Q = W^T (d, r), both row-major; for the edited rows h (S, d),
// u = h (Q - P) + b (S, r) and the edit is u P^T.

// [0, 1) from raw mt19937_64 output, the same on every platform.
double Uniform(std::mt19937_64& rng) { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }

// Standard normal by Box-Muller on raw output (std::normal_distribution differs between libraries).
double Normal(std::mt19937_64& rng) {
    const double u1 = 1.0 - Uniform(rng), u2 = Uniform(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// Orthonormalizes the columns of a row-major (d, r) matrix in place, by modified Gram-Schmidt.
void OrthonormalColumns(std::vector<double>& m, int64_t d, int64_t r) {
    for (int64_t j = 0; j < r; ++j) {
        for (int64_t k = 0; k < j; ++k) {
            double dot = 0;
            for (int64_t i = 0; i < d; ++i) dot += m[static_cast<std::size_t>(i * r + j)] * m[static_cast<std::size_t>(i * r + k)];
            for (int64_t i = 0; i < d; ++i) m[static_cast<std::size_t>(i * r + j)] -= dot * m[static_cast<std::size_t>(i * r + k)];
        }
        double norm = 0;
        for (int64_t i = 0; i < d; ++i) norm += m[static_cast<std::size_t>(i * r + j)] * m[static_cast<std::size_t>(i * r + j)];
        norm = std::sqrt(norm);
        if (!(norm > 1e-12)) throw std::invalid_argument("LoReFTIntervention: the rotation's columns are linearly dependent");
        for (int64_t i = 0; i < d; ++i) m[static_cast<std::size_t>(i * r + j)] /= norm;
    }
}

std::vector<float> ToFloat(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

std::vector<float> Transposed(const std::vector<float>& m, int64_t rows, int64_t cols) {
    std::vector<float> t(m.size());
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) t[static_cast<std::size_t>(j * rows + i)] = m[static_cast<std::size_t>(i * cols + j)];
    return t;
}

Tensor Copy(const Tensor& t) {
    Tensor out(t.shape(), t.backend(), t.device());
    t.backend()->copy(out.data(), t.data(), static_cast<std::size_t>(t.numel()) * sizeof(float), CopyDirection::DeviceToDevice);
    return out;
}

}  // namespace

LoReFTIntervention::LoReFTIntervention(int64_t d_model, int64_t rank, int64_t first, int64_t last, DeviceBackend* backend,
                                       uint64_t seed)
    : d_(d_model),
      rank_(rank),
      first_(first),
      last_(last),
      backend_(backend),
      r_(Shape({std::max<int64_t>(d_model, 1), std::max<int64_t>(rank, 1)}), backend, backend->device()),
      w_(Shape({std::max<int64_t>(d_model, 1), std::max<int64_t>(rank, 1)}), backend, backend->device()),
      b_(Shape({std::max<int64_t>(rank, 1)}), backend, backend->device()),
      r_grad_(r_.shape(), backend, backend->device()),
      w_grad_(w_.shape(), backend, backend->device()),
      b_grad_(b_.shape(), backend, backend->device()),
      last_indices_(Shape({0}), backend, backend->device()),
      last_selected_(Shape({0}), backend, backend->device()),
      last_coords_(Shape({0}), backend, backend->device()),
      last_edit_relevance_(Shape({0}), backend, backend->device()) {
    if (rank < 1 || rank > d_model) throw std::invalid_argument("LoReFTIntervention: rank must be in [1, d_model]");
    if (first < 0 || last < 0) throw std::invalid_argument("LoReFTIntervention: first and last must be >= 0");
    std::mt19937_64 rng(seed);
    std::vector<double> p(static_cast<std::size_t>(d_model * rank));
    for (double& v : p) v = Normal(rng);
    OrthonormalColumns(p, d_model, rank);
    const double bound = 1.0 / std::sqrt(static_cast<double>(d_model));
    std::vector<float> q(static_cast<std::size_t>(d_model * rank)), b(static_cast<std::size_t>(rank));
    for (float& v : q) v = static_cast<float>((2 * Uniform(rng) - 1) * bound);
    for (float& v : b) v = static_cast<float>((2 * Uniform(rng) - 1) * bound);
    set_parameters(ToFloat(p), q, b);
}

void LoReFTIntervention::set_parameters(const std::vector<float>& rotation, const std::vector<float>& source_weight,
                                        const std::vector<float>& source_bias) {
    if (static_cast<int64_t>(rotation.size()) != d_ * rank_ || static_cast<int64_t>(source_weight.size()) != d_ * rank_ ||
        static_cast<int64_t>(source_bias.size()) != rank_) {
        throw std::invalid_argument("LoReFTIntervention::set_parameters: expected (d, r), (d, r) and (r) values");
    }
    const DeviceType device = backend_->device();
    const bool train_r = r_.requires_grad(), train_w = w_.requires_grad(), train_b = b_.requires_grad();
    r_ = Tensor(r_.shape(), backend_, rotation, device);
    w_ = Tensor(w_.shape(), backend_, source_weight, device);
    b_ = Tensor(b_.shape(), backend_, source_bias, device);
    r_.set_requires_grad(train_r);
    w_.set_requires_grad(train_w);
    b_.set_requires_grad(train_b);
}

void LoReFTIntervention::orthonormalize() {
    const std::vector<float> r = r_.to_host_vector();
    std::vector<double> p(r.begin(), r.end());
    OrthonormalColumns(p, d_, rank_);
    set_parameters(ToFloat(p), w_.to_host_vector(), b_.to_host_vector());
}

double LoReFTIntervention::orthonormality_error() const {
    const std::vector<float> p = r_.to_host_vector();
    double worst = 0;
    for (int64_t j = 0; j < rank_; ++j) {
        for (int64_t k = 0; k <= j; ++k) {
            double dot = 0;
            for (int64_t i = 0; i < d_; ++i) dot += static_cast<double>(p[static_cast<std::size_t>(i * rank_ + j)]) * p[static_cast<std::size_t>(i * rank_ + k)];
            worst = std::max(worst, std::fabs(dot - (j == k ? 1.0 : 0.0)));
        }
    }
    return worst;
}

std::vector<NamedParamRef> LoReFTIntervention::named_parameters() {
    return {{"rotate_layer", {&r_, &r_grad_}}, {"learned_source.weight", {&w_, &w_grad_}}, {"learned_source.bias", {&b_, &b_grad_}}};
}

Tensor LoReFTIntervention::forward_at(const Tensor& hidden, int64_t start, int64_t prompt_length) {
    if (hidden.rank() != 3 || hidden.shape().dim(2) != d_) {
        throw std::invalid_argument("LoReFTIntervention::forward: hidden states must be (N, L, " + std::to_string(d_) + ")");
    }
    const int64_t n = hidden.shape().dim(0), len = hidden.shape().dim(1);
    last_shape_ = hidden.shape();
    release_tensor(last_edit_relevance_);
    last_rows_.clear();
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t p = start; p < start + len; ++p) {
            if (p < prompt_length && (p < first_ || p >= prompt_length - last_)) last_rows_.push_back(s * len + (p - start));
        }
    }
    Tensor out = Copy(hidden);
    const auto rows = static_cast<std::size_t>(last_rows_.size()), d = static_cast<std::size_t>(d_), r = static_cast<std::size_t>(rank_);
    if (rows == 0) {
        release_tensor(last_selected_);
        release_tensor(last_coords_);
        return out;
    }
    const DeviceType device = backend_->device();
    last_indices_ = Tensor(Shape({static_cast<int64_t>(rows)}), backend_, std::vector<float>(last_rows_.begin(), last_rows_.end()), device);
    last_selected_ = Tensor(Shape({static_cast<int64_t>(rows), d_}), backend_, device);
    backend_->gather_rows(hidden.data(), last_indices_.data(), last_selected_.data(), rows, d);
    Tensor q_minus_p(w_.shape(), backend_, device);
    backend_->axpby(1.0f, w_.data(), -1.0f, r_.data(), q_minus_p.data(), d * r);
    Tensor coords(Shape({static_cast<int64_t>(rows), rank_}), backend_, device);
    backend_->gemm(last_selected_.data(), q_minus_p.data(), coords.data(), rows, d, r);
    last_coords_ = Tensor(coords.shape(), backend_, device);
    backend_->add_row_vector(coords.data(), b_.data(), last_coords_.data(), rows, r);
    Tensor edit(last_selected_.shape(), backend_, device);
    backend_->gemm_ex(last_coords_.data(), false, r_.data(), true, edit.data(), rows, r, d, 0.0f);
    backend_->scatter_add_rows(edit.data(), last_indices_.data(), out.data(), rows, d);
    return out;
}

Tensor LoReFTIntervention::backward(const Tensor& grad_output) {
    if (grad_output.shape() != last_shape_) throw std::invalid_argument("LoReFTIntervention::backward: grad_output must match the last forward's shape");
    Tensor grad_input = Copy(grad_output);
    const auto rows = last_rows_.size(), d = static_cast<std::size_t>(d_), r = static_cast<std::size_t>(rank_);
    if (rows == 0) return grad_input;
    const DeviceType device = backend_->device();
    Tensor g(last_selected_.shape(), backend_, device);
    backend_->gather_rows(grad_output.data(), last_indices_.data(), g.data(), rows, d);
    Tensor gu(last_coords_.shape(), backend_, device);  // dL/du = g P
    backend_->gemm(g.data(), r_.data(), gu.data(), rows, d, r);
    if (r_.requires_grad()) {  // dL/dP = g^T u - h^T (g P)
        backend_->gemm_ex(g.data(), true, last_coords_.data(), false, r_grad_.data(), d, rows, r, 1.0f);
        Tensor through_coords(r_.shape(), backend_, device);
        backend_->gemm_ex(last_selected_.data(), true, gu.data(), false, through_coords.data(), d, rows, r, 0.0f);
        backend_->axpby(-1.0f, through_coords.data(), 1.0f, r_grad_.data(), r_grad_.data(), d * r);
    }
    if (w_.requires_grad()) backend_->gemm_ex(last_selected_.data(), true, gu.data(), false, w_grad_.data(), d, rows, r, 1.0f);
    if (b_.requires_grad()) backend_->column_sums(gu.data(), b_grad_.data(), rows, r, 1.0f);
    Tensor q_minus_p(w_.shape(), backend_, device);
    backend_->axpby(1.0f, w_.data(), -1.0f, r_.data(), q_minus_p.data(), d * r);
    Tensor gh(last_selected_.shape(), backend_, device);  // dL/dh beyond the identity: (g P) (Q - P)^T
    backend_->gemm_ex(gu.data(), false, q_minus_p.data(), true, gh.data(), rows, r, d, 0.0f);
    backend_->scatter_add_rows(gh.data(), last_indices_.data(), grad_input.data(), rows, d);
    return grad_input;
}

Tensor LoReFTIntervention::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (relevance_out.shape() != last_shape_) {
        throw std::invalid_argument("LoReFTIntervention::propagate_relevance: relevance must match the last forward's shape");
    }
    lrp_rules::validate(config, "LoReFTIntervention");
    const DeviceType device = backend_->device();
    Tensor relevance_in = Copy(relevance_out);
    last_edit_relevance_ = Tensor(last_shape_, backend_, device);
    backend_->fill(last_edit_relevance_.data(), 0.0f, static_cast<std::size_t>(last_edit_relevance_.numel()));
    const auto rows = last_rows_.size(), d = static_cast<std::size_t>(d_), r = static_cast<std::size_t>(rank_);
    if (rows == 0) return relevance_in;

    // The edited rows as one Linear layer on [h, h] with weights [I; (Q - P) P^T], bias b P^T, and the
    // pre-bias output z = h + h (Q - P) P^T.
    Tensor q_minus_p(w_.shape(), backend_, device);
    backend_->axpby(1.0f, w_.data(), -1.0f, r_.data(), q_minus_p.data(), d * r);
    std::vector<float> stacked_host(2 * d * d, 0.0f);
    for (std::size_t i = 0; i < d; ++i) stacked_host[i * d + i] = 1.0f;
    Tensor stacked(Shape({2 * d_, d_}), backend_, stacked_host, device);
    backend_->gemm_ex(q_minus_p.data(), false, r_.data(), true, stacked.data() + d * d, d, r, d, 0.0f);
    Tensor bias(Shape({d_}), backend_, device);
    backend_->gemm_ex(b_.data(), false, r_.data(), true, bias.data(), 1, r, d, 0.0f);
    Tensor z(last_selected_.shape(), backend_, device);
    backend_->gemm(last_selected_.data(), stacked.data() + d * d, z.data(), rows, d, d);
    backend_->add(z.data(), last_selected_.data(), z.data(), rows * d);
    Tensor doubled(Shape({static_cast<int64_t>(rows), 2 * d_}), backend_, device);
    backend_->copy_2d(doubled.data(), 2 * d, last_selected_.data(), d, rows, d);
    backend_->copy_2d(doubled.data() + d, 2 * d, last_selected_.data(), d, rows, d);
    Tensor selected_relevance(last_selected_.shape(), backend_, device);
    backend_->gather_rows(relevance_out.data(), last_indices_.data(), selected_relevance.data(), rows, d);
    const Tensor both = lrp_rules::linear(backend_, device, doubled, stacked, bias, z, selected_relevance, rows, 2 * d, d, config);

    Tensor kept(last_selected_.shape(), backend_, device), edit(last_selected_.shape(), backend_, device);
    backend_->copy_2d(kept.data(), d, both.data(), 2 * d, rows, d);
    backend_->copy_2d(edit.data(), d, both.data() + d, 2 * d, rows, d);
    // relevance_in's edited rows become kept + edit: add (kept + edit - relevance_out) there.
    Tensor change(last_selected_.shape(), backend_, device);
    backend_->add(kept.data(), edit.data(), change.data(), rows * d);
    backend_->axpby(1.0f, change.data(), -1.0f, selected_relevance.data(), change.data(), rows * d);
    backend_->scatter_add_rows(change.data(), last_indices_.data(), relevance_in.data(), rows, d);
    backend_->scatter_add_rows(edit.data(), last_indices_.data(), last_edit_relevance_.data(), rows, d);
    return relevance_in;
}

const Tensor& LoReFTIntervention::last_edit_relevance() const {
    if (last_edit_relevance_.numel() == 0) throw std::logic_error("LoReFTIntervention::last_edit_relevance: no relevance pass since the last forward()");
    return last_edit_relevance_;
}

void LoReFTIntervention::release_activations() {
    release_tensor(last_selected_);
    release_tensor(last_coords_);
    release_tensor(last_indices_);
    release_tensor(last_edit_relevance_);
    last_rows_.clear();
}

std::vector<LoReFTIntervention*> ApplyLoReFT(CausalLM& model, const LoReFTConfig& config) {
    std::vector<int64_t> layers = config.layers;
    if (layers.empty()) {
        for (int64_t i = 0; i < model.num_layers(); ++i) layers.push_back(i);
    }
    std::set<int64_t> seen;
    for (int64_t layer : layers) {
        if (layer < 0 || layer >= model.num_layers()) throw std::invalid_argument("ApplyLoReFT: no layer " + std::to_string(layer));
        if (!seen.insert(layer).second) throw std::invalid_argument("ApplyLoReFT: layer " + std::to_string(layer) + " is listed twice");
        if (model.intervention(layer) != nullptr) throw std::logic_error("ApplyLoReFT: layer " + std::to_string(layer) + " already has an intervention");
    }
    std::sort(layers.begin(), layers.end());
    const int64_t d = model.config().hidden_size;
    DeviceBackend* backend = model.embed_tokens().weight().backend();
    std::vector<std::unique_ptr<LoReFTIntervention>> made;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        made.push_back(std::make_unique<LoReFTIntervention>(d, config.rank, config.first, config.last, backend,
                                                            config.seed * 0x9E3779B97F4A7C15ULL + i + 1));
    }
    model.set_requires_grad(false);
    std::vector<LoReFTIntervention*> out;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        out.push_back(made[i].get());
        model.set_intervention(layers[i], std::move(made[i]));
    }
    return out;
}

void OrthonormalizeLoReFT(CausalLM& model) {
    for (int64_t i = 0; i < model.num_layers(); ++i) {
        if (auto* intervention = dynamic_cast<LoReFTIntervention*>(model.intervention(i))) intervention->orthonormalize();
    }
}

void SaveLoReFT(CausalLM& model, const std::string& path) {
    std::vector<Tensor> owned;
    owned.reserve(static_cast<std::size_t>(3 * model.num_layers()));
    std::vector<std::pair<std::string, const Tensor*>> tensors;
    const LoReFTIntervention* first = nullptr;
    for (int64_t i = 0; i < model.num_layers(); ++i) {
        auto* intervention = dynamic_cast<LoReFTIntervention*>(model.intervention(i));
        if (intervention == nullptr) continue;
        if (first == nullptr) first = intervention;
        if (intervention->first() != first->first() || intervention->last() != first->last()) {
            throw std::invalid_argument("SaveLoReFT: the interventions edit different positions");
        }
        const std::string key = "layer." + std::to_string(i) + ".";
        const int64_t d = intervention->rotation().shape().dim(0), r = intervention->rank();
        DeviceBackend* backend = intervention->rotation().backend();
        owned.emplace_back(Shape({d, r}), backend, intervention->rotation().to_host_vector());
        tensors.emplace_back(key + "rotate_layer", &owned.back());
        owned.emplace_back(Shape({r, d}), backend, Transposed(intervention->source_weight().to_host_vector(), d, r));
        tensors.emplace_back(key + "learned_source.weight", &owned.back());
        owned.emplace_back(Shape({r}), backend, intervention->source_bias().to_host_vector());
        tensors.emplace_back(key + "learned_source.bias", &owned.back());
    }
    if (first == nullptr) throw std::invalid_argument("SaveLoReFT: the model has no LoReFT interventions");
    WriteSafetensors(path, tensors, {{"first", std::to_string(first->first())}, {"last", std::to_string(first->last())}});
}

std::vector<LoReFTIntervention*> LoadLoReFT(CausalLM& model, const std::string& path) {
    const SafetensorsFile file = SafetensorsFile::Read(path);
    auto count = [&](const char* key) {
        const auto it = file.metadata().find(key);
        if (it == file.metadata().end()) throw std::invalid_argument(std::string("LoadLoReFT: no '") + key + "' in the file's metadata");
        std::size_t used = 0;
        const long long v = std::stoll(it->second, &used);
        if (used != it->second.size() || v < 0) throw std::invalid_argument(std::string("LoadLoReFT: bad '") + key + "' metadata");
        return static_cast<int64_t>(v);
    };
    const int64_t first = count("first"), last = count("last");
    const int64_t d = model.config().hidden_size;
    std::map<int64_t, std::map<std::string, std::string>> by_layer;  // layer -> part -> key
    for (const std::string& key : file.names()) {
        const std::size_t dot = key.find('.', 6);
        if (key.rfind("layer.", 0) != 0 || dot == std::string::npos) throw std::invalid_argument("LoadLoReFT: unexpected key '" + key + "'");
        const std::string index = key.substr(6, dot - 6), part = key.substr(dot + 1);
        if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos ||
            (part != "rotate_layer" && part != "learned_source.weight" && part != "learned_source.bias")) {
            throw std::invalid_argument("LoadLoReFT: unexpected key '" + key + "'");
        }
        const int64_t layer = std::stoll(index);
        if (layer >= model.num_layers()) throw std::invalid_argument("LoadLoReFT: the model has no layer " + index);
        if (model.intervention(layer) != nullptr) throw std::invalid_argument("LoadLoReFT: layer " + index + " already has an intervention");
        by_layer[layer][part] = key;
    }
    if (by_layer.empty()) throw std::invalid_argument("LoadLoReFT: the file has no interventions");
    DeviceBackend* backend = model.embed_tokens().weight().backend();
    std::vector<std::pair<int64_t, std::unique_ptr<LoReFTIntervention>>> made;
    for (const auto& [layer, parts] : by_layer) {
        if (parts.size() != 3) throw std::invalid_argument("LoadLoReFT: layer " + std::to_string(layer) + " lacks a tensor");
        const std::vector<int64_t>& p = file.info(parts.at("rotate_layer")).shape;
        const std::vector<int64_t>& w = file.info(parts.at("learned_source.weight")).shape;
        const std::vector<int64_t>& b = file.info(parts.at("learned_source.bias")).shape;
        if (p.size() != 2 || p[0] != d || w != std::vector<int64_t>{p[1], d} || b != std::vector<int64_t>{p[1]}) {
            throw std::invalid_argument("LoadLoReFT: layer " + std::to_string(layer) + "'s shapes don't fit hidden size " + std::to_string(d));
        }
        auto intervention = std::make_unique<LoReFTIntervention>(d, p[1], first, last, backend, 0);
        intervention->set_parameters(file.tensor(parts.at("rotate_layer"), backend).to_host_vector(),
                                     Transposed(file.tensor(parts.at("learned_source.weight"), backend).to_host_vector(), p[1], d),
                                     file.tensor(parts.at("learned_source.bias"), backend).to_host_vector());
        made.emplace_back(layer, std::move(intervention));
    }
    model.set_requires_grad(false);
    std::vector<LoReFTIntervention*> out;
    for (auto& [layer, intervention] : made) {
        out.push_back(intervention.get());
        model.set_intervention(layer, std::move(intervention));
    }
    return out;
}

}  // namespace pulsatrix
