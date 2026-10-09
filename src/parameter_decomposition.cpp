#include "pulsatrix/parameter_decomposition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

constexpr double kSqrt2 = 1.4142135623730951;
constexpr double kInvSqrt2Pi = 0.3989422804014327;

double Gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / kSqrt2)); }
double GeluGrad(double x) { return 0.5 * (1.0 + std::erf(x / kSqrt2)) + x * kInvSqrt2Pi * std::exp(-0.5 * x * x); }

Tensor* GradOf(LinearModule& m, const char* name) {
    for (const NamedParamRef& p : m.named_parameters()) {
        if (p.name == name) return p.ref.grad;
    }
    throw std::logic_error("ComponentLinear: no parameter named " + std::string(name));
}

void AddTo(Tensor* t, const std::vector<float>& add) {
    std::vector<float> v = t->to_host_vector();
    for (size_t i = 0; i < v.size(); ++i) v[i] += add[i];
    *t = Tensor(t->shape(), t->backend(), v, t->device());
}

Tensor HostTensor(const Shape& shape, DeviceBackend* backend, std::vector<float> v) { return Tensor(shape, backend, std::move(v), backend->device()); }

/** @brief Zeros `(rows, cols)`, or `(rows,)` when cols is 0; at least one row. */
Tensor Zeros(int64_t rows, int64_t cols, DeviceBackend* backend) {
    rows = std::max<int64_t>(rows, 1);
    if (cols == 0) return HostTensor(Shape({rows}), backend, std::vector<float>(static_cast<size_t>(rows), 0.0f));
    cols = std::max<int64_t>(cols, 1);
    return HostTensor(Shape({rows, cols}), backend, std::vector<float>(static_cast<size_t>(rows * cols), 0.0f));
}

/** @brief The trainable parameters of every layer, as one Module for optimizers. */
class ParamsModule : public Module {
public:
    explicit ParamsModule(std::vector<ComponentLinear*> layers) : layers_(std::move(layers)) {}
    std::vector<NamedParamRef> named_parameters() override {
        std::vector<NamedParamRef> out;
        for (size_t i = 0; i < layers_.size(); ++i) append_named_parameters(out, "layers." + std::to_string(i), *layers_[i]);
        return out;
    }
    Tensor backward(const Tensor&) override { throw std::logic_error("ParameterDecomposition: the parameters module has no forward"); }
    Tensor propagate_relevance(const Tensor&, const LRPRuleConfig&) override {
        throw std::logic_error("ParameterDecomposition: the parameters module has no forward");
    }
    OpType op_type() const override { return OpType::Composite; }

protected:
    Tensor forward_impl(const Tensor&) override { throw std::logic_error("ParameterDecomposition: the parameters module has no forward"); }

private:
    std::vector<ComponentLinear*> layers_;
};

}  // namespace

// ---- ComponentLinear ---------------------------------------------------------------------------

ComponentLinear::ComponentLinear(LinearModule& target, int64_t num_components, DeviceBackend* backend, int64_t gate_hidden, uint64_t seed)
    : in_(target.weight().shape().dim(0)),
      out_(target.weight().shape().dim(1)),
      C_(num_components),
      G_(gate_hidden),
      backend_(backend),
      target_w_(target.weight().to_host_vector()),
      w_(in_, out_, backend, /*use_bias=*/false),
      v_(in_, num_components > 0 ? num_components : 1, backend, false),
      u_(num_components > 0 ? num_components : 1, out_, backend, false),
      gate_in_w_(Zeros(num_components, gate_hidden, backend)),
      gate_in_b_(Zeros(num_components, gate_hidden, backend)),
      gate_out_w_(Zeros(num_components, gate_hidden, backend)),
      gate_out_b_(Zeros(num_components, 0, backend)),
      gate_in_w_grad_(Zeros(num_components, gate_hidden, backend)),
      gate_in_b_grad_(Zeros(num_components, gate_hidden, backend)),
      gate_out_w_grad_(Zeros(num_components, gate_hidden, backend)),
      gate_out_b_grad_(Zeros(num_components, 0, backend)) {
    if (num_components < 1 || gate_hidden < 1) throw std::invalid_argument("ComponentLinear: num_components and gate_hidden must be positive");
    bias_ = target.uses_bias() ? target.bias().to_host_vector() : std::vector<float>(static_cast<size_t>(out_), 0.0f);
    w_.set_weight(target_w_);
    PortableRng rng{seed};
    // SPD v1: unit V columns and U rows, each U row scaled by its overlap with the target.
    std::vector<float> v(static_cast<size_t>(in_ * C_)), u(static_cast<size_t>(C_ * out_));
    for (int64_t c = 0; c < C_; ++c) {
        double sv = 0, su = 0;
        for (int64_t j = 0; j < in_; ++j) {
            const double g = rng.gaussian();
            v[static_cast<size_t>(j * C_ + c)] = static_cast<float>(g);
            sv += g * g;
        }
        for (int64_t k = 0; k < out_; ++k) {
            const double g = rng.gaussian();
            u[static_cast<size_t>(c * out_ + k)] = static_cast<float>(g);
            su += g * g;
        }
        for (int64_t j = 0; j < in_; ++j) v[static_cast<size_t>(j * C_ + c)] = static_cast<float>(v[static_cast<size_t>(j * C_ + c)] / std::sqrt(sv));
        for (int64_t k = 0; k < out_; ++k) u[static_cast<size_t>(c * out_ + k)] = static_cast<float>(u[static_cast<size_t>(c * out_ + k)] / std::sqrt(su));
        double overlap = 0;
        for (int64_t j = 0; j < in_; ++j) {
            for (int64_t k = 0; k < out_; ++k) {
                overlap += static_cast<double>(v[static_cast<size_t>(j * C_ + c)]) * target_w_[static_cast<size_t>(j * out_ + k)] * u[static_cast<size_t>(c * out_ + k)];
            }
        }
        for (int64_t k = 0; k < out_; ++k) u[static_cast<size_t>(c * out_ + k)] = static_cast<float>(u[static_cast<size_t>(c * out_ + k)] * overlap);
    }
    v_.set_weight(v);
    u_.set_weight(u);
    std::vector<float> in_w(static_cast<size_t>(C_ * G_)), out_w(static_cast<size_t>(C_ * G_));
    for (float& e : in_w) e = static_cast<float>(rng.gaussian() * kSqrt2);
    for (float& e : out_w) e = static_cast<float>(rng.gaussian() / std::sqrt(static_cast<double>(G_)));
    gate_in_w_ = HostTensor(Shape({C_, G_}), backend, in_w);
    gate_out_w_ = HostTensor(Shape({C_, G_}), backend, out_w);
}

void ComponentLinear::use_target() { target_mode_ = true; }

void ComponentLinear::use_components(std::vector<float> masks, std::vector<float> delta_masks) {
    target_mode_ = false;
    masks_ = std::move(masks);
    delta_masks_ = std::move(delta_masks);
}

Tensor ComponentLinear::forward_impl(const Tensor& x) {
    if (x.rank() != 2 || x.shape().dim(1) != in_ || x.shape().dim(0) < 1) throw std::invalid_argument("ComponentLinear: the input must be (N, in_features)");
    const int64_t N = x.shape().dim(0);
    rows_ = N;
    std::vector<float> y;
    if (target_mode_) {
        target_input_ = x;
        y = w_.forward(x).to_host_vector();
    } else {
        if (static_cast<int64_t>(masks_.size()) != N * C_) throw std::invalid_argument("ComponentLinear: the masks must be (N, num_components)");
        const bool delta = !delta_masks_.empty();
        if (delta && static_cast<int64_t>(delta_masks_.size()) != N) throw std::invalid_argument("ComponentLinear: the Δ masks must be (N,)");
        h_ = v_.forward(x).to_host_vector();
        scaled_ = masks_;
        if (delta) {
            for (int64_t r = 0; r < N; ++r) {
                for (int64_t c = 0; c < C_; ++c) scaled_[static_cast<size_t>(r * C_ + c)] -= delta_masks_[static_cast<size_t>(r)];
            }
        }
        std::vector<float> hm(h_.size());
        for (size_t i = 0; i < hm.size(); ++i) hm[i] = h_[i] * scaled_[i];
        y = u_.forward(HostTensor(Shape({N, C_}), backend_, hm)).to_host_vector();
        if (delta) {
            xw_ = w_.forward(x).to_host_vector();
            for (int64_t r = 0; r < N; ++r) {
                for (int64_t k = 0; k < out_; ++k) y[static_cast<size_t>(r * out_ + k)] += delta_masks_[static_cast<size_t>(r)] * xw_[static_cast<size_t>(r * out_ + k)];
            }
        }
    }
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t k = 0; k < out_; ++k) y[static_cast<size_t>(r * out_ + k)] += bias_[static_cast<size_t>(k)];
    }
    return HostTensor(Shape({N, out_}), backend_, y);
}

Tensor ComponentLinear::backward(const Tensor& grad_output) {
    if (rows_ == 0) throw std::logic_error("ComponentLinear::backward: called before any forward()");
    if (target_mode_) return w_.backward(grad_output);
    const int64_t N = rows_;
    const std::vector<float> dhm = u_.backward(grad_output).to_host_vector();
    mask_grad_.assign(static_cast<size_t>(N * C_), 0.0f);
    std::vector<float> dh(dhm.size());
    for (size_t i = 0; i < dhm.size(); ++i) {
        mask_grad_[i] = dhm[i] * h_[i];
        dh[i] = dhm[i] * scaled_[i];
    }
    Tensor dx = v_.backward(HostTensor(Shape({N, C_}), backend_, dh));
    if (delta_masks_.empty()) return dx;
    // m_Δ scales x W and, through m - m_Δ, takes from every subcomponent.
    const std::vector<float> g = grad_output.to_host_vector();
    delta_mask_grad_.assign(static_cast<size_t>(N), 0.0f);
    std::vector<float> gw(g.size());
    for (int64_t r = 0; r < N; ++r) {
        double s = 0;
        for (int64_t c = 0; c < C_; ++c) s -= mask_grad_[static_cast<size_t>(r * C_ + c)];
        for (int64_t k = 0; k < out_; ++k) {
            const size_t q = static_cast<size_t>(r * out_ + k);
            s += static_cast<double>(g[q]) * xw_[q];
            gw[q] = delta_masks_[static_cast<size_t>(r)] * g[q];
        }
        delta_mask_grad_[static_cast<size_t>(r)] = static_cast<float>(s);
    }
    const Tensor dxw = w_.backward(HostTensor(Shape({N, out_}), backend_, gw));
    Tensor out(dx.shape(), backend_, dx.device());
    backend_->add(dx.data(), dxw.data(), out.data(), static_cast<size_t>(dx.numel()));
    return out;
}

std::vector<float> ComponentLinear::gate_outputs() {
    if (!target_input_) throw std::logic_error("ComponentLinear::gate_outputs: needs a target-mode forward() first");
    gate_input_ = *target_input_;  // later target-mode passes (SPD's layerwise ones) mustn't move it
    const int64_t N = gate_input_->shape().dim(0);
    gate_h_ = v_.forward(*gate_input_).to_host_vector();
    const std::vector<float> wi = gate_in_w_.to_host_vector(), bi = gate_in_b_.to_host_vector(), wo = gate_out_w_.to_host_vector(),
                             bo = gate_out_b_.to_host_vector();
    // GELU and its slope at each hidden unit, kept for gate_backward(). Rows whose inner
    // activation is exactly 0 (common on sparse inputs) share one evaluation per subcomponent.
    gate_act_.assign(static_cast<size_t>(N * C_ * G_), 0.0f);
    gate_slope_.assign(gate_act_.size(), 0.0f);
    std::vector<float> zero_act(static_cast<size_t>(C_ * G_)), zero_slope(zero_act.size()), zero_z(static_cast<size_t>(C_));
    for (int64_t c = 0; c < C_; ++c) {
        double s = bo[static_cast<size_t>(c)];
        for (int64_t j = 0; j < G_; ++j) {
            const size_t q = static_cast<size_t>(c * G_ + j);
            zero_act[q] = static_cast<float>(Gelu(bi[q]));
            zero_slope[q] = static_cast<float>(GeluGrad(bi[q]));
            s += wo[q] * zero_act[q];
        }
        zero_z[static_cast<size_t>(c)] = static_cast<float>(s);
    }
    std::vector<float> z(static_cast<size_t>(N * C_));
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t c = 0; c < C_; ++c) {
            const double h = gate_h_[static_cast<size_t>(r * C_ + c)];
            float* act = gate_act_.data() + (r * C_ + c) * G_;
            float* slope = gate_slope_.data() + (r * C_ + c) * G_;
            if (h == 0) {
                std::copy_n(zero_act.data() + c * G_, G_, act);
                std::copy_n(zero_slope.data() + c * G_, G_, slope);
                z[static_cast<size_t>(r * C_ + c)] = zero_z[static_cast<size_t>(c)];
                continue;
            }
            double s = bo[static_cast<size_t>(c)];
            for (int64_t j = 0; j < G_; ++j) {
                const size_t q = static_cast<size_t>(c * G_ + j);
                const double pre = wi[q] * h + bi[q], e = std::erf(pre / kSqrt2);
                act[j] = static_cast<float>(0.5 * pre * (1.0 + e));
                slope[j] = static_cast<float>(0.5 * (1.0 + e) + pre * kInvSqrt2Pi * std::exp(-0.5 * pre * pre));
                s += wo[q] * act[j];
            }
            z[static_cast<size_t>(r * C_ + c)] = static_cast<float>(s);
        }
    }
    return z;
}

void ComponentLinear::gate_backward(const std::vector<float>& grad_z) {
    if (!gate_input_ || gate_h_.empty()) throw std::logic_error("ComponentLinear::gate_backward: needs gate_outputs() first");
    const int64_t N = gate_input_->shape().dim(0);
    if (static_cast<int64_t>(grad_z.size()) != N * C_) throw std::invalid_argument("ComponentLinear::gate_backward: needs (N, num_components) values");
    const std::vector<float> wi = gate_in_w_.to_host_vector(), wo = gate_out_w_.to_host_vector();
    std::vector<double> gwi(wi.size(), 0.0), gbi(wi.size(), 0.0), gwo(wi.size(), 0.0), gbo(static_cast<size_t>(C_), 0.0);
    std::vector<float> dh(static_cast<size_t>(N * C_), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t c = 0; c < C_; ++c) {
            const double gz = grad_z[static_cast<size_t>(r * C_ + c)];
            if (gz == 0) continue;
            const double h = gate_h_[static_cast<size_t>(r * C_ + c)];
            const float* act = gate_act_.data() + (r * C_ + c) * G_;
            const float* slope = gate_slope_.data() + (r * C_ + c) * G_;
            gbo[static_cast<size_t>(c)] += gz;
            double gh = 0;
            for (int64_t j = 0; j < G_; ++j) {
                const size_t q = static_cast<size_t>(c * G_ + j);
                gwo[q] += gz * act[j];
                const double gpre = gz * wo[q] * slope[j];
                gwi[q] += gpre * h;
                gbi[q] += gpre;
                gh += gpre * wi[q];
            }
            dh[static_cast<size_t>(r * C_ + c)] = static_cast<float>(gh);
        }
    }
    auto to_float = [](const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); };
    AddTo(&gate_in_w_grad_, to_float(gwi));
    AddTo(&gate_in_b_grad_, to_float(gbi));
    AddTo(&gate_out_w_grad_, to_float(gwo));
    AddTo(&gate_out_b_grad_, to_float(gbo));
    // Into V through h = x V, on the target pass's input.
    (void)v_.forward(*gate_input_);
    (void)v_.backward(HostTensor(Shape({N, C_}), backend_, dh));
}

std::vector<float> ComponentLinear::component_weight() {
    const std::vector<float> v = v_.weight().to_host_vector(), u = u_.weight().to_host_vector();
    std::vector<float> w(static_cast<size_t>(in_ * out_), 0.0f);
    for (int64_t j = 0; j < in_; ++j) {
        for (int64_t c = 0; c < C_; ++c) {
            const float a = v[static_cast<size_t>(j * C_ + c)];
            if (a == 0.0f) continue;
            for (int64_t k = 0; k < out_; ++k) w[static_cast<size_t>(j * out_ + k)] += a * u[static_cast<size_t>(c * out_ + k)];
        }
    }
    return w;
}

Tensor ComponentLinear::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (rows_ == 0) throw std::logic_error("ComponentLinear::propagate_relevance: called before any forward()");
    if (target_mode_ || !delta_masks_.empty()) {
        // The target weight, or the masked subcomponents plus Δ: relevance through x W is a
        // first-order approximation there.
        return w_.propagate_relevance(relevance_out, config);
    }
    return v_.propagate_relevance(u_.propagate_relevance(relevance_out, config), config);
}

std::vector<NamedParamRef> ComponentLinear::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "V", v_);
    append_named_parameters(out, "U", u_);
    out.push_back({"gate.in_weight", {&gate_in_w_, &gate_in_w_grad_}});
    out.push_back({"gate.in_bias", {&gate_in_b_, &gate_in_b_grad_}});
    out.push_back({"gate.out_weight", {&gate_out_w_, &gate_out_w_grad_}});
    out.push_back({"gate.out_bias", {&gate_out_b_, &gate_out_b_grad_}});
    return out;
}

void ComponentLinear::release_activations() {
    w_.release_activations();
    v_.release_activations();
    u_.release_activations();
    h_.clear();
    xw_.clear();
    scaled_.clear();
    gate_h_.clear();
    gate_act_.clear();
    gate_slope_.clear();
    target_input_.reset();
    gate_input_.reset();
    rows_ = 0;
}

// ---- ParameterDecomposition -------------------------------------------------------------------

ParameterDecomposition::ParameterDecomposition(Module& model, std::vector<ComponentLinear*> layers, DeviceBackend* backend,
                                               DecompositionOptions options)
    : model_(model), layers_(std::move(layers)), backend_(backend), options_(options), rng_state_(options.seed * 2654435761ULL + 1) {
    if (layers_.empty() || std::any_of(layers_.begin(), layers_.end(), [](ComponentLinear* l) { return l == nullptr; })) {
        throw std::invalid_argument("ParameterDecomposition: needs one or more layers, none null");
    }
    params_ = std::make_unique<ParamsModule>(layers_);
}

Module& ParameterDecomposition::parameters_module() { return *params_; }

std::vector<float> ParameterDecomposition::Draw(const char* pass, int64_t layer, int64_t count) {
    if (noise_) {
        std::vector<float> v = noise_(pass, layer, count);
        if (static_cast<int64_t>(v.size()) != count) throw std::invalid_argument("ParameterDecomposition: the noise function returned the wrong count");
        return v;
    }
    std::vector<float> v(static_cast<size_t>(count));
    for (float& e : v) {
        rng_state_ = rng_state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        e = static_cast<float>(static_cast<double>(rng_state_ >> 40) / 16777216.0);
    }
    return v;
}

float ParameterDecomposition::Divergence(const std::vector<float>& pred, const std::vector<float>& target, int64_t rows,
                                         std::vector<float>* grad) const {
    grad->assign(pred.size(), 0.0f);
    if (options_.divergence == OutputDivergence::MeanSquaredError) {
        double s = 0;
        const auto n = static_cast<double>(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            const double d = static_cast<double>(pred[i]) - target[i];
            s += d * d;
            (*grad)[i] = static_cast<float>(2.0 * d / n);
        }
        return static_cast<float>(s / n);
    }
    // KL(softmax(t) || softmax(p)) per row: Σ q (log q - log p); d/dp = (softmax(p) - q) / rows.
    const auto V = static_cast<int64_t>(pred.size()) / rows;
    double total = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const float* p = pred.data() + r * V;
        const float* t = target.data() + r * V;
        const double mp = *std::max_element(p, p + V), mt = *std::max_element(t, t + V);
        double zp = 0, zt = 0;
        for (int64_t j = 0; j < V; ++j) {
            zp += std::exp(p[j] - mp);
            zt += std::exp(t[j] - mt);
        }
        const double lzp = mp + std::log(zp), lzt = mt + std::log(zt);
        for (int64_t j = 0; j < V; ++j) {
            const double lq = t[j] - lzt, lp = p[j] - lzp, q = std::exp(lq);
            total += q * (lq - lp);
            (*grad)[static_cast<size_t>(r * V + j)] = static_cast<float>((std::exp(lp) - q) / static_cast<double>(rows));
        }
    }
    return static_cast<float>(total / static_cast<double>(rows));
}

float ParameterDecomposition::Pass(const Tensor& x, const std::vector<float>& target, float weight) {
    const Tensor y = model_.forward(x);
    std::vector<float> g;
    const float loss = Divergence(y.to_host_vector(), target, x.shape().dim(0), &g);
    for (float& e : g) e *= weight;
    (void)model_.backward(Tensor(y.shape(), backend_, g, y.device()));
    return loss;
}

Tensor ParameterDecomposition::target_output(const Tensor& x) {
    for (ComponentLinear* l : layers_) l->use_target();
    return model_.forward(x);
}

std::vector<std::vector<float>> ParameterDecomposition::causal_importances(const Tensor& x) {
    (void)target_output(x);
    std::vector<std::vector<float>> out;
    for (ComponentLinear* l : layers_) {
        std::vector<float> z = l->gate_outputs();
        for (float& e : z) e = std::clamp(e, 0.0f, 1.0f);
        out.push_back(std::move(z));
    }
    return out;
}

DecompositionLoss ParameterDecomposition::loss_and_backward(const Tensor& x) {
    const bool vpd = options_.method == DecompositionMethod::VPD;
    const int64_t N = x.shape().dim(0);
    const auto L = static_cast<int64_t>(layers_.size());
    const float leak = options_.leak;
    DecompositionLoss loss;

    // The target pass, and each layer's causal importances.
    const std::vector<float> target = target_output(x).to_host_vector();
    std::vector<Importance> imp(static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l) {
        Importance& im = imp[static_cast<size_t>(l)];
        im.z = layers_[static_cast<size_t>(l)]->gate_outputs();
        im.lower.resize(im.z.size());
        im.upper.resize(im.z.size());
        for (size_t i = 0; i < im.z.size(); ++i) {
            const float z = im.z[i];
            im.lower[i] = z > 0.0f ? std::min(z, 1.0f) : (vpd ? 0.0f : leak * z);
            im.upper[i] = z > 1.0f ? 1.0f + leak * (z - 1.0f) : std::max(z, 0.0f);
        }
    }
    std::vector<std::vector<float>> d_lower(static_cast<size_t>(L)), d_upper(static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l) {
        const int64_t C = layers_[static_cast<size_t>(l)]->num_components();
        d_lower[static_cast<size_t>(l)].assign(static_cast<size_t>(N * C), 0.0f);
        d_upper[static_cast<size_t>(l)].assign(static_cast<size_t>(N * C), 0.0f);
    }

    // Faithfulness: Σ_l |V U - W|² over the number of weights.
    {
        int64_t n_weights = 0;
        for (ComponentLinear* l : layers_) n_weights += l->in_features() * l->out_features();
        double total = 0;
        for (ComponentLinear* layer : layers_) {
            const int64_t in = layer->in_features(), out = layer->out_features(), C = layer->num_components();
            const std::vector<float> wc = layer->component_weight(), &w = layer->target_weight();
            const std::vector<float> v = layer->V().weight().to_host_vector(), u = layer->U().weight().to_host_vector();
            std::vector<float> diff(wc.size());
            for (size_t i = 0; i < wc.size(); ++i) {
                diff[i] = wc[i] - w[i];
                total += static_cast<double>(diff[i]) * diff[i];
            }
            const double s = 2.0 * options_.faithfulness_coefficient / static_cast<double>(n_weights);
            std::vector<float> gv(v.size(), 0.0f), gu(u.size(), 0.0f);
            for (int64_t j = 0; j < in; ++j) {
                for (int64_t c = 0; c < C; ++c) {
                    double a = 0;
                    for (int64_t k = 0; k < out; ++k) a += static_cast<double>(diff[static_cast<size_t>(j * out + k)]) * u[static_cast<size_t>(c * out + k)];
                    gv[static_cast<size_t>(j * C + c)] = static_cast<float>(s * a);
                }
            }
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t k = 0; k < out; ++k) {
                    double a = 0;
                    for (int64_t j = 0; j < in; ++j) a += static_cast<double>(v[static_cast<size_t>(j * C + c)]) * diff[static_cast<size_t>(j * out + k)];
                    gu[static_cast<size_t>(c * out + k)] = static_cast<float>(s * a);
                }
            }
            AddTo(GradOf(layer->V(), "weight"), gv);
            AddTo(GradOf(layer->U(), "weight"), gu);
        }
        loss.faithfulness = static_cast<float>(total / static_cast<double>(n_weights));
    }

    auto masks_from = [&](int64_t l, const std::vector<float>& r) {
        const std::vector<float>& g = imp[static_cast<size_t>(l)].lower;
        std::vector<float> m(g.size());
        for (size_t i = 0; i < g.size(); ++i) m[i] = g[i] + (1.0f - g[i]) * r[i];
        return m;
    };
    // dL/dg += dL/dm (1 - r), on the rows given (all when empty).
    auto carry = [&](int64_t l, const std::vector<float>& r, const std::vector<uint8_t>& rows) {
        const ComponentLinear* layer = layers_[static_cast<size_t>(l)];
        const int64_t C = layer->num_components();
        const std::vector<float>& dm = layer->mask_grad();
        std::vector<float>& dg = d_lower[static_cast<size_t>(l)];
        for (int64_t q = 0; q < N; ++q) {
            if (!rows.empty() && rows[static_cast<size_t>(q)] == 0) continue;
            for (int64_t c = 0; c < C; ++c) {
                const size_t i = static_cast<size_t>(q * C + c);
                dg[i] += dm[i] * (1.0f - r[i]);
            }
        }
    };

    if (!vpd) {
        // Every layer masked.
        std::vector<std::vector<float>> r(static_cast<size_t>(L));
        for (int64_t l = 0; l < L; ++l) {
            r[static_cast<size_t>(l)] = Draw("stochastic", l, N * layers_[static_cast<size_t>(l)]->num_components());
            layers_[static_cast<size_t>(l)]->use_components(masks_from(l, r[static_cast<size_t>(l)]));
        }
        loss.stochastic = Pass(x, target, options_.stochastic_coefficient);
        for (int64_t l = 0; l < L; ++l) carry(l, r[static_cast<size_t>(l)], {});
        // One layer masked at a time.
        double lw = 0;
        for (int64_t l = 0; l < L; ++l) {
            const std::vector<float> rl = Draw("layerwise", l, N * layers_[static_cast<size_t>(l)]->num_components());
            for (int64_t k = 0; k < L; ++k) {
                if (k == l) layers_[static_cast<size_t>(k)]->use_components(masks_from(k, rl));
                else layers_[static_cast<size_t>(k)]->use_target();
            }
            lw += Pass(x, target, options_.layerwise_coefficient / static_cast<float>(L));
            carry(l, rl, {});
        }
        loss.layerwise = static_cast<float>(lw / static_cast<double>(L));
    } else {
        // Each row through the components of k random layers (k uniform in 1..L), the rest of the
        // rows and layers at masks of 1 (the target exactly, with Δ).
        const std::vector<float> route = Draw("route", -1, N * (1 + L));
        std::vector<std::vector<uint8_t>> routed(static_cast<size_t>(L), std::vector<uint8_t>(static_cast<size_t>(N), 0));
        for (int64_t q = 0; q < N; ++q) {
            const float* v = route.data() + q * (1 + L);
            const int64_t k = std::min<int64_t>(L, 1 + static_cast<int64_t>(v[0] * static_cast<float>(L)));
            std::vector<int64_t> order(static_cast<size_t>(L));
            std::iota(order.begin(), order.end(), int64_t{0});
            std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return v[1 + a] < v[1 + b]; });
            for (int64_t i = 0; i < k; ++i) routed[static_cast<size_t>(order[static_cast<size_t>(i)])][static_cast<size_t>(q)] = 1;
        }
        std::vector<std::vector<float>> r(static_cast<size_t>(L));
        for (int64_t l = 0; l < L; ++l) {
            const int64_t C = layers_[static_cast<size_t>(l)]->num_components();
            r[static_cast<size_t>(l)] = Draw("stochastic", l, N * C);
            const std::vector<float> rd = Draw("delta", l, N);
            std::vector<float> m = masks_from(l, r[static_cast<size_t>(l)]), md(static_cast<size_t>(N), 1.0f);
            for (int64_t q = 0; q < N; ++q) {
                if (routed[static_cast<size_t>(l)][static_cast<size_t>(q)] != 0) {
                    md[static_cast<size_t>(q)] = rd[static_cast<size_t>(q)];
                } else {
                    std::fill(m.begin() + q * C, m.begin() + (q + 1) * C, 1.0f);
                }
            }
            layers_[static_cast<size_t>(l)]->use_components(std::move(m), std::move(md));
        }
        loss.stochastic = Pass(x, target, options_.stochastic_coefficient);
        for (int64_t l = 0; l < L; ++l) carry(l, r[static_cast<size_t>(l)], routed[static_cast<size_t>(l)]);

        // Adversarial masks: persistent sources in [0, 1], raised by Adam ascent on the loss.
        if (options_.adversarial_coefficient > 0.0f) {
            if (sources_.size() != static_cast<size_t>(L) || static_cast<int64_t>(sources_[0].size()) != N * (layers_[0]->num_components() + 1)) {
                sources_.assign(static_cast<size_t>(L), {});
                source_m_.assign(static_cast<size_t>(L), {});
                source_v_.assign(static_cast<size_t>(L), {});
                for (int64_t l = 0; l < L; ++l) {
                    const int64_t n = N * (layers_[static_cast<size_t>(l)]->num_components() + 1);
                    sources_[static_cast<size_t>(l)] = Draw("adversarial", l, n);
                    source_m_[static_cast<size_t>(l)].assign(static_cast<size_t>(n), 0.0f);
                    source_v_[static_cast<size_t>(l)].assign(static_cast<size_t>(n), 0.0f);
                }
                pgd_t_ = 0;
            }
            if (source_m_.size() != sources_.size()) {  // sources set from outside: fresh moments
                source_m_.assign(sources_.size(), {});
                source_v_.assign(sources_.size(), {});
                for (size_t l = 0; l < sources_.size(); ++l) {
                    source_m_[l].assign(sources_[l].size(), 0.0f);
                    source_v_[l].assign(sources_[l].size(), 0.0f);
                }
                pgd_t_ = 0;
            }
            auto set_adversarial = [&]() {
                std::vector<std::vector<float>> rs(static_cast<size_t>(L));
                for (int64_t l = 0; l < L; ++l) {
                    const int64_t C = layers_[static_cast<size_t>(l)]->num_components();
                    const std::vector<float>& s = sources_[static_cast<size_t>(l)];
                    std::vector<float> rl(static_cast<size_t>(N * C)), md(static_cast<size_t>(N));
                    for (int64_t q = 0; q < N; ++q) {
                        for (int64_t c = 0; c < C; ++c) rl[static_cast<size_t>(q * C + c)] = s[static_cast<size_t>(q * (C + 1) + c)];
                        md[static_cast<size_t>(q)] = s[static_cast<size_t>(q * (C + 1) + C)];
                    }
                    layers_[static_cast<size_t>(l)]->use_components(masks_from(l, rl), std::move(md));
                    rs[static_cast<size_t>(l)] = std::move(rl);
                }
                return rs;
            };
            // One Adam ascent step on the sources from the layers' last mask gradients.
            auto ascend = [&](double scale) {
                ++pgd_t_;
                const double b1 = options_.pgd_beta1, b2 = options_.pgd_beta2;
                const double c1 = 1.0 - std::pow(b1, static_cast<double>(pgd_t_)), c2 = 1.0 - std::pow(b2, static_cast<double>(pgd_t_));
                for (int64_t l = 0; l < L; ++l) {
                    const ComponentLinear* layer = layers_[static_cast<size_t>(l)];
                    const int64_t C = layer->num_components();
                    const std::vector<float>& g = imp[static_cast<size_t>(l)].lower;
                    std::vector<float>& s = sources_[static_cast<size_t>(l)];
                    std::vector<float>& m = source_m_[static_cast<size_t>(l)];
                    std::vector<float>& v = source_v_[static_cast<size_t>(l)];
                    for (int64_t q = 0; q < N; ++q) {
                        for (int64_t c = 0; c <= C; ++c) {
                            const size_t i = static_cast<size_t>(q * (C + 1) + c);
                            const double grad = scale * (c < C ? layer->mask_grad()[static_cast<size_t>(q * C + c)] * (1.0 - g[static_cast<size_t>(q * C + c)])
                                                               : layer->delta_mask_grad()[static_cast<size_t>(q)]);
                            m[i] = static_cast<float>(b1 * m[i] + (1 - b1) * grad);
                            v[i] = static_cast<float>(b2 * v[i] + (1 - b2) * grad * grad);
                            const double step = options_.pgd_lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-8);
                            s[i] = std::clamp(static_cast<float>(s[i] + step), 0.0f, 1.0f);
                        }
                    }
                }
            };
            // The ascent passes mustn't train the components: keep V's and U's gradients.
            std::vector<std::vector<float>> saved;
            for (ComponentLinear* layer : layers_) {
                saved.push_back(GradOf(layer->V(), "weight")->to_host_vector());
                saved.push_back(GradOf(layer->U(), "weight")->to_host_vector());
            }
            for (int64_t step = 0; step < options_.pgd_steps; ++step) {
                (void)set_adversarial();
                (void)Pass(x, target, 1.0f);
                ascend(1.0);
            }
            for (size_t i = 0; i < layers_.size(); ++i) {
                Tensor* gv = GradOf(layers_[i]->V(), "weight");
                Tensor* gu = GradOf(layers_[i]->U(), "weight");
                *gv = Tensor(gv->shape(), gv->backend(), saved[2 * i], gv->device());
                *gu = Tensor(gu->shape(), gu->backend(), saved[2 * i + 1], gu->device());
            }
            const std::vector<std::vector<float>> rs = set_adversarial();
            loss.adversarial = Pass(x, target, options_.adversarial_coefficient);
            for (int64_t l = 0; l < L; ++l) carry(l, rs[static_cast<size_t>(l)], {});
            // The main pass's gradient moves the sources once more (unscaled by the coefficient).
            ascend(1.0 / options_.adversarial_coefficient);
        }
    }

    // Importance minimality on the upper clamp.
    {
        const double p = options_.p, eps = vpd ? 1e-12 : 0.0;
        double total = 0;
        for (int64_t l = 0; l < L; ++l) {
            const int64_t C = layers_[static_cast<size_t>(l)]->num_components();
            const std::vector<float>& u = imp[static_cast<size_t>(l)].upper;
            std::vector<float>& du = d_upper[static_cast<size_t>(l)];
            for (int64_t c = 0; c < C; ++c) {
                double sum = 0;
                for (int64_t q = 0; q < N; ++q) sum += std::pow(static_cast<double>(u[static_cast<size_t>(q * C + c)]) + eps, p);
                const double mean = sum / static_cast<double>(N);
                double ds;  // d(term)/d(sum)
                if (vpd) {
                    const double lg = std::log2(1.0 + sum);
                    total += mean + options_.frequency_beta * mean * lg;
                    ds = (1.0 + options_.frequency_beta * (lg + sum / ((1.0 + sum) * std::log(2.0)))) / static_cast<double>(N);
                } else {
                    total += mean;
                    ds = 1.0 / static_cast<double>(N);
                }
                for (int64_t q = 0; q < N; ++q) {
                    const double v = static_cast<double>(u[static_cast<size_t>(q * C + c)]) + eps;
                    if (v <= 0) continue;
                    du[static_cast<size_t>(q * C + c)] += static_cast<float>(options_.importance_coefficient * ds * p * std::pow(v, p - 1.0));
                }
            }
        }
        loss.importance = static_cast<float>(total);
    }

    // Through the clamps into the gates.
    for (int64_t l = 0; l < L; ++l) {
        const Importance& im = imp[static_cast<size_t>(l)];
        std::vector<float> dz(im.z.size());
        for (size_t i = 0; i < dz.size(); ++i) {
            const float z = im.z[i], gl = d_lower[static_cast<size_t>(l)][i], gu = d_upper[static_cast<size_t>(l)][i];
            float lower;
            if (z > 0.0f) {
                lower = z <= 1.0f ? gl : 0.0f;
            } else {
                lower = vpd ? (gl < 0.0f ? leak * gl : 0.0f) : leak * gl;
            }
            const float upper = z > 1.0f ? leak * gu : (z > 0.0f ? gu : 0.0f);
            dz[i] = lower + upper;
        }
        layers_[static_cast<size_t>(l)]->gate_backward(dz);
    }
    for (ComponentLinear* l : layers_) l->use_target();
    loss.total = options_.faithfulness_coefficient * loss.faithfulness + options_.stochastic_coefficient * loss.stochastic +
                 (vpd ? options_.adversarial_coefficient * loss.adversarial : options_.layerwise_coefficient * loss.layerwise) +
                 options_.importance_coefficient * loss.importance;
    return loss;
}

// ---- Measures ----------------------------------------------------------------------------------

ComponentAlignment AlignComponentsToRows(ComponentLinear& layer) {
    const int64_t in = layer.in_features(), out = layer.out_features(), C = layer.num_components();
    const std::vector<float> v = layer.V().weight().to_host_vector(), u = layer.U().weight().to_host_vector();
    const std::vector<float>& w = layer.target_weight();
    std::vector<double> unorm(static_cast<size_t>(C), 0.0);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t k = 0; k < out; ++k) unorm[static_cast<size_t>(c)] += static_cast<double>(u[static_cast<size_t>(c * out + k)]) * u[static_cast<size_t>(c * out + k)];
        unorm[static_cast<size_t>(c)] = std::sqrt(unorm[static_cast<size_t>(c)]);
    }
    ComponentAlignment a;
    for (int64_t j = 0; j < in; ++j) {
        double wn = 0;
        for (int64_t k = 0; k < out; ++k) wn += static_cast<double>(w[static_cast<size_t>(j * out + k)]) * w[static_cast<size_t>(j * out + k)];
        wn = std::sqrt(wn);
        double best = -2, ratio = 0;
        int64_t arg = 0;
        for (int64_t c = 0; c < C; ++c) {
            const double vj = v[static_cast<size_t>(j * C + c)];
            const double n = std::abs(vj) * unorm[static_cast<size_t>(c)];
            if (n == 0 || wn == 0) continue;
            double dot = 0;
            for (int64_t k = 0; k < out; ++k) dot += vj * u[static_cast<size_t>(c * out + k)] * w[static_cast<size_t>(j * out + k)];
            const double cos = dot / (n * wn);
            if (cos > best) {
                best = cos;
                ratio = n / wn;
                arg = c;
            }
        }
        a.mean_max_cosine += best / static_cast<double>(in);
        a.mean_norm_ratio += ratio / static_cast<double>(in);
        a.best.push_back(arg);
    }
    return a;
}

std::vector<ImportanceStats> MeasureImportance(ParameterDecomposition& d, const Tensor& x) {
    const std::vector<std::vector<float>> ci = d.causal_importances(x);
    const int64_t N = x.shape().dim(0);
    std::vector<ImportanceStats> out;
    for (size_t l = 0; l < ci.size(); ++l) {
        const int64_t C = d.layers()[l]->num_components();
        ImportanceStats s;
        s.max_importance.assign(static_cast<size_t>(C), 0.0f);
        double above = 0;
        for (int64_t q = 0; q < N; ++q) {
            for (int64_t c = 0; c < C; ++c) {
                const float g = ci[l][static_cast<size_t>(q * C + c)];
                above += g > 0.01f ? 1 : 0;
                s.max_importance[static_cast<size_t>(c)] = std::max(s.max_importance[static_cast<size_t>(c)], g);
            }
        }
        s.l0 = above / static_cast<double>(N);
        s.alive = std::count_if(s.max_importance.begin(), s.max_importance.end(), [](float g) { return g > 0.1f; });
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace pulsatrix
