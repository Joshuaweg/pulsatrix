#include "pulsatrix/circuit_tracing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {

namespace {

/** @brief `a (m, k) @ b` on the backend, where b is stored `(k, n)`, or `(n, k)` when
 *         @p transpose_b; the result `(m, n)` on the host. */
std::vector<float> MatMul(DeviceBackend* backend, const std::vector<float>& a, int64_t m, int64_t k, const Tensor& b, bool transpose_b,
                          int64_t n) {
    const Tensor at(Shape({m, k}), backend, a, backend->device());
    Tensor out(Shape({m, n}), backend, backend->device());
    backend->gemm_ex(at.data(), false, b.data(), transpose_b, out.data(), static_cast<size_t>(m), static_cast<size_t>(k),
                     static_cast<size_t>(n), 0.0f);
    return out.to_host_vector();
}

/** @brief A frozen RMSNorm's scale per row and feature: `(offset + gamma) / rms`, `(rows, d)`. */
std::vector<float> FrozenScale(const RMSNormModule& norm, int64_t rows, int64_t d) {
    const std::vector<float> gamma = norm.gamma().to_host_vector(), rms = norm.last_rms().to_host_vector();
    if (static_cast<int64_t>(rms.size()) != rows) throw std::logic_error("TraceCircuit: a norm's cached rows don't match the prompt");
    std::vector<float> s(static_cast<size_t>(rows * d));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < d; ++j) s[static_cast<size_t>(r * d + j)] = (norm.weight_offset() + gamma[static_cast<size_t>(j)]) / rms[static_cast<size_t>(r)];
    }
    return s;
}

std::string LayerFile(const std::string& dir, const std::string& stem, int64_t i) {
    return (std::filesystem::path(dir) / (stem + std::to_string(i) + ".safetensors")).string();
}

}  // namespace

// ---- CrossLayerTranscoder ----------------------------------------------------------------------

CrossLayerTranscoder::CrossLayerTranscoder(int64_t num_layers, int64_t d_model, DeviceBackend* backend)
    : L_(num_layers), d_(d_model), backend_(backend) {
    if (num_layers < 1 || d_model < 1) throw std::invalid_argument("CrossLayerTranscoder: sizes must be positive");
    layers_.reserve(static_cast<size_t>(L_));
    for (int64_t l = 0; l < L_; ++l) {
        layers_.push_back({Tensor(Shape({1, d_}), backend, std::vector<float>(static_cast<size_t>(d_), 0.0f), backend->device()),
                           {0.0f}, {}, std::vector<float>(static_cast<size_t>(d_), 0.0f), {}, {}});
        layers_.back().b_enc.assign(1, 0.0f);
    }
}

void CrossLayerTranscoder::set_layer(int64_t layer, Tensor w_enc, std::vector<float> b_enc, std::vector<float> threshold, std::vector<Tensor> w_dec) {
    if (layer < 0 || layer >= L_) throw std::invalid_argument("CrossLayerTranscoder::set_layer: no such layer");
    if (w_enc.rank() != 2 || w_enc.shape().dim(1) != d_) throw std::invalid_argument("CrossLayerTranscoder::set_layer: W_enc must be (F, d)");
    const int64_t F = w_enc.shape().dim(0);
    if (static_cast<int64_t>(b_enc.size()) != F || (!threshold.empty() && static_cast<int64_t>(threshold.size()) != F)) {
        throw std::invalid_argument("CrossLayerTranscoder::set_layer: b_enc and threshold need one value per feature");
    }
    if (w_dec.empty() || static_cast<int64_t>(w_dec.size()) > L_ - layer) {
        throw std::invalid_argument("CrossLayerTranscoder::set_layer: needs 1 to L - layer decoders");
    }
    for (const Tensor& w : w_dec) {
        if (w.rank() != 2 || w.shape().dim(0) != F || w.shape().dim(1) != d_) throw std::invalid_argument("CrossLayerTranscoder::set_layer: each W_dec must be (F, d)");
    }
    Layer& l = layers_[static_cast<size_t>(layer)];
    l.w_enc = std::move(w_enc);
    l.b_enc = std::move(b_enc);
    l.threshold = std::move(threshold);
    l.w_dec = std::move(w_dec);
}

void CrossLayerTranscoder::set_output_bias(int64_t layer, std::vector<float> b_dec) {
    if (layer < 0 || layer >= L_ || static_cast<int64_t>(b_dec.size()) != d_) throw std::invalid_argument("CrossLayerTranscoder::set_output_bias: bad layer or size");
    layers_[static_cast<size_t>(layer)].b_dec = std::move(b_dec);
}

void CrossLayerTranscoder::set_skip(int64_t layer, Tensor w_skip) {
    if (layer < 0 || layer >= L_ || w_skip.rank() != 2 || w_skip.shape().dim(0) != d_ || w_skip.shape().dim(1) != d_) {
        throw std::invalid_argument("CrossLayerTranscoder::set_skip: bad layer, or W_skip isn't (d, d)");
    }
    layers_[static_cast<size_t>(layer)].w_skip = {std::move(w_skip)};
}

int64_t CrossLayerTranscoder::num_features(int64_t layer) const {
    const Layer& l = layers_.at(static_cast<size_t>(layer));
    return l.w_dec.empty() ? 0 : l.w_enc.shape().dim(0);
}

int64_t CrossLayerTranscoder::span(int64_t layer) const { return static_cast<int64_t>(layers_.at(static_cast<size_t>(layer)).w_dec.size()); }

bool CrossLayerTranscoder::has_skip(int64_t layer) const { return !layers_.at(static_cast<size_t>(layer)).w_skip.empty(); }

std::vector<float> CrossLayerTranscoder::preactivations(int64_t layer, const Tensor& x) const {
    const Layer& l = layers_.at(static_cast<size_t>(layer));
    const int64_t F = num_features(layer), rows = x.shape().dim(0);
    if (F == 0) return {};
    if (x.rank() != 2 || x.shape().dim(1) != d_) throw std::invalid_argument("CrossLayerTranscoder: inputs must be (rows, d)");
    std::vector<float> pre = MatMul(backend_, x.to_host_vector(), rows, d_, l.w_enc, true, F);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t f = 0; f < F; ++f) pre[static_cast<size_t>(r * F + f)] += l.b_enc[static_cast<size_t>(f)];
    }
    return pre;
}

std::vector<float> CrossLayerTranscoder::encode(int64_t layer, const Tensor& x) const {
    std::vector<float> a = preactivations(layer, x);
    const Layer& l = layers_.at(static_cast<size_t>(layer));
    const int64_t F = num_features(layer);
    for (size_t i = 0; i < a.size(); ++i) {
        const float theta = l.threshold.empty() ? 0.0f : l.threshold[i % static_cast<size_t>(F)];
        if (!(a[i] > theta) || a[i] <= 0.0f) a[i] = 0.0f;
    }
    return a;
}

std::vector<float> CrossLayerTranscoder::reconstruct(int64_t layer, const std::vector<std::vector<float>>& activations, const Tensor* mlp_input) const {
    if (layer < 0 || layer >= L_) throw std::invalid_argument("CrossLayerTranscoder::reconstruct: no such layer");
    int64_t rows = -1;
    std::vector<float> y;
    for (int64_t src = 0; src <= layer; ++src) {
        const int64_t k = layer - src, F = num_features(src);
        if (F == 0 || k >= span(src) || static_cast<int64_t>(activations.size()) <= src || activations[static_cast<size_t>(src)].empty()) continue;
        const std::vector<float>& a = activations[static_cast<size_t>(src)];
        if (rows < 0) rows = static_cast<int64_t>(a.size()) / F;
        const std::vector<float> part = MatMul(backend_, a, rows, F, layers_[static_cast<size_t>(src)].w_dec[static_cast<size_t>(k)], false, d_);
        if (y.empty()) y.assign(part.size(), 0.0f);
        for (size_t i = 0; i < y.size(); ++i) y[i] += part[i];
    }
    if (rows < 0) {
        if (mlp_input == nullptr) throw std::invalid_argument("CrossLayerTranscoder::reconstruct: no activations reach this layer");
        rows = mlp_input->shape().dim(0);
        y.assign(static_cast<size_t>(rows * d_), 0.0f);
    }
    const Layer& l = layers_[static_cast<size_t>(layer)];
    if (!l.w_skip.empty() && mlp_input != nullptr) {
        const std::vector<float> s = MatMul(backend_, mlp_input->to_host_vector(), rows, d_, l.w_skip[0], true, d_);
        for (size_t i = 0; i < y.size(); ++i) y[i] += s[i];
    }
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < d_; ++j) y[static_cast<size_t>(r * d_ + j)] += l.b_dec[static_cast<size_t>(j)];
    }
    return y;
}

const Tensor& CrossLayerTranscoder::encoder(int64_t layer) const { return layers_.at(static_cast<size_t>(layer)).w_enc; }

std::vector<float> CrossLayerTranscoder::encoder_row(int64_t layer, int64_t f) const {
    const std::vector<float> w = layers_.at(static_cast<size_t>(layer)).w_enc.to_host_vector();
    return {w.begin() + f * d_, w.begin() + (f + 1) * d_};
}

std::vector<float> CrossLayerTranscoder::decoder_row(int64_t layer, int64_t f, int64_t out_layer) const {
    const std::vector<float> w = decoder(layer, out_layer).to_host_vector();
    return {w.begin() + f * d_, w.begin() + (f + 1) * d_};
}

const Tensor& CrossLayerTranscoder::decoder(int64_t layer, int64_t out_layer) const {
    const Layer& l = layers_.at(static_cast<size_t>(layer));
    const int64_t k = out_layer - layer;
    if (k < 0 || k >= static_cast<int64_t>(l.w_dec.size())) throw std::invalid_argument("CrossLayerTranscoder::decoder: the layer doesn't write there");
    return l.w_dec[static_cast<size_t>(k)];
}

const Tensor& CrossLayerTranscoder::skip(int64_t layer) const {
    const Layer& l = layers_.at(static_cast<size_t>(layer));
    if (l.w_skip.empty()) throw std::invalid_argument("CrossLayerTranscoder::skip: no skip at this layer");
    return l.w_skip[0];
}

CrossLayerTranscoder LoadTranscoders(const std::string& dir, int64_t num_layers, DeviceBackend* backend) {
    const bool per_layer = std::filesystem::exists(LayerFile(dir, "layer_", 0));
    const bool lazy = std::filesystem::exists(LayerFile(dir, "W_enc_", 0));
    if (!per_layer && !lazy) throw std::invalid_argument("LoadTranscoders: " + dir + " holds neither layer_0.safetensors nor W_enc_0.safetensors");
    auto host = [&](const SafetensorsFile& f, const std::string& name) {
        CPUBackend cpu;
        return f.tensor(name, &cpu).to_host_vector();
    };
    auto device = [&](const SafetensorsFile& f, const std::string& name, Shape shape) {
        return Tensor(shape, backend, host(f, name), backend->device());
    };
    std::optional<CrossLayerTranscoder> clt;
    for (int64_t i = 0; i < num_layers; ++i) {
        const SafetensorsFile enc = SafetensorsFile::Map(per_layer ? LayerFile(dir, "layer_", i) : LayerFile(dir, "W_enc_", i));
        const std::string sfx = per_layer ? "" : "_" + std::to_string(i);
        const std::vector<int64_t> shape = enc.info("W_enc" + sfx).shape;
        if (shape.size() != 2) throw std::invalid_argument("LoadTranscoders: W_enc must be (F, d)");
        const int64_t F = shape[0], d = shape[1];
        if (!clt) clt.emplace(num_layers, d, backend);
        if (d != clt->d_model()) throw std::invalid_argument("LoadTranscoders: layers disagree on d_model");
        std::vector<float> threshold;
        for (const std::string& name : {"activation_function.threshold" + sfx, "threshold" + sfx}) {
            if (enc.contains(name)) threshold = host(enc, name);
        }
        std::vector<Tensor> w_dec;
        if (per_layer) {
            w_dec.push_back(device(enc, "W_dec", Shape({F, d})));
        } else {
            const SafetensorsFile dec = SafetensorsFile::Map(LayerFile(dir, "W_dec_", i));
            const std::vector<int64_t> ds = dec.info("W_dec" + sfx).shape;
            if (ds.size() != 3 || ds[0] != F || ds[2] != d || ds[1] < 1 || ds[1] > num_layers - i) {
                throw std::invalid_argument("LoadTranscoders: W_dec_" + std::to_string(i) + " must be (F, L - i, d)");
            }
            const std::vector<float> all = host(dec, "W_dec" + sfx);
            for (int64_t k = 0; k < ds[1]; ++k) {
                std::vector<float> w(static_cast<size_t>(F * d));
                for (int64_t f = 0; f < F; ++f) std::copy_n(all.begin() + (f * ds[1] + k) * d, d, w.begin() + f * d);
                w_dec.emplace_back(Shape({F, d}), backend, w, backend->device());
            }
        }
        clt->set_layer(i, device(enc, "W_enc" + sfx, Shape({F, d})), host(enc, "b_enc" + sfx), std::move(threshold), std::move(w_dec));
        clt->set_output_bias(i, host(enc, "b_dec" + sfx));
        if (enc.contains("W_skip" + sfx)) clt->set_skip(i, device(enc, "W_skip" + sfx, Shape({d, d})));
    }
    return std::move(*clt);
}

// ---- The trace ---------------------------------------------------------------------------------

int64_t CircuitTrace::num_nodes() const {
    return static_cast<int64_t>(features.size()) + num_layers * num_positions + num_positions + static_cast<int64_t>(logit_tokens.size());
}
int64_t CircuitTrace::error_node(int64_t layer, int64_t position) const {
    return static_cast<int64_t>(features.size()) + layer * num_positions + position;
}
int64_t CircuitTrace::token_node(int64_t position) const { return static_cast<int64_t>(features.size()) + num_layers * num_positions + position; }
int64_t CircuitTrace::logit_node(int64_t k) const { return token_node(num_positions) + k; }

CircuitTrace TraceCircuit(CausalLM& model, const CrossLayerTranscoder& clt, const std::vector<int64_t>& ids, const CircuitTraceOptions& o) {
    const int64_t L = model.num_layers(), d = model.config().hidden_size, P = static_cast<int64_t>(ids.size()), V = model.config().vocab_size;
    if (P == 0) throw std::invalid_argument("TraceCircuit: the prompt is empty");
    if (clt.num_layers() != L || clt.d_model() != d) throw std::invalid_argument("TraceCircuit: the transcoder doesn't fit the model");
    if (o.batch < 1 || o.max_logits < 1) throw std::invalid_argument("TraceCircuit: batch and max_logits must be positive");
    DeviceBackend* backend = clt.backend();

    // The model's own forward pass, reading every MLP's input and output and the embeddings.
    std::vector<Tensor> mlp_in;
    std::vector<std::vector<float>> mlp_out(static_cast<size_t>(L));
    std::vector<float> embeddings;
    for (int64_t l = 0; l < L; ++l) {
        model.layer(l).set_mlp_hook([&, l](const Tensor& in, const Tensor& out) {
            mlp_in.push_back(Tensor(Shape({P, d}), backend, in.to_host_vector(), backend->device()));
            mlp_out[static_cast<size_t>(l)] = out.to_host_vector();
            return out;
        });
    }
    model.set_hidden_state_hook([&](int64_t position, const Tensor& h) {
        if (position == 0) embeddings = h.to_host_vector();
        return h;
    });
    const std::vector<float> ids_f(ids.begin(), ids.end());
    const std::vector<float> logits = model.forward(Tensor(Shape({1, P}), backend, ids_f, backend->device())).to_host_vector();
    model.set_hidden_state_hook({});
    for (int64_t l = 0; l < L; ++l) model.layer(l).set_mlp_hook({});
    if (static_cast<int64_t>(mlp_in.size()) != L) throw std::logic_error("TraceCircuit: an MLP hook didn't run");

    CircuitTrace t;
    t.num_layers = L;
    t.num_positions = P;
    t.tokens = ids;
    // Features and errors, with the first skip_positions positions left out.
    std::vector<std::vector<float>> acts(static_cast<size_t>(L));
    std::vector<std::vector<float>> errors(static_cast<size_t>(L));
    for (int64_t l = 0; l < L; ++l) {
        acts[static_cast<size_t>(l)] = clt.encode(l, mlp_in[static_cast<size_t>(l)]);
        const int64_t F = clt.num_features(l);
        for (int64_t p = 0; p < std::min(P, o.skip_positions); ++p) std::fill_n(acts[static_cast<size_t>(l)].begin() + p * F, F, 0.0f);
    }
    for (int64_t l = 0; l < L; ++l) {
        const std::vector<float> rec = clt.reconstruct(l, acts, &mlp_in[static_cast<size_t>(l)]);
        std::vector<float>& e = errors[static_cast<size_t>(l)];
        e.resize(rec.size());
        for (size_t i = 0; i < e.size(); ++i) e[i] = mlp_out[static_cast<size_t>(l)][i] - rec[i];
        for (int64_t p = 0; p < std::min(P, o.skip_positions); ++p) std::fill_n(e.begin() + p * d, d, 0.0f);
    }
    for (int64_t l = 0; l < L; ++l) {
        const int64_t F = clt.num_features(l);
        for (int64_t p = 0; p < P; ++p) {
            for (int64_t f = 0; f < F; ++f) {
                const float a = acts[static_cast<size_t>(l)][static_cast<size_t>(p * F + f)];
                if (a > 0.0f) t.features.push_back({l, p, f, a});
            }
        }
    }
    const auto nF = static_cast<int64_t>(t.features.size());

    // The salient logits at the last position.
    {
        const float* z = logits.data() + (P - 1) * V;
        const double mx = *std::max_element(z, z + V);
        std::vector<double> prob(static_cast<size_t>(V));
        double sum = 0;
        for (int64_t v = 0; v < V; ++v) sum += prob[static_cast<size_t>(v)] = std::exp(z[v] - mx);
        for (double& p : prob) p /= sum;
        std::vector<int64_t> order(static_cast<size_t>(V));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::partial_sort(order.begin(), order.begin() + std::min<int64_t>(o.max_logits, V), order.end(),
                          [&](int64_t a, int64_t b) { return prob[static_cast<size_t>(a)] > prob[static_cast<size_t>(b)]; });
        double cum = 0;
        for (int64_t k = 0; k < std::min<int64_t>(o.max_logits, V); ++k) {
            t.logit_tokens.push_back(order[static_cast<size_t>(k)]);
            t.logit_probabilities.push_back(prob[static_cast<size_t>(order[static_cast<size_t>(k)])]);
            cum += prob[static_cast<size_t>(order[static_cast<size_t>(k)])];
            if (cum >= o.logit_probability) break;
        }
    }
    const auto nLogits = static_cast<int64_t>(t.logit_tokens.size());
    // Unembedding vectors minus their mean, (vocab rows of d).
    std::vector<float> unembed;
    if (LinearModule* head = model.lm_head(); head != nullptr) {
        const std::vector<float> w = head->weight().to_host_vector();  // (d, V)
        unembed.resize(static_cast<size_t>(V * d));
        for (int64_t j = 0; j < d; ++j) {
            for (int64_t v = 0; v < V; ++v) unembed[static_cast<size_t>(v * d + j)] = w[static_cast<size_t>(j * V + v)];
        }
    } else {
        unembed = model.embed_tokens().weight().to_host_vector();  // (V, d), tied
    }
    std::vector<double> mean_u(static_cast<size_t>(d), 0.0);
    for (int64_t v = 0; v < V; ++v) {
        for (int64_t j = 0; j < d; ++j) mean_u[static_cast<size_t>(j)] += unembed[static_cast<size_t>(v * d + j)] / static_cast<double>(V);
    }

    // Frozen pieces of each block.
    struct Frozen {
        std::vector<float> s1, s2, s_post_attn, attn;  // (P, d) scales; attn (H, P, P)
        bool post_attn = false;
    };
    std::vector<Frozen> fz(static_cast<size_t>(L));
    int64_t H = 0, Hkv = 0, D = 0;
    for (int64_t l = 0; l < L; ++l) {
        TransformerBlock& b = model.layer(l);
        Frozen& f = fz[static_cast<size_t>(l)];
        f.s1 = FrozenScale(b.norm1(), P, d);
        f.s2 = FrozenScale(b.norm2(), P, d);
        if (b.post_attn_norm() != nullptr) {
            f.post_attn = true;
            f.s_post_attn = FrozenScale(*b.post_attn_norm(), P, d);
        }
        f.attn = b.mha().last_attention_weights().to_host_vector();
        H = b.mha().num_heads();
        Hkv = b.mha().num_kv_heads();
        D = b.mha().head_dim();
    }
    const std::vector<float> s_final = FrozenScale(model.norm(), P, d);
    const int64_t group = H / Hkv;

    // Decoder rows of the active features into each layer, grouped by output layer, and their
    // encoder rows (feature targets).
    struct Writer {
        int64_t node, position;
        std::vector<float> vec;  ///< a_s times the decoder row
    };
    std::vector<std::vector<Writer>> writers(static_cast<size_t>(L));
    for (int64_t src = 0; src < L; ++src) {
        std::vector<int64_t> mine;
        for (int64_t n = 0; n < nF; ++n) {
            if (t.features[static_cast<size_t>(n)].layer == src) mine.push_back(n);
        }
        if (mine.empty()) continue;
        for (int64_t k = 0; k < clt.span(src); ++k) {
            const std::vector<float> w = clt.decoder(src, src + k).to_host_vector();
            for (int64_t n : mine) {
                const auto& f = t.features[static_cast<size_t>(n)];
                Writer wr{n, f.position, std::vector<float>(static_cast<size_t>(d))};
                for (int64_t j = 0; j < d; ++j) wr.vec[static_cast<size_t>(j)] = f.activation * w[static_cast<size_t>(f.index * d + j)];
                writers[static_cast<size_t>(src + k)].push_back(std::move(wr));
            }
        }
    }
    const int64_t n = t.num_nodes();
    t.adjacency.assign(static_cast<size_t>(n * n), 0.0);
    t.target_values.assign(static_cast<size_t>(n), 0.0);

    // Targets: logits first, then features from the deepest layer down.
    struct Target {
        int64_t node, layer, position;  ///< layer L for logits
        std::vector<float> vec;         ///< the input vector, at the norm2 output or final norm output
    };
    std::vector<Target> targets;
    for (int64_t k = 0; k < nLogits; ++k) {
        std::vector<float> u(static_cast<size_t>(d));
        for (int64_t j = 0; j < d; ++j) u[static_cast<size_t>(j)] = static_cast<float>(unembed[static_cast<size_t>(t.logit_tokens[static_cast<size_t>(k)] * d + j)] - mean_u[static_cast<size_t>(j)]);
        targets.push_back({t.logit_node(k), L, P - 1, std::move(u)});
    }
    {
        std::vector<std::vector<float>> enc_host(static_cast<size_t>(L));  // one host copy of each W_enc
        for (int64_t nn = 0; nn < nF; ++nn) {
            const auto& f = t.features[static_cast<size_t>(nn)];
            std::vector<float>& w = enc_host[static_cast<size_t>(f.layer)];
            if (w.empty()) w = clt.encoder(f.layer).to_host_vector();
            targets.push_back({nn, f.layer, f.position, std::vector<float>(w.begin() + f.index * d, w.begin() + (f.index + 1) * d)});
        }
    }
    std::stable_sort(targets.begin() + nLogits, targets.end(), [](const Target& a, const Target& b) { return a.layer > b.layer; });

    for (size_t b0 = 0; b0 < targets.size(); b0 += static_cast<size_t>(o.batch)) {
        const auto B = static_cast<int64_t>(std::min<size_t>(static_cast<size_t>(o.batch), targets.size() - b0));
        const Target* tg = targets.data() + b0;
        std::vector<float> G(static_cast<size_t>(B * P * d), 0.0f);  // gradient at the residual stream
        auto at = [&](int64_t b, int64_t p) { return G.data() + (b * P + p) * d; };
        for (int64_t b = 0; b < B; ++b) {
            if (tg[b].layer != L) continue;
            float* g = at(b, tg[b].position);
            for (int64_t j = 0; j < d; ++j) g[j] = tg[b].vec[static_cast<size_t>(j)] * s_final[static_cast<size_t>(tg[b].position * d + j)];
        }
        auto add_edge = [&](int64_t b, int64_t source, double value) { t.adjacency[static_cast<size_t>(tg[b].node * n + source)] += value; };
        for (int64_t l = L - 1; l >= 0; --l) {
            const Frozen& f = fz[static_cast<size_t>(l)];
            // MLP output of layer l: errors and every feature writing here.
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t p = 0; p < P; ++p) {
                    const float* g = at(b, p);
                    double s = 0;
                    const float* e = errors[static_cast<size_t>(l)].data() + p * d;
                    for (int64_t j = 0; j < d; ++j) s += static_cast<double>(g[j]) * e[j];
                    if (s != 0) add_edge(b, t.error_node(l, p), s);
                }
                for (const Writer& w : writers[static_cast<size_t>(l)]) {
                    const float* g = at(b, w.position);
                    double s = 0;
                    for (int64_t j = 0; j < d; ++j) s += static_cast<double>(g[j]) * w.vec[static_cast<size_t>(j)];
                    if (s != 0) add_edge(b, w.node, s);
                }
            }
            // The skip, and the feature targets of this layer, at the MLP input; then through norm2.
            std::vector<float> gin(G.size(), 0.0f);
            bool any_in = false;
            if (clt.has_skip(l)) {
                gin = MatMul(backend, G, B * P, d, clt.skip(l), false, d);  // y = x W_skipᵀ: dx = g W_skip
                any_in = true;
            }
            for (int64_t b = 0; b < B; ++b) {
                if (tg[b].layer != l) continue;
                float* g = gin.data() + (b * P + tg[b].position) * d;
                for (int64_t j = 0; j < d; ++j) g[j] += tg[b].vec[static_cast<size_t>(j)];
                any_in = true;
            }
            if (any_in) {
                for (int64_t b = 0; b < B; ++b) {
                    for (int64_t p = 0; p < P; ++p) {
                        for (int64_t j = 0; j < d; ++j) G[static_cast<size_t>((b * P + p) * d + j)] += gin[static_cast<size_t>((b * P + p) * d + j)] * f.s2[static_cast<size_t>(p * d + j)];
                    }
                }
            }
            // Attention with its pattern frozen: back along the value path.
            std::vector<float> gao = G;
            if (f.post_attn) {
                for (int64_t b = 0; b < B; ++b) {
                    for (int64_t i = 0; i < P * d; ++i) gao[static_cast<size_t>(b * P * d + i)] *= f.s_post_attn[static_cast<size_t>(i)];
                }
            }
            TransformerBlock& blk = model.layer(l);
            const std::vector<float> gctx = MatMul(backend, gao, B * P, d, blk.mha().out_proj().weight(), true, H * D);  // (B·P, H·D)
            std::vector<float> gv(static_cast<size_t>(B * P * Hkv * D), 0.0f);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    const int64_t kv = h / group;
                    for (int64_t i = 0; i < P; ++i) {
                        const float* gi = gctx.data() + (b * P + i) * H * D + h * D;
                        for (int64_t j = 0; j <= i; ++j) {
                            const float a = f.attn[static_cast<size_t>((h * P + i) * P + j)];
                            if (a == 0.0f) continue;
                            float* gj = gv.data() + (b * P + j) * Hkv * D + kv * D;
                            for (int64_t q = 0; q < D; ++q) gj[q] += a * gi[q];
                        }
                    }
                }
            }
            const std::vector<float> gn1 = MatMul(backend, gv, B * P, Hkv * D, blk.mha().v_proj().weight(), true, d);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t i = 0; i < P * d; ++i) G[static_cast<size_t>(b * P * d + i)] += gn1[static_cast<size_t>(b * P * d + i)] * f.s1[static_cast<size_t>(i)];
            }
        }
        // The token embeddings (scaled, as the residual stream receives them).
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t p = 0; p < P; ++p) {
                const float* g = at(b, p);
                double s = 0;
                for (int64_t j = 0; j < d; ++j) s += static_cast<double>(g[j]) * embeddings[static_cast<size_t>(p * d + j)];
                if (s != 0) add_edge(b, t.token_node(p), s);
            }
        }
    }

    // Each target's value through the frozen model: what its input vector reads.
    for (const Target& tg : targets) {
        double s = 0;
        if (tg.layer == L) {
            // The final norm's output at the last position: the residual stream times its scale.
            // Recover it from the logits: Σ_j u_j y_j = logit - mean logit (no head bias in these models).
            const float* z = logits.data() + (P - 1) * V;
            double mean = 0;
            for (int64_t v = 0; v < V; ++v) mean += z[v] / static_cast<double>(V);
            s = z[t.logit_tokens[static_cast<size_t>(tg.node - t.logit_node(0))]] - mean;
        } else {
            const std::vector<float> x = mlp_in[static_cast<size_t>(tg.layer)].to_host_vector();
            for (int64_t j = 0; j < d; ++j) s += static_cast<double>(tg.vec[static_cast<size_t>(j)]) * x[static_cast<size_t>(tg.position * d + j)];
        }
        t.target_values[static_cast<size_t>(tg.node)] = s;
    }
    return t;
}

// ---- Pruning and scores (circuit-tracer's graph.py) ------------------------------------------

namespace {

std::vector<double> Normalize(const std::vector<double>& a, int64_t n) {
    std::vector<double> out(a.size());
    for (int64_t r = 0; r < n; ++r) {
        double s = 0;
        for (int64_t c = 0; c < n; ++c) s += std::abs(a[static_cast<size_t>(r * n + c)]);
        s = std::max(s, 1e-10);
        for (int64_t c = 0; c < n; ++c) out[static_cast<size_t>(r * n + c)] = std::abs(a[static_cast<size_t>(r * n + c)]) / s;
    }
    return out;
}

/** @brief w Â + w Â² + ...: each node's influence on the weighted logits. */
std::vector<double> Influence(const std::vector<double>& a, const std::vector<double>& w, int64_t n) {
    std::vector<double> cur(static_cast<size_t>(n), 0.0), total(static_cast<size_t>(n), 0.0);
    auto step = [&](const std::vector<double>& v) {
        std::vector<double> out(static_cast<size_t>(n), 0.0);
        for (int64_t r = 0; r < n; ++r) {
            if (v[static_cast<size_t>(r)] == 0) continue;
            for (int64_t c = 0; c < n; ++c) out[static_cast<size_t>(c)] += v[static_cast<size_t>(r)] * a[static_cast<size_t>(r * n + c)];
        }
        return out;
    };
    cur = step(w);
    for (int64_t it = 0; it <= n + 1; ++it) {
        bool any = false;
        for (int64_t i = 0; i < n; ++i) {
            total[static_cast<size_t>(i)] += cur[static_cast<size_t>(i)];
            any = any || cur[static_cast<size_t>(i)] != 0;
        }
        if (!any) return total;
        cur = step(cur);
    }
    throw std::logic_error("circuit influence: the graph has a cycle");
}

double FindThreshold(std::vector<double> scores, double threshold) {
    std::sort(scores.begin(), scores.end(), std::greater<>());
    double total = 0;
    for (double s : scores) total += s;
    int64_t idx = static_cast<int64_t>(scores.size());
    double cum = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        cum += scores[i];
        if (total > 0 && cum / total >= threshold) {
            idx = static_cast<int64_t>(i);
            break;
        }
    }
    const auto nonzero = std::count_if(scores.begin(), scores.end(), [](double s) { return s > 0; });
    idx = std::min<int64_t>(idx, std::max<int64_t>(nonzero - 1, 0));
    return scores.empty() ? 0.0 : scores[static_cast<size_t>(idx)];
}

std::vector<double> LogitWeights(const CircuitTrace& t) {
    std::vector<double> w(static_cast<size_t>(t.num_nodes()), 0.0);
    for (size_t k = 0; k < t.logit_probabilities.size(); ++k) w[static_cast<size_t>(t.logit_node(static_cast<int64_t>(k)))] = t.logit_probabilities[k];
    return w;
}

}  // namespace

CircuitScores ScoreCircuit(const CircuitTrace& t) {
    const int64_t n = t.num_nodes();
    const std::vector<double> w = LogitWeights(t), a = Normalize(t.adjacency, n), infl = Influence(a, w, n);
    const int64_t e0 = t.error_node(0, 0), e1 = t.token_node(0), t1 = t.logit_node(0);
    double tok = 0, err = 0;
    for (int64_t i = e0; i < e1; ++i) err += infl[static_cast<size_t>(i)];
    for (int64_t i = e1; i < t1; ++i) tok += infl[static_cast<size_t>(i)];
    CircuitScores s;
    s.replacement = tok + err > 0 ? tok / (tok + err) : 0;
    double num = 0, den = 0;
    for (int64_t r = 0; r < n; ++r) {
        double from_errors = 0;
        for (int64_t c = e0; c < e1; ++c) from_errors += a[static_cast<size_t>(r * n + c)];
        const double out = infl[static_cast<size_t>(r)] + w[static_cast<size_t>(r)];
        num += (1 - from_errors) * out;
        den += out;
    }
    s.completeness = den > 0 ? num / den : 0;
    return s;
}

AttributionGraph ToAttributionGraph(const CircuitTrace& t, const CircuitTraceOptions& o, const std::vector<std::string>& logit_labels) {
    if (!(o.node_threshold >= 0 && o.node_threshold <= 1) || !(o.edge_threshold >= 0 && o.edge_threshold <= 1)) {
        throw std::invalid_argument("ToAttributionGraph: thresholds must be in [0, 1]");
    }
    const int64_t n = t.num_nodes(), nF = static_cast<int64_t>(t.features.size()), first_token = t.token_node(0);
    const std::vector<double> w = LogitWeights(t);
    const std::vector<double> node_infl = Influence(Normalize(t.adjacency, n), w, n);
    const double node_cut = FindThreshold(node_infl, o.node_threshold);
    std::vector<uint8_t> keep(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) keep[static_cast<size_t>(i)] = i >= first_token || node_infl[static_cast<size_t>(i)] >= node_cut ? 1 : 0;
    std::vector<double> pruned = t.adjacency;
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            if (keep[static_cast<size_t>(r)] == 0 || keep[static_cast<size_t>(c)] == 0) pruned[static_cast<size_t>(r * n + c)] = 0;
        }
    }
    const std::vector<double> na = Normalize(pruned, n);
    std::vector<double> pi = Influence(na, w, n);
    for (int64_t i = 0; i < n; ++i) pi[static_cast<size_t>(i)] += w[static_cast<size_t>(i)];
    std::vector<double> edge_scores(static_cast<size_t>(n * n));
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < n; ++c) edge_scores[static_cast<size_t>(r * n + c)] = na[static_cast<size_t>(r * n + c)] * pi[static_cast<size_t>(r)];
    }
    const double edge_cut = FindThreshold(edge_scores, o.edge_threshold);
    std::vector<uint8_t> edge(static_cast<size_t>(n * n));
    for (size_t i = 0; i < edge.size(); ++i) edge[i] = edge_scores[i] >= edge_cut ? 1 : 0;
    // Features and errors need outgoing edges, features incoming ones, until nothing changes.
    for (bool changed = true; changed;) {
        changed = false;
        for (int64_t i = 0; i < n; ++i) {
            if (keep[static_cast<size_t>(i)] == 0) {
                for (int64_t j = 0; j < n; ++j) edge[static_cast<size_t>(i * n + j)] = edge[static_cast<size_t>(j * n + i)] = 0;
            }
        }
        for (int64_t i = 0; i < first_token; ++i) {
            if (keep[static_cast<size_t>(i)] == 0) continue;
            bool out = false, in = false;
            for (int64_t j = 0; j < n; ++j) {
                out = out || edge[static_cast<size_t>(j * n + i)] != 0;
                in = in || edge[static_cast<size_t>(i * n + j)] != 0;
            }
            if (!out || (i < nF && !in)) {
                keep[static_cast<size_t>(i)] = 0;
                changed = true;
            }
        }
    }

    AttributionGraph g;
    g.slug = o.slug;
    g.scan = o.scan;
    g.prompt = o.prompt;
    g.node_threshold = o.node_threshold;
    g.prompt_tokens = o.prompt_tokens;
    if (g.prompt_tokens.empty()) {
        for (int64_t id : t.tokens) g.prompt_tokens.push_back(std::to_string(id));
    }
    std::vector<std::string> ids(static_cast<size_t>(n));
    std::vector<double> scores;
    for (int64_t i = 0; i < n; ++i) {
        if (keep[static_cast<size_t>(i)] == 0) continue;
        AttributionGraph::Node node;
        if (i < nF) {
            const auto& f = t.features[static_cast<size_t>(i)];
            node.node_id = std::to_string(f.layer) + "_" + std::to_string(f.index) + "_" + std::to_string(f.position);
            node.feature = (f.layer + f.index) * (f.layer + f.index + 1) / 2 + f.index;  // circuit-tracer's Cantor pairing
            node.layer = std::to_string(f.layer);
            node.ctx_idx = f.position;
            node.feature_type = "cross layer transcoder";
            node.js_node_id = std::to_string(f.layer) + "_" + std::to_string(f.index) + "-0";
            node.clerp = "L" + std::to_string(f.layer) + " feature " + std::to_string(f.index);
            node.activation = f.activation;
        } else if (i < first_token) {
            const int64_t l = (i - nF) / t.num_positions, p = (i - nF) % t.num_positions;
            node.node_id = std::to_string(l) + "_-1_" + std::to_string(p);
            node.feature = -1;
            node.layer = std::to_string(l);
            node.ctx_idx = p;
            node.feature_type = "mlp reconstruction error";
            node.js_node_id = std::to_string(l) + "_-1-" + std::to_string(p);
            node.clerp = "Error L" + std::to_string(l);
        } else if (i < t.logit_node(0)) {
            const int64_t p = i - first_token;
            const int64_t tok = t.tokens[static_cast<size_t>(p)];
            node.node_id = "E_" + std::to_string(tok) + "_" + std::to_string(p);
            node.feature = p;
            node.layer = "E";
            node.ctx_idx = p;
            node.feature_type = "embedding";
            node.js_node_id = node.node_id;
            node.clerp = "Emb: \"" + g.prompt_tokens[static_cast<size_t>(p)] + "\"";
        } else {
            const auto k = static_cast<size_t>(i - t.logit_node(0));
            const int64_t tok = t.logit_tokens[k];
            char prob[32];
            std::snprintf(prob, sizeof prob, "%.4f", t.logit_probabilities[k]);
            node.node_id = std::to_string(t.num_layers + 1) + "_" + std::to_string(tok) + "_" + std::to_string(t.num_positions - 1);
            node.layer = std::to_string(t.num_layers + 1);
            node.ctx_idx = t.num_positions - 1;
            node.feature_type = "logit";
            node.js_node_id = "L_" + std::to_string(tok) + "-" + std::to_string(t.num_positions - 1);
            const std::string label = k < logit_labels.size() ? logit_labels[k] : std::to_string(tok);
            node.clerp = "Output \"" + label + "\" (p=" + prob + ")";
            node.token_prob = t.logit_probabilities[k];
            node.is_target_logit = k == 0;
        }
        ids[static_cast<size_t>(i)] = node.node_id;
        g.nodes.push_back(std::move(node));
        scores.push_back(node_infl[static_cast<size_t>(i)]);
    }
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            if (edge[static_cast<size_t>(r * n + c)] == 0 || keep[static_cast<size_t>(r)] == 0 || keep[static_cast<size_t>(c)] == 0) continue;
            const double v = t.adjacency[static_cast<size_t>(r * n + c)];
            if (v == 0) continue;
            g.links.push_back({ids[static_cast<size_t>(c)], ids[static_cast<size_t>(r)], v});
        }
    }
    ComputeInfluence(g, scores);
    return g;
}

}  // namespace pulsatrix
