#include "pulsatrix/topk_sparse_autoencoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"
#include "pulsatrix/top_k.hpp"

namespace pulsatrix {

namespace {

Tensor* GradOf(LinearModule& m, const char* name) {
    for (const NamedParamRef& p : m.named_parameters()) {
        if (p.name == name) return p.ref.grad;
    }
    throw std::logic_error("TopKSparseAutoencoder: no parameter named " + std::string(name));
}

}  // namespace

TopKSparseAutoencoder::TopKSparseAutoencoder(int64_t dim, int64_t num_features, DeviceBackend* backend, TopKSaeOptions options)
    : dim_(dim),
      m_(num_features),
      backend_(backend),
      options_(options),
      encoder_(dim > 0 ? dim : 1, num_features > 0 ? num_features : 1, backend),
      decoder_(num_features > 0 ? num_features : 1, dim > 0 ? dim : 1, backend) {
    if (dim < 1 || num_features < 1) throw std::invalid_argument("TopKSparseAutoencoder: dim and num_features must be positive");
    if (options.k < 1 || options.k > num_features) throw std::invalid_argument("TopKSparseAutoencoder: k must be in [1, num_features]");
    if (options.k_aux < 0 || options.aux_coefficient < 0 || options.dead_after < 1) {
        throw std::invalid_argument("TopKSparseAutoencoder: k_aux and aux_coefficient must not be negative, dead_after must be positive");
    }
    if (options_.k_aux == 0) options_.k_aux = std::max<int64_t>(1, dim / 2);
    // Random unit decoder directions; the encoder starts as their transpose.
    PortableRng rng{options.seed};
    std::vector<float> dec(static_cast<size_t>(m_ * dim_)), enc(static_cast<size_t>(dim_ * m_));
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < dim_; ++j) {
            const double v = rng.gaussian();
            dec[static_cast<size_t>(i * dim_ + j)] = static_cast<float>(v);
            sq += v * v;
        }
        const double norm = std::sqrt(sq);
        for (int64_t j = 0; j < dim_; ++j) {
            dec[static_cast<size_t>(i * dim_ + j)] = static_cast<float>(dec[static_cast<size_t>(i * dim_ + j)] / norm);
            enc[static_cast<size_t>(j * m_ + i)] = dec[static_cast<size_t>(i * dim_ + j)];
        }
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    decoder_.set_bias(std::vector<float>(static_cast<size_t>(dim_), 0.0f));
    encoder_.set_bias(std::vector<float>(static_cast<size_t>(m_), 0.0f));
    since_fired_.assign(static_cast<size_t>(m_), 0);
}

void TopKSparseAutoencoder::Check(const Tensor& x, const char* who) const {
    if (x.rank() != 2 || x.shape().dim(1) != dim_ || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the batch must be (N, dim) with N > 0");
    }
}

void TopKSparseAutoencoder::initialize_bias(const Tensor& x) {
    Check(x, "TopKSparseAutoencoder::initialize_bias");
    const std::vector<float> v = x.to_host_vector();
    const int64_t N = x.shape().dim(0);
    std::vector<double> mean(static_cast<size_t>(dim_), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < dim_; ++j) mean[static_cast<size_t>(j)] += v[static_cast<size_t>(r * dim_ + j)];
    }
    std::vector<float> b(static_cast<size_t>(dim_));
    for (int64_t j = 0; j < dim_; ++j) b[static_cast<size_t>(j)] = static_cast<float>(mean[static_cast<size_t>(j)] / static_cast<double>(N));
    decoder_.set_bias(b);
}

Tensor TopKSparseAutoencoder::Centered(const Tensor& x) {
    std::vector<float> v = x.to_host_vector();
    const std::vector<float> b = decoder_.bias().to_host_vector();
    for (size_t i = 0; i < v.size(); ++i) v[i] -= b[i % static_cast<size_t>(dim_)];
    return Tensor(x.shape(), backend_, v, backend_->device());
}

TopKSparseAutoencoder::Pass TopKSparseAutoencoder::Encode(const Tensor& x, Tensor* centered) {
    const Tensor xc = Centered(x);
    if (centered != nullptr) *centered = xc;
    const Tensor pre = encoder_.forward(xc);
    const TopKResult top = top_k(pre, options_.k);
    const std::vector<float> values = top.values.to_host_vector(), indices = top.indices.to_host_vector();
    const int64_t N = x.shape().dim(0), k = options_.k;
    Pass p;
    p.pre = pre.to_host_vector();
    p.codes.assign(static_cast<size_t>(N * m_), 0.0f);
    p.kept.resize(static_cast<size_t>(N * k));
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const auto i = static_cast<int64_t>(indices[static_cast<size_t>(r * k + j)]);
            p.kept[static_cast<size_t>(r * k + j)] = i;
            p.codes[static_cast<size_t>(r * m_ + i)] = std::max(0.0f, values[static_cast<size_t>(r * k + j)]);  // ReLU(TopK)
        }
    }
    return p;
}

Tensor TopKSparseAutoencoder::encode(const Tensor& x) {
    Check(x, "TopKSparseAutoencoder::encode");
    const Pass p = Encode(x);
    return Tensor(Shape({x.shape().dim(0), m_}), backend_, p.codes, backend_->device());
}

Tensor TopKSparseAutoencoder::decode(const Tensor& codes) {
    if (codes.rank() != 2 || codes.shape().dim(1) != m_ || codes.shape().dim(0) < 1) {
        throw std::invalid_argument("TopKSparseAutoencoder::decode: codes must be (N, num_features) with N > 0");
    }
    return decoder_.forward(codes);
}

std::vector<int64_t> TopKSparseAutoencoder::dead_latents() const {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < m_; ++i) {
        if (since_fired_[static_cast<size_t>(i)] >= options_.dead_after) out.push_back(i);
    }
    return out;
}

void TopKSparseAutoencoder::set_inputs_since_fired(const std::vector<int64_t>& since) {
    if (static_cast<int64_t>(since.size()) != m_ || std::any_of(since.begin(), since.end(), [](int64_t v) { return v < 0; })) {
        throw std::invalid_argument("TopKSparseAutoencoder::set_inputs_since_fired: needs num_features non-negative values");
    }
    since_fired_ = since;
}

void TopKSparseAutoencoder::ProjectDecoderGradient() {
    Tensor* g = GradOf(decoder_, "weight");
    std::vector<float> grad = g->to_host_vector();
    const std::vector<float> w = decoder_.weight().to_host_vector();
    for (int64_t i = 0; i < m_; ++i) {
        double norm2 = 0, dot = 0;
        for (int64_t j = 0; j < dim_; ++j) {
            const size_t k = static_cast<size_t>(i * dim_ + j);
            norm2 += static_cast<double>(w[k]) * w[k];
            dot += static_cast<double>(w[k]) * grad[k];
        }
        if (norm2 == 0) continue;
        for (int64_t j = 0; j < dim_; ++j) {
            const size_t k = static_cast<size_t>(i * dim_ + j);
            grad[k] = static_cast<float>(grad[k] - dot / norm2 * w[k]);
        }
    }
    *g = Tensor(g->shape(), g->backend(), grad, g->device());
}

FeaturizerLoss TopKSparseAutoencoder::loss_and_backward(const Tensor& x, std::vector<float>* codes_out) {
    Check(x, "TopKSparseAutoencoder::loss_and_backward");
    const int64_t N = x.shape().dim(0), k = options_.k;
    const std::vector<int64_t> dead = dead_latents();  // from the inputs before this batch
    Tensor xc = x;
    const Pass p = Encode(x, &xc);
    const Tensor codes(Shape({N, m_}), backend_, p.codes, backend_->device());
    const std::vector<float> xh = decoder_.forward(codes).to_host_vector(), xv = x.to_host_vector();
    FeaturizerLoss loss;
    const double n_el = static_cast<double>(N * dim_);
    std::vector<float> g_main(xh.size()), residual(xh.size());
    double mse = 0;
    for (size_t i = 0; i < xh.size(); ++i) {
        const double d = static_cast<double>(xh[i]) - xv[i];
        mse += d * d;
        g_main[i] = static_cast<float>(2.0 * d / n_el);
        residual[i] = static_cast<float>(-d);  // e = x - x̂
    }
    loss.reconstruction = static_cast<float>(mse / n_el);
    // Main path: back through the decoder; only the kept, positive latents pass it on.
    const std::vector<float> g_codes = decoder_.backward(Tensor(Shape({N, dim_}), backend_, g_main, backend_->device())).to_host_vector();
    std::vector<float> g_pre(static_cast<size_t>(N * m_), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const auto i = static_cast<size_t>(r * m_ + p.kept[static_cast<size_t>(r * k + j)]);
            if (p.codes[i] > 0.0f) g_pre[i] = g_codes[i];
        }
    }
    // Auxiliary path: the k_aux largest dead latents reconstruct the residual.
    if (options_.aux_coefficient > 0.0f && !dead.empty()) {
        const auto kaux = static_cast<size_t>(std::min<int64_t>(options_.k_aux, static_cast<int64_t>(dead.size())));
        std::vector<float> aux(static_cast<size_t>(N * m_), 0.0f);
        std::vector<int64_t> order(dead);
        for (int64_t r = 0; r < N; ++r) {
            const float* pre = p.pre.data() + r * m_;
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(kaux), order.end(), [&](int64_t a, int64_t b) {
                return pre[a] > pre[b] || (pre[a] == pre[b] && a < b);
            });
            for (size_t j = 0; j < kaux; ++j) aux[static_cast<size_t>(r * m_ + order[j])] = std::max(0.0f, pre[order[j]]);
        }
        // ê = z_aux W_dec, without the bias.
        std::vector<float> e_hat = decoder_.forward(Tensor(Shape({N, m_}), backend_, aux, backend_->device())).to_host_vector();
        const std::vector<float> b = decoder_.bias().to_host_vector();
        std::vector<float> g_aux(e_hat.size(), 0.0f);
        double total = 0;
        for (int64_t r = 0; r < N; ++r) {
            double err = 0, energy = 0;
            for (int64_t j = 0; j < dim_; ++j) {
                const size_t i = static_cast<size_t>(r * dim_ + j);
                e_hat[i] -= b[static_cast<size_t>(j)];
                err += (static_cast<double>(e_hat[i]) - residual[i]) * (e_hat[i] - residual[i]);
                energy += static_cast<double>(residual[i]) * residual[i];
            }
            if (energy == 0) continue;
            total += err / energy;
            for (int64_t j = 0; j < dim_; ++j) {
                const size_t i = static_cast<size_t>(r * dim_ + j);
                g_aux[i] = static_cast<float>(options_.aux_coefficient * 2.0 * (e_hat[i] - residual[i]) / energy / static_cast<double>(N));
            }
        }
        loss.sparsity = static_cast<float>(options_.aux_coefficient * total / static_cast<double>(N));
        // The bias isn't part of ê, so its gradient from this pass is taken back out.
        Tensor* bias_grad = GradOf(decoder_, "bias");
        const std::vector<float> bias_before = bias_grad->to_host_vector();
        const std::vector<float> g_aux_codes =
            decoder_.backward(Tensor(Shape({N, dim_}), backend_, g_aux, backend_->device())).to_host_vector();
        *bias_grad = Tensor(bias_grad->shape(), bias_grad->backend(), bias_before, bias_grad->device());
        for (size_t i = 0; i < aux.size(); ++i) {
            if (aux[i] > 0.0f) g_pre[i] += g_aux_codes[i];
        }
    }
    loss.total = loss.reconstruction + loss.sparsity;
    // Through the encoder, and the bias subtraction: x - b_dec sends minus its gradient to b_dec.
    const std::vector<float> g_xc = encoder_.backward(Tensor(Shape({N, m_}), backend_, g_pre, backend_->device())).to_host_vector();
    Tensor* bias_grad = GradOf(decoder_, "bias");
    std::vector<float> bg = bias_grad->to_host_vector();
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < dim_; ++j) bg[static_cast<size_t>(j)] -= g_xc[static_cast<size_t>(r * dim_ + j)];
    }
    *bias_grad = Tensor(bias_grad->shape(), bias_grad->backend(), bg, bias_grad->device());
    ProjectDecoderGradient();
    // Dead-latent tracking: inputs since each latent last fired.
    for (int64_t i = 0; i < m_; ++i) {
        int64_t last = -1;
        for (int64_t r = 0; r < N; ++r) {
            if (p.codes[static_cast<size_t>(r * m_ + i)] > 0.0f) last = r;
        }
        since_fired_[static_cast<size_t>(i)] = last < 0 ? since_fired_[static_cast<size_t>(i)] + N : N - 1 - last;
    }
    if (codes_out != nullptr) *codes_out = p.codes;
    return loss;
}

void TopKSparseAutoencoder::normalize_decoder() {
    std::vector<float> w = decoder_.weight().to_host_vector();
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < dim_; ++j) sq += static_cast<double>(w[static_cast<size_t>(i * dim_ + j)]) * w[static_cast<size_t>(i * dim_ + j)];
        const double norm = std::sqrt(sq);
        if (norm == 0) continue;
        for (int64_t j = 0; j < dim_; ++j) w[static_cast<size_t>(i * dim_ + j)] = static_cast<float>(w[static_cast<size_t>(i * dim_ + j)] / norm);
    }
    decoder_.set_weight(w);
}

std::vector<float> TopKSparseAutoencoder::decoder_direction(int64_t i) {
    if (i < 0 || i >= m_) throw std::invalid_argument("TopKSparseAutoencoder::decoder_direction: no such feature");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    return {w.begin() + static_cast<std::ptrdiff_t>(i * dim_), w.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim_)};
}

Tensor TopKSparseAutoencoder::forward_impl(const Tensor& input) {
    Check(input, "TopKSparseAutoencoder::forward");
    const Pass p = Encode(input);
    last_codes_ = p.codes;
    last_kept_ = p.kept;
    last_rows_ = input.shape().dim(0);
    return decoder_.forward(Tensor(Shape({last_rows_, m_}), backend_, p.codes, backend_->device()));
}

Tensor TopKSparseAutoencoder::backward(const Tensor& grad_output) {
    if (last_rows_ == 0) throw std::logic_error("TopKSparseAutoencoder::backward: called before any forward()");
    const int64_t N = last_rows_, k = options_.k;
    const std::vector<float> g_codes = decoder_.backward(grad_output).to_host_vector();
    std::vector<float> g_pre(static_cast<size_t>(N * m_), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const auto i = static_cast<size_t>(r * m_ + last_kept_[static_cast<size_t>(r * k + j)]);
            if (last_codes_[i] > 0.0f) g_pre[i] = g_codes[i];
        }
    }
    const Tensor g_xc = encoder_.backward(Tensor(Shape({N, m_}), backend_, g_pre, backend_->device()));
    const std::vector<float> gx = g_xc.to_host_vector();
    Tensor* bias_grad = GradOf(decoder_, "bias");
    std::vector<float> bg = bias_grad->to_host_vector();
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < dim_; ++j) bg[static_cast<size_t>(j)] -= gx[static_cast<size_t>(r * dim_ + j)];
    }
    *bias_grad = Tensor(bias_grad->shape(), bias_grad->backend(), bg, bias_grad->device());
    return g_xc;
}

Tensor TopKSparseAutoencoder::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_rows_ == 0) throw std::logic_error("TopKSparseAutoencoder::propagate_relevance: called before any forward()");
    std::vector<float> r_codes = decoder_.propagate_relevance(relevance_out, config).to_host_vector();
    for (size_t i = 0; i < r_codes.size(); ++i) {
        if (last_codes_[i] <= 0.0f) r_codes[i] = 0.0f;  // only the kept latents carry relevance
    }
    return encoder_.propagate_relevance(Tensor(Shape({last_rows_, m_}), backend_, r_codes, backend_->device()), config);
}

std::vector<NamedParamRef> TopKSparseAutoencoder::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "encoder", encoder_);
    append_named_parameters(out, "decoder", decoder_);
    return out;
}

void TopKSparseAutoencoder::set_training(bool training) {
    Module::set_training(training);
    encoder_.set_training(training);
    decoder_.set_training(training);
}

void TopKSparseAutoencoder::release_activations() {
    encoder_.release_activations();
    decoder_.release_activations();
    last_codes_.clear();
    last_kept_.clear();
    last_rows_ = 0;
}

}  // namespace pulsatrix
