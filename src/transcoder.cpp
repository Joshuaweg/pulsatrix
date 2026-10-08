#include "pulsatrix/transcoder.hpp"

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
    throw std::logic_error("Transcoder: no parameter named " + std::string(name));
}

}  // namespace

Transcoder::Transcoder(int64_t input_dim, int64_t output_dim, int64_t num_features, DeviceBackend* backend, TranscoderOptions options)
    : in_(input_dim),
      out_(output_dim),
      m_(num_features),
      backend_(backend),
      options_(options),
      encoder_(input_dim > 0 ? input_dim : 1, num_features > 0 ? num_features : 1, backend),
      decoder_(num_features > 0 ? num_features : 1, output_dim > 0 ? output_dim : 1, backend) {
    if (input_dim < 1 || output_dim < 1 || num_features < 1) throw std::invalid_argument("Transcoder: sizes must be positive");
    if (options.k < 1 || options.k > num_features) throw std::invalid_argument("Transcoder: k must be in [1, num_features]");
    if (options.k_aux < 0 || options.aux_coefficient < 0 || options.dead_after < 1) {
        throw std::invalid_argument("Transcoder: k_aux and aux_coefficient must not be negative, dead_after must be positive");
    }
    if (options_.k_aux == 0) options_.k_aux = std::max<int64_t>(1, output_dim / 2);
    PortableRng rng{options.seed};
    auto unit_rows = [&](int64_t rows, int64_t cols) {
        std::vector<float> w(static_cast<size_t>(rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            double sq = 0;
            for (int64_t c = 0; c < cols; ++c) {
                const double v = rng.gaussian();
                w[static_cast<size_t>(r * cols + c)] = static_cast<float>(v);
                sq += v * v;
            }
            for (int64_t c = 0; c < cols; ++c) w[static_cast<size_t>(r * cols + c)] = static_cast<float>(w[static_cast<size_t>(r * cols + c)] / std::sqrt(sq));
        }
        return w;
    };
    const std::vector<float> dec = unit_rows(m_, out_);  // (m, out)
    std::vector<float> enc(static_cast<size_t>(in_ * m_));  // (in, m)
    if (in_ == out_) {
        for (int64_t i = 0; i < m_; ++i) {
            for (int64_t j = 0; j < in_; ++j) enc[static_cast<size_t>(j * m_ + i)] = dec[static_cast<size_t>(i * out_ + j)];
        }
    } else {
        const std::vector<float> rows = unit_rows(m_, in_);
        for (int64_t i = 0; i < m_; ++i) {
            for (int64_t j = 0; j < in_; ++j) enc[static_cast<size_t>(j * m_ + i)] = rows[static_cast<size_t>(i * in_ + j)];
        }
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    decoder_.set_bias(std::vector<float>(static_cast<size_t>(out_), 0.0f));
    encoder_.set_bias(std::vector<float>(static_cast<size_t>(m_), 0.0f));
    if (options.skip) skip_ = std::make_unique<LinearModule>(in_, out_, backend, /*use_bias=*/false);  // starts at zero
    since_fired_.assign(static_cast<size_t>(m_), 0);
}

void Transcoder::CheckInput(const Tensor& x, const char* who) const {
    if (x.rank() != 2 || x.shape().dim(1) != in_ || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the input must be (N, input_dim) with N > 0");
    }
}

void Transcoder::initialize_bias(const Tensor& target) {
    if (target.rank() != 2 || target.shape().dim(1) != out_ || target.shape().dim(0) < 1) {
        throw std::invalid_argument("Transcoder::initialize_bias: the targets must be (N, output_dim) with N > 0");
    }
    const std::vector<float> v = target.to_host_vector();
    const int64_t N = target.shape().dim(0);
    std::vector<double> mean(static_cast<size_t>(out_), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < out_; ++j) mean[static_cast<size_t>(j)] += v[static_cast<size_t>(r * out_ + j)];
    }
    std::vector<float> b(static_cast<size_t>(out_));
    for (int64_t j = 0; j < out_; ++j) b[static_cast<size_t>(j)] = static_cast<float>(mean[static_cast<size_t>(j)] / static_cast<double>(N));
    decoder_.set_bias(b);
}

std::vector<float> Transcoder::Codes(const Tensor& pre) const {
    const int64_t N = pre.shape().dim(0), k = options_.k;
    const TopKResult top = top_k(pre, k);
    const std::vector<float> values = top.values.to_host_vector(), indices = top.indices.to_host_vector();
    std::vector<float> codes(static_cast<size_t>(N * m_), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const auto i = static_cast<int64_t>(indices[static_cast<size_t>(r * k + j)]);
            codes[static_cast<size_t>(r * m_ + i)] = std::max(0.0f, values[static_cast<size_t>(r * k + j)]);
        }
    }
    return codes;
}

Tensor Transcoder::encode(const Tensor& x) {
    CheckInput(x, "Transcoder::encode");
    return Tensor(Shape({x.shape().dim(0), m_}), backend_, Codes(encoder_.forward(x)), backend_->device());
}

Tensor Transcoder::decode(const Tensor& codes) {
    if (codes.rank() != 2 || codes.shape().dim(1) != m_ || codes.shape().dim(0) < 1) {
        throw std::invalid_argument("Transcoder::decode: codes must be (N, num_features) with N > 0");
    }
    return decoder_.forward(codes);
}

Tensor Transcoder::predict(const Tensor& x) {
    CheckInput(x, "Transcoder::predict");
    Tensor y = decoder_.forward(Tensor(Shape({x.shape().dim(0), m_}), backend_, Codes(encoder_.forward(x)), backend_->device()));
    if (!skip_) return y;
    const Tensor s = skip_->forward(x);
    Tensor out(y.shape(), backend_, y.device());
    backend_->add(y.data(), s.data(), out.data(), static_cast<size_t>(y.numel()));
    return out;
}

FeaturizerLoss Transcoder::loss_and_backward(const Tensor&, std::vector<float>*) {
    throw std::invalid_argument("Transcoder::loss_and_backward: a transcoder needs a target; use loss_and_backward_with_target()");
}

std::vector<int64_t> Transcoder::dead_latents() const {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < m_; ++i) {
        if (since_fired_[static_cast<size_t>(i)] >= options_.dead_after) out.push_back(i);
    }
    return out;
}

void Transcoder::set_inputs_since_fired(const std::vector<int64_t>& since) {
    if (static_cast<int64_t>(since.size()) != m_ || std::any_of(since.begin(), since.end(), [](int64_t v) { return v < 0; })) {
        throw std::invalid_argument("Transcoder::set_inputs_since_fired: needs num_features non-negative values");
    }
    since_fired_ = since;
}

Tensor Transcoder::BackwardFromCodes(const std::vector<float>& codes, const std::vector<float>& g_codes) {
    std::vector<float> g_pre(codes.size(), 0.0f);
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] > 0.0f) g_pre[i] = g_codes[i];
    }
    const auto N = static_cast<int64_t>(codes.size()) / m_;
    return encoder_.backward(Tensor(Shape({N, m_}), backend_, g_pre, backend_->device()));
}

FeaturizerLoss Transcoder::loss_and_backward_with_target(const Tensor& x, const Tensor& target, std::vector<float>* codes_out) {
    CheckInput(x, "Transcoder::loss_and_backward_with_target");
    const int64_t N = x.shape().dim(0);
    if (target.rank() != 2 || target.shape().dim(0) != N || target.shape().dim(1) != out_) {
        throw std::invalid_argument("Transcoder::loss_and_backward_with_target: the target must be (N, output_dim)");
    }
    const std::vector<int64_t> dead = dead_latents();
    const Tensor pre_t = encoder_.forward(x);
    const std::vector<float> pre = pre_t.to_host_vector();
    const std::vector<float> codes = Codes(pre_t);
    std::vector<float> yh = decoder_.forward(Tensor(Shape({N, m_}), backend_, codes, backend_->device())).to_host_vector();
    if (skip_) {
        const std::vector<float> s = skip_->forward(x).to_host_vector();
        for (size_t i = 0; i < yh.size(); ++i) yh[i] += s[i];
    }
    const std::vector<float> y = target.to_host_vector();
    const double n_el = static_cast<double>(N * out_);
    std::vector<float> g(yh.size()), residual(yh.size());
    double mse = 0;
    for (size_t i = 0; i < yh.size(); ++i) {
        const double d = static_cast<double>(yh[i]) - y[i];
        mse += d * d;
        g[i] = static_cast<float>(2.0 * d / n_el);
        residual[i] = static_cast<float>(-d);
    }
    FeaturizerLoss loss;
    loss.reconstruction = static_cast<float>(mse / n_el);
    const Tensor g_t(Shape({N, out_}), backend_, g, backend_->device());
    if (skip_) (void)skip_->backward(g_t);
    std::vector<float> g_codes = decoder_.backward(g_t).to_host_vector();
    for (size_t i = 0; i < g_codes.size(); ++i) {
        if (codes[i] <= 0.0f) g_codes[i] = 0.0f;
    }
    // AuxK: the k_aux largest dead latents predict the error, without the bias.
    std::vector<float> aux_codes(codes.size(), 0.0f);
    if (options_.aux_coefficient > 0.0f && !dead.empty()) {
        const auto kaux = static_cast<size_t>(std::min<int64_t>(options_.k_aux, static_cast<int64_t>(dead.size())));
        std::vector<int64_t> order(dead);
        for (int64_t r = 0; r < N; ++r) {
            const float* p = pre.data() + r * m_;
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(kaux), order.end(),
                              [&](int64_t a, int64_t b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
            for (size_t j = 0; j < kaux; ++j) aux_codes[static_cast<size_t>(r * m_ + order[j])] = std::max(0.0f, p[order[j]]);
        }
        std::vector<float> e_hat = decoder_.forward(Tensor(Shape({N, m_}), backend_, aux_codes, backend_->device())).to_host_vector();
        const std::vector<float> b = decoder_.bias().to_host_vector();
        std::vector<float> g_aux(e_hat.size(), 0.0f);
        double total = 0;
        for (int64_t r = 0; r < N; ++r) {
            double err = 0, energy = 0;
            for (int64_t j = 0; j < out_; ++j) {
                const size_t i = static_cast<size_t>(r * out_ + j);
                e_hat[i] -= b[static_cast<size_t>(j)];
                err += (static_cast<double>(e_hat[i]) - residual[i]) * (e_hat[i] - residual[i]);
                energy += static_cast<double>(residual[i]) * residual[i];
            }
            if (energy == 0) continue;
            total += err / energy;
            for (int64_t j = 0; j < out_; ++j) {
                const size_t i = static_cast<size_t>(r * out_ + j);
                g_aux[i] = static_cast<float>(options_.aux_coefficient * 2.0 * (e_hat[i] - residual[i]) / energy / static_cast<double>(N));
            }
        }
        loss.sparsity = static_cast<float>(options_.aux_coefficient * total / static_cast<double>(N));
        Tensor* bias_grad = GradOf(decoder_, "bias");
        const std::vector<float> bias_before = bias_grad->to_host_vector();
        const std::vector<float> g_aux_codes = decoder_.backward(Tensor(Shape({N, out_}), backend_, g_aux, backend_->device())).to_host_vector();
        *bias_grad = Tensor(bias_grad->shape(), bias_grad->backend(), bias_before, bias_grad->device());
        for (size_t i = 0; i < aux_codes.size(); ++i) {
            if (aux_codes[i] > 0.0f) g_codes[i] += g_aux_codes[i];
        }
    }
    loss.total = loss.reconstruction + loss.sparsity;
    // Into the encoder: the main codes and the auxiliary ones both route through pre.
    std::vector<float> active(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) active[i] = codes[i] > 0.0f || aux_codes[i] > 0.0f ? 1.0f : 0.0f;
    (void)BackwardFromCodes(active, g_codes);
    // The decoder's gradient loses its component along each direction.
    Tensor* gw = GradOf(decoder_, "weight");
    std::vector<float> grad = gw->to_host_vector();
    const std::vector<float> w = decoder_.weight().to_host_vector();
    for (int64_t i = 0; i < m_; ++i) {
        double n2 = 0, dot = 0;
        for (int64_t j = 0; j < out_; ++j) {
            const size_t q = static_cast<size_t>(i * out_ + j);
            n2 += static_cast<double>(w[q]) * w[q];
            dot += static_cast<double>(w[q]) * grad[q];
        }
        if (n2 == 0) continue;
        for (int64_t j = 0; j < out_; ++j) {
            const size_t q = static_cast<size_t>(i * out_ + j);
            grad[q] = static_cast<float>(grad[q] - dot / n2 * w[q]);
        }
    }
    *gw = Tensor(gw->shape(), gw->backend(), grad, gw->device());
    for (int64_t i = 0; i < m_; ++i) {
        int64_t last = -1;
        for (int64_t r = 0; r < N; ++r) {
            if (codes[static_cast<size_t>(r * m_ + i)] > 0.0f) last = r;
        }
        since_fired_[static_cast<size_t>(i)] = last < 0 ? since_fired_[static_cast<size_t>(i)] + N : N - 1 - last;
    }
    if (codes_out != nullptr) *codes_out = codes;
    return loss;
}

void Transcoder::normalize_decoder() {
    std::vector<float> w = decoder_.weight().to_host_vector();
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < out_; ++j) sq += static_cast<double>(w[static_cast<size_t>(i * out_ + j)]) * w[static_cast<size_t>(i * out_ + j)];
        const double norm = std::sqrt(sq);
        if (norm == 0) continue;
        for (int64_t j = 0; j < out_; ++j) w[static_cast<size_t>(i * out_ + j)] = static_cast<float>(w[static_cast<size_t>(i * out_ + j)] / norm);
    }
    decoder_.set_weight(w);
}

std::vector<float> Transcoder::decoder_direction(int64_t i) {
    if (i < 0 || i >= m_) throw std::invalid_argument("Transcoder::decoder_direction: no such feature");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    return {w.begin() + static_cast<std::ptrdiff_t>(i * out_), w.begin() + static_cast<std::ptrdiff_t>((i + 1) * out_)};
}

Tensor Transcoder::forward_impl(const Tensor& input) {
    CheckInput(input, "Transcoder::forward");
    last_rows_ = input.shape().dim(0);
    last_codes_ = Codes(encoder_.forward(input));
    Tensor y = decoder_.forward(Tensor(Shape({last_rows_, m_}), backend_, last_codes_, backend_->device()));
    if (!skip_) return y;
    const Tensor s = skip_->forward(input);
    last_skip_out_ = s.to_host_vector();
    Tensor out(y.shape(), backend_, y.device());
    backend_->add(y.data(), s.data(), out.data(), static_cast<size_t>(y.numel()));
    return out;
}

Tensor Transcoder::backward(const Tensor& grad_output) {
    if (last_rows_ == 0) throw std::logic_error("Transcoder::backward: called before any forward()");
    const Tensor gx = BackwardFromCodes(last_codes_, decoder_.backward(grad_output).to_host_vector());
    if (!skip_) return gx;
    const Tensor gs = skip_->backward(grad_output);
    Tensor out(gx.shape(), backend_, gx.device());
    backend_->add(gx.data(), gs.data(), out.data(), static_cast<size_t>(gx.numel()));
    return out;
}

Tensor Transcoder::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_rows_ == 0) throw std::logic_error("Transcoder::propagate_relevance: called before any forward()");
    auto through_latents = [&](const Tensor& r) {
        std::vector<float> rc = decoder_.propagate_relevance(r, config).to_host_vector();
        for (size_t i = 0; i < rc.size(); ++i) {
            if (last_codes_[i] <= 0.0f) rc[i] = 0.0f;
        }
        return encoder_.propagate_relevance(Tensor(Shape({last_rows_, m_}), backend_, rc, backend_->device()), config);
    };
    if (!skip_) return through_latents(relevance_out);
    // Split each output's relevance by the two branches' contributions (epsilon-stabilized).
    const std::vector<float> r = relevance_out.to_host_vector();
    const std::vector<float> dec = decoder_.forward(Tensor(Shape({last_rows_, m_}), backend_, last_codes_, backend_->device())).to_host_vector();
    std::vector<float> r_dec(r.size()), r_skip(r.size());
    const double eps = config.epsilon;
    for (size_t i = 0; i < r.size(); ++i) {
        const double total = static_cast<double>(dec[i]) + last_skip_out_[i];
        const double denom = total + (total >= 0 ? eps : -eps);
        r_dec[i] = static_cast<float>(r[i] * dec[i] / denom);
        r_skip[i] = static_cast<float>(r[i] * last_skip_out_[i] / denom);
    }
    const Tensor a = through_latents(Tensor(relevance_out.shape(), backend_, r_dec, backend_->device()));
    const Tensor b = skip_->propagate_relevance(Tensor(relevance_out.shape(), backend_, r_skip, backend_->device()), config);
    Tensor out(a.shape(), backend_, a.device());
    backend_->add(a.data(), b.data(), out.data(), static_cast<size_t>(a.numel()));
    return out;
}

std::vector<NamedParamRef> Transcoder::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "encoder", encoder_);
    append_named_parameters(out, "decoder", decoder_);
    if (skip_) append_named_parameters(out, "skip", *skip_);
    return out;
}

void Transcoder::set_training(bool training) {
    Module::set_training(training);
    encoder_.set_training(training);
    decoder_.set_training(training);
    if (skip_) skip_->set_training(training);
}

void Transcoder::release_activations() {
    encoder_.release_activations();
    decoder_.release_activations();
    if (skip_) skip_->release_activations();
    last_codes_.clear();
    last_skip_out_.clear();
    last_rows_ = 0;
}

}  // namespace pulsatrix
