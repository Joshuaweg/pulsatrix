#include "pulsatrix/crosscoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

Tensor* GradOf(LinearModule& m, const char* name) {
    for (const NamedParamRef& p : m.named_parameters()) {
        if (p.name == name) return p.ref.grad;
    }
    throw std::logic_error("Crosscoder: no parameter named " + std::string(name));
}

void AddToGrad(LinearModule& m, const char* name, const std::vector<float>& add) {
    Tensor* g = GradOf(m, name);
    std::vector<float> v = g->to_host_vector();
    for (size_t i = 0; i < v.size(); ++i) v[i] += add[i];
    *g = Tensor(g->shape(), g->backend(), v, g->device());
}

}  // namespace

Crosscoder::Crosscoder(int64_t num_sources, int64_t source_dim, int64_t num_features, DeviceBackend* backend, CrosscoderOptions options)
    : S_(num_sources),
      d_(source_dim),
      m_(num_features),
      backend_(backend),
      options_(options),
      encoder_(num_sources > 0 && source_dim > 0 ? num_sources * source_dim : 1, num_features > 0 ? num_features : 1, backend),
      decoder_(num_features > 0 ? num_features : 1, num_sources > 0 && source_dim > 0 ? num_sources * source_dim : 1, backend),
      threshold_(Shape({options.delta ? 2 : 1}), backend, std::vector<float>(options.delta ? 2 : 1, -1.0f), backend->device()) {
    if (num_sources < 1 || source_dim < 1 || num_features < 1) throw std::invalid_argument("Crosscoder: sizes must be positive");
    if (options.l1_coefficient < 0 || options.aux_coefficient < 0 || options.delta_coefficient < 0 || options.k_aux < 0 ||
        options.decoder_init_norm < 0 || options.k_shared < 0) {
        throw std::invalid_argument("Crosscoder: coefficients, k_aux, k_shared and decoder_init_norm must not be negative");
    }
    if (options.dead_after < 1) throw std::invalid_argument("Crosscoder: dead_after must be positive");
    if (options.threshold_decay < 0.0f || options.threshold_decay >= 1.0f) throw std::invalid_argument("Crosscoder: threshold_decay must be in [0, 1)");
    if (options.delta) {
        if (num_sources != 2 || options.sparsity != CrosscoderSparsity::BatchTopK) {
            throw std::invalid_argument("Crosscoder: the Delta-Crosscoder needs two sources and BatchTopK");
        }
        shared_ = static_cast<int64_t>(std::llround(options.shared_fraction * static_cast<double>(num_features)));
        if (shared_ < 1 || shared_ >= num_features) throw std::invalid_argument("Crosscoder: both the shared and the delta partition need latents");
        if (options_.k_shared == 0) options_.k_shared = 2 * options.k;
        if (options.k < 1 || options.k > num_features - shared_ || options_.k_shared > shared_) {
            throw std::invalid_argument("Crosscoder: k must be in [1, delta latents] and k_shared at most the shared latents");
        }
    } else if (options.sparsity == CrosscoderSparsity::BatchTopK && (options.k < 1 || options.k > num_features)) {
        throw std::invalid_argument("Crosscoder: k must be in [1, num_features]");
    }
    if (options_.k_aux == 0) options_.k_aux = std::max<int64_t>(1, S_ * d_ / 2);
    if (options_.decoder_init_norm == 0.0f) options_.decoder_init_norm = options.sparsity == CrosscoderSparsity::L1 ? 0.05f : 1.0f;
    // The same random direction for a latent in every source, at decoder_init_norm.
    PortableRng rng{options.seed};
    const int64_t D = S_ * d_;
    std::vector<float> dec(static_cast<size_t>(m_ * D)), enc(static_cast<size_t>(D * m_));
    std::vector<double> v(static_cast<size_t>(d_));
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (double& e : v) {
            e = rng.gaussian();
            sq += e * e;
        }
        const double scale = options_.decoder_init_norm / std::sqrt(sq);
        for (int64_t s = 0; s < S_; ++s) {
            for (int64_t j = 0; j < d_; ++j) {
                const auto w = static_cast<float>(v[static_cast<size_t>(j)] * scale);
                dec[static_cast<size_t>(i * D + s * d_ + j)] = w;
                enc[static_cast<size_t>((s * d_ + j) * m_ + i)] = w;
            }
        }
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    decoder_.set_bias(std::vector<float>(static_cast<size_t>(D), 0.0f));
    encoder_.set_bias(std::vector<float>(static_cast<size_t>(m_), 0.0f));
    since_fired_.assign(static_cast<size_t>(m_), 0);
}

void Crosscoder::Check(const Tensor& x, const char* who) const {
    if (x.rank() != 2 || x.shape().dim(1) != S_ * d_ || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the input must be (N, num_sources * source_dim) with N > 0");
    }
}

void Crosscoder::initialize_bias(const Tensor& x) {
    Check(x, "Crosscoder::initialize_bias");
    const std::vector<float> v = x.to_host_vector();
    const int64_t N = x.shape().dim(0), D = S_ * d_;
    std::vector<double> mean(static_cast<size_t>(D), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < D; ++j) mean[static_cast<size_t>(j)] += v[static_cast<size_t>(r * D + j)];
    }
    std::vector<float> b(static_cast<size_t>(D));
    for (int64_t j = 0; j < D; ++j) b[static_cast<size_t>(j)] = static_cast<float>(mean[static_cast<size_t>(j)] / static_cast<double>(N));
    decoder_.set_bias(b);
}

Tensor Crosscoder::EncoderInput(const Tensor& x) const {
    if (!options_.delta) return x;
    std::vector<float> h = x.to_host_vector();
    for (float& e : h) e *= 0.5f;
    return Tensor(x.shape(), backend_, h, backend_->device());
}

std::vector<double> Crosscoder::NormSums() const {
    const std::vector<float> w = decoder_.weight().to_host_vector();
    const int64_t D = S_ * d_;
    std::vector<double> out(static_cast<size_t>(m_), 0.0);
    for (int64_t i = 0; i < m_; ++i) {
        for (int64_t s = 0; s < S_; ++s) {
            double sq = 0;
            for (int64_t j = 0; j < d_; ++j) sq += static_cast<double>(w[static_cast<size_t>(i * D + s * d_ + j)]) * w[static_cast<size_t>(i * D + s * d_ + j)];
            out[static_cast<size_t>(i)] += std::sqrt(sq);
        }
    }
    return out;
}

std::vector<float> Crosscoder::thresholds() const { return threshold_.to_host_vector(); }

void Crosscoder::set_thresholds(const std::vector<float>& thresholds) {
    if (static_cast<int64_t>(thresholds.size()) != threshold_.numel()) {
        throw std::invalid_argument("Crosscoder::set_thresholds: needs one threshold, two with delta");
    }
    threshold_ = Tensor(threshold_.shape(), backend_, thresholds, backend_->device());
}

Crosscoder::Pass Crosscoder::Encode(const Tensor& x, bool training) {
    const int64_t N = x.shape().dim(0);
    Pass p;
    p.pre = encoder_.forward(EncoderInput(x)).to_host_vector();
    p.codes.assign(p.pre.size(), 0.0f);
    if (options_.sparsity == CrosscoderSparsity::L1) {
        for (size_t i = 0; i < p.pre.size(); ++i) p.codes[i] = std::max(0.0f, p.pre[i]);
        return p;
    }
    // BatchTopK on f_i · Σ_s |d_i^s|, per partition.
    const std::vector<double> w = NormSums();
    std::vector<double> v(p.pre.size());
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t i = 0; i < m_; ++i) {
            const size_t q = static_cast<size_t>(r * m_ + i);
            v[q] = std::max(0.0f, p.pre[q]) * w[static_cast<size_t>(i)];
        }
    }
    const std::vector<float> theta = thresholds();
    struct Part {
        int64_t begin, end, k;
    };
    std::vector<Part> parts;
    if (options_.delta) {
        parts = {{0, shared_, options_.k_shared}, {shared_, m_, options_.k}};
    } else {
        parts = {{0, m_, options_.k}};
    }
    p.min_kept.assign(parts.size(), -1.0f);
    for (size_t pi = 0; pi < parts.size(); ++pi) {
        const Part& part = parts[pi];
        if (!training && theta[pi] >= 0.0f) {
            for (int64_t r = 0; r < N; ++r) {
                for (int64_t i = part.begin; i < part.end; ++i) {
                    const size_t q = static_cast<size_t>(r * m_ + i);
                    if (v[q] > theta[pi]) p.codes[q] = p.pre[q];
                }
            }
            continue;
        }
        // The batch's N · k largest scaled values in the partition, ties to the lower index.
        std::vector<int64_t> order;
        order.reserve(static_cast<size_t>(N * (part.end - part.begin)));
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t i = part.begin; i < part.end; ++i) order.push_back(r * m_ + i);
        }
        const auto keep = static_cast<std::ptrdiff_t>(N * part.k);
        std::nth_element(order.begin(), order.begin() + keep - 1, order.end(), [&](int64_t a, int64_t b) {
            const double va = v[static_cast<size_t>(a)], vb = v[static_cast<size_t>(b)];
            return va > vb || (va == vb && a < b);
        });
        for (std::ptrdiff_t q = 0; q < keep; ++q) {
            const auto i = static_cast<size_t>(order[static_cast<size_t>(q)]);
            if (v[i] <= 0.0) continue;
            p.codes[i] = p.pre[i];
            const auto vi = static_cast<float>(v[i]);
            p.min_kept[pi] = p.min_kept[pi] < 0.0f ? vi : std::min(p.min_kept[pi], vi);
        }
    }
    return p;
}

Tensor Crosscoder::encode(const Tensor& x) {
    Check(x, "Crosscoder::encode");
    return Tensor(Shape({x.shape().dim(0), m_}), backend_, Encode(x, false).codes, backend_->device());
}

Tensor Crosscoder::decode(const Tensor& codes) {
    if (codes.rank() != 2 || codes.shape().dim(1) != m_ || codes.shape().dim(0) < 1) {
        throw std::invalid_argument("Crosscoder::decode: codes must be (N, num_features) with N > 0");
    }
    return decoder_.forward(codes);
}

std::vector<int64_t> Crosscoder::dead_latents() const {
    std::vector<int64_t> out;
    for (int64_t i = 0; i < m_; ++i) {
        if (since_fired_[static_cast<size_t>(i)] >= options_.dead_after) out.push_back(i);
    }
    return out;
}

void Crosscoder::set_inputs_since_fired(const std::vector<int64_t>& since) {
    if (static_cast<int64_t>(since.size()) != m_ || std::any_of(since.begin(), since.end(), [](int64_t v) { return v < 0; })) {
        throw std::invalid_argument("Crosscoder::set_inputs_since_fired: needs num_features non-negative values");
    }
    since_fired_ = since;
}

FeaturizerLoss Crosscoder::loss_and_backward(const Tensor& x, std::vector<float>* codes_out) {
    Check(x, "Crosscoder::loss_and_backward");
    const int64_t N = x.shape().dim(0), D = S_ * d_;
    const std::vector<int64_t> dead = dead_latents();
    const std::vector<float> w_dec = decoder_.weight().to_host_vector();
    const std::vector<double> norm_sums = NormSums();
    const Pass p = Encode(x, true);
    const std::vector<float>& codes = p.codes;
    const std::vector<float> xh = decoder_.forward(Tensor(Shape({N, m_}), backend_, codes, backend_->device())).to_host_vector();
    const std::vector<float> xv = x.to_host_vector();
    const double n_el = static_cast<double>(N * D);
    std::vector<float> g(xh.size()), residual(xh.size());
    double mse = 0;
    for (size_t i = 0; i < xh.size(); ++i) {
        const double e = static_cast<double>(xh[i]) - xv[i];
        mse += e * e;
        g[i] = static_cast<float>(2.0 * e / n_el);
        residual[i] = static_cast<float>(-e);
    }
    FeaturizerLoss loss;
    loss.reconstruction = static_cast<float>(mse / n_el);
    std::vector<float> g_codes = decoder_.backward(Tensor(Shape({N, D}), backend_, g, backend_->device())).to_host_vector();
    for (size_t i = 0; i < g_codes.size(); ++i) {
        if (codes[i] <= 0.0f) g_codes[i] = 0.0f;  // only selected latents pass the main gradient
    }
    std::vector<float> g_w_dec(w_dec.size(), 0.0f);  // added to the decoder's weight gradient at the end

    std::vector<float> aux_codes(codes.size(), 0.0f);
    if (options_.sparsity == CrosscoderSparsity::L1) {
        // λ/N Σ_r Σ_i f_ri Σ_s |d_i^s|: into f, λ/N Σ_s |d_i^s|; into d_i^s, λ/N (Σ_r f_ri) d_i^s / |d_i^s|.
        const double lam = options_.l1_coefficient / static_cast<double>(N);
        double penalty = 0;
        std::vector<double> f_sum(static_cast<size_t>(m_), 0.0);
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t i = 0; i < m_; ++i) {
                const size_t q = static_cast<size_t>(r * m_ + i);
                if (codes[q] <= 0.0f) continue;
                f_sum[static_cast<size_t>(i)] += codes[q];
                g_codes[q] += static_cast<float>(lam * norm_sums[static_cast<size_t>(i)]);
            }
        }
        for (int64_t i = 0; i < m_; ++i) {
            penalty += f_sum[static_cast<size_t>(i)] * norm_sums[static_cast<size_t>(i)];
            if (f_sum[static_cast<size_t>(i)] == 0) continue;
            for (int64_t s = 0; s < S_; ++s) {
                double sq = 0;
                for (int64_t j = 0; j < d_; ++j) sq += static_cast<double>(w_dec[static_cast<size_t>(i * D + s * d_ + j)]) * w_dec[static_cast<size_t>(i * D + s * d_ + j)];
                if (sq == 0) continue;
                const double c = lam * f_sum[static_cast<size_t>(i)] / std::sqrt(sq);
                for (int64_t j = 0; j < d_; ++j) {
                    const size_t q = static_cast<size_t>(i * D + s * d_ + j);
                    g_w_dec[q] += static_cast<float>(c * w_dec[q]);
                }
            }
        }
        loss.sparsity = static_cast<float>(lam * penalty);
    } else if (options_.aux_coefficient > 0.0f && !dead.empty()) {
        // AuxK: the k_aux dead latents of largest scaled value reconstruct the error, without the bias.
        const auto kaux = static_cast<size_t>(std::min<int64_t>(options_.k_aux, static_cast<int64_t>(dead.size())));
        std::vector<int64_t> order(dead);
        for (int64_t r = 0; r < N; ++r) {
            const float* pr = p.pre.data() + r * m_;
            auto scaled = [&](int64_t i) { return std::max(0.0f, pr[i]) * norm_sums[static_cast<size_t>(i)]; };
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(kaux), order.end(), [&](int64_t a, int64_t b) {
                const double va = scaled(a), vb = scaled(b);
                return va > vb || (va == vb && a < b);
            });
            for (size_t j = 0; j < kaux; ++j) aux_codes[static_cast<size_t>(r * m_ + order[j])] = std::max(0.0f, pr[order[j]]);
        }
        std::vector<float> e_hat = decoder_.forward(Tensor(Shape({N, m_}), backend_, aux_codes, backend_->device())).to_host_vector();
        const std::vector<float> b = decoder_.bias().to_host_vector();
        std::vector<float> g_aux(e_hat.size(), 0.0f);
        double total = 0;
        for (int64_t r = 0; r < N; ++r) {
            double err = 0, energy = 0;
            for (int64_t j = 0; j < D; ++j) {
                const size_t i = static_cast<size_t>(r * D + j);
                e_hat[i] -= b[static_cast<size_t>(j)];
                err += (static_cast<double>(e_hat[i]) - residual[i]) * (e_hat[i] - residual[i]);
                energy += static_cast<double>(residual[i]) * residual[i];
            }
            if (energy == 0) continue;
            total += err / energy;
            for (int64_t j = 0; j < D; ++j) {
                const size_t i = static_cast<size_t>(r * D + j);
                g_aux[i] = static_cast<float>(options_.aux_coefficient * 2.0 * (e_hat[i] - residual[i]) / energy / static_cast<double>(N));
            }
        }
        loss.sparsity = static_cast<float>(options_.aux_coefficient * total / static_cast<double>(N));
        Tensor* bias_grad = GradOf(decoder_, "bias");
        const std::vector<float> bias_before = bias_grad->to_host_vector();
        const std::vector<float> g_aux_codes = decoder_.backward(Tensor(Shape({N, D}), backend_, g_aux, backend_->device())).to_host_vector();
        *bias_grad = Tensor(bias_grad->shape(), bias_grad->backend(), bias_before, bias_grad->device());
        for (size_t i = 0; i < aux_codes.size(); ++i) {
            if (aux_codes[i] > 0.0f) g_codes[i] += g_aux_codes[i];
        }
    }

    last_delta_loss_ = 0;
    if (options_.delta) {
        // The delta latents alone predict x^1 - x^0 through W_dec^1 - W_dec^0.
        const double lam = options_.delta_coefficient, n_d = static_cast<double>(N * d_);
        double sq = 0;
        for (int64_t r = 0; r < N; ++r) {
            std::vector<double> diff(static_cast<size_t>(d_));
            for (int64_t j = 0; j < d_; ++j) diff[static_cast<size_t>(j)] = -(static_cast<double>(xv[static_cast<size_t>(r * D + d_ + j)]) - xv[static_cast<size_t>(r * D + j)]);
            for (int64_t i = shared_; i < m_; ++i) {
                const float z = codes[static_cast<size_t>(r * m_ + i)];
                if (z <= 0.0f) continue;
                for (int64_t j = 0; j < d_; ++j) {
                    diff[static_cast<size_t>(j)] += z * (static_cast<double>(w_dec[static_cast<size_t>(i * D + d_ + j)]) - w_dec[static_cast<size_t>(i * D + j)]);
                }
            }
            for (int64_t j = 0; j < d_; ++j) sq += diff[static_cast<size_t>(j)] * diff[static_cast<size_t>(j)];
            for (int64_t i = shared_; i < m_; ++i) {
                const size_t q = static_cast<size_t>(r * m_ + i);
                const float z = codes[q];
                if (z <= 0.0f) continue;
                double gz = 0;
                for (int64_t j = 0; j < d_; ++j) {
                    const double gd = 2.0 * lam * diff[static_cast<size_t>(j)] / n_d;
                    gz += gd * (static_cast<double>(w_dec[static_cast<size_t>(i * D + d_ + j)]) - w_dec[static_cast<size_t>(i * D + j)]);
                    g_w_dec[static_cast<size_t>(i * D + d_ + j)] += static_cast<float>(gd * z);
                    g_w_dec[static_cast<size_t>(i * D + j)] -= static_cast<float>(gd * z);
                }
                g_codes[q] += static_cast<float>(gz);
            }
        }
        last_delta_loss_ = static_cast<float>(lam * sq / n_d);
    }
    loss.total = loss.reconstruction + loss.sparsity + last_delta_loss_;
    AddToGrad(decoder_, "weight", g_w_dec);

    // Into the encoder, through every latent that passed a value (main or auxiliary).
    std::vector<float> g_pre(codes.size(), 0.0f);
    for (size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] > 0.0f || aux_codes[i] > 0.0f) g_pre[i] = g_codes[i];
    }
    (void)encoder_.backward(Tensor(Shape({N, m_}), backend_, g_pre, backend_->device()));

    if (options_.sparsity == CrosscoderSparsity::BatchTopK) {
        std::vector<float> theta = thresholds();
        for (size_t pi = 0; pi < theta.size(); ++pi) {
            if (p.min_kept[pi] <= 0.0f) continue;
            theta[pi] = theta[pi] < 0.0f ? p.min_kept[pi] : options_.threshold_decay * theta[pi] + (1.0f - options_.threshold_decay) * p.min_kept[pi];
        }
        set_thresholds(theta);
    }
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

void Crosscoder::normalize_decoder() {
    std::vector<float> dec = decoder_.weight().to_host_vector(), enc = encoder_.weight().to_host_vector(), b_enc = encoder_.bias().to_host_vector();
    const int64_t D = S_ * d_;
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < D; ++j) sq += static_cast<double>(dec[static_cast<size_t>(i * D + j)]) * dec[static_cast<size_t>(i * D + j)];
        const double n = std::sqrt(sq);
        if (n == 0) continue;
        for (int64_t j = 0; j < D; ++j) dec[static_cast<size_t>(i * D + j)] = static_cast<float>(dec[static_cast<size_t>(i * D + j)] / n);
        for (int64_t j = 0; j < D; ++j) enc[static_cast<size_t>(j * m_ + i)] = static_cast<float>(enc[static_cast<size_t>(j * m_ + i)] * n);
        b_enc[static_cast<size_t>(i)] = static_cast<float>(b_enc[static_cast<size_t>(i)] * n);
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    encoder_.set_bias(b_enc);
}

std::vector<float> Crosscoder::decoder_direction(int64_t i) {
    if (i < 0 || i >= m_) throw std::invalid_argument("Crosscoder::decoder_direction: no such feature");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    const int64_t D = S_ * d_;
    return {w.begin() + static_cast<std::ptrdiff_t>(i * D), w.begin() + static_cast<std::ptrdiff_t>((i + 1) * D)};
}

std::vector<float> Crosscoder::decoder_direction(int64_t i, int64_t s) {
    if (s < 0 || s >= S_) throw std::invalid_argument("Crosscoder::decoder_direction: no such source");
    const std::vector<float> all = decoder_direction(i);
    return {all.begin() + static_cast<std::ptrdiff_t>(s * d_), all.begin() + static_cast<std::ptrdiff_t>((s + 1) * d_)};
}

Tensor Crosscoder::forward_impl(const Tensor& input) {
    Check(input, "Crosscoder::forward");
    last_rows_ = input.shape().dim(0);
    last_codes_ = Encode(input, false).codes;
    return decoder_.forward(Tensor(Shape({last_rows_, m_}), backend_, last_codes_, backend_->device()));
}

Tensor Crosscoder::backward(const Tensor& grad_output) {
    if (last_rows_ == 0) throw std::logic_error("Crosscoder::backward: called before any forward()");
    std::vector<float> g = decoder_.backward(grad_output).to_host_vector();
    for (size_t i = 0; i < g.size(); ++i) {
        if (last_codes_[i] <= 0.0f) g[i] = 0.0f;
    }
    Tensor gx = encoder_.backward(Tensor(Shape({last_rows_, m_}), backend_, g, backend_->device()));
    return options_.delta ? EncoderInput(gx) : gx;  // the encoder read x / 2
}

Tensor Crosscoder::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_rows_ == 0) throw std::logic_error("Crosscoder::propagate_relevance: called before any forward()");
    std::vector<float> rc = decoder_.propagate_relevance(relevance_out, config).to_host_vector();
    for (size_t i = 0; i < rc.size(); ++i) {
        if (last_codes_[i] <= 0.0f) rc[i] = 0.0f;
    }
    // Relevance is scale-free in the input, so the encoder's x / 2 passes it to x unchanged.
    return encoder_.propagate_relevance(Tensor(Shape({last_rows_, m_}), backend_, rc, backend_->device()), config);
}

std::vector<NamedParamRef> Crosscoder::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "encoder", encoder_);
    append_named_parameters(out, "decoder", decoder_);
    return out;
}

std::vector<NamedBufferRef> Crosscoder::named_buffers() { return {{"threshold", &threshold_}}; }

void Crosscoder::set_training(bool training) {
    Module::set_training(training);
    encoder_.set_training(training);
    decoder_.set_training(training);
}

void Crosscoder::release_activations() {
    encoder_.release_activations();
    decoder_.release_activations();
    last_codes_.clear();
    last_rows_ = 0;
}

// ---- Model diffing -------------------------------------------------------------------------

namespace {

void CheckSources(const Crosscoder& c, int64_t a, int64_t b, const char* who) {
    if (a < 0 || b < 0 || a >= c.num_sources() || b >= c.num_sources() || a == b) {
        throw std::invalid_argument(std::string(who) + ": needs two different sources in range");
    }
}

}  // namespace

std::vector<CrosscoderLatentStats> CrosscoderLatents(Crosscoder& c, int64_t a, int64_t b) {
    CheckSources(c, a, b, "CrosscoderLatents");
    const std::vector<float> w = c.decoder().weight().to_host_vector();
    const int64_t m = c.num_features(), d = c.source_dim(), D = c.input_dim();
    std::vector<CrosscoderLatentStats> out(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        double aa = 0, bb = 0, ab = 0;
        for (int64_t j = 0; j < d; ++j) {
            const double x = w[static_cast<size_t>(i * D + a * d + j)], y = w[static_cast<size_t>(i * D + b * d + j)];
            aa += x * x;
            bb += y * y;
            ab += x * y;
        }
        CrosscoderLatentStats& s = out[static_cast<size_t>(i)];
        s.norm_a = std::sqrt(aa);
        s.norm_b = std::sqrt(bb);
        const double sum = s.norm_a + s.norm_b, mx = std::max(s.norm_a, s.norm_b);
        s.relative_norm = sum > 0 ? s.norm_b / sum : 0.5;
        s.delta_norm = mx > 0 ? 0.5 * (1.0 + (s.norm_b - s.norm_a) / mx) : 0.5;
        s.cosine = s.norm_a > 0 && s.norm_b > 0 ? ab / (s.norm_a * s.norm_b) : 0.0;
    }
    return out;
}

LatentClass ClassifyLatent(const CrosscoderLatentStats& s) {
    if (s.delta_norm < 0.1) return LatentClass::AOnly;
    if (s.delta_norm > 0.9) return LatentClass::BOnly;
    if (s.delta_norm >= 0.4 && s.delta_norm <= 0.6) return LatentClass::Shared;
    return LatentClass::Other;
}

std::vector<LatentScaling> MeasureLatentScaling(Crosscoder& c, const Tensor& x, const std::vector<int64_t>& latents, int64_t a, int64_t b,
                                                int64_t batch) {
    CheckSources(c, a, b, "MeasureLatentScaling");
    const int64_t m = c.num_features(), d = c.source_dim(), D = c.input_dim();
    if (x.rank() != 2 || x.shape().dim(1) != D || x.shape().dim(0) < 1) throw std::invalid_argument("MeasureLatentScaling: x must be (N, S·d)");
    if (batch < 1) throw std::invalid_argument("MeasureLatentScaling: batch must be positive");
    for (int64_t j : latents) {
        if (j < 0 || j >= m) throw std::invalid_argument("MeasureLatentScaling: no such latent");
    }
    const std::vector<float> w = c.decoder().weight().to_host_vector();
    // Per latent: Σ f ⟨Y, d⟩ for the four targets, and Σ f².
    struct Acc {
        double err_a = 0, err_b = 0, rec_a = 0, rec_b = 0, f2 = 0;
        int64_t active = 0;
    };
    std::vector<Acc> acc(latents.size());
    const std::vector<float> xv = x.to_host_vector();
    const int64_t N = x.shape().dim(0);
    for (int64_t r0 = 0; r0 < N; r0 += batch) {
        const int64_t n = std::min(batch, N - r0);
        const Tensor xb(Shape({n, D}), x.backend(), std::vector<float>(xv.begin() + r0 * D, xv.begin() + (r0 + n) * D), x.device());
        const Tensor codes_t = c.encode(xb);
        const std::vector<float> codes = codes_t.to_host_vector(), xh = c.decode(codes_t).to_host_vector();
        for (size_t li = 0; li < latents.size(); ++li) {
            const int64_t j = latents[li];
            const float* db = w.data() + j * D + b * d;  // the b-side direction, used for both
            const float* da = w.data() + j * D + a * d;
            Acc& s = acc[li];
            for (int64_t r = 0; r < n; ++r) {
                const double f = codes[static_cast<size_t>(r * m + j)];
                if (f == 0) continue;
                ++s.active;
                s.f2 += f * f;
                double ea = 0, eb = 0, ra = 0, rb = 0;
                for (int64_t k = 0; k < d; ++k) {
                    const size_t qa = static_cast<size_t>(r * D + a * d + k), qb = static_cast<size_t>(r * D + b * d + k);
                    // x^m - x̂^m + f d_j^m: what the other latents leave.
                    const double ya = static_cast<double>(xv[static_cast<size_t>((r0 + r) * D + a * d + k)]) - xh[qa] + f * da[k];
                    const double yb = static_cast<double>(xv[static_cast<size_t>((r0 + r) * D + b * d + k)]) - xh[qb] + f * db[k];
                    ea += ya * db[k];
                    eb += yb * db[k];
                    ra += static_cast<double>(xh[qa]) * db[k];
                    rb += static_cast<double>(xh[qb]) * db[k];
                }
                s.err_a += f * ea;
                s.err_b += f * eb;
                s.rec_a += f * ra;
                s.rec_b += f * rb;
            }
        }
    }
    std::vector<LatentScaling> out(latents.size());
    for (size_t li = 0; li < latents.size(); ++li) {
        const int64_t j = latents[li];
        double dd = 0;
        for (int64_t k = 0; k < d; ++k) dd += static_cast<double>(w[static_cast<size_t>(j * D + b * d + k)]) * w[static_cast<size_t>(j * D + b * d + k)];
        const Acc& s = acc[li];
        LatentScaling& o = out[li];
        o.latent = j;
        o.active = s.active;
        const double den = s.f2 * dd;
        if (den == 0) continue;
        o.beta_error_a = s.err_a / den;
        o.beta_error_b = s.err_b / den;
        o.beta_reconstruction_a = s.rec_a / den;
        o.beta_reconstruction_b = s.rec_b / den;
        o.nu_error = o.beta_error_b != 0 ? o.beta_error_a / o.beta_error_b : 0.0;
        o.nu_reconstruction = o.beta_reconstruction_b != 0 ? o.beta_reconstruction_a / o.beta_reconstruction_b : 0.0;
    }
    return out;
}

std::vector<double> ExplainedVarianceBySource(Crosscoder& c, const Tensor& x, int64_t batch) {
    const int64_t S = c.num_sources(), d = c.source_dim(), D = c.input_dim();
    if (x.rank() != 2 || x.shape().dim(1) != D || x.shape().dim(0) < 1) throw std::invalid_argument("ExplainedVarianceBySource: x must be (N, S·d)");
    if (batch < 1) throw std::invalid_argument("ExplainedVarianceBySource: batch must be positive");
    const std::vector<float> xv = x.to_host_vector();
    const int64_t N = x.shape().dim(0);
    std::vector<double> sum(static_cast<size_t>(D), 0.0), sum2(static_cast<size_t>(D), 0.0), err(static_cast<size_t>(S), 0.0);
    for (int64_t r0 = 0; r0 < N; r0 += batch) {
        const int64_t n = std::min(batch, N - r0);
        const Tensor xb(Shape({n, D}), x.backend(), std::vector<float>(xv.begin() + r0 * D, xv.begin() + (r0 + n) * D), x.device());
        const std::vector<float> xh = c.decode(c.encode(xb)).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < D; ++j) {
                const double v = xv[static_cast<size_t>((r0 + r) * D + j)], e = v - xh[static_cast<size_t>(r * D + j)];
                sum[static_cast<size_t>(j)] += v;
                sum2[static_cast<size_t>(j)] += v * v;
                err[static_cast<size_t>(j / d)] += e * e;
            }
        }
    }
    std::vector<double> out(static_cast<size_t>(S));
    for (int64_t s = 0; s < S; ++s) {
        double var = 0;
        for (int64_t j = s * d; j < (s + 1) * d; ++j) var += sum2[static_cast<size_t>(j)] - sum[static_cast<size_t>(j)] * sum[static_cast<size_t>(j)] / static_cast<double>(N);
        out[static_cast<size_t>(s)] = var > 0 ? 1.0 - err[static_cast<size_t>(s)] / var : 0.0;
    }
    return out;
}

}  // namespace pulsatrix
