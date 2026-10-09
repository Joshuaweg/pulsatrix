#include "pulsatrix/block_sparse_featurizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/matrix_decompositions.hpp"

namespace pulsatrix {

namespace {

Tensor Scalar(float v, DeviceBackend* backend) { return Tensor(Shape({1}), backend, std::vector<float>{v}, backend->device()); }
Tensor Filled(int64_t n, float v, DeviceBackend* backend) {
    return Tensor(Shape({n}), backend, std::vector<float>(static_cast<size_t>(n), v), backend->device());
}

Tensor* GradOf(LinearModule& m, const char* name) {
    for (const NamedParamRef& p : m.named_parameters()) {
        if (p.name == name) return p.ref.grad;
    }
    throw std::logic_error("BlockSparseFeaturizer: no parameter named " + std::string(name));
}

double Softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/** @brief The reference code's Gaussian kernel: `exp(-u²/2) / (bandwidth √(2π))`, u = x / (bandwidth/2). */
double GaussianKernel(double x, double bandwidth) {
    const double h = bandwidth / 2, u = x / h;
    return std::exp(-0.5 * u * u) / (bandwidth * std::sqrt(2.0 * 3.14159265358979323846));
}

/** @brief Gram–Schmidt on the b rows of a `b x d` block, in place (QR of its transpose with a
 *         positive diagonal). */
void Orthonormalize(float* rows, int64_t b, int64_t d) {
    for (int64_t i = 0; i < b; ++i) {
        float* r = rows + i * d;
        for (int64_t j = 0; j < i; ++j) {
            const float* q = rows + j * d;
            double dot = 0;
            for (int64_t t = 0; t < d; ++t) dot += static_cast<double>(r[t]) * q[t];
            for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] - dot * q[t]);
        }
        double n2 = 0;
        for (int64_t t = 0; t < d; ++t) n2 += static_cast<double>(r[t]) * r[t];
        const double n = std::sqrt(n2);
        if (n == 0) continue;
        for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] / n);
    }
}

/** @brief torch.quantile's linear interpolation. */
double Quantile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(pos)), hi = static_cast<size_t>(std::ceil(pos));
    return v[lo] + (pos - static_cast<double>(lo)) * (static_cast<double>(v[hi]) - v[lo]);
}

/** @brief The unbiased standard deviation, as torch.std. */
double StdDev(const std::vector<float>& v) {
    double mean = 0;
    for (float x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double s = 0;
    for (float x : v) s += (x - mean) * (x - mean);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

}  // namespace

BlockSparseFeaturizer::BlockSparseFeaturizer(int64_t dim, int64_t num_blocks, DeviceBackend* backend, BlockSparseOptions options)
    : d_(dim),
      G_(num_blocks),
      b_(options.block_size),
      backend_(backend),
      options_(options),
      encoder_(dim > 0 ? dim : 1, std::max<int64_t>(1, num_blocks * options.block_size), backend, options.variant != BsfVariant::Grassmannian),
      decoder_(std::max<int64_t>(1, num_blocks * options.block_size), dim > 0 ? dim : 1, backend, /*use_bias=*/false),
      gamma_(Filled(std::max<int64_t>(1, num_blocks), 1.0f, backend)),
      gamma_grad_(Filled(std::max<int64_t>(1, num_blocks), 0.0f, backend)),
      threshold_raw_(Filled(std::max<int64_t>(1, num_blocks), 0.0f, backend)),
      threshold_raw_grad_(Filled(std::max<int64_t>(1, num_blocks), 0.0f, backend)),
      bandwidth_(Scalar(1.0f, backend)),
      log_lambda_(Scalar(std::log(options.initial_lambda > 0 ? options.initial_lambda : 1.0f), backend)),
      initialized_(Scalar(0.0f, backend)) {
    if (dim < 1 || num_blocks < 1 || options.block_size < 1) throw std::invalid_argument("BlockSparseFeaturizer: sizes must be positive");
    if (options.variant != BsfVariant::GroupLasso && (options.k < 1 || options.k > num_blocks)) {
        throw std::invalid_argument("BlockSparseFeaturizer: k must be in [1, num_blocks]");
    }
    if (options.variant == BsfVariant::GroupLasso &&
        (options.target_l0 <= 0 || options.target_l0 > static_cast<double>(num_blocks) || options.gain <= 0 || options.bandwidth <= 0 ||
         options.initial_lambda <= 0)) {
        throw std::invalid_argument("BlockSparseFeaturizer: target_l0 must be in (0, num_blocks]; gain, bandwidth and initial_lambda positive");
    }
    if (options.duel_overlap <= 0 || options.duel_overlap > 1 || options.verdict_overlap <= 0 || options.verdict_overlap > 1) {
        throw std::invalid_argument("BlockSparseFeaturizer: overlaps must be in (0, 1]");
    }
    PortableRng rng{options.seed};
    std::vector<float> atoms(static_cast<size_t>(G_ * b_ * d_));
    for (float& v : atoms) v = static_cast<float>(rng.gaussian());
    if (options.variant == BsfVariant::Grassmannian) {
        for (int64_t g = 0; g < G_; ++g) Orthonormalize(atoms.data() + g * b_ * d_, b_, d_);
    } else {
        for (int64_t i = 0; i < G_ * b_; ++i) {
            double n2 = 0;
            for (int64_t t = 0; t < d_; ++t) n2 += static_cast<double>(atoms[static_cast<size_t>(i * d_ + t)]) * atoms[static_cast<size_t>(i * d_ + t)];
            for (int64_t t = 0; t < d_; ++t) atoms[static_cast<size_t>(i * d_ + t)] = static_cast<float>(atoms[static_cast<size_t>(i * d_ + t)] / std::sqrt(n2));
        }
    }
    decoder_.set_weight(atoms);
    SyncTiedEncoder();  // every variant starts tied: W_enc = Dᵀ
    if (options.variant != BsfVariant::Grassmannian) encoder_.set_bias(std::vector<float>(static_cast<size_t>(G_ * b_), 0.0f));
    wins_.assign(static_cast<size_t>(G_), 0);
}

void BlockSparseFeaturizer::Check(const Tensor& x, const char* who) const {
    if (x.rank() != 2 || x.shape().dim(1) != d_ || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the batch must be (N, dim) with N > 0");
    }
}

void BlockSparseFeaturizer::SyncTiedEncoder() {
    const std::vector<float> w = decoder_.weight().to_host_vector();  // (G·b, d)
    const int64_t m = G_ * b_;
    std::vector<float> t(w.size());  // (d, G·b)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < d_; ++j) t[static_cast<size_t>(j * m + i)] = w[static_cast<size_t>(i * d_ + j)];
    }
    encoder_.set_weight(t);
}

std::vector<float> BlockSparseFeaturizer::block_frame(int64_t g) {
    if (g < 0 || g >= G_) throw std::invalid_argument("BlockSparseFeaturizer::block_frame: no such block");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    return {w.begin() + g * b_ * d_, w.begin() + (g + 1) * b_ * d_};
}

std::vector<float> BlockSparseFeaturizer::gamma() const {
    return options_.variant == BsfVariant::Grassmannian ? gamma_.to_host_vector() : std::vector<float>{};
}

std::vector<float> BlockSparseFeaturizer::thresholds() const {
    if (options_.variant != BsfVariant::GroupLasso) return {};
    std::vector<float> t = threshold_raw_.to_host_vector();
    for (float& v : t) v = static_cast<float>(Softplus(options_.gain * static_cast<double>(v)));
    return t;
}

float BlockSparseFeaturizer::l0_multiplier() const { return std::exp(log_lambda_.to_host_vector()[0]); }

std::vector<std::vector<float>> BlockSparseFeaturizer::Frames() {
    const std::vector<float> w = decoder_.weight().to_host_vector();
    std::vector<std::vector<float>> out(static_cast<size_t>(G_));
    for (int64_t g = 0; g < G_; ++g) {
        out[static_cast<size_t>(g)].assign(w.begin() + g * b_ * d_, w.begin() + (g + 1) * b_ * d_);
        Orthonormalize(out[static_cast<size_t>(g)].data(), b_, d_);
    }
    return out;
}

std::vector<int64_t> BlockSparseFeaturizer::Select(const std::vector<float>& norms, const std::vector<std::vector<float>>* frames, bool training) {
    std::vector<int64_t> order(static_cast<size_t>(G_));
    std::iota(order.begin(), order.end(), int64_t{0});
    const int64_t k = options_.k;
    const bool tournament = options_.selection == BlockSelection::Tournament && frames != nullptr &&
                            !(training && steps_ < options_.tournament_warmup);
    if (!tournament) {
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int64_t a, int64_t c) {
            return norms[static_cast<size_t>(a)] > norms[static_cast<size_t>(c)] || (norms[static_cast<size_t>(a)] == norms[static_cast<size_t>(c)] && a < c);
        });
        return {order.begin(), order.begin() + k};
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t c) { return norms[static_cast<size_t>(a)] > norms[static_cast<size_t>(c)]; });
    auto overlap = [&](int64_t g, int64_t h) {
        const std::vector<float>& qg = (*frames)[static_cast<size_t>(g)];
        const std::vector<float>& qh = (*frames)[static_cast<size_t>(h)];
        double s = 0;
        for (int64_t i = 0; i < b_; ++i) {
            for (int64_t j = 0; j < b_; ++j) {
                double dot = 0;
                for (int64_t t = 0; t < d_; ++t) dot += static_cast<double>(qg[static_cast<size_t>(i * d_ + t)]) * qh[static_cast<size_t>(j * d_ + t)];
                s += dot * dot;
            }
        }
        return s / static_cast<double>(b_);
    };
    std::vector<int64_t> taken;
    for (int64_t g : order) {
        if (static_cast<int64_t>(taken.size()) == k) break;
        if (lost_to_.count(g) != 0) continue;  // a permanent loser is never selected again
        int64_t rival = -1;
        double best = options_.duel_overlap;
        for (int64_t h : taken) {
            const double o = overlap(g, h);
            if (o > best) best = o, rival = h;
        }
        if (rival < 0) {
            taken.push_back(g);
            continue;
        }
        if (!training) continue;
        ++wins_[static_cast<size_t>(rival)];
        if (best > options_.verdict_overlap) {
            const bool g_wins = wins_[static_cast<size_t>(g)] > wins_[static_cast<size_t>(rival)];
            lost_to_[g_wins ? rival : g] = g_wins ? g : rival;
        }
    }
    return taken;
}

BlockSparseFeaturizer::Pass BlockSparseFeaturizer::Run(const Tensor& x, bool training) {
    if (options_.variant == BsfVariant::Grassmannian) SyncTiedEncoder();
    const int64_t N = x.shape().dim(0), m = G_ * b_;
    Pass p;
    p.a = encoder_.forward(x).to_host_vector();
    std::vector<float> pre = p.a;
    if (options_.variant == BsfVariant::Grassmannian) {
        const std::vector<float> gm = gamma_.to_host_vector();
        for (size_t i = 0; i < pre.size(); ++i) pre[i] *= gm[(i % static_cast<size_t>(m)) / static_cast<size_t>(b_)];
    }
    std::vector<float> norm(static_cast<size_t>(N * G_));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < G_; ++g) {
            double s = 0;
            for (int64_t t = 0; t < b_; ++t) s += static_cast<double>(pre[static_cast<size_t>(n * m + g * b_ + t)]) * pre[static_cast<size_t>(n * m + g * b_ + t)];
            norm[static_cast<size_t>(n * G_ + g)] = static_cast<float>(std::sqrt(s));
        }
    }
    p.gate.assign(static_cast<size_t>(N * G_), 0.0f);
    if (options_.variant == BsfVariant::GroupLasso) {
        if (training) {
            if (initialized_.to_host_vector()[0] == 0.0f) {
                // Cold start: θ at the norm quantile that fires target_l0 blocks on this batch.
                const double thr = std::max(1e-3, Quantile(norm, 1.0 - options_.target_l0 / static_cast<double>(G_)));
                threshold_raw_ = Filled(G_, static_cast<float>(std::log(std::expm1(thr)) / options_.gain), backend_);
                bandwidth_ = Scalar(static_cast<float>(std::max(StdDev(norm), 1e-6) * options_.bandwidth), backend_);
                initialized_ = Scalar(1.0f, backend_);
            }
            const float bw = bandwidth_.to_host_vector()[0];
            bandwidth_ = Scalar(static_cast<float>(0.99 * bw + 0.01 * std::max(StdDev(norm), 1e-6) * options_.bandwidth), backend_);
        }
        const std::vector<float> theta = thresholds();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < G_; ++g) p.gate[static_cast<size_t>(n * G_ + g)] = norm[static_cast<size_t>(n * G_ + g)] > theta[static_cast<size_t>(g)] ? 1.0f : 0.0f;
        }
        p.norm = std::move(norm);
    } else {
        std::vector<std::vector<float>> frames;
        if (options_.selection == BlockSelection::Tournament) frames = Frames();
        for (int64_t n = 0; n < N; ++n) {
            const std::vector<float> row(norm.begin() + n * G_, norm.begin() + (n + 1) * G_);
            for (int64_t g : Select(row, frames.empty() ? nullptr : &frames, training)) p.gate[static_cast<size_t>(n * G_ + g)] = 1.0f;
        }
    }
    p.z.resize(pre.size());
    for (size_t i = 0; i < pre.size(); ++i) {
        const size_t n = i / static_cast<size_t>(m), g = (i % static_cast<size_t>(m)) / static_cast<size_t>(b_);
        p.z[i] = pre[i] * p.gate[n * static_cast<size_t>(G_) + g];
    }
    return p;
}

Tensor BlockSparseFeaturizer::encode(const Tensor& x) {
    Check(x, "BlockSparseFeaturizer::encode");
    return Tensor(Shape({x.shape().dim(0), G_ * b_}), backend_, Run(x, false).z, backend_->device());
}

Tensor BlockSparseFeaturizer::decode(const Tensor& codes) {
    if (codes.rank() != 2 || codes.shape().dim(1) != G_ * b_ || codes.shape().dim(0) < 1) {
        throw std::invalid_argument("BlockSparseFeaturizer::decode: codes must be (N, num_features) with N > 0");
    }
    return decoder_.forward(codes);
}

Tensor BlockSparseFeaturizer::BackwardThroughEncoder(const Pass& p, const std::vector<float>& g_z, double l0_weight, bool straight_through) {
    const auto m = static_cast<size_t>(G_ * b_), b = static_cast<size_t>(b_), G = static_cast<size_t>(G_);
    const size_t N = p.z.size() / m;
    std::vector<float> g_a(g_z.size(), 0.0f);
    if (options_.variant == BsfVariant::Grassmannian) {
        const std::vector<float> gm = gamma_.to_host_vector();
        std::vector<float> gg = gamma_grad_.to_host_vector();
        for (size_t i = 0; i < g_z.size(); ++i) {
            const size_t g = (i % m) / b;
            if (p.gate[(i / m) * G + g] == 0.0f) continue;
            gg[g] += g_z[i] * p.a[i];
            g_a[i] = g_z[i] * gm[g];
        }
        gamma_grad_ = Tensor(Shape({G_}), backend_, gg, backend_->device());
    } else if (options_.variant == BsfVariant::GroupLasso && straight_through) {
        const std::vector<float> theta = thresholds(), raw = threshold_raw_.to_host_vector();
        const double bw = bandwidth_.to_host_vector()[0];
        std::vector<double> g_theta(G, 0.0);
        for (size_t n = 0; n < N; ++n) {
            for (size_t g = 0; g < G; ++g) {
                double g_gate = l0_weight;
                for (size_t t = 0; t < b; ++t) g_gate += static_cast<double>(g_z[n * m + g * b + t]) * p.a[n * m + g * b + t];
                const double gn = p.norm[n * G + g], k = GaussianKernel(gn - theta[g], bw), g_gn = g_gate * k;
                g_theta[g] -= g_gn;
                for (size_t t = 0; t < b; ++t) {
                    const size_t i = n * m + g * b + t;
                    g_a[i] = static_cast<float>(g_z[i] * p.gate[n * G + g] + (gn > 0 ? g_gn * p.a[i] / gn : 0.0));
                }
            }
        }
        std::vector<float> gr = threshold_raw_grad_.to_host_vector();
        for (size_t g = 0; g < G; ++g) {
            gr[g] += static_cast<float>(g_theta[g] * options_.gain * Sigmoid(options_.gain * static_cast<double>(raw[g])));
        }
        threshold_raw_grad_ = Tensor(Shape({G_}), backend_, gr, backend_->device());
    } else {
        for (size_t i = 0; i < g_z.size(); ++i) g_a[i] = g_z[i] * p.gate[(i / m) * G + (i % m) / b];
    }
    Tensor grad_x = encoder_.backward(Tensor(Shape({static_cast<int64_t>(N), G_ * b_}), backend_, g_a, backend_->device()));
    if (options_.variant == BsfVariant::Grassmannian) {
        // The encoder is the decoder's transpose: its gradient belongs to the frames.
        Tensor* ge = GradOf(encoder_, "weight");
        Tensor* gd = GradOf(decoder_, "weight");
        const std::vector<float> e = ge->to_host_vector();  // (d, G·b)
        std::vector<float> dd = gd->to_host_vector();       // (G·b, d)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < static_cast<size_t>(d_); ++j) dd[i * static_cast<size_t>(d_) + j] += e[j * m + i];
        }
        *gd = Tensor(gd->shape(), gd->backend(), dd, gd->device());
        *ge = Tensor(ge->shape(), ge->backend(), std::vector<float>(e.size(), 0.0f), ge->device());
    }
    return grad_x;
}

FeaturizerLoss BlockSparseFeaturizer::loss_and_backward(const Tensor& x, std::vector<float>* codes_out) {
    Check(x, "BlockSparseFeaturizer::loss_and_backward");
    const int64_t N = x.shape().dim(0);
    const Pass p = Run(x, true);
    const std::vector<float> xh = decoder_.forward(Tensor(Shape({N, G_ * b_}), backend_, p.z, backend_->device())).to_host_vector();
    const std::vector<float> xv = x.to_host_vector();
    const double n_el = static_cast<double>(N * d_);
    std::vector<float> g(xh.size());
    double mse = 0;
    for (size_t i = 0; i < xh.size(); ++i) {
        const double diff = static_cast<double>(xh[i]) - xv[i];
        mse += diff * diff;
        g[i] = static_cast<float>(2.0 * diff / n_el);
    }
    double active = 0;
    for (float v : p.gate) active += v;
    const double l0 = active / static_cast<double>(N);
    FeaturizerLoss loss;
    loss.reconstruction = static_cast<float>(mse / n_el);
    const std::vector<float> g_z = decoder_.backward(Tensor(Shape({N, d_}), backend_, g, backend_->device())).to_host_vector();
    const bool group_lasso = options_.variant == BsfVariant::GroupLasso;
    const double lambda = group_lasso ? l0_multiplier() : 0.0;
    (void)BackwardThroughEncoder(p, g_z, lambda / static_cast<double>(N), group_lasso);
    if (group_lasso) {
        loss.sparsity = static_cast<float>(lambda * l0);
        // Dual ascent on log λ toward target_l0, the relative error clamped to ±1.
        const double err = std::clamp((l0 - options_.target_l0) / std::max(options_.target_l0, 1e-8), -1.0, 1.0);
        const double next = std::clamp(static_cast<double>(log_lambda_.to_host_vector()[0]) + options_.dual_lr * err, std::log(1e-8), std::log(1e2));
        log_lambda_ = Scalar(static_cast<float>(next), backend_);
    }
    loss.total = loss.reconstruction + loss.sparsity;
    ++steps_;
    if (codes_out != nullptr) *codes_out = p.z;
    return loss;
}

void BlockSparseFeaturizer::normalize_decoder() {
    std::vector<float> w = decoder_.weight().to_host_vector();
    if (options_.variant == BsfVariant::Grassmannian) {
        for (int64_t g = 0; g < G_; ++g) Orthonormalize(w.data() + g * b_ * d_, b_, d_);
        decoder_.set_weight(w);
        SyncTiedEncoder();
        return;
    }
    for (int64_t i = 0; i < G_ * b_; ++i) {
        double n2 = 0;
        for (int64_t t = 0; t < d_; ++t) n2 += static_cast<double>(w[static_cast<size_t>(i * d_ + t)]) * w[static_cast<size_t>(i * d_ + t)];
        const double n = std::max(std::sqrt(n2), 1e-8);
        for (int64_t t = 0; t < d_; ++t) w[static_cast<size_t>(i * d_ + t)] = static_cast<float>(w[static_cast<size_t>(i * d_ + t)] / n);
    }
    decoder_.set_weight(w);
}

std::vector<float> BlockSparseFeaturizer::decoder_direction(int64_t i) {
    if (i < 0 || i >= G_ * b_) throw std::invalid_argument("BlockSparseFeaturizer::decoder_direction: no such atom");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    return {w.begin() + i * d_, w.begin() + (i + 1) * d_};
}

Tensor BlockSparseFeaturizer::forward_impl(const Tensor& input) {
    Check(input, "BlockSparseFeaturizer::forward");
    last_ = Run(input, false);
    last_rows_ = input.shape().dim(0);
    return decoder_.forward(Tensor(Shape({last_rows_, G_ * b_}), backend_, last_.z, backend_->device()));
}

Tensor BlockSparseFeaturizer::backward(const Tensor& grad_output) {
    if (last_rows_ == 0) throw std::logic_error("BlockSparseFeaturizer::backward: called before any forward()");
    return BackwardThroughEncoder(last_, decoder_.backward(grad_output).to_host_vector(), 0.0, false);
}

Tensor BlockSparseFeaturizer::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_rows_ == 0) throw std::logic_error("BlockSparseFeaturizer::propagate_relevance: called before any forward()");
    std::vector<float> r = decoder_.propagate_relevance(relevance_out, config).to_host_vector();
    const auto m = static_cast<size_t>(G_ * b_);
    for (size_t i = 0; i < r.size(); ++i) {
        if (last_.gate[(i / m) * static_cast<size_t>(G_) + (i % m) / static_cast<size_t>(b_)] == 0.0f) r[i] = 0.0f;  // a fixed gate
    }
    return encoder_.propagate_relevance(Tensor(Shape({last_rows_, G_ * b_}), backend_, r, backend_->device()), config);
}

std::vector<NamedParamRef> BlockSparseFeaturizer::named_parameters() {
    std::vector<NamedParamRef> out;
    if (options_.variant != BsfVariant::Grassmannian) append_named_parameters(out, "encoder", encoder_);
    append_named_parameters(out, "decoder", decoder_);
    if (options_.variant == BsfVariant::Grassmannian) out.push_back({"gamma", {&gamma_, &gamma_grad_}});
    if (options_.variant == BsfVariant::GroupLasso) out.push_back({"threshold_raw", {&threshold_raw_, &threshold_raw_grad_}});
    return out;
}

std::vector<NamedBufferRef> BlockSparseFeaturizer::named_buffers() {
    if (options_.variant != BsfVariant::GroupLasso) return {};
    return {{"bandwidth", &bandwidth_}, {"log_lambda", &log_lambda_}, {"initialized", &initialized_}};
}

void BlockSparseFeaturizer::set_training(bool training) {
    Module::set_training(training);
    encoder_.set_training(training);
    decoder_.set_training(training);
}

void BlockSparseFeaturizer::release_activations() {
    encoder_.release_activations();
    decoder_.release_activations();
    last_ = Pass{};
    last_rows_ = 0;
}

// ---- geometry and description length --------------------------------------------------------

namespace {

/** @brief Per-block sums of the active codes and their outer products, and counts. */
struct BlockMoments {
    int64_t G = 0, b = 0, N = 0;
    std::vector<double> count, sum, outer;  ///< (G), (G·b), (G·b·b)
};

BlockMoments Moments(Featurizer& f, const Tensor& x, int64_t batch) {
    if (x.rank() != 2 || x.shape().dim(1) != f.input_dim() || x.shape().dim(0) < 1) {
        throw std::invalid_argument("block metrics: the batch must be (N, input_dim) with N > 0");
    }
    BlockMoments mo;
    mo.b = f.block_size();
    mo.G = f.num_features() / mo.b;
    mo.N = x.shape().dim(0);
    const int64_t d = f.input_dim(), m = f.num_features(), b = mo.b;
    mo.count.assign(static_cast<size_t>(mo.G), 0.0);
    mo.sum.assign(static_cast<size_t>(m), 0.0);
    mo.outer.assign(static_cast<size_t>(mo.G * b * b), 0.0);
    const std::vector<float> xv = x.to_host_vector();
    for (int64_t r0 = 0; r0 < mo.N; r0 += batch) {
        const int64_t n = std::min(batch, mo.N - r0);
        const std::vector<float> c =
            f.encode(Tensor(Shape({n, d}), x.backend(), std::vector<float>(xv.begin() + r0 * d, xv.begin() + (r0 + n) * d), x.device())).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t g = 0; g < mo.G; ++g) {
                const float* z = c.data() + r * m + g * b;
                bool on = false;
                for (int64_t s = 0; s < b; ++s) on = on || z[s] != 0.0f;
                if (!on) continue;
                mo.count[static_cast<size_t>(g)] += 1;
                for (int64_t s = 0; s < b; ++s) {
                    mo.sum[static_cast<size_t>(g * b + s)] += z[s];
                    for (int64_t t = 0; t < b; ++t) mo.outer[static_cast<size_t>((g * b + s) * b + t)] += static_cast<double>(z[s]) * z[t];
                }
            }
        }
    }
    return mo;
}

/** @brief Eigenvalues (non-negative) of block g's code covariance where it is active. */
std::vector<double> BlockEigenvalues(const BlockMoments& mo, int64_t g) {
    const int64_t b = mo.b;
    const double n = mo.count[static_cast<size_t>(g)];
    std::vector<float> cov(static_cast<size_t>(b * b));
    for (int64_t s = 0; s < b; ++s) {
        for (int64_t t = 0; t < b; ++t) {
            const double ms = mo.sum[static_cast<size_t>(g * b + s)] / n, mt = mo.sum[static_cast<size_t>(g * b + t)] / n;
            cov[static_cast<size_t>(s * b + t)] = static_cast<float>(mo.outer[static_cast<size_t>((g * b + s) * b + t)] / n - ms * mt);
        }
    }
    for (int64_t s = 0; s < b; ++s) {
        for (int64_t t = 0; t < s; ++t) cov[static_cast<size_t>(s * b + t)] = cov[static_cast<size_t>(t * b + s)];  // exactly symmetric
    }
    std::vector<double> out;
    if (b == 1) return {std::max(0.0, static_cast<double>(cov[0]))};
    static CPUBackend cpu;
    for (float v : SymmetricEigen(Tensor(Shape({b, b}), &cpu, cov)).values.to_host_vector()) out.push_back(std::max(0.0, static_cast<double>(v)));
    return out;
}

}  // namespace

BlockGeometry MeasureBlockGeometry(Featurizer& f, const Tensor& x, int64_t batch) {
    const BlockMoments mo = Moments(f, x, std::max<int64_t>(1, batch));
    BlockGeometry out;
    out.firing_rate.resize(static_cast<size_t>(mo.G));
    out.stable_rank.assign(static_cast<size_t>(mo.G), 0.0);
    out.participation_ratio.assign(static_cast<size_t>(mo.G), 0.0);
    out.effective_rank.assign(static_cast<size_t>(mo.G), 0.0);
    double weight = 0;
    for (int64_t g = 0; g < mo.G; ++g) {
        const double rate = mo.count[static_cast<size_t>(g)] / static_cast<double>(mo.N);
        out.firing_rate[static_cast<size_t>(g)] = rate;
        if (mo.count[static_cast<size_t>(g)] < 2) continue;
        const std::vector<double> ev = BlockEigenvalues(mo, g);
        double sum = 0, sq = 0, top = 0;
        for (double v : ev) sum += v, sq += v * v, top = std::max(top, v);
        if (top <= 0) continue;
        double entropy = 0;
        for (double v : ev) {
            if (v > 0) entropy -= v / sum * std::log(v / sum);
        }
        out.stable_rank[static_cast<size_t>(g)] = sq / (top * top);
        out.participation_ratio[static_cast<size_t>(g)] = sum * sum / sq;
        out.effective_rank[static_cast<size_t>(g)] = std::exp(entropy);
        out.mean_stable_rank += rate * out.stable_rank[static_cast<size_t>(g)];
        out.mean_participation_ratio += rate * out.participation_ratio[static_cast<size_t>(g)];
        out.mean_effective_rank += rate * out.effective_rank[static_cast<size_t>(g)];
        weight += rate;
    }
    if (weight > 0) {
        out.mean_stable_rank /= weight;
        out.mean_participation_ratio /= weight;
        out.mean_effective_rank /= weight;
    }
    return out;
}

DescriptionLength MeasureDescriptionLength(Featurizer& f, const Tensor& x, double distortion_fraction, int64_t batch) {
    if (distortion_fraction <= 0 || distortion_fraction >= 1) throw std::invalid_argument("MeasureDescriptionLength: distortion_fraction must be in (0, 1)");
    if (x.rank() != 2 || x.shape().dim(0) < 2) throw std::invalid_argument("MeasureDescriptionLength: needs at least 2 inputs");
    batch = std::max<int64_t>(1, batch);
    const BlockMoments mo = Moments(f, x, batch);
    const int64_t N = mo.N, d = f.input_dim();
    const std::vector<float> xv = x.to_host_vector();
    // δ from the inputs' mean per-dimension variance.
    double var = 0;
    for (int64_t j = 0; j < d; ++j) {
        double s = 0, s2 = 0;
        for (int64_t r = 0; r < N; ++r) {
            const double v = xv[static_cast<size_t>(r * d + j)];
            s += v, s2 += v * v;
        }
        var += (s2 / static_cast<double>(N) - (s / static_cast<double>(N)) * (s / static_cast<double>(N))) / static_cast<double>(d);
    }
    DescriptionLength out;
    out.delta = distortion_fraction * var;
    auto bits = [&](double sigma2) { return 0.5 * std::log2(1.0 + sigma2 / out.delta); };
    // Support: log2 C(G, ℓ) for the mean number of active blocks ℓ.
    double active = 0;
    for (double c : mo.count) active += c;
    const double l = active / static_cast<double>(N), G = static_cast<double>(mo.G);
    out.support = (std::lgamma(G + 1) - std::lgamma(l + 1) - std::lgamma(G - l + 1)) / std::log(2.0);
    for (int64_t g = 0; g < mo.G; ++g) {
        if (mo.count[static_cast<size_t>(g)] < 2) continue;
        double s = 0;
        for (double v : BlockEigenvalues(mo, g)) s += bits(v);
        out.code += mo.count[static_cast<size_t>(g)] / static_cast<double>(N) * s;
    }
    // Residual covariance, d x d.
    std::vector<double> mean(static_cast<size_t>(d), 0.0), outer(static_cast<size_t>(d * d), 0.0);
    for (int64_t r0 = 0; r0 < N; r0 += batch) {
        const int64_t n = std::min(batch, N - r0);
        const std::vector<float> xb(xv.begin() + r0 * d, xv.begin() + (r0 + n) * d);
        const std::vector<float> xh = f.predict(Tensor(Shape({n, d}), x.backend(), xb, x.device())).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            std::vector<double> e(static_cast<size_t>(d));
            for (int64_t j = 0; j < d; ++j) e[static_cast<size_t>(j)] = static_cast<double>(xb[static_cast<size_t>(r * d + j)]) - xh[static_cast<size_t>(r * d + j)];
            for (int64_t i = 0; i < d; ++i) {
                mean[static_cast<size_t>(i)] += e[static_cast<size_t>(i)];
                for (int64_t j = i; j < d; ++j) outer[static_cast<size_t>(i * d + j)] += e[static_cast<size_t>(i)] * e[static_cast<size_t>(j)];
            }
        }
    }
    std::vector<float> cov(static_cast<size_t>(d * d));
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = i; j < d; ++j) {
            const double c = outer[static_cast<size_t>(i * d + j)] / static_cast<double>(N) -
                             mean[static_cast<size_t>(i)] / static_cast<double>(N) * mean[static_cast<size_t>(j)] / static_cast<double>(N);
            cov[static_cast<size_t>(i * d + j)] = cov[static_cast<size_t>(j * d + i)] = static_cast<float>(c);
        }
    }
    static CPUBackend cpu;
    for (float v : SymmetricEigen(Tensor(Shape({d, d}), &cpu, cov)).values.to_host_vector()) out.residual += bits(std::max(0.0f, v));
    const double b = static_cast<double>(mo.b);
    out.dictionary = 0.5 * G * b * (static_cast<double>(d) - b) / static_cast<double>(N) * std::log2(static_cast<double>(N));
    out.total = out.support + out.code + out.residual + out.dictionary;
    return out;
}

}  // namespace pulsatrix
