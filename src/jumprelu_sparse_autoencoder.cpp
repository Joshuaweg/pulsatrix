#include "pulsatrix/jumprelu_sparse_autoencoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

Tensor Filled(int64_t n, float value, DeviceBackend* backend) {
    return Tensor(Shape({n}), backend, std::vector<float>(static_cast<size_t>(n), value), backend->device());
}

/** @brief The rectangle kernel: 1 on (-1/2, 1/2). */
double Rectangle(double u) { return std::abs(u) < 0.5 ? 1.0 : 0.0; }

}  // namespace

JumpReLUSparseAutoencoder::JumpReLUSparseAutoencoder(int64_t dim, int64_t num_features, DeviceBackend* backend, JumpReLUSaeOptions options)
    : dim_(dim),
      m_(num_features),
      backend_(backend),
      options_(options),
      encoder_(dim > 0 ? dim : 1, num_features > 0 ? num_features : 1, backend),
      decoder_(num_features > 0 ? num_features : 1, dim > 0 ? dim : 1, backend),
      log_threshold_(Filled(num_features > 0 ? num_features : 1, std::log(options.initial_threshold > 0 ? options.initial_threshold : 1.0f), backend)),
      log_threshold_grad_(Filled(num_features > 0 ? num_features : 1, 0.0f, backend)) {
    if (dim < 1 || num_features < 1) throw std::invalid_argument("JumpReLUSparseAutoencoder: dim and num_features must be positive");
    if (options.l0_coefficient < 0 || options.bandwidth <= 0 || options.initial_threshold <= 0) {
        throw std::invalid_argument("JumpReLUSparseAutoencoder: l0_coefficient must not be negative, bandwidth and initial_threshold must be positive");
    }
    PortableRng rng{options.seed};
    std::vector<float> dec(static_cast<size_t>(m_ * dim_)), enc(static_cast<size_t>(dim_ * m_));
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        std::vector<double> row(static_cast<size_t>(dim_));
        for (double& v : row) {
            v = rng.gaussian();
            sq += v * v;
        }
        for (int64_t j = 0; j < dim_; ++j) {
            const auto v = static_cast<float>(row[static_cast<size_t>(j)] / std::sqrt(sq));
            dec[static_cast<size_t>(i * dim_ + j)] = v;
            enc[static_cast<size_t>(j * m_ + i)] = v;
        }
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    decoder_.set_bias(std::vector<float>(static_cast<size_t>(dim_), 0.0f));
    encoder_.set_bias(std::vector<float>(static_cast<size_t>(m_), 0.0f));
}

void JumpReLUSparseAutoencoder::Check(const Tensor& x, const char* who) const {
    if (x.rank() != 2 || x.shape().dim(1) != dim_ || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the batch must be (N, dim) with N > 0");
    }
}

void JumpReLUSparseAutoencoder::initialize_bias(const Tensor& x) {
    Check(x, "JumpReLUSparseAutoencoder::initialize_bias");
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

void JumpReLUSparseAutoencoder::set_l0_coefficient(float lambda) {
    if (lambda < 0) throw std::invalid_argument("JumpReLUSparseAutoencoder::set_l0_coefficient: must not be negative");
    options_.l0_coefficient = lambda;
}

std::vector<float> JumpReLUSparseAutoencoder::thresholds() const {
    std::vector<float> t = log_threshold_.to_host_vector();
    for (float& v : t) v = std::exp(v);
    return t;
}

std::vector<float> JumpReLUSparseAutoencoder::Codes(const std::vector<float>& pre) const {
    const std::vector<float> theta = thresholds();
    std::vector<float> codes(pre.size());
    for (size_t i = 0; i < pre.size(); ++i) codes[i] = pre[i] > theta[i % static_cast<size_t>(m_)] ? pre[i] : 0.0f;
    return codes;
}

Tensor JumpReLUSparseAutoencoder::encode(const Tensor& x) {
    Check(x, "JumpReLUSparseAutoencoder::encode");
    return Tensor(Shape({x.shape().dim(0), m_}), backend_, Codes(encoder_.forward(x).to_host_vector()), backend_->device());
}

Tensor JumpReLUSparseAutoencoder::decode(const Tensor& codes) {
    if (codes.rank() != 2 || codes.shape().dim(1) != m_ || codes.shape().dim(0) < 1) {
        throw std::invalid_argument("JumpReLUSparseAutoencoder::decode: codes must be (N, num_features) with N > 0");
    }
    return decoder_.forward(codes);
}

Tensor JumpReLUSparseAutoencoder::BackwardFromCodes(const std::vector<float>& pre, const std::vector<float>& g_codes, double l0_weight) {
    const std::vector<float> theta = thresholds();
    const auto N = static_cast<int64_t>(pre.size()) / m_;
    const double eps = options_.bandwidth;
    std::vector<float> g_pre(pre.size(), 0.0f);
    std::vector<double> g_theta(static_cast<size_t>(m_), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t i = 0; i < m_; ++i) {
            const size_t k = static_cast<size_t>(r * m_ + i);
            const double t = theta[static_cast<size_t>(i)], window = Rectangle((pre[k] - t) / eps);
            if (pre[k] > t) g_pre[k] = g_codes[k];
            if (window == 0.0) continue;
            // ∂z/∂θ = -(θ/ε) K, and the L0 term's ∂H/∂θ = -(1/ε) K.
            g_theta[static_cast<size_t>(i)] += -(t / eps) * window * g_codes[k] - l0_weight / eps * window;
        }
    }
    // Into log θ: ∂θ/∂log θ = θ.
    std::vector<float> g = log_threshold_grad_.to_host_vector();
    for (int64_t i = 0; i < m_; ++i) g[static_cast<size_t>(i)] += static_cast<float>(g_theta[static_cast<size_t>(i)] * theta[static_cast<size_t>(i)]);
    log_threshold_grad_ = Tensor(Shape({m_}), backend_, g, backend_->device());
    return encoder_.backward(Tensor(Shape({N, m_}), backend_, g_pre, backend_->device()));
}

FeaturizerLoss JumpReLUSparseAutoencoder::loss_and_backward(const Tensor& x, std::vector<float>* codes_out) {
    Check(x, "JumpReLUSparseAutoencoder::loss_and_backward");
    const int64_t N = x.shape().dim(0);
    const std::vector<float> pre = encoder_.forward(x).to_host_vector();
    const std::vector<float> codes = Codes(pre);
    const std::vector<float> xh = decoder_.forward(Tensor(Shape({N, m_}), backend_, codes, backend_->device())).to_host_vector();
    const std::vector<float> xv = x.to_host_vector();
    const double n_el = static_cast<double>(N * dim_);
    std::vector<float> g(xh.size());
    double mse = 0;
    for (size_t i = 0; i < xh.size(); ++i) {
        const double d = static_cast<double>(xh[i]) - xv[i];
        mse += d * d;
        g[i] = static_cast<float>(2.0 * d / n_el);
    }
    double active = 0;
    for (float c : codes) active += c > 0.0f ? 1.0 : 0.0;
    FeaturizerLoss loss;
    loss.reconstruction = static_cast<float>(mse / n_el);
    loss.sparsity = static_cast<float>(options_.l0_coefficient * active / static_cast<double>(N));
    loss.total = loss.reconstruction + loss.sparsity;
    const std::vector<float> g_codes = decoder_.backward(Tensor(Shape({N, dim_}), backend_, g, backend_->device())).to_host_vector();
    (void)BackwardFromCodes(pre, g_codes, options_.l0_coefficient / static_cast<double>(N));
    if (codes_out != nullptr) *codes_out = codes;
    return loss;
}

void JumpReLUSparseAutoencoder::normalize_decoder() {
    std::vector<float> dec = decoder_.weight().to_host_vector(), enc = encoder_.weight().to_host_vector();
    std::vector<float> b = encoder_.bias().to_host_vector(), lt = log_threshold_.to_host_vector();
    for (int64_t i = 0; i < m_; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < dim_; ++j) sq += static_cast<double>(dec[static_cast<size_t>(i * dim_ + j)]) * dec[static_cast<size_t>(i * dim_ + j)];
        const double s = std::sqrt(sq);
        if (s == 0) continue;
        // Direction /= s, code *= s: the encoder column, bias and threshold all scale by s.
        for (int64_t j = 0; j < dim_; ++j) {
            dec[static_cast<size_t>(i * dim_ + j)] = static_cast<float>(dec[static_cast<size_t>(i * dim_ + j)] / s);
            enc[static_cast<size_t>(j * m_ + i)] = static_cast<float>(enc[static_cast<size_t>(j * m_ + i)] * s);
        }
        b[static_cast<size_t>(i)] = static_cast<float>(b[static_cast<size_t>(i)] * s);
        lt[static_cast<size_t>(i)] = static_cast<float>(lt[static_cast<size_t>(i)] + std::log(s));
    }
    decoder_.set_weight(dec);
    encoder_.set_weight(enc);
    encoder_.set_bias(b);
    log_threshold_ = Tensor(Shape({m_}), backend_, lt, backend_->device());
}

std::vector<float> JumpReLUSparseAutoencoder::decoder_direction(int64_t i) {
    if (i < 0 || i >= m_) throw std::invalid_argument("JumpReLUSparseAutoencoder::decoder_direction: no such feature");
    const std::vector<float> w = decoder_.weight().to_host_vector();
    return {w.begin() + static_cast<std::ptrdiff_t>(i * dim_), w.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim_)};
}

Tensor JumpReLUSparseAutoencoder::forward_impl(const Tensor& input) {
    Check(input, "JumpReLUSparseAutoencoder::forward");
    last_pre_ = encoder_.forward(input).to_host_vector();
    last_rows_ = input.shape().dim(0);
    return decoder_.forward(Tensor(Shape({last_rows_, m_}), backend_, Codes(last_pre_), backend_->device()));
}

Tensor JumpReLUSparseAutoencoder::backward(const Tensor& grad_output) {
    if (last_rows_ == 0) throw std::logic_error("JumpReLUSparseAutoencoder::backward: called before any forward()");
    return BackwardFromCodes(last_pre_, decoder_.backward(grad_output).to_host_vector(), 0.0);
}

Tensor JumpReLUSparseAutoencoder::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_rows_ == 0) throw std::logic_error("JumpReLUSparseAutoencoder::propagate_relevance: called before any forward()");
    std::vector<float> r = decoder_.propagate_relevance(relevance_out, config).to_host_vector();
    const std::vector<float> codes = Codes(last_pre_);
    for (size_t i = 0; i < r.size(); ++i) {
        if (codes[i] <= 0.0f) r[i] = 0.0f;  // only active latents carry relevance
    }
    return encoder_.propagate_relevance(Tensor(Shape({last_rows_, m_}), backend_, r, backend_->device()), config);
}

std::vector<NamedParamRef> JumpReLUSparseAutoencoder::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "encoder", encoder_);
    append_named_parameters(out, "decoder", decoder_);
    out.push_back({"log_threshold", {&log_threshold_, &log_threshold_grad_}});
    return out;
}

void JumpReLUSparseAutoencoder::set_training(bool training) {
    Module::set_training(training);
    encoder_.set_training(training);
    decoder_.set_training(training);
}

void JumpReLUSparseAutoencoder::release_activations() {
    encoder_.release_activations();
    decoder_.release_activations();
    last_pre_.clear();
    last_rows_ = 0;
}

}  // namespace pulsatrix
