#include "pulsatrix/tda/significance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tda/diagram_distance.hpp"

namespace pulsatrix {
namespace {

constexpr double kEulerMascheroni = 0.57721566490153286061;
constexpr double kPi = 3.14159265358979323846;

// Uniform in (0, 1) from mt19937's raw output, which the standard fixes on every platform (unlike
// std::uniform_real_distribution and std::normal_distribution).
double Uniform(std::mt19937& rng) { return (static_cast<double>(rng()) + 0.5) / 4294967296.0; }

// A standard normal by Box-Muller.
double Normal(std::mt19937& rng) {
    const double u = Uniform(rng), v = Uniform(rng);
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * kPi * v);
}

// An index in [0, n) by Lemire's multiply-shift: deterministic on every platform.
int64_t Index(std::mt19937& rng, int64_t n) {
    return static_cast<int64_t>((static_cast<uint64_t>(rng()) * static_cast<uint64_t>(n)) >> 32);
}

void CheckAlpha(double alpha, const char* function) {
    if (!(alpha > 0 && alpha < 1)) throw std::invalid_argument(std::string(function) + ": alpha must be in (0, 1)");
}

}  // namespace

UniversalNullResult UniversalNullTest(const PersistenceDiagram& diagram, double alpha) {
    CheckAlpha(alpha, "UniversalNullTest");
    if (diagram.dimension < 1) {
        throw std::invalid_argument("UniversalNullTest: H0 features are all born at 0, so death / birth is undefined");
    }
    UniversalNullResult out;
    for (std::size_t k = 0; k < diagram.pairs.size(); ++k) {
        const PersistencePair& p = diagram.pairs[k];
        if (p.essential() || !(p.birth > 0) || !(p.death > p.birth)) continue;
        FeatureSignificance f;
        f.index = static_cast<int64_t>(k);
        f.ratio = p.death / p.birth;
        f.ell = std::log(std::log(f.ratio));  // A = 1 for Vietoris-Rips; B is added below
        out.features.push_back(f);
    }
    if (out.features.size() < 2) {
        throw std::invalid_argument("UniversalNullTest: needs at least 2 finite features born above 0");
    }
    for (const FeatureSignificance& f : out.features) out.mean_log_log += f.ell;
    out.mean_log_log /= static_cast<double>(out.features.size());
    out.threshold = alpha / static_cast<double>(out.features.size());
    for (FeatureSignificance& f : out.features) {
        f.ell += -kEulerMascheroni - out.mean_log_log;
        f.p_value = std::exp(-std::exp(f.ell));
        f.significant = f.p_value < out.threshold;
    }
    return out;
}

DistanceMatrix SubDistanceMatrix(const DistanceMatrix& distances, const std::vector<int64_t>& indices) {
    DistanceMatrix out;
    out.size = static_cast<int64_t>(indices.size());
    out.error_bound = distances.error_bound;
    out.values.reserve(indices.size() * (indices.empty() ? 0 : indices.size() - 1) / 2);
    for (int64_t i : indices) {
        if (i < 0 || i >= distances.size) throw std::invalid_argument("SubDistanceMatrix: an index is out of range");
    }
    for (std::size_t a = 0; a < indices.size(); ++a) {
        for (std::size_t b = a + 1; b < indices.size(); ++b) {
            const float d = distances.at(indices[a], indices[b]);
            out.values.push_back(d);
            out.max_distance = std::max(out.max_distance, d);
        }
    }
    return out;
}

BootstrapBandResult BootstrapBand(const DistanceMatrix& distances, int dimension, const BootstrapOptions& options) {
    if (options.resamples < 1) throw std::invalid_argument("BootstrapBand: resamples must be at least 1");
    CheckAlpha(options.alpha, "BootstrapBand");
    if (dimension < 0 || dimension > options.rips.max_dimension) {
        throw std::invalid_argument("BootstrapBand: dimension must be in [0, rips.max_dimension]");
    }
    BootstrapBandResult out;
    const std::size_t dim = static_cast<std::size_t>(dimension);
    out.diagram = ComputeRipsPersistence(distances, options.rips).diagrams[dim];
    std::mt19937 rng(options.seed);
    std::vector<int64_t> sample(static_cast<std::size_t>(distances.size));
    for (int64_t r = 0; r < options.resamples; ++r) {
        for (int64_t& i : sample) i = Index(rng, distances.size);
        const PersistenceDiagram resampled = ComputeRipsPersistence(SubDistanceMatrix(distances, sample), options.rips).diagrams[dim];
        out.distances.push_back(BottleneckDistance(resampled, out.diagram).value);
    }
    std::sort(out.distances.begin(), out.distances.end());
    const double rank = std::ceil((1 - options.alpha) * static_cast<double>(options.resamples));
    const std::size_t k = static_cast<std::size_t>(std::clamp(rank, 1.0, static_cast<double>(options.resamples))) - 1;
    out.band = out.distances[k];
    for (std::size_t i = 0; i < out.diagram.pairs.size(); ++i) {
        const PersistencePair& p = out.diagram.pairs[i];
        if (!p.essential() && p.persistence() > 2 * out.band) out.significant.push_back(static_cast<int64_t>(i));
    }
    return out;
}

Tensor GaussianBaseline(const Tensor& points, int64_t count, uint32_t seed) {
    if (count < 1) throw std::invalid_argument("GaussianBaseline: count must be at least 1");
    if (points.rank() < 2) throw std::invalid_argument("GaussianBaseline: points must have shape (n, ...)");
    const int64_t n = points.shape().dim(0);
    if (n < 2) throw std::invalid_argument("GaussianBaseline: needs at least 2 points");
    const int64_t d = points.numel() / n;
    const std::vector<float> x = points.to_host_vector();
    std::vector<double> mean(static_cast<std::size_t>(d), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t k = 0; k < d; ++k) {
            const float v = x[static_cast<std::size_t>(i * d + k)];
            if (!std::isfinite(v)) throw std::invalid_argument("GaussianBaseline: points have a NaN or infinite value");
            mean[static_cast<std::size_t>(k)] += v;
        }
    }
    for (double& m : mean) m /= static_cast<double>(n);
    // Centered rows scaled by 1 / sqrt(n - 1): then Z times them has the sample covariance.
    const double scale = 1.0 / std::sqrt(static_cast<double>(n - 1));
    std::vector<float> centered(x.size());
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t k = 0; k < d; ++k) {
            const std::size_t at = static_cast<std::size_t>(i * d + k);
            centered[at] = static_cast<float>((x[at] - mean[static_cast<std::size_t>(k)]) * scale);
        }
    }
    std::mt19937 rng(seed);
    std::vector<float> z(static_cast<std::size_t>(count * n));
    for (float& v : z) v = static_cast<float>(Normal(rng));

    DeviceBackend* backend = points.backend();
    const Tensor zt(Shape({count, n}), backend, z), ct(Shape({n, d}), backend, centered);
    Tensor product(Shape({count, d}), backend);
    backend->gemm_ex(zt.data(), false, ct.data(), false, product.data(), static_cast<std::size_t>(count),
                     static_cast<std::size_t>(n), static_cast<std::size_t>(d), 0.0f);
    std::vector<float> out = product.to_host_vector();
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t k = 0; k < d; ++k) {
            out[static_cast<std::size_t>(i * d + k)] += static_cast<float>(mean[static_cast<std::size_t>(k)]);
        }
    }
    return Tensor(Shape({count, d}), backend, out);
}

double BaselinePValue(double observed, const std::vector<double>& baselines) {
    if (baselines.empty()) throw std::invalid_argument("BaselinePValue: no baselines");
    if (std::isnan(observed)) throw std::invalid_argument("BaselinePValue: observed is NaN");
    int64_t at_least = 0;
    for (double b : baselines) {
        if (std::isnan(b)) throw std::invalid_argument("BaselinePValue: a baseline is NaN");
        if (b >= observed) ++at_least;
    }
    return static_cast<double>(1 + at_least) / static_cast<double>(1 + baselines.size());
}

double MaxPersistence(const PersistenceDiagram& diagram) {
    double best = 0;
    for (const PersistencePair& p : diagram.pairs) {
        if (!p.essential()) best = std::max(best, p.persistence());
    }
    return best;
}

}  // namespace pulsatrix
