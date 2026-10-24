#include "pulsatrix/trajectory_lrp.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "pulsatrix/sinusoidal_timestep_embedding.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Bad(const std::string& message) { throw std::invalid_argument("trajectory LRP: " + message); }

double Sum(const std::vector<float>& v) {
    double total = 0;
    for (float x : v) total += x;
    return total;
}

/** @brief DDIM's timesteps: round(T k / steps) for k = steps..1, deduplicated, descending. */
std::vector<int64_t> DdimTimesteps(int64_t T, int64_t steps) {
    std::vector<int64_t> out;
    for (int64_t k = steps; k >= 1; --k) {
        const auto t = static_cast<int64_t>(std::llround(static_cast<double>(T) * static_cast<double>(k) / static_cast<double>(steps)));
        if (t >= 1 && (out.empty() || t < out.back())) out.push_back(t);
    }
    return out;
}

}  // namespace

// ---- ConcatDenoiser ---------------------------------------------------------------------------

LRPRuleConfig TrajectoryNetworkRules() {
    LRPRuleConfig config;
    config.epsilon_bias_in_denominator = true;
    return config;
}

ConcatDenoiser::ConcatDenoiser(Module& network, int64_t dim, int64_t time_dim, std::vector<float> condition, DeviceBackend* backend,
                               LRPRuleConfig config)
    : network_(network), dim_(dim), time_dim_(time_dim), condition_(std::move(condition)), backend_(backend), config_(std::move(config)) {
    if (dim <= 0) Bad("ConcatDenoiser needs dim > 0");
    if (time_dim < 0) Bad("ConcatDenoiser needs time_dim >= 0");
}

void ConcatDenoiser::set_condition(std::vector<float> condition) {
    if (condition.size() != condition_.size()) Bad("set_condition: the conditioning changed size");
    condition_ = std::move(condition);
}

std::vector<float> ConcatDenoiser::input_row(const std::vector<float>& x_t, int64_t t) const {
    if (static_cast<int64_t>(x_t.size()) != dim_) Bad("ConcatDenoiser: x_t has " + std::to_string(x_t.size()) + " values, not " + std::to_string(dim_));
    std::vector<float> row(x_t);
    if (time_dim_ > 0) {
        const std::vector<float> embedding = SinusoidalTimestepEmbedding(t, time_dim_, backend_).to_host_vector();
        row.insert(row.end(), embedding.begin(), embedding.end());
    }
    row.insert(row.end(), condition_.begin(), condition_.end());
    return row;
}

std::vector<float> ConcatDenoiser::predict(const std::vector<float>& x_t, int64_t t) {
    const std::vector<float> row = input_row(x_t, t);
    const int64_t width = static_cast<int64_t>(row.size());
    std::vector<float> out = network_.forward(Tensor(Shape({1, width}), backend_, row, backend_->device())).to_host_vector();
    if (static_cast<int64_t>(out.size()) != dim_) Bad("ConcatDenoiser: the network gave " + std::to_string(out.size()) + " values, not " + std::to_string(dim_));
    return out;
}

DenoiserRelevance ConcatDenoiser::relevance(const std::vector<float>& relevance_eps) {
    if (static_cast<int64_t>(relevance_eps.size()) != dim_) Bad("ConcatDenoiser: relevance on the prediction has the wrong size");
    const std::vector<float> in =
        network_.propagate_relevance(Tensor(Shape({1, dim_}), backend_, relevance_eps, backend_->device()), config_).to_host_vector();
    const auto width = static_cast<size_t>(dim_ + time_dim_) + condition_.size();
    if (in.size() != width) Bad("ConcatDenoiser: the input relevance has the wrong size");
    DenoiserRelevance r;
    r.x.assign(in.begin(), in.begin() + dim_);
    for (int64_t e = 0; e < time_dim_; ++e) r.time += in[static_cast<size_t>(dim_ + e)];
    r.condition.assign(in.begin() + dim_ + time_dim_, in.end());
    return r;
}

// ---- sampling ---------------------------------------------------------------------------------

std::vector<float> StandardNormal(size_t n, uint64_t seed) {
    // Box-Muller over the engine's raw bits: std::normal_distribution differs between standard
    // libraries, and a trajectory must be the same everywhere.
    std::mt19937_64 engine(seed);
    const auto uniform = [&]() { return (static_cast<double>(engine() >> 11) + 0.5) * 0x1.0p-53; };  // (0, 1)
    constexpr double kTwoPi = 6.28318530717958647692;
    std::vector<float> out;
    out.reserve(n);
    while (out.size() < n) {
        const double r = std::sqrt(-2.0 * std::log(uniform())), angle = kTwoPi * uniform();
        out.push_back(static_cast<float>(r * std::cos(angle)));
        if (out.size() < n) out.push_back(static_cast<float>(r * std::sin(angle)));
    }
    return out;
}

ContinuousTrajectory SampleContinuousDiffusion(ContinuousDenoiser& model, const NoiseSchedule& schedule, const std::vector<float>& x_T,
                                               const ContinuousSamplerConfig& config) {
    if (x_T.empty()) Bad("x_T is empty");
    const int64_t T = schedule.num_timesteps();
    ContinuousTrajectory out;
    if (config.sampler == ContinuousSampler::Ddpm) {
        for (int64_t t = T; t >= 1; --t) out.timesteps.push_back(t);
    } else {
        if (config.steps < 1) Bad("DDIM needs steps >= 1");
        if (!(config.eta >= 0.0f)) Bad("DDIM needs eta >= 0");
        out.timesteps = DdimTimesteps(T, config.steps);
    }
    const size_t n = x_T.size();
    std::mt19937_64 seeds(config.seed);
    out.x.push_back(x_T);
    for (size_t k = 0; k < out.timesteps.size(); ++k) {
        const int64_t t = out.timesteps[k];
        ReverseStepCoefficients c;
        if (config.sampler == ContinuousSampler::Ddpm) {
            c = schedule.ddpm_coefficients(t);
            if (t == 1) c.noise = 0.0f;  // the last DDPM step is deterministic
        } else {
            c = schedule.ddim_coefficients(t, k + 1 < out.timesteps.size() ? out.timesteps[k + 1] : 0, config.eta);
        }
        const uint64_t step_seed = seeds();  // drawn every step, so a step's noise doesn't depend on the others' eta
        std::vector<float> z = c.noise != 0.0f ? StandardNormal(n, step_seed) : std::vector<float>(n, 0.0f);
        const std::vector<float>& x_t = out.x.back();
        const std::vector<float> eps = model.predict(x_t, t);
        if (eps.size() != n) Bad("the model's prediction has " + std::to_string(eps.size()) + " values, not " + std::to_string(n));
        std::vector<float> next(n);
        for (size_t i = 0; i < n; ++i) next[i] = c.skip * x_t[i] - c.network * eps[i] + c.noise * z[i];
        out.coefficients.push_back(c);
        out.noise.push_back(std::move(z));
        out.x.push_back(std::move(next));
    }
    return out;
}

// ---- seeds ------------------------------------------------------------------------------------

std::vector<float> RegionSeed(const std::vector<float>& sample, const std::vector<int64_t>& region) {
    std::vector<float> seed(sample.size(), 0.0f);
    for (int64_t i : region) {
        if (i < 0 || i >= static_cast<int64_t>(sample.size())) Bad("region index " + std::to_string(i) + " is outside the sample");
        seed[static_cast<size_t>(i)] = sample[static_cast<size_t>(i)];
    }
    return seed;
}

std::vector<float> ReconstructionSeed(const std::vector<float>& sample, const std::vector<float>& reference) {
    if (sample.size() != reference.size()) Bad("the reference's size differs from the sample's");
    std::vector<float> seed(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        const float d = sample[i] - reference[i];
        seed[i] = d * d;
    }
    return seed;
}

// ---- the explainer ----------------------------------------------------------------------------

double TrajectoryRelevance::x_T_total() const { return Sum(x_T); }
double TrajectoryRelevance::condition_total() const { return Sum(condition); }
double TrajectoryRelevance::time_total() const {
    double total = 0;
    for (const TrajectoryHop& h : hops) total += h.network_to_time;
    return total;
}
double TrajectoryRelevance::noise_total() const {
    double total = 0;
    for (const TrajectoryHop& h : hops) total += h.noise;
    return total;
}
double TrajectoryRelevance::absorbed_total() const {
    double total = 0;
    for (const TrajectoryHop& h : hops) total += h.absorbed;
    return total;
}
double TrajectoryRelevance::leak_total() const {
    double total = 0;
    for (const TrajectoryHop& h : hops) total += h.network_leak();
    return total;
}

TrajectoryRelevance ExplainTrajectory(ContinuousDenoiser& model, const ContinuousTrajectory& trajectory, const std::vector<float>& seed,
                                      const TrajectoryLrpConfig& config) {
    const int64_t K = trajectory.steps();
    if (K < 1 || trajectory.x.size() != static_cast<size_t>(K + 1) || trajectory.coefficients.size() != static_cast<size_t>(K) ||
        trajectory.noise.size() != static_cast<size_t>(K)) {
        Bad("the trajectory is incomplete");
    }
    const size_t n = trajectory.sample().size();
    if (seed.size() != n) Bad("the seed has " + std::to_string(seed.size()) + " values, the sample " + std::to_string(n));

    TrajectoryRelevance out;
    out.value = Sum(seed);
    out.hops.resize(static_cast<size_t>(K));
    out.condition_by_step.resize(static_cast<size_t>(K));
    std::vector<double> r(seed.begin(), seed.end());  // relevance on the current x_prev
    std::vector<double> condition;
    std::vector<float> to_eps(n);
    for (int64_t k = K - 1; k >= 0; --k) {
        const auto ks = static_cast<size_t>(k);
        const std::vector<float>& x_t = trajectory.x[ks];
        const std::vector<float>& z = trajectory.noise[ks];
        const ReverseStepCoefficients& c = trajectory.coefficients[ks];
        const std::vector<float> eps = model.predict(x_t, trajectory.timesteps[ks]);
        if (eps.size() != n) Bad("the model's prediction has the wrong size");

        TrajectoryHop& hop = out.hops[ks];
        hop.step = k;
        hop.timestep = trajectory.timesteps[ks];
        std::vector<double> skip(n);
        double scale = 0, error = 0;
        for (size_t i = 0; i < n; ++i) {
            const double h[3] = {static_cast<double>(c.skip) * x_t[i], -static_cast<double>(c.network) * eps[i],
                                 static_cast<double>(c.noise) * z[i]};
            const double s = h[0] + h[1] + h[2], a = std::abs(h[0]) + std::abs(h[1]) + std::abs(h[2]);
            scale = std::max(scale, std::abs(static_cast<double>(trajectory.x[ks + 1][i])));
            error = std::max(error, std::abs(s - static_cast<double>(trajectory.x[ks + 1][i])));
            double share[3];
            if (config.split == TrajectorySplit::AbsoluteRatio) {
                for (int j = 0; j < 3; ++j) share[j] = r[i] * std::abs(h[j]) / (a + config.epsilon);
            } else if (config.split == TrajectorySplit::SignedRatio) {
                const double sign = s >= 0 ? 1.0 : -1.0;
                for (int j = 0; j < 3; ++j) share[j] = r[i] * h[j] * sign / (a + config.epsilon);
            } else {
                const double e = config.epsilon * a;
                const double d = s + (s >= 0 ? e : -e);
                for (int j = 0; j < 3; ++j) share[j] = d != 0 ? r[i] * h[j] / d : 0.0;
            }
            if (s != 0) {
                hop.signed_skip += r[i] * h[0] / s;
                hop.signed_network += r[i] * h[1] / s;
                hop.signed_noise += r[i] * h[2] / s;
            }
            hop.seeded += r[i];
            hop.skip += share[0];
            hop.network += share[1];
            hop.noise += share[2];
            hop.absorbed += r[i] - share[0] - share[1] - share[2];
            skip[i] = share[0];
            to_eps[i] = static_cast<float>(share[1]);
        }
        if (config.recompute_tolerance >= 0 && error > config.recompute_tolerance * std::max(1.0, scale)) {
            throw std::runtime_error("trajectory LRP: step " + std::to_string(k) + " (t = " + std::to_string(hop.timestep) +
                                     ") rerun from the stored state differs from the stored next state by " + std::to_string(error) +
                                     "; the model isn't the one that sampled, or isn't deterministic");
        }

        const DenoiserRelevance net = model.relevance(to_eps);
        if (net.x.size() != n) Bad("the model's relevance on x_t has the wrong size");
        if (condition.empty()) condition.assign(net.condition.size(), 0.0);
        if (net.condition.size() != condition.size()) Bad("the conditioning's relevance changed size between steps");
        hop.network_to_x = Sum(net.x);
        hop.network_to_condition = Sum(net.condition);
        hop.network_to_time = net.time;
        out.condition_by_step[ks] = net.condition;
        for (size_t j = 0; j < condition.size(); ++j) condition[j] += net.condition[j];
        for (size_t i = 0; i < n; ++i) r[i] = skip[i] + net.x[i];
    }
    out.x_T.assign(r.begin(), r.end());
    out.condition.assign(condition.begin(), condition.end());
    return out;
}

}  // namespace pulsatrix
