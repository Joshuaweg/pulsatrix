#include "pulsatrix/steering.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

/** @brief Least-squares slope of @p y against @p x. */
double Slope(const std::vector<double>& x, const std::vector<double>& y) {
    double mx = 0, my = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= static_cast<double>(x.size());
    my /= static_cast<double>(y.size());
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
    }
    return sxy / sxx;
}

/** @brief Each input's behavior per coefficient along @p direction: `[input][coefficient]`. */
std::vector<std::vector<double>> Sweep(const std::function<double(int64_t, const HiddenStateHook&)>& behavior, int64_t n, int64_t position,
                                       const std::vector<float>& direction, const std::vector<double>& coefficients) {
    std::vector<std::vector<double>> out(static_cast<size_t>(n), std::vector<double>(coefficients.size()));
    for (size_t c = 0; c < coefficients.size(); ++c) {
        const HiddenStateHook hook = SteeringHook(position, direction, static_cast<float>(coefficients[c]));
        for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)][c] = behavior(i, hook);
    }
    return out;
}

}  // namespace

std::vector<float> DifferenceOfMeans(const std::vector<float>& positive, const std::vector<float>& negative, int64_t d) {
    if (d < 1 || positive.empty() || negative.empty() || positive.size() % static_cast<size_t>(d) != 0 ||
        negative.size() % static_cast<size_t>(d) != 0) {
        throw std::invalid_argument("DifferenceOfMeans: both sets need at least one row of d values");
    }
    std::vector<double> sum(static_cast<size_t>(d), 0.0);
    const double np = static_cast<double>(positive.size() / static_cast<size_t>(d)), nn = static_cast<double>(negative.size() / static_cast<size_t>(d));
    for (size_t i = 0; i < positive.size(); ++i) sum[i % static_cast<size_t>(d)] += positive[i] / np;
    for (size_t i = 0; i < negative.size(); ++i) sum[i % static_cast<size_t>(d)] -= negative[i] / nn;
    return {sum.begin(), sum.end()};
}

std::vector<float> FeaturizerDirection(Featurizer& featurizer, int64_t feature) { return featurizer.decoder_direction(feature); }

HiddenStateHook SteeringHook(int64_t position, std::vector<float> direction, float coefficient, std::vector<bool> rows) {
    return [position, direction = std::move(direction), coefficient, rows = std::move(rows)](int64_t at, const Tensor& hidden) -> Tensor {
        if (at != position) return hidden;
        const auto d = static_cast<int64_t>(direction.size());
        if (hidden.rank() < 2 || hidden.shape().dim(hidden.rank() - 1) != d) {
            throw std::invalid_argument("SteeringHook: the hidden size isn't the direction's");
        }
        const int64_t n = hidden.numel() / d;
        if (!rows.empty() && static_cast<int64_t>(rows.size()) != n) throw std::invalid_argument("SteeringHook: rows must have one entry per (n, l)");
        std::vector<float> h = hidden.to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            if (!rows.empty() && !rows[static_cast<size_t>(r)]) continue;
            for (int64_t j = 0; j < d; ++j) h[static_cast<size_t>(r * d + j)] += coefficient * direction[static_cast<size_t>(j)];
        }
        return Tensor(hidden.shape(), hidden.backend(), h, hidden.device());
    };
}

SteeringReport MeasureSteering(const std::function<double(int64_t, const HiddenStateHook&)>& behavior, int64_t num_inputs, int64_t position,
                               const std::vector<float>& direction, const SteeringOptions& o) {
    if (num_inputs < 1) throw std::invalid_argument("MeasureSteering: needs at least one input");
    if (std::set<double>(o.coefficients.begin(), o.coefficients.end()).size() < 2) {
        throw std::invalid_argument("MeasureSteering: needs at least two distinct coefficients");
    }
    double norm = 0;
    for (float v : direction) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);
    if (direction.empty() || norm == 0) throw std::invalid_argument("MeasureSteering: the direction is empty or zero");
    if (o.random_directions < 0) throw std::invalid_argument("MeasureSteering: random_directions must not be negative");

    SteeringReport r;
    r.coefficients = o.coefficients;
    const size_t nc = o.coefficients.size();
    const std::vector<std::vector<double>> main = Sweep(behavior, num_inputs, position, direction, o.coefficients);
    r.mean_behavior.assign(nc, 0.0);
    int64_t anti = 0;
    for (const std::vector<double>& b : main) {
        for (size_t c = 0; c < nc; ++c) r.mean_behavior[c] += b[c] / static_cast<double>(num_inputs);
        const double s = Slope(o.coefficients, b);
        r.steerability.push_back(s);
        r.mean_steerability += s / static_cast<double>(num_inputs);
        anti += s < 0 ? 1 : 0;
    }
    for (double s : r.steerability) r.steerability_sd += (s - r.mean_steerability) * (s - r.mean_steerability);
    r.steerability_sd = std::sqrt(r.steerability_sd / static_cast<double>(num_inputs));
    r.anti_steerable_fraction = static_cast<double>(anti) / static_cast<double>(num_inputs);

    // The control: random directions of the same norm.
    r.random_mean_behavior.assign(nc, 0.0);
    PortableRng rng{o.seed};
    for (int64_t k = 0; k < o.random_directions; ++k) {
        std::vector<float> random(direction.size());
        double sq = 0;
        for (float& v : random) {
            v = static_cast<float>(rng.gaussian());
            sq += static_cast<double>(v) * v;
        }
        for (float& v : random) v = static_cast<float>(v * norm / std::sqrt(sq));
        const std::vector<std::vector<double>> b = Sweep(behavior, num_inputs, position, random, o.coefficients);
        double mean_slope = 0;
        for (const std::vector<double>& row : b) {
            for (size_t c = 0; c < nc; ++c) r.random_mean_behavior[c] += row[c] / static_cast<double>(num_inputs * o.random_directions);
            mean_slope += Slope(o.coefficients, row) / static_cast<double>(num_inputs);
        }
        r.random_mean_steerability += mean_slope / static_cast<double>(o.random_directions);
        r.random_max_abs_steerability = std::max(r.random_max_abs_steerability, std::abs(mean_slope));
    }
    r.over_random = r.random_max_abs_steerability > 0 ? r.mean_steerability / r.random_max_abs_steerability : 0.0;
    return r;
}

}  // namespace pulsatrix
