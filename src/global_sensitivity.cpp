#include "pulsatrix/global_sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>

namespace pulsatrix {
namespace {

// A platform-independent generator: std::mt19937_64's output is fixed by the standard, the
// std:: distributions are not.
class Rng {
public:
    explicit Rng(uint64_t seed) : engine_(seed) {}
    double uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }
    uint64_t below(uint64_t n) { return engine_() % n; }

private:
    std::mt19937_64 engine_;
};

// The standard normal quantile (Acklam's rational approximation, relative error < 1.2e-9).
double NormalQuantile(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01,  -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double low = 0.02425;
    if (p < low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > 1.0 - low) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

double StdDdof1(const std::vector<double>& v) {
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double ss = 0.0;
    for (double x : v) ss += (x - mean) * (x - mean);
    return std::sqrt(ss / static_cast<double>(v.size() - 1));
}

void CheckProblem(const GlobalSensitivityProblem& p, const char* where) {
    const size_t d = p.features.size();
    if (d == 0) {
        throw std::invalid_argument(std::string(where) + ": no features to vary");
    }
    if (p.lower.size() != d || p.upper.size() != d) {
        throw std::invalid_argument(std::string(where) + ": lower and upper need one value per varied feature");
    }
    std::set<int64_t> seen;
    for (size_t i = 0; i < d; ++i) {
        if (p.features[i] < 0 || p.features[i] >= p.base.numel()) {
            throw std::invalid_argument(std::string(where) + ": feature index " + std::to_string(p.features[i]) +
                                        " is out of range");
        }
        if (!seen.insert(p.features[i]).second) {
            throw std::invalid_argument(std::string(where) + ": a feature is listed twice");
        }
        if (!(p.lower[i] < p.upper[i])) {
            throw std::invalid_argument(std::string(where) + ": every range needs lower < upper");
        }
    }
}

void CheckStatistics(int num_resamples, float confidence, const char* where) {
    if (num_resamples < 2 || !(confidence > 0.0f && confidence < 1.0f)) {
        throw std::invalid_argument(std::string(where) + ": needs num_resamples >= 2 and 0 < confidence < 1");
    }
}

// The target output for every row, with the varied features set from the row.
std::vector<float> Evaluate(const std::function<Tensor(const Tensor&)>& predict, const GlobalSensitivityProblem& p,
                            int64_t target, const std::vector<std::vector<float>>& rows, const char* where) {
    std::vector<float> values = p.base.to_host_vector();
    std::vector<float> out;
    out.reserve(rows.size());
    for (const auto& row : rows) {
        for (size_t i = 0; i < p.features.size(); ++i) {
            values[static_cast<size_t>(p.features[i])] = row[i];
        }
        Tensor y = predict(Tensor(p.base.shape(), p.base.backend(), values, p.base.device()));
        if (target < 0 || target >= y.numel()) {
            throw std::invalid_argument(std::string(where) + ": target index " + std::to_string(target) +
                                        " is out of range for an output of " + std::to_string(y.numel()));
        }
        out.push_back(y.read_element(target));
    }
    return out;
}

}  // namespace

// ---- Morris ---------------------------------------------------------------------------------

std::vector<std::vector<float>> MorrisSample(const GlobalSensitivityProblem& problem, const MorrisOptions& options) {
    CheckProblem(problem, "MorrisSample");
    if (options.num_trajectories < 2 || options.num_levels < 2 || options.num_levels % 2 != 0) {
        throw std::invalid_argument("MorrisSample: needs num_trajectories >= 2 and an even num_levels >= 2");
    }
    const size_t d = problem.features.size();
    const int p = options.num_levels;
    const double delta = p / (2.0 * (p - 1));
    Rng rng(options.seed);
    std::vector<std::vector<float>> rows;
    for (int64_t t = 0; t < options.num_trajectories; ++t) {
        // A start on the lower half of the grid, a direction per feature, and a random order.
        std::vector<double> x(d);
        std::vector<int> dir(d);
        for (size_t j = 0; j < d; ++j) {
            const auto level = static_cast<double>(rng.below(static_cast<uint64_t>(p / 2)));
            dir[j] = rng.below(2) == 0 ? 1 : -1;
            x[j] = level / (p - 1) + (dir[j] < 0 ? delta : 0.0);
        }
        std::vector<size_t> order(d);
        for (size_t j = 0; j < d; ++j) order[j] = j;
        for (size_t j = d; j > 1; --j) std::swap(order[j - 1], order[rng.below(j)]);  // Fisher-Yates

        auto emit = [&]() {
            std::vector<float> row(d);
            for (size_t j = 0; j < d; ++j) {
                row[j] = static_cast<float>(problem.lower[j] + x[j] * (problem.upper[j] - problem.lower[j]));
            }
            rows.push_back(std::move(row));
        };
        emit();
        for (size_t j : order) {
            x[j] += dir[j] * delta;
            emit();
        }
    }
    return rows;
}

MorrisResult AnalyzeMorris(const std::vector<std::vector<float>>& samples, const std::vector<float>& outputs,
                           const MorrisOptions& options) {
    CheckStatistics(options.num_resamples, options.confidence, "AnalyzeMorris");
    if (options.num_levels < 2) {
        throw std::invalid_argument("AnalyzeMorris: num_levels must be at least 2");
    }
    if (samples.empty() || samples.size() != outputs.size()) {
        throw std::invalid_argument("AnalyzeMorris: needs one output per sample row");
    }
    const size_t d = samples.front().size();
    const size_t traj = d + 1;
    if (d == 0 || samples.size() % traj != 0) {
        throw std::invalid_argument("AnalyzeMorris: the rows don't form whole trajectories of num_features + 1 rows");
    }
    const size_t r = samples.size() / traj;
    if (r < 2) {
        throw std::invalid_argument("AnalyzeMorris: needs at least 2 trajectories");
    }
    const double delta = options.num_levels / (2.0 * (options.num_levels - 1));
    std::vector<std::vector<double>> ee(d, std::vector<double>(r, 0.0));
    for (size_t t = 0; t < r; ++t) {
        std::vector<bool> done(d, false);
        for (size_t k = 0; k < d; ++k) {
            const auto& a = samples[t * traj + k];
            const auto& b = samples[t * traj + k + 1];
            if (a.size() != d || b.size() != d) {
                throw std::invalid_argument("AnalyzeMorris: rows differ in length");
            }
            size_t changed = d;
            for (size_t j = 0; j < d; ++j) {
                if (a[j] != b[j]) {
                    if (changed != d) throw std::invalid_argument("AnalyzeMorris: a step changes more than one feature");
                    changed = j;
                }
            }
            if (changed == d || done[changed]) {
                throw std::invalid_argument("AnalyzeMorris: each step must change exactly one new feature");
            }
            done[changed] = true;
            const double dy = static_cast<double>(outputs[t * traj + k + 1]) - outputs[t * traj + k];
            ee[changed][t] = (b[changed] > a[changed] ? dy : -dy) / delta;
        }
    }
    MorrisResult res;
    res.num_trajectories = static_cast<int64_t>(r);
    const double z = NormalQuantile(0.5 + options.confidence / 2.0);
    Rng rng(options.seed);
    for (size_t j = 0; j < d; ++j) {
        res.features.push_back(static_cast<int64_t>(j));
        double mean = 0.0, mean_abs = 0.0;
        for (double e : ee[j]) {
            mean += e;
            mean_abs += std::fabs(e);
        }
        mean /= static_cast<double>(r);
        mean_abs /= static_cast<double>(r);
        res.mu.push_back(static_cast<float>(mean));
        res.mu_star.push_back(static_cast<float>(mean_abs));
        res.sigma.push_back(static_cast<float>(StdDdof1(ee[j])));
        std::vector<double> boot(static_cast<size_t>(options.num_resamples));
        for (double& m : boot) {
            double s = 0.0;
            for (size_t i = 0; i < r; ++i) s += std::fabs(ee[j][rng.below(r)]);
            m = s / static_cast<double>(r);
        }
        res.mu_star_conf.push_back(static_cast<float>(z * StdDdof1(boot)));
    }
    return res;
}

MorrisResult ComputeMorris(const std::function<Tensor(const Tensor&)>& predict, const GlobalSensitivityProblem& problem,
                           int64_t target_index, const MorrisOptions& options) {
    const auto rows = MorrisSample(problem, options);
    MorrisResult r = AnalyzeMorris(rows, Evaluate(predict, problem, target_index, rows, "ComputeMorris"), options);
    r.features = problem.features;
    return r;
}

// ---- Sobol ----------------------------------------------------------------------------------

SobolResult AnalyzeSobol(const std::vector<float>& outputs, int64_t num_features, const SobolOptions& options) {
    CheckStatistics(options.num_resamples, options.confidence, "AnalyzeSobol");
    if (num_features < 1) {
        throw std::invalid_argument("AnalyzeSobol: num_features must be at least 1");
    }
    const auto d = static_cast<size_t>(num_features);
    const size_t step = d + 2;
    if (outputs.empty() || outputs.size() % step != 0) {
        throw std::invalid_argument("AnalyzeSobol: the outputs don't form whole blocks of num_features + 2");
    }
    const size_t n = outputs.size() / step;
    // Standardize, as SALib does: (Y - mean) / std with ddof 0.
    double mean = 0.0;
    for (float y : outputs) mean += y;
    mean /= static_cast<double>(outputs.size());
    double var = 0.0;
    for (float y : outputs) var += (y - mean) * (y - mean);
    const double sd = std::sqrt(var / static_cast<double>(outputs.size()));
    auto Y = [&](size_t block, size_t slot) { return sd > 0 ? (outputs[block * step + slot] - mean) / sd : 0.0; };

    // Indices over a list of blocks (all of them, or a bootstrap resample).
    auto indices = [&](const std::vector<size_t>& blocks, size_t j, double& s1, double& st) {
        double m = 0.0;
        for (size_t b : blocks) m += Y(b, 0) + Y(b, step - 1);
        m /= 2.0 * static_cast<double>(blocks.size());
        double v = 0.0, num1 = 0.0, numt = 0.0;
        for (size_t b : blocks) {
            const double a = Y(b, 0), bb = Y(b, step - 1), ab = Y(b, j + 1);
            v += (a - m) * (a - m) + (bb - m) * (bb - m);
            num1 += bb * (ab - a);
            numt += (a - ab) * (a - ab);
        }
        v /= 2.0 * static_cast<double>(blocks.size());
        const double k = static_cast<double>(blocks.size());
        s1 = v > 0 ? (num1 / k) / v : 0.0;
        st = v > 0 ? 0.5 * (numt / k) / v : 0.0;
    };

    SobolResult res;
    res.num_samples = static_cast<int64_t>(n);
    const double z = NormalQuantile(0.5 + options.confidence / 2.0);
    std::vector<size_t> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = i;
    // One set of resamples shared by every feature, as SALib draws r once.
    Rng rng(options.seed);
    std::vector<std::vector<size_t>> resamples(static_cast<size_t>(options.num_resamples), std::vector<size_t>(n));
    for (auto& rs : resamples) {
        for (size_t& b : rs) b = rng.below(n);
    }
    for (size_t j = 0; j < d; ++j) {
        double s1 = 0, st = 0;
        indices(all, j, s1, st);
        std::vector<double> b1, bt;
        for (const auto& rs : resamples) {
            double x1 = 0, xt = 0;
            indices(rs, j, x1, xt);
            b1.push_back(x1);
            bt.push_back(xt);
        }
        res.features.push_back(static_cast<int64_t>(j));
        res.first_order.push_back(static_cast<float>(s1));
        res.total_order.push_back(static_cast<float>(st));
        res.first_order_conf.push_back(static_cast<float>(z * StdDdof1(b1)));
        res.total_order_conf.push_back(static_cast<float>(z * StdDdof1(bt)));
    }
    return res;
}

SobolResult ComputeSobol(const std::function<Tensor(const Tensor&)>& predict, const GlobalSensitivityProblem& problem,
                         int64_t target_index, const SobolOptions& options) {
    CheckProblem(problem, "ComputeSobol");
    if (options.num_samples < 2) {
        throw std::invalid_argument("ComputeSobol: num_samples must be at least 2");
    }
    const size_t d = problem.features.size();
    Rng rng(options.seed);
    auto draw = [&]() {
        std::vector<float> row(d);
        for (size_t j = 0; j < d; ++j) {
            row[j] = static_cast<float>(problem.lower[j] + rng.uniform() * (problem.upper[j] - problem.lower[j]));
        }
        return row;
    };
    std::vector<std::vector<float>> rows;
    rows.reserve(static_cast<size_t>(options.num_samples) * (d + 2));
    for (int64_t i = 0; i < options.num_samples; ++i) {
        const std::vector<float> a = draw();
        const std::vector<float> b = draw();
        rows.push_back(a);
        for (size_t j = 0; j < d; ++j) {
            std::vector<float> ab = a;
            ab[j] = b[j];
            rows.push_back(std::move(ab));
        }
        rows.push_back(b);
    }
    SobolResult r = AnalyzeSobol(Evaluate(predict, problem, target_index, rows, "ComputeSobol"),
                                 static_cast<int64_t>(d), options);
    r.features = problem.features;
    return r;
}

}  // namespace pulsatrix
