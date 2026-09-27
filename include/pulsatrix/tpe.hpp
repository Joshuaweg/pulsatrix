/** @file tpe.hpp
 *  @brief Tree-structured Parzen Estimator (Bergstra, Bardenet, Bengio, Kegl, "Algorithms for
 *         Hyper-Parameter Optimization," NeurIPS 2011) -- a structurally distinct second
 *         surrogate family from GP-BO (gp_bo.hpp), handling mixed/categorical search spaces
 *         GP-BO's own vanilla kernel cannot (gp_bo.hpp restricts to Continuous/LogUniform;
 *         this file supports every ParameterKind).
 *  @ingroup hyperparameter_optimization
 *  @note Scope simplifications from the original paper, both logged here rather than left
 *        implicit: (1) per-parameter densities use a *fixed*-bandwidth Gaussian mixture
 *        (bandwidth derived once from each parameter's own declared range), not the original
 *        paper's adaptive per-observation bandwidth (spacing to each point's nearest
 *        neighbors); (2) candidates are proposed by drawing uniformly from the SearchSpace's
 *        own prior (RandomSample, hpo_sampling.hpp) and scoring each by l(x)/g(x), not by
 *        drawing directly from the fitted l(x) density itself (the original paper's actual
 *        strategy, which requires being able to sample from an arbitrary fitted mixture, not
 *        just evaluate its density). Both are well-precedented, simpler variants that
 *        preserve TPE's defining idea (score candidates by a good/bad density ratio, built
 *        independently per parameter so mixed/categorical spaces compose naturally) without
 *        the original paper's full generative-sampling machinery.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/trial.hpp"

namespace pulsatrix {

/**
 * @brief Fixed-bandwidth Gaussian KDE: density(x) = mean over every observation o of
 *        N(x; o, bandwidth^2).
 * @throws std::invalid_argument if observations is empty or bandwidth <= 0.
 */
inline double GaussianKdeDensity(const std::vector<double>& observations, double bandwidth, double x) {
    if (observations.empty()) {
        throw std::invalid_argument("GaussianKdeDensity: observations must not be empty");
    }
    if (bandwidth <= 0.0) {
        throw std::invalid_argument("GaussianKdeDensity: bandwidth must be positive");
    }
    constexpr double kInvSqrt2Pi = 0.3989422804014327;
    double sum = 0.0;
    for (double obs : observations) {
        double z = (x - obs) / bandwidth;
        sum += kInvSqrt2Pi * std::exp(-0.5 * z * z) / bandwidth;
    }
    return sum / static_cast<double>(observations.size());
}

/**
 * @brief Laplace(add-one)-smoothed empirical probability of category among observations.
 * @throws std::invalid_argument if num_categories == 0.
 * @note observations may be empty (probability reduces to the uniform prior 1/num_categories).
 */
inline double CategoricalDensity(const std::vector<std::string>& observations, size_t num_categories,
                                  const std::string& category) {
    if (num_categories == 0) {
        throw std::invalid_argument("CategoricalDensity: num_categories must be positive");
    }
    size_t count = static_cast<size_t>(std::count(observations.begin(), observations.end(), category));
    return (static_cast<double>(count) + 1.0) /
           (static_cast<double>(observations.size()) + static_cast<double>(num_categories));
}

/**
 * @brief log(l(candidate)) - log(g(candidate)): the TPE scoring function, summed
 *        independently over every parameter in space (so a candidate's mixed
 *        continuous/categorical parameters each contribute their own term, composing
 *        naturally rather than needing a joint density over the whole space).
 * @param good_configs,bad_configs Observed configurations split by objective quantile
 *        (RunTPELoop's own job); every config must contain every parameter named in space.
 * @throws std::invalid_argument if good_configs or bad_configs is empty.
 * @note Per-parameter bandwidth for Continuous is 0.2 * (upper - lower); for LogUniform,
 *       0.2 * (log(upper) - log(lower)), applied after taking the log of both the bandwidth
 *       basis and every observed/candidate value (consistent with this campaign's own
 *       established log-space convention for LogUniform elsewhere -- GridSample/RandomSample/
 *       UnitCubeToConfiguration); for Integer, 0.2 * (upper - lower), floored at 1.0 (a
 *       sub-1.0 bandwidth over integer-valued data would make the KDE unreasonably peaked).
 */
inline double LogDensityRatio(const SearchSpace& space, const Configuration& candidate,
                               const std::vector<Configuration>& good_configs,
                               const std::vector<Configuration>& bad_configs) {
    if (good_configs.empty() || bad_configs.empty()) {
        throw std::invalid_argument("LogDensityRatio: good_configs and bad_configs must both be non-empty");
    }

    double log_ratio = 0.0;
    for (const auto& spec : space.parameters()) {
        if (spec.kind == ParameterKind::Categorical) {
            std::vector<std::string> good_values, bad_values;
            for (const auto& c : good_configs) good_values.push_back(std::get<std::string>(c.at(spec.name)));
            for (const auto& c : bad_configs) bad_values.push_back(std::get<std::string>(c.at(spec.name)));
            std::string candidate_value = std::get<std::string>(candidate.at(spec.name));

            double good_density = CategoricalDensity(good_values, spec.categories.size(), candidate_value);
            double bad_density = CategoricalDensity(bad_values, spec.categories.size(), candidate_value);
            log_ratio += std::log(good_density) - std::log(bad_density);
            continue;
        }

        // Continuous, LogUniform, Integer -- all evaluated as a 1D KDE, LogUniform taken in
        // log-space, Integer floored at bandwidth 1.0.
        std::vector<double> good_values, bad_values;
        double candidate_value;
        double bandwidth;
        if (spec.kind == ParameterKind::LogUniform) {
            for (const auto& c : good_configs) good_values.push_back(std::log(std::get<double>(c.at(spec.name))));
            for (const auto& c : bad_configs) bad_values.push_back(std::log(std::get<double>(c.at(spec.name))));
            candidate_value = std::log(std::get<double>(candidate.at(spec.name)));
            bandwidth = 0.2 * (std::log(spec.upper) - std::log(spec.lower));
        } else if (spec.kind == ParameterKind::Integer) {
            for (const auto& c : good_configs) good_values.push_back(static_cast<double>(std::get<int64_t>(c.at(spec.name))));
            for (const auto& c : bad_configs) bad_values.push_back(static_cast<double>(std::get<int64_t>(c.at(spec.name))));
            candidate_value = static_cast<double>(std::get<int64_t>(candidate.at(spec.name)));
            bandwidth = std::max(1.0, 0.2 * (spec.upper - spec.lower));
        } else {
            for (const auto& c : good_configs) good_values.push_back(std::get<double>(c.at(spec.name)));
            for (const auto& c : bad_configs) bad_values.push_back(std::get<double>(c.at(spec.name)));
            candidate_value = std::get<double>(candidate.at(spec.name));
            bandwidth = 0.2 * (spec.upper - spec.lower);
        }

        double good_density = GaussianKdeDensity(good_values, bandwidth, candidate_value);
        double bad_density = GaussianKdeDensity(bad_values, bandwidth, candidate_value);
        log_ratio += std::log(good_density) - std::log(bad_density);
    }
    return log_ratio;
}

/**
 * @brief Runs TPE: num_initial_random uniformly-random trials (RandomSample), then
 *        num_iterations trials each chosen by splitting all trials so far into good/bad by
 *        the gamma quantile (maximization convention: good = highest objective values) and
 *        picking, among num_candidates uniformly-random candidates, the one maximizing
 *        LogDensityRatio.
 * @param gamma Fraction of trials-so-far classified "good" (e.g. 0.2 = top 20%); at least one
 *        trial is always classified good and at least one bad, regardless of gamma, once
 *        num_initial_random >= 2.
 * @throws std::invalid_argument if num_initial_random < 2 (a quantile split needs at least
 *         one good and one bad observation), or gamma is outside (0, 1).
 */
template <typename ObjectiveFn, typename RNG>
std::vector<Trial> RunTPELoop(const SearchSpace& space, ObjectiveFn objective_fn,
                               size_t num_initial_random, size_t num_iterations, double gamma,
                               size_t num_candidates, RNG& rng) {
    if (num_initial_random < 2) {
        throw std::invalid_argument("RunTPELoop: num_initial_random must be >= 2");
    }
    if (gamma <= 0.0 || gamma >= 1.0) {
        throw std::invalid_argument("RunTPELoop: gamma must be in (0, 1)");
    }

    std::vector<Trial> trials;

    auto evaluate = [&](const Configuration& config) {
        Trial trial(config);
        double value = objective_fn(config);
        trial.RecordMetric("objective", value, 0);
        trials.push_back(std::move(trial));
    };

    for (size_t i = 0; i < num_initial_random; ++i) {
        evaluate(RandomSample(space, rng));
    }

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        std::vector<size_t> order(trials.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return *trials[a].LatestMetric("objective") > *trials[b].LatestMetric("objective");
        });

        size_t num_good = std::max<size_t>(1, static_cast<size_t>(gamma * static_cast<double>(order.size())));
        num_good = std::min(num_good, order.size() - 1);

        std::vector<Configuration> good_configs, bad_configs;
        for (size_t i = 0; i < num_good; ++i) good_configs.push_back(trials[order[i]].configuration());
        for (size_t i = num_good; i < order.size(); ++i) bad_configs.push_back(trials[order[i]].configuration());

        Configuration best_candidate;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < num_candidates; ++c) {
            Configuration candidate = RandomSample(space, rng);
            double score = LogDensityRatio(space, candidate, good_configs, bad_configs);
            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }
        evaluate(best_candidate);
    }

    return trials;
}

}  // namespace pulsatrix
