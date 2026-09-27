/** @file gp_bo.hpp
 *  @brief The GP-BO trial loop: random-initialize, then repeatedly fit a GP to every trial
 *         observed so far and propose the next point by maximizing an acquisition function
 *         over random candidates in the unit hypercube.
 *  @ingroup hyperparameter_optimization
 *  @note GP-BO here supports Continuous/LogUniform parameters only -- Integer/Categorical
 *        parameters (mixed/conditional search spaces) are explicitly TPE's job (Phase 2
 *        Mission 2), not vanilla GP-BO's, per this campaign's own scope note. Attempting to
 *        run GP-BO over a SearchSpace containing an Integer/Categorical parameter throws.
 *  @note Internally operates entirely in the unit hypercube [0, 1]^d, not the SearchSpace's
 *        own (possibly wildly different per-dimension) bounds -- this is not just
 *        convenience: the GP kernel's single length_scale hyperparameter is only meaningfully
 *        comparable across dimensions if every dimension is on the same scale.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/acquisition_functions.hpp"
#include "pulsatrix/gaussian_process.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/trial.hpp"

namespace pulsatrix {

/** @brief Which acquisition function RunGPBOLoop maximizes over candidates each iteration. */
enum class AcquisitionKind { ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound };

/**
 * @brief Maps a unit-hypercube point (one value per parameter, each in [0, 1]) to a
 *        Configuration, using space's own declared bounds.
 * @throws std::invalid_argument if t.size() != space.size(), any t[i] is outside [0, 1], or
 *         space contains a parameter kind other than Continuous/LogUniform.
 */
inline Configuration UnitCubeToConfiguration(const SearchSpace& space, const std::vector<float>& t) {
    if (t.size() != space.size()) {
        throw std::invalid_argument("UnitCubeToConfiguration: t.size() must equal space.size()");
    }
    Configuration config;
    for (size_t i = 0; i < space.parameters().size(); ++i) {
        const auto& spec = space.parameters()[i];
        float u = t[i];
        if (u < 0.0f || u > 1.0f) {
            throw std::invalid_argument("UnitCubeToConfiguration: every t value must be in [0, 1]");
        }
        switch (spec.kind) {
            case ParameterKind::Continuous:
                config[spec.name] = spec.lower + static_cast<double>(u) * (spec.upper - spec.lower);
                break;
            case ParameterKind::LogUniform: {
                double log_lower = std::log(spec.lower);
                double log_upper = std::log(spec.upper);
                config[spec.name] = std::exp(log_lower + static_cast<double>(u) * (log_upper - log_lower));
                break;
            }
            default:
                throw std::invalid_argument(
                    "UnitCubeToConfiguration: GP-BO supports only Continuous/LogUniform parameters "
                    "(Integer/Categorical are TPE's job)");
        }
    }
    return config;
}

/**
 * @brief Runs GP-BO: num_initial_random uniformly-random trials, then num_iterations trials
 *        each chosen by fitting a GP to every trial so far and maximizing acquisition over
 *        num_candidates random points in the unit hypercube.
 * @param objective_fn Callable `double(const Configuration&)` -- the value to maximize (a
 *        caller minimizing a loss negates it first, per this file's maximization convention).
 * @return Every Trial run, in order (the initial random trials first, then each GP-BO-
 *         proposed trial), each with one "objective" metric recorded.
 * @throws std::invalid_argument if num_initial_random == 0, or (propagated from
 *         UnitCubeToConfiguration) space contains an Integer/Categorical parameter.
 */
template <typename ObjectiveFn, typename RNG>
std::vector<Trial> RunGPBOLoop(const SearchSpace& space, ObjectiveFn objective_fn,
                                size_t num_initial_random, size_t num_iterations,
                                AcquisitionKind acquisition, size_t num_candidates, RNG& rng) {
    if (num_initial_random == 0) {
        throw std::invalid_argument("RunGPBOLoop: num_initial_random must be >= 1");
    }

    std::vector<Trial> trials;
    std::vector<std::vector<float>> observed_points;
    std::vector<float> observed_values;
    std::uniform_real_distribution<float> unit_dist(0.0f, 1.0f);

    auto evaluate = [&](const std::vector<float>& t) {
        Configuration config = UnitCubeToConfiguration(space, t);
        Trial trial(config);
        double value = objective_fn(config);
        trial.RecordMetric("objective", value, 0);
        trials.push_back(std::move(trial));
        observed_points.push_back(t);
        observed_values.push_back(static_cast<float>(value));
    };

    for (size_t i = 0; i < num_initial_random; ++i) {
        std::vector<float> t(space.size());
        for (auto& v : t) {
            v = unit_dist(rng);
        }
        evaluate(t);
    }

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        GaussianProcessRegressor gp(/*sigma_f=*/1.0f, /*length_scale=*/0.3f, /*noise_variance=*/1e-6f);
        gp.Fit(observed_points, observed_values);

        float best_value = *std::max_element(observed_values.begin(), observed_values.end());

        std::vector<float> best_candidate;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < num_candidates; ++c) {
            std::vector<float> candidate(space.size());
            for (auto& v : candidate) {
                v = unit_dist(rng);
            }
            auto posterior = gp.Predict(candidate);
            double score = 0.0;
            switch (acquisition) {
                case AcquisitionKind::ExpectedImprovement:
                    score = ExpectedImprovement(posterior.mean, posterior.variance, best_value);
                    break;
                case AcquisitionKind::ProbabilityOfImprovement:
                    score = ProbabilityOfImprovement(posterior.mean, posterior.variance, best_value);
                    break;
                case AcquisitionKind::UpperConfidenceBound:
                    score = UpperConfidenceBound(posterior.mean, posterior.variance);
                    break;
            }
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
