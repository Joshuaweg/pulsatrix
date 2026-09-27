/** @file hpo_sampling.hpp
 *  @brief Grid search and random search over a SearchSpace: the two simplest, baseline
 *         hyperparameter-optimization algorithms, proving the SearchSpace/Trial abstraction's
 *         instantiate/train/read-back-metric loop end-to-end before Phase 2's GP-BO/TPE.
 *  @ingroup hyperparameter_optimization
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/search_space.hpp"

namespace pulsatrix {

/**
 * @brief Draws one configuration uniformly at random from space: Continuous parameters
 *        uniform over [lower, upper]; LogUniform parameters uniform in log-space (so e.g.
 *        [0.001, 1.0] gives 0.001-0.01, 0.01-0.1, and 0.1-1.0 equal probability, not the
 *        top decade 90% of the draws); Integer parameters uniform over the inclusive integer
 *        range; Categorical parameters uniform over the category list.
 */
template <typename RNG>
Configuration RandomSample(const SearchSpace& space, RNG& rng) {
    Configuration config;
    for (const auto& spec : space.parameters()) {
        switch (spec.kind) {
            case ParameterKind::Continuous: {
                std::uniform_real_distribution<double> dist(spec.lower, spec.upper);
                config[spec.name] = dist(rng);
                break;
            }
            case ParameterKind::LogUniform: {
                std::uniform_real_distribution<double> dist(std::log(spec.lower), std::log(spec.upper));
                config[spec.name] = std::exp(dist(rng));
                break;
            }
            case ParameterKind::Integer: {
                std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(spec.lower),
                                                              static_cast<int64_t>(spec.upper));
                config[spec.name] = dist(rng);
                break;
            }
            case ParameterKind::Categorical: {
                std::uniform_int_distribution<size_t> dist(0, spec.categories.size() - 1);
                config[spec.name] = spec.categories[dist(rng)];
                break;
            }
        }
    }
    return config;
}

/**
 * @brief Enumerates the full cartesian-product grid over space.
 * @param points_per_continuous_dimension Number of points (inclusive of both bounds) used
 *        for every Continuous (linearly spaced) and LogUniform (log-spaced) parameter.
 *        Integer parameters always enumerate every integer in [lower, upper]; Categorical
 *        parameters always enumerate every category -- neither is affected by this
 *        parameter.
 * @throws std::invalid_argument if points_per_continuous_dimension < 2 (fewer than 2 points
 *         cannot include both bounds).
 * @note Grid size is the product of every parameter's own point count -- grows combinatorially
 *       with both the number of parameters and points_per_continuous_dimension; the caller's
 *       responsibility to keep the search space small enough for grid search specifically
 *       (this is grid search's own well-known scaling limit, not a bug).
 */
inline std::vector<Configuration> GridSample(const SearchSpace& space,
                                              size_t points_per_continuous_dimension) {
    if (points_per_continuous_dimension < 2) {
        throw std::invalid_argument("GridSample: points_per_continuous_dimension must be >= 2");
    }

    std::vector<std::vector<ConfigValue>> axis_values;
    axis_values.reserve(space.parameters().size());
    for (const auto& spec : space.parameters()) {
        std::vector<ConfigValue> values;
        const double n_minus_1 = static_cast<double>(points_per_continuous_dimension - 1);
        switch (spec.kind) {
            case ParameterKind::Continuous: {
                for (size_t i = 0; i < points_per_continuous_dimension; ++i) {
                    double t = static_cast<double>(i) / n_minus_1;
                    values.push_back(spec.lower + t * (spec.upper - spec.lower));
                }
                break;
            }
            case ParameterKind::LogUniform: {
                double log_lower = std::log(spec.lower);
                double log_upper = std::log(spec.upper);
                for (size_t i = 0; i < points_per_continuous_dimension; ++i) {
                    double t = static_cast<double>(i) / n_minus_1;
                    values.push_back(std::exp(log_lower + t * (log_upper - log_lower)));
                }
                break;
            }
            case ParameterKind::Integer: {
                for (int64_t v = static_cast<int64_t>(spec.lower); v <= static_cast<int64_t>(spec.upper); ++v) {
                    values.push_back(v);
                }
                break;
            }
            case ParameterKind::Categorical: {
                for (const auto& category : spec.categories) {
                    values.push_back(category);
                }
                break;
            }
        }
        axis_values.push_back(std::move(values));
    }

    std::vector<Configuration> grid;
    grid.emplace_back();
    for (size_t param_idx = 0; param_idx < space.parameters().size(); ++param_idx) {
        const auto& name = space.parameters()[param_idx].name;
        const auto& values = axis_values[param_idx];
        std::vector<Configuration> next_grid;
        next_grid.reserve(grid.size() * values.size());
        for (const auto& partial : grid) {
            for (const auto& value : values) {
                Configuration extended = partial;
                extended[name] = value;
                next_grid.push_back(std::move(extended));
            }
        }
        grid = std::move(next_grid);
    }
    return grid;
}

}  // namespace pulsatrix
