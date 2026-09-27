/** @file pbt.hpp
 *  @brief Population Based Training (Jaderberg et al. 2017, "Population Based Training of
 *         Neural Networks"): a population of live, incrementally-trained trials periodically
 *         truncation-selected -- the bottom fraction exploits (copies weights and
 *         hyperparameters from a uniformly-randomly-chosen top performer) then explores
 *         (perturbs the copied hyperparameters) -- producing a hyperparameter *schedule*
 *         (different effective hyperparameters at different points in training) rather than a
 *         single fixed configuration.
 *  @ingroup evolutionary
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

#include "pulsatrix/pbt_trial.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {

/** @brief Indices of the population's current worst- and best-performing members. */
struct PBTTruncationGroups {
    std::vector<size_t> bottom_indices;
    std::vector<size_t> top_indices;
};

/**
 * @brief Pure core: identifies the bottom and top truncation_fraction of the population by
 *        metric (higher is better). Ties are broken by a stable sort on descending metric, so
 *        the earlier index among equal values sorts toward "top." At least one individual is
 *        always selected on each end, even if floor(size*fraction) would be 0.
 * @throws std::invalid_argument if metrics has fewer than 2 entries, or truncation_fraction is
 *         not in (0, 0.5].
 */
inline PBTTruncationGroups ComputeTruncationGroups(const std::vector<double>& metrics,
                                                    double truncation_fraction) {
    if (metrics.size() < 2) {
        throw std::invalid_argument("ComputeTruncationGroups: metrics must have at least 2 entries");
    }
    if (!(truncation_fraction > 0.0) || truncation_fraction > 0.5) {
        throw std::invalid_argument("ComputeTruncationGroups: truncation_fraction must be in (0, 0.5]");
    }

    size_t n = metrics.size();
    size_t num_selected = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(n) * truncation_fraction));

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return metrics[a] > metrics[b]; });

    PBTTruncationGroups groups;
    groups.top_indices.assign(order.begin(), order.begin() + static_cast<long>(num_selected));
    groups.bottom_indices.assign(order.end() - static_cast<long>(num_selected), order.end());
    return groups;
}

/**
 * @brief Pure core: applies an explicit per-parameter multiplicative factor to every
 *        Continuous/LogUniform/Integer parameter in config, clamped to that parameter's own
 *        bounds -- PBT's own "explore" step (Jaderberg et al.'s own simple perturbation:
 *        multiply by 0.8 or 1.2). Categorical parameters are left unchanged (explore, in its
 *        original form, perturbs numeric hyperparameters only -- a deliberate, logged scope
 *        decision, not an oversight). Integer results are rounded to the nearest integer.
 * @throws std::invalid_argument if config is missing a value for any parameter in space, or
 *         factors is missing an entry for any non-categorical parameter in space.
 */
inline Configuration ExploreConfigurationGivenFactors(const Configuration& config, const SearchSpace& space,
                                                       const std::map<std::string, double>& factors) {
    Configuration result = config;
    for (const auto& spec : space.parameters()) {
        auto config_it = config.find(spec.name);
        if (config_it == config.end()) {
            throw std::invalid_argument("ExploreConfigurationGivenFactors: config missing parameter '" +
                                         spec.name + "'");
        }
        if (spec.kind == ParameterKind::Categorical) {
            continue;
        }
        auto factor_it = factors.find(spec.name);
        if (factor_it == factors.end()) {
            throw std::invalid_argument("ExploreConfigurationGivenFactors: factors missing parameter '" +
                                         spec.name + "'");
        }
        double factor = factor_it->second;
        if (spec.kind == ParameterKind::Integer) {
            int64_t current = std::get<int64_t>(config_it->second);
            double perturbed = std::clamp(static_cast<double>(current) * factor, spec.lower, spec.upper);
            result[spec.name] = static_cast<int64_t>(std::llround(perturbed));
        } else {
            double current = std::get<double>(config_it->second);
            result[spec.name] = std::clamp(current * factor, spec.lower, spec.upper);
        }
    }
    return result;
}

/**
 * @brief RNG-driven wrapper: draws each non-categorical parameter's factor uniformly from
 *        {0.8, 1.2} (Jaderberg et al.'s own standard explore perturbation), then applies
 *        ExploreConfigurationGivenFactors.
 */
template <typename RNG>
Configuration ExploreConfiguration(const Configuration& config, const SearchSpace& space, RNG& rng) {
    std::bernoulli_distribution coin(0.5);
    std::map<std::string, double> factors;
    for (const auto& spec : space.parameters()) {
        if (spec.kind == ParameterKind::Categorical) {
            continue;
        }
        factors[spec.name] = coin(rng) ? 1.2 : 0.8;
    }
    return ExploreConfigurationGivenFactors(config, space, factors);
}

/**
 * @brief Runs one PBT generation: trains every live trial for num_epochs, then exploits+explores
 *        the bottom truncation_fraction of the population from a uniformly-randomly-chosen
 *        member of the top truncation_fraction. Individuals outside both groups are left
 *        running untouched. Returns each trial's metric as of this generation (post
 *        exploit/explore for any trial that was replaced) -- the value to feed into the next
 *        generation's own truncation.
 * @throws std::invalid_argument if trials is empty or num_epochs is not positive (delegates to
 *         ComputeTruncationGroups for its own preconditions once trials.size() >= 2).
 */
template <typename RNG>
std::vector<double> RunPBTGeneration(std::vector<std::unique_ptr<PBTResumableTrial>>& trials, const SearchSpace& space,
                                      int num_epochs, double truncation_fraction, RNG& rng) {
    if (trials.empty()) {
        throw std::invalid_argument("RunPBTGeneration: trials must not be empty");
    }
    if (num_epochs <= 0) {
        throw std::invalid_argument("RunPBTGeneration: num_epochs must be positive");
    }

    std::vector<double> metrics(trials.size());
    for (size_t i = 0; i < trials.size(); ++i) {
        metrics[i] = trials[i]->TrainForEpochs(num_epochs);
    }

    if (trials.size() < 2) {
        return metrics;  // nothing to exploit/explore against with a single trial
    }

    PBTTruncationGroups groups = ComputeTruncationGroups(metrics, truncation_fraction);
    std::uniform_int_distribution<size_t> pick_top(0, groups.top_indices.size() - 1);
    for (size_t bottom_index : groups.bottom_indices) {
        size_t source_index = groups.top_indices[pick_top(rng)];
        if (source_index == bottom_index) {
            continue;  // only possible if top/bottom overlap at a tiny population size
        }
        trials[bottom_index]->SetWeights(trials[source_index]->GetWeights());
        Configuration copied = trials[source_index]->GetHyperparameters();
        trials[bottom_index]->SetHyperparameters(ExploreConfiguration(copied, space, rng));
        metrics[bottom_index] = metrics[source_index];
    }

    return metrics;
}

/** @brief The best-performing trial's index, its metric, and how many generations ran. */
struct PBTResult {
    size_t best_trial_index;
    double best_metric;
    int generations_run;
};

/**
 * @brief Runs num_generations of RunPBTGeneration in sequence.
 * @throws std::invalid_argument if num_generations is not positive (delegates to
 *         RunPBTGeneration for its own preconditions).
 */
template <typename RNG>
PBTResult RunPBT(std::vector<std::unique_ptr<PBTResumableTrial>>& trials, const SearchSpace& space,
                  int num_generations, int epochs_per_generation, double truncation_fraction, RNG& rng) {
    if (num_generations <= 0) {
        throw std::invalid_argument("RunPBT: num_generations must be positive");
    }

    std::vector<double> metrics;
    for (int gen = 0; gen < num_generations; ++gen) {
        metrics = RunPBTGeneration(trials, space, epochs_per_generation, truncation_fraction, rng);
    }

    size_t best_index = static_cast<size_t>(std::max_element(metrics.begin(), metrics.end()) - metrics.begin());
    return PBTResult{best_index, metrics[best_index], num_generations};
}

}  // namespace pulsatrix
