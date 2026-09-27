/** @file hyperband.hpp
 *  @brief Hyperband (Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, JMLR 2018): runs several
 *         Successive Halving "brackets" with different (num_configs, initial_budget)
 *         trade-offs -- covering the tension between "few configs trained long" and "many
 *         configs trained short, halved down" that a single Successive Halving run commits to
 *         in advance -- and returns the best result across every bracket.
 *  @ingroup hyperparameter_optimization
 *  @note Scope simplification, logged rather than silently assumed: the original paper's
 *        inner Successive Halving loop runs exactly (s+1) rounds per bracket, ending at
 *        precisely max_resource on its final rung (some brackets may finish with more than
 *        one surviving candidate at that point). This implementation instead runs each
 *        bracket via successive_halving.hpp's own RunSuccessiveHalving, which always halves
 *        down to exactly one survivor regardless of a resource cap -- for larger brackets
 *        (small s), this can train the final survivor somewhat past max_resource rather than
 *        stopping exactly there. The core idea Hyperband is actually named for -- explore the
 *        num_configs/initial_budget trade-off via multiple brackets, keep the best result
 *        across all of them -- is preserved exactly; only the inner loop's precise stopping
 *        point differs from the original paper's.
 */
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pulsatrix/successive_halving.hpp"

namespace pulsatrix {

/** @brief One bracket's own (num_configs, initial_budget) trade-off point; s is the bracket
 *         index (s_max = most configs/smallest budget, down to s=0 = fewest configs/largest
 *         budget, matching the original paper's own naming). */
struct HyperbandBracket {
    int s;
    size_t num_configs;
    int initial_budget;
};

/**
 * @brief Computes the classic Hyperband bracket schedule: s_max = floor(log_eta(max_resource)),
 *        B = (s_max + 1) * max_resource; for s from s_max down to 0, num_configs =
 *        ceil((B / max_resource) * (eta^s / (s + 1))), initial_budget = round(max_resource /
 *        eta^s) (each floored at 1).
 * @throws std::invalid_argument if max_resource <= 0 or eta <= 1.0.
 */
inline std::vector<HyperbandBracket> ComputeHyperbandBrackets(int max_resource, double eta) {
    if (max_resource <= 0) {
        throw std::invalid_argument("ComputeHyperbandBrackets: max_resource must be positive");
    }
    if (eta <= 1.0) {
        throw std::invalid_argument("ComputeHyperbandBrackets: eta must be > 1.0");
    }

    int s_max = static_cast<int>(std::floor(std::log(static_cast<double>(max_resource)) / std::log(eta)));
    double b = static_cast<double>(s_max + 1) * static_cast<double>(max_resource);

    std::vector<HyperbandBracket> brackets;
    for (int s = s_max; s >= 0; --s) {
        double eta_pow_s = std::pow(eta, s);
        size_t num_configs =
            std::max<size_t>(1, static_cast<size_t>(std::ceil((b / max_resource) * (eta_pow_s / (s + 1)))));
        int initial_budget = std::max(1, static_cast<int>(std::round(max_resource / eta_pow_s)));
        brackets.push_back(HyperbandBracket{s, num_configs, initial_budget});
    }
    return brackets;
}

/** @brief The best configuration/metric found across every bracket, and the total epoch
 *         budget spent summed across all of them. */
struct HyperbandResult {
    Configuration best_configuration;
    double best_metric;
    size_t total_epochs_trained;
};

/**
 * @brief Runs one Successive Halving bracket (successive_halving.hpp) per
 *        ComputeHyperbandBrackets(max_resource, eta), keeping the best result across all of
 *        them.
 * @throws std::invalid_argument -- propagated from ComputeHyperbandBrackets.
 */
template <typename RNG>
HyperbandResult RunHyperband(const SearchSpace& space, const TrialFactory& make_trial, int max_resource,
                              double eta, RNG& rng) {
    auto brackets = ComputeHyperbandBrackets(max_resource, eta);

    HyperbandResult overall{Configuration{}, -std::numeric_limits<double>::infinity(), 0};
    for (const auto& bracket : brackets) {
        auto result = RunSuccessiveHalving(space, make_trial, bracket.num_configs, bracket.initial_budget, eta, rng);
        overall.total_epochs_trained += result.total_epochs_trained;
        if (result.best_metric > overall.best_metric) {
            overall.best_metric = result.best_metric;
            overall.best_configuration = result.best_configuration;
        }
    }
    return overall;
}

}  // namespace pulsatrix
