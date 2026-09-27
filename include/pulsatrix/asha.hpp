/** @file asha.hpp
 *  @brief ASHA (Asynchronous Successive Halving Algorithm; Li, Jamieson, Rostamizadeh,
 *         Gonina, Hardt, Recht, Talwalkar, "A System for Massively Parallel Hyperparameter
 *         Tuning," MLSys 2020): unlike synchronous Successive Halving/Hyperband (which
 *         process one full rung across every candidate before any candidate advances to the
 *         next), ASHA promotes a candidate to the next rung as soon as it qualifies (its
 *         metric is in the top 1/eta fraction of every candidate that has *ever* completed
 *         that rung so far), never waiting on the rest of the current rung's population.
 *  @ingroup hyperparameter_optimization
 *  @note This codebase's HPO loops are all single-threaded/sequential (no genuine parallel
 *        workers) -- what "asynchronous" means concretely here is the *algorithmic*
 *        distinction from synchronous Successive Halving: promotion decisions are made
 *        opportunistically against the current state of every rung, one action (promote one
 *        candidate, or start one new one) per step, never gated on a full population
 *        finishing a rung together. This is the real mechanism ASHA is named for, independent
 *        of whether it runs across literal parallel workers or one sequential loop.
 */
#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pulsatrix/successive_halving.hpp"

namespace pulsatrix {

/** @brief The best configuration/metric found, the total epoch budget spent, and how many
 *         distinct configurations were ever started (as opposed to promoted). */
struct ASHAResult {
    Configuration best_configuration;
    double best_metric;
    size_t total_epochs_trained;
    size_t num_configs_started;
};

namespace detail {
struct ASHACandidate {
    Configuration config;
    std::unique_ptr<ResumableTrial> trial;
    int rung;
    double metric;
};
}  // namespace detail

/**
 * @brief Runs ASHA, drawing new configurations from an explicit, ordered queue (rather than
 *        sampling indefinitely) -- the pure, deterministic core; RunASHA (below) is the thin
 *        SearchSpace/RNG-sampling wrapper around it.
 * @param initial_epoch_budget Rung 0's epoch budget; rung k's budget is
 *        initial_epoch_budget * eta^k.
 * @param num_rungs Total number of rungs (>= 2 -- at least one promotion opportunity must
 *        exist).
 * @note Each step does exactly one of: (a) promote the single best-qualifying candidate at
 *       the highest rung with a promotable candidate (scanned top-down, so nearly-finished
 *       candidates are preferred over starting fresh ones -- ASHA's own stated preference),
 *       or (b) if no promotion is possible, start the next queued configuration at rung 0.
 *       A rung only becomes eligible for promotion decisions once at least eta candidates
 *       have completed it (not enough data to identify a meaningful top fraction before
 *       that).
 * @throws std::invalid_argument if config_queue is empty, initial_epoch_budget <= 0, eta <=
 *         1.0, or num_rungs < 2.
 */
inline ASHAResult RunASHAOnConfigQueue(std::vector<Configuration> config_queue,
                                       const TrialFactory& make_trial, int initial_epoch_budget,
                                       double eta, int num_rungs) {
    if (config_queue.empty()) {
        throw std::invalid_argument("RunASHAOnConfigQueue: config_queue must not be empty");
    }
    if (initial_epoch_budget <= 0) {
        throw std::invalid_argument("RunASHAOnConfigQueue: initial_epoch_budget must be positive");
    }
    if (eta <= 1.0) {
        throw std::invalid_argument("RunASHAOnConfigQueue: eta must be > 1.0");
    }
    if (num_rungs < 2) {
        throw std::invalid_argument("RunASHAOnConfigQueue: num_rungs must be >= 2");
    }

    std::vector<int> budgets(static_cast<size_t>(num_rungs));
    budgets[0] = initial_epoch_budget;
    for (int i = 1; i < num_rungs; ++i) {
        budgets[static_cast<size_t>(i)] = static_cast<int>(budgets[static_cast<size_t>(i - 1)] * eta);
    }

    std::vector<std::vector<double>> rung_history(static_cast<size_t>(num_rungs));
    std::vector<detail::ASHACandidate> candidates;
    size_t queue_pos = 0;
    size_t total_epochs_trained = 0;

    while (true) {
        int promote_rung = -1;
        int promote_idx = -1;

        for (int k = num_rungs - 2; k >= 0; --k) {
            auto& history = rung_history[static_cast<size_t>(k)];
            if (static_cast<double>(history.size()) < eta) {
                continue;
            }
            std::vector<double> sorted_desc = history;
            std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<double>());
            size_t keep = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(sorted_desc.size()) / eta));
            double cutoff = sorted_desc[keep - 1];

            int local_best_idx = -1;
            double local_best_metric = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (candidates[i].rung == k && candidates[i].metric >= cutoff) {
                    if (local_best_idx == -1 || candidates[i].metric > local_best_metric) {
                        local_best_idx = static_cast<int>(i);
                        local_best_metric = candidates[i].metric;
                    }
                }
            }
            if (local_best_idx != -1) {
                promote_rung = k;
                promote_idx = local_best_idx;
                break;  // highest-k promotable candidate wins -- prefer finishing over starting fresh
            }
        }

        if (promote_idx != -1) {
            int k = promote_rung;
            int additional = budgets[static_cast<size_t>(k + 1)] - budgets[static_cast<size_t>(k)];
            double new_metric = candidates[static_cast<size_t>(promote_idx)].trial->TrainForEpochs(additional);
            total_epochs_trained += static_cast<size_t>(additional);
            candidates[static_cast<size_t>(promote_idx)].metric = new_metric;
            candidates[static_cast<size_t>(promote_idx)].rung = k + 1;
            rung_history[static_cast<size_t>(k + 1)].push_back(new_metric);
            continue;
        }

        if (queue_pos < config_queue.size()) {
            Configuration config = std::move(config_queue[queue_pos++]);
            auto trial = make_trial(config);
            double metric = trial->TrainForEpochs(budgets[0]);
            total_epochs_trained += static_cast<size_t>(budgets[0]);
            rung_history[0].push_back(metric);
            candidates.push_back(detail::ASHACandidate{std::move(config), std::move(trial), 0, metric});
            continue;
        }

        break;  // nothing left to promote or start
    }

    auto best_it = std::max_element(
        candidates.begin(), candidates.end(),
        [](const detail::ASHACandidate& a, const detail::ASHACandidate& b) { return a.metric < b.metric; });
    return ASHAResult{best_it->config, best_it->metric, total_epochs_trained, queue_pos};
}

/**
 * @brief RNG-driven wrapper: draws max_configs_started configurations from space via
 *        RandomSample to serve as the queue, then runs RunASHAOnConfigQueue.
 * @throws std::invalid_argument if max_configs_started == 0.
 */
template <typename RNG>
ASHAResult RunASHA(const SearchSpace& space, const TrialFactory& make_trial, size_t max_configs_started,
                    int initial_epoch_budget, double eta, int num_rungs, RNG& rng) {
    if (max_configs_started == 0) {
        throw std::invalid_argument("RunASHA: max_configs_started must be positive");
    }
    std::vector<Configuration> queue;
    queue.reserve(max_configs_started);
    for (size_t i = 0; i < max_configs_started; ++i) {
        queue.push_back(RandomSample(space, rng));
    }
    return RunASHAOnConfigQueue(std::move(queue), make_trial, initial_epoch_budget, eta, num_rungs);
}

}  // namespace pulsatrix
