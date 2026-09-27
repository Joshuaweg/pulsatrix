/** @file successive_halving.hpp
 *  @brief Successive Halving (the rung-based promotion mechanism underlying Hyperband and
 *         ASHA, Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, "Hyperband: A Novel
 *         Bandit-Based Approach to Hyperparameter Optimization," JMLR 2018): train a
 *         population of configurations for a small budget, keep the best fraction, repeat
 *         with a larger budget, until one survives.
 *  @ingroup hyperparameter_optimization
 *  @note Decision Point 4 resolution (campaign_exai_dl_library_hyperparameter_optimization,
 *        Phase 3 activation, 2026-09-27): recon (Phase 1 Mission 0) already confirmed
 *        MetricsSink is observation-only (log_scalar/log_histogram) -- no mid-training
 *        external stop-signal hook exists. Rather than adding one to MetricsSink itself
 *        (a core interface every module/training loop depends on), the resolution is a new,
 *        purely-additive HPO-side abstraction: ResumableTrial, below. Every training loop in
 *        this codebase (XorNetwork::train_step and friends) is already an ordinary,
 *        synchronous C++ function the *caller* loops over -- "stopping early" needs no new
 *        capability from the network/optimizer/MetricsSink at all, just a caller that owns
 *        the trial's live state across multiple short training calls instead of one long one,
 *        and simply chooses not to call it again for pruned configurations. Zero changes to
 *        Module/Tensor/DeviceBackend/MetricsSink.
 *  @note This interface is scoped to what Successive Halving/Hyperband/ASHA (Phase 3) need --
 *        train for N more epochs, report the current metric. The sibling
 *        campaign_exai_dl_library_evolutionary_deep_learning's Population Based Training
 *        mission is documented (this campaign's own Phase 3 exit gate section) as consuming
 *        this same "mid-training hook," but PBT also needs to read/replace a live trial's
 *        weights (its exploit/explore step) -- not assumed here. Per this project's own
 *        precedent (RL's Agent interface was extended per-consumer, e.g. DQNAgent's
 *        epsilon-greedy methods, CategoricalPolicyAgent's log-probability accessor, rather
 *        than speculatively upfront), that campaign's own PBT mission must re-verify by recon
 *        whether this interface suffices or needs extending -- not guessed here ahead of need.
 */
#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {

/**
 * @brief A single hyperparameter configuration's live, resumable training state -- own
 *        whatever network/optimizer/dataset a concrete trial needs, and train it
 *        incrementally across multiple calls rather than all at once.
 */
class ResumableTrial {
public:
    virtual ~ResumableTrial() = default;

    /**
     * @brief Trains this trial for num_epochs additional epochs (continuing from wherever
     *        this trial's own training left off, not restarting), then returns the current
     *        validation metric (maximization convention, matching every other HPO algorithm
     *        in this campaign -- a caller minimizing a loss negates it).
     */
    virtual double TrainForEpochs(int num_epochs) = 0;
};

/** @brief Builds a fresh ResumableTrial for a given configuration. */
using TrialFactory = std::function<std::unique_ptr<ResumableTrial>(const Configuration&)>;

/** @brief The winning configuration, its final metric, and the total epoch-budget actually
 *         spent across every trial/rung (the exit-gate's own "reduces total training
 *         compute" measure). */
struct SuccessiveHalvingResult {
    Configuration best_configuration;
    double best_metric;
    size_t total_epochs_trained;
};

/**
 * @brief Runs Successive Halving over an explicit, caller-supplied list of configurations --
 *        the pure, deterministic core; RunSuccessiveHalving (below) is the thin
 *        SearchSpace/RNG-sampling wrapper around it.
 * @param initial_epoch_budget Epochs trained in the first rung.
 * @param eta Reduction factor: after each rung, floor(count / eta) configurations survive
 *        (at least 1), and the next rung's budget is the previous budget * eta.
 * @throws std::invalid_argument if configs is empty, initial_epoch_budget <= 0, or eta <= 1.0.
 */
inline SuccessiveHalvingResult RunSuccessiveHalvingOnConfigs(std::vector<Configuration> configs,
                                                              const TrialFactory& make_trial,
                                                              int initial_epoch_budget, double eta) {
    if (configs.empty()) {
        throw std::invalid_argument("RunSuccessiveHalvingOnConfigs: configs must not be empty");
    }
    if (initial_epoch_budget <= 0) {
        throw std::invalid_argument("RunSuccessiveHalvingOnConfigs: initial_epoch_budget must be positive");
    }
    if (eta <= 1.0) {
        throw std::invalid_argument("RunSuccessiveHalvingOnConfigs: eta must be > 1.0");
    }

    struct Candidate {
        Configuration config;
        std::unique_ptr<ResumableTrial> trial;
        double metric;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(configs.size());
    for (auto& config : configs) {
        auto trial = make_trial(config);
        candidates.push_back(Candidate{std::move(config), std::move(trial),
                                        -std::numeric_limits<double>::infinity()});
    }

    size_t total_epochs_trained = 0;
    int budget = initial_epoch_budget;
    bool trained_at_least_once = false;

    while (true) {
        for (auto& c : candidates) {
            c.metric = c.trial->TrainForEpochs(budget);
        }
        total_epochs_trained += candidates.size() * static_cast<size_t>(budget);
        trained_at_least_once = true;

        if (candidates.size() <= 1) {
            break;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.metric > b.metric; });

        size_t num_survivors = std::max<size_t>(1, static_cast<size_t>(candidates.size() / eta));
        if (num_survivors >= candidates.size()) {
            break;  // eta too small to actually reduce the population further -- stop here
        }
        candidates.resize(num_survivors);
        budget = static_cast<int>(static_cast<double>(budget) * eta);
    }
    (void)trained_at_least_once;  // always true by this point -- documents the loop's own invariant

    auto best_it = std::max_element(candidates.begin(), candidates.end(),
                                     [](const Candidate& a, const Candidate& b) { return a.metric < b.metric; });
    return SuccessiveHalvingResult{best_it->config, best_it->metric, total_epochs_trained};
}

/**
 * @brief RNG-driven wrapper: draws num_configs configurations from space via RandomSample,
 *        then runs RunSuccessiveHalvingOnConfigs.
 * @throws std::invalid_argument if num_configs == 0 -- see RunSuccessiveHalvingOnConfigs for
 *         the other validated preconditions.
 */
template <typename RNG>
SuccessiveHalvingResult RunSuccessiveHalving(const SearchSpace& space, const TrialFactory& make_trial,
                                              size_t num_configs, int initial_epoch_budget, double eta,
                                              RNG& rng) {
    if (num_configs == 0) {
        throw std::invalid_argument("RunSuccessiveHalving: num_configs must be positive");
    }
    std::vector<Configuration> configs;
    configs.reserve(num_configs);
    for (size_t i = 0; i < num_configs; ++i) {
        configs.push_back(RandomSample(space, rng));
    }
    return RunSuccessiveHalvingOnConfigs(std::move(configs), make_trial, initial_epoch_budget, eta);
}

}  // namespace pulsatrix
