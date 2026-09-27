/** @file selection.hpp
 *  @brief Genetic-algorithm selection operators (tournament, roulette/fitness-proportionate,
 *         linear-rank) over a population of Individual<Genotype, FitnessT>.
 *  @ingroup evolutionary
 *  @note All three operators assume higher fitness is better (maximization) -- the same
 *        convention as DEAP's own weights=(1.0,) FitnessMax default (research doc
 *        research_2026_evolutionary_deep_learning.md, campaign
 *        campaign_exai_dl_library_evolutionary_deep_learning, Phase 1 Mission 0).
 *  @note Each stochastic operator (TournamentSelect, RouletteSelect, RankSelect) is a thin
 *        RNG-driven wrapper around a pure, deterministic core (RouletteSelectByDraw,
 *        RankSelectByDraw) or is itself exactly reproducible given its RNG stream
 *        (TournamentSelect) -- this is what makes each operator's correctness
 *        hand-derivable/closed-form testable independent of any particular RNG's internal
 *        implementation, per this project's TDD discipline.
 */
#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/individual.hpp"

namespace pulsatrix {

/**
 * @brief Tournament selection: draw tournament_size individuals without replacement from
 *        population and return the index of the fittest among them.
 * @param population Non-empty population to select from.
 * @param tournament_size Number of distinct competitors, in [1, population.size()].
 * @param rng Any UniformRandomBitGenerator (e.g. std::mt19937).
 * @return Index into population of the selected individual.
 * @throws std::invalid_argument if population is empty, tournament_size is 0, or
 *         tournament_size exceeds population.size() -- external-boundary malformed input
 *         (caller-supplied population/config), not an internal invariant.
 * @note tournament_size == population.size() draws every individual exactly once (without
 *       replacement), so it always returns the global-best index regardless of the RNG
 *       stream -- this operator's own hand-verifiable correctness case, independent of
 *       which RNG or seed is used.
 */
template <typename Genotype, typename FitnessT, typename RNG>
size_t TournamentSelect(const std::vector<Individual<Genotype, FitnessT>>& population,
                         size_t tournament_size, RNG& rng) {
    if (population.empty()) {
        throw std::invalid_argument("TournamentSelect: population must not be empty");
    }
    if (tournament_size == 0 || tournament_size > population.size()) {
        throw std::invalid_argument(
            "TournamentSelect: tournament_size must be in [1, population.size()]");
    }

    std::vector<size_t> indices(population.size());
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t best_index = indices[0];
    for (size_t i = 1; i < tournament_size; ++i) {
        if (population[indices[i]].fitness > population[best_index].fitness) {
            best_index = indices[i];
        }
    }
    return best_index;
}

/**
 * @brief Fitness-proportionate ("roulette wheel") selection given an explicit draw in
 *        [0, total_fitness). Pure and deterministic -- the hand-testable core RouletteSelect
 *        wraps with an RNG-generated draw.
 * @param population Non-empty population with strictly non-negative fitness values summing
 *        to a strictly positive total.
 * @param draw A value in [0, total_fitness), where total_fitness = sum of population fitness.
 * @return Index into population whose cumulative-fitness bucket contains draw.
 * @throws std::invalid_argument if population is empty, any fitness is negative, the total
 *         fitness is not strictly positive, or draw is outside [0, total_fitness).
 */
template <typename Genotype, typename FitnessT>
size_t RouletteSelectByDraw(const std::vector<Individual<Genotype, FitnessT>>& population,
                            FitnessT draw) {
    if (population.empty()) {
        throw std::invalid_argument("RouletteSelectByDraw: population must not be empty");
    }
    FitnessT total{};
    for (const auto& ind : population) {
        if (ind.fitness < FitnessT{0}) {
            throw std::invalid_argument("RouletteSelectByDraw: fitness must be non-negative");
        }
        total += ind.fitness;
    }
    if (!(total > FitnessT{0})) {
        throw std::invalid_argument("RouletteSelectByDraw: total fitness must be positive");
    }
    if (draw < FitnessT{0} || draw >= total) {
        throw std::invalid_argument("RouletteSelectByDraw: draw must be in [0, total_fitness)");
    }

    FitnessT cumulative{};
    for (size_t i = 0; i < population.size(); ++i) {
        cumulative += population[i].fitness;
        if (draw < cumulative) {
            return i;
        }
    }
    // Unreachable: draw < total is enforced above, and cumulative reaches total by the last
    // element -- an internal invariant, not a condition a caller can trigger.
    PULSATRIX_ASSERT(false);
    return population.size() - 1;
}

/**
 * @brief RNG-driven wrapper around RouletteSelectByDraw: draws uniformly from
 *        [0, total_fitness) and selects accordingly.
 * @throws std::invalid_argument -- see RouletteSelectByDraw (validation happens there; this
 *         wrapper adds no separate checks so the two never disagree on what is malformed).
 */
template <typename Genotype, typename FitnessT, typename RNG>
size_t RouletteSelect(const std::vector<Individual<Genotype, FitnessT>>& population, RNG& rng) {
    FitnessT total{};
    for (const auto& ind : population) {
        total += ind.fitness;
    }
    std::uniform_real_distribution<FitnessT> dist(FitnessT{0}, total);
    return RouletteSelectByDraw(population, dist(rng));
}

/**
 * @brief Linear-rank selection given an explicit draw in [0, total_weight). Individuals are
 *        ranked ascending by fitness (worst = rank 1, best = rank population.size()); each
 *        rank's selection weight equals its rank, so the best individual is
 *        population.size() times as likely to be drawn as the worst. Pure and
 *        deterministic -- the hand-testable core RankSelect wraps with an RNG-generated draw.
 * @param population Non-empty population.
 * @param draw A value in [0, total_weight), where total_weight = n*(n+1)/2 for n individuals.
 * @return Index into population of the individual occupying the drawn rank.
 * @throws std::invalid_argument if population is empty or draw is outside [0, total_weight).
 */
template <typename Genotype, typename FitnessT>
size_t RankSelectByDraw(const std::vector<Individual<Genotype, FitnessT>>& population,
                        double draw) {
    if (population.empty()) {
        throw std::invalid_argument("RankSelectByDraw: population must not be empty");
    }

    std::vector<size_t> order(population.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&population](size_t a, size_t b) {
        return population[a].fitness < population[b].fitness;
    });

    const double n = static_cast<double>(population.size());
    const double total_weight = n * (n + 1.0) / 2.0;  // 1 + 2 + ... + n
    if (draw < 0.0 || draw >= total_weight) {
        throw std::invalid_argument("RankSelectByDraw: draw must be in [0, total_weight)");
    }

    double cumulative = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        cumulative += static_cast<double>(i + 1);  // rank of order[i] is i+1 (1-indexed)
        if (draw < cumulative) {
            return order[i];
        }
    }
    // Unreachable: draw < total_weight is enforced above -- an internal invariant.
    PULSATRIX_ASSERT(false);
    return order.back();
}

/**
 * @brief RNG-driven wrapper around RankSelectByDraw: draws uniformly from [0, total_weight)
 *        and selects accordingly.
 * @throws std::invalid_argument -- see RankSelectByDraw.
 */
template <typename Genotype, typename FitnessT, typename RNG>
size_t RankSelect(const std::vector<Individual<Genotype, FitnessT>>& population, RNG& rng) {
    const double n = static_cast<double>(population.size());
    const double total_weight = n * (n + 1.0) / 2.0;
    std::uniform_real_distribution<double> dist(0.0, total_weight);
    return RankSelectByDraw(population, dist(rng));
}

}  // namespace pulsatrix
