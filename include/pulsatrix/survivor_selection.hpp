/** @file survivor_selection.hpp
 *  @brief Survivor-selection policies for the evolutionary-loop skeleton (evolutionary_loop.hpp):
 *         generational replacement, (mu+lambda), (mu,lambda).
 *  @ingroup evolutionary
 *  @note Research doc research_2026_evolutionary_deep_learning.md §1: DEAP's eaSimple,
 *        eaMuPlusLambda, and eaMuCommaLambda "reduce to the same six-step skeleton..
 *        differing only in the survivor-selection policy" -- this file is exactly that
 *        differing piece, factored out so evolutionary_loop.hpp's own loop is written once,
 *        not three times.
 */
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "pulsatrix/individual.hpp"

namespace pulsatrix {

/**
 * @brief Generational replacement (DEAP's eaSimple): the offspring pool becomes the entire
 *        next generation; population is ignored (parents never survive).
 * @throws std::invalid_argument if offspring.size() != mu.
 */
template <typename Genotype, typename FitnessT>
std::vector<Individual<Genotype, FitnessT>> GenerationalReplacement(
    const std::vector<Individual<Genotype, FitnessT>>& /*population*/,
    std::vector<Individual<Genotype, FitnessT>> offspring, size_t mu) {
    if (offspring.size() != mu) {
        throw std::invalid_argument("GenerationalReplacement: offspring.size() must equal mu");
    }
    return offspring;
}

/**
 * @brief (mu+lambda) replacement: the next generation is the fittest mu individuals from
 *        population union offspring (parents may survive) -- more exploitative than
 *        (mu,lambda), since a fit parent is never discarded just for being old.
 * @throws std::invalid_argument if population.size() + offspring.size() < mu.
 */
template <typename Genotype, typename FitnessT>
std::vector<Individual<Genotype, FitnessT>> MuPlusLambdaReplacement(
    const std::vector<Individual<Genotype, FitnessT>>& population,
    std::vector<Individual<Genotype, FitnessT>> offspring, size_t mu) {
    if (population.size() + offspring.size() < mu) {
        throw std::invalid_argument(
            "MuPlusLambdaReplacement: population.size() + offspring.size() must be >= mu");
    }
    std::vector<Individual<Genotype, FitnessT>> combined = population;
    combined.insert(combined.end(), std::make_move_iterator(offspring.begin()),
                     std::make_move_iterator(offspring.end()));
    std::partial_sort(combined.begin(), combined.begin() + static_cast<std::ptrdiff_t>(mu),
                       combined.end(),
                       [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
    combined.resize(mu);
    return combined;
}

/**
 * @brief (mu,lambda) replacement: the next generation is the fittest mu individuals from
 *        offspring only (population/parents are always discarded) -- more explorative than
 *        (mu+lambda), since it cannot get stuck re-selecting the same elite parent forever.
 * @throws std::invalid_argument if offspring.size() < mu.
 */
template <typename Genotype, typename FitnessT>
std::vector<Individual<Genotype, FitnessT>> MuCommaLambdaReplacement(
    const std::vector<Individual<Genotype, FitnessT>>& /*population*/,
    std::vector<Individual<Genotype, FitnessT>> offspring, size_t mu) {
    if (offspring.size() < mu) {
        throw std::invalid_argument("MuCommaLambdaReplacement: offspring.size() must be >= mu");
    }
    std::partial_sort(offspring.begin(), offspring.begin() + static_cast<std::ptrdiff_t>(mu),
                       offspring.end(),
                       [](const auto& a, const auto& b) { return a.fitness > b.fitness; });
    offspring.resize(mu);
    return offspring;
}

}  // namespace pulsatrix
