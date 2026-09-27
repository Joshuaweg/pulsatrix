/** @file neat_evolution.hpp
 *  @brief NEAT's own generational evolutionary loop: fitness evaluation, speciation, fitness
 *         sharing, proportional offspring allocation, and mutation-only reproduction.
 *  @ingroup evolutionary
 *  @note Crossover between two genomes is deliberately NOT built here -- a logged scope cut
 *        matching Mission 0/1's own notes (the innovation-tracking system exists to eventually
 *        support it, but it remains a future mission's concern). Reproduction in this file is
 *        therefore mutation-only: clone a parent, then apply weight and/or structural
 *        mutation. This is a real scope reduction from NEAT's own original design (which uses
 *        crossover as its primary variation operator), not an oversight.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_speciation.hpp"

namespace pulsatrix {

/**
 * @brief Pure core: allocates population_size offspring slots across species proportionally
 *        to each species' own adjusted-fitness sum, using the largest-remainder (Hamilton)
 *        apportionment method so the total always sums to exactly population_size (ties in
 *        fractional remainder broken by species index, earliest first). Falls back to an
 *        equal split (remainder to the earliest species, by index) if every species sum is
 *        non-positive, rather than dividing by zero.
 * @throws std::invalid_argument if species_adjusted_fitness_sums is empty or population_size
 *         is not positive.
 */
inline std::vector<int> AllocateOffspringCounts(const std::vector<double>& species_adjusted_fitness_sums,
                                                 int population_size) {
    if (species_adjusted_fitness_sums.empty()) {
        throw std::invalid_argument("AllocateOffspringCounts: species_adjusted_fitness_sums must not be empty");
    }
    if (population_size <= 0) {
        throw std::invalid_argument("AllocateOffspringCounts: population_size must be positive");
    }

    size_t n = species_adjusted_fitness_sums.size();
    double total = std::accumulate(species_adjusted_fitness_sums.begin(), species_adjusted_fitness_sums.end(), 0.0);

    std::vector<int> counts(n, 0);
    if (total <= 0.0) {
        int base = population_size / static_cast<int>(n);
        int remainder = population_size - base * static_cast<int>(n);
        for (size_t i = 0; i < n; ++i) {
            counts[i] = base + (static_cast<int>(i) < remainder ? 1 : 0);
        }
        return counts;
    }

    std::vector<double> exact(n);
    int allocated = 0;
    for (size_t i = 0; i < n; ++i) {
        exact[i] = species_adjusted_fitness_sums[i] / total * static_cast<double>(population_size);
        counts[i] = static_cast<int>(std::floor(exact[i]));
        allocated += counts[i];
    }

    int remaining = population_size - allocated;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (exact[a] - std::floor(exact[a])) > (exact[b] - std::floor(exact[b]));
    });
    for (int i = 0; i < remaining; ++i) {
        counts[order[i]] += 1;
    }
    return counts;
}

/**
 * @brief RNG-driven wrapper: clones parent, then applies weight mutation (gated per-connection
 *        by weight_mutation_probability inside MutateWeights itself) and, independently, one
 *        attempt each at the two structural mutations, each gated by its own probability.
 */
template <typename RNG>
NEATGenome ReproduceOffspring(const NEATGenome& parent, InnovationTracker& tracker, RNG& rng,
                               double weight_mutation_sigma, double weight_mutation_probability,
                               double add_connection_probability, double add_node_probability) {
    NEATGenome child = parent;
    child.MutateWeights(weight_mutation_sigma, weight_mutation_probability, rng);
    std::bernoulli_distribution add_connection_roll(add_connection_probability);
    if (add_connection_roll(rng)) {
        child.AddConnection(tracker, rng);
    }
    std::bernoulli_distribution add_node_roll(add_node_probability);
    if (add_node_roll(rng)) {
        child.AddNode(tracker, rng);
    }
    return child;
}

/** @brief Result of a full NEAT evolutionary run. */
struct NEATEvolutionResult {
    NEATGenome best_genome;
    double best_fitness;
    int generations_run;
};

/**
 * @brief Runs num_generations of speciated, mutation-only NEAT evolution. Each generation:
 *        evaluates every genome's fitness via fitness_fn, tracks the best genome seen across
 *        the whole run so far (global elitism -- best_fitness never decreases generation to
 *        generation), speciates the population, applies fitness sharing, allocates each
 *        species a share of the next generation proportional to its adjusted-fitness sum, and
 *        fills that share with one unmutated species-champion copy plus mutated copies of
 *        uniformly-randomly chosen species members.
 * @throws std::invalid_argument if population is empty or num_generations is not positive.
 */
template <typename FitnessFn, typename RNG>
NEATEvolutionResult RunNEATEvolution(std::vector<NEATGenome> population, FitnessFn fitness_fn, int num_generations,
                                      double compatibility_threshold, double c1, double c2, double c3,
                                      double weight_mutation_sigma, double weight_mutation_probability,
                                      double add_connection_probability, double add_node_probability,
                                      InnovationTracker& tracker, RNG& rng) {
    if (population.empty()) {
        throw std::invalid_argument("RunNEATEvolution: population must not be empty");
    }
    if (num_generations <= 0) {
        throw std::invalid_argument("RunNEATEvolution: num_generations must be positive");
    }

    NEATGenome best_genome = population[0];
    double best_fitness = -std::numeric_limits<double>::infinity();

    for (int gen = 0; gen < num_generations; ++gen) {
        std::vector<double> fitnesses(population.size());
        for (size_t i = 0; i < population.size(); ++i) {
            fitnesses[i] = fitness_fn(population[i]);
            if (fitnesses[i] > best_fitness) {
                best_fitness = fitnesses[i];
                best_genome = population[i];
            }
        }

        if (gen == num_generations - 1) {
            break;
        }

        SpeciesAssignment assignment = SpeciatePopulation(population, compatibility_threshold, c1, c2, c3);
        std::vector<double> adjusted = ComputeAdjustedFitness(fitnesses, assignment.species);

        std::vector<double> species_sums(assignment.species.size(), 0.0);
        for (size_t s = 0; s < assignment.species.size(); ++s) {
            for (size_t idx : assignment.species[s]) {
                species_sums[s] += adjusted[idx];
            }
        }
        std::vector<int> counts = AllocateOffspringCounts(species_sums, static_cast<int>(population.size()));

        std::vector<NEATGenome> next_population;
        next_population.reserve(population.size());
        for (size_t s = 0; s < assignment.species.size(); ++s) {
            int count = counts[s];
            if (count <= 0) {
                continue;
            }
            std::vector<size_t> members = assignment.species[s];
            std::sort(members.begin(), members.end(),
                      [&](size_t a, size_t b) { return fitnesses[a] > fitnesses[b]; });

            next_population.push_back(population[members.front()]);
            std::uniform_int_distribution<size_t> pick_parent(0, members.size() - 1);
            for (int offspring_index = 1; offspring_index < count; ++offspring_index) {
                const NEATGenome& parent = population[members[pick_parent(rng)]];
                next_population.push_back(ReproduceOffspring(parent, tracker, rng, weight_mutation_sigma,
                                                               weight_mutation_probability,
                                                               add_connection_probability, add_node_probability));
            }
        }

        population = std::move(next_population);
    }

    return NEATEvolutionResult{best_genome, best_fitness, num_generations};
}

}  // namespace pulsatrix
