/** @file evolutionary_loop.hpp
 *  @brief The shared six-step evolutionary-loop skeleton (evaluate -> select parents -> vary
 *         via crossover/mutation -> evaluate offspring -> survivor-select -> loop), generic
 *         over which survivor-selection policy (survivor_selection.hpp) and which
 *         offspring-production recipe (selection + crossover + mutation, composed by the
 *         caller) is used.
 *  @ingroup evolutionary
 *  @note Deliberately does NOT bundle a specific selection/crossover/mutation composition
 *        into this loop -- how a caller composes those (which selection operator, which
 *        crossover, whether/how to mutate) is left to the caller's own offspring-production
 *        callable, the same "wire manually, don't build a generic training abstraction"
 *        discipline this project's RL campaign already established for its own training
 *        loops. What genuinely is shared across generational/(mu+lambda)/(mu,lambda) -- the
 *        loop shape itself and fitness evaluation -- is written once, here.
 *  @note Single-node thread-pool parallel fitness evaluation reuses DataThreadPool
 *        (data_thread_pool.hpp, built for the data-pipeline campaign's DataLoader) rather
 *        than a second thread pool -- found by this campaign's own Mission 0 activation
 *        recon.
 */
#pragma once

#include <future>
#include <utility>
#include <vector>

#include "pulsatrix/data_thread_pool.hpp"
#include "pulsatrix/individual.hpp"

namespace pulsatrix {

/**
 * @brief Evaluates (or re-evaluates) every individual's fitness in place via fitness_fn.
 * @param thread_pool If non-null, each individual's fitness is submitted as an independent
 *        task; if null, evaluation runs sequentially on the calling thread. fitness_fn must
 *        be safely callable concurrently from multiple threads when a thread_pool is
 *        supplied (an ordinary pure function of its genotype argument, per this project's
 *        explainer-layer functional-purity convention, satisfies this trivially).
 */
template <typename Genotype, typename FitnessT, typename FitnessFn>
void EvaluatePopulation(std::vector<Individual<Genotype, FitnessT>>& population,
                        FitnessFn& fitness_fn, DataThreadPool* thread_pool) {
    if (thread_pool == nullptr) {
        for (auto& ind : population) {
            ind.fitness = fitness_fn(ind.genes);
        }
        return;
    }

    std::vector<std::future<FitnessT>> futures;
    futures.reserve(population.size());
    for (auto& ind : population) {
        Genotype genes_copy = ind.genes;  // copied into the task -- avoids a dangling
                                           // reference to population while other threads run
        futures.push_back(
            thread_pool->submit([&fitness_fn, genes_copy]() { return fitness_fn(genes_copy); }));
    }
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].fitness = futures[i].get();
    }
}

/**
 * @brief Runs num_generations of the shared evolutionary-loop skeleton: evaluate -> produce
 *        lambda_size offspring (via produce_offspring_genotype, called once per offspring) ->
 *        evaluate offspring -> survivor-select mu individuals for the next generation.
 * @param population Initial population (its own fitness values are (re-)computed, not
 *        trusted, before the loop begins). mu = population.size().
 * @param produce_offspring_genotype Callable `Genotype(const std::vector<Individual<Genotype,
 *        FitnessT>>& current_population)` -- composes whichever selection + crossover +
 *        mutation recipe the caller wants; called once per offspring needed.
 * @param survivor_selector Callable `std::vector<Individual<Genotype, FitnessT>>(const
 *        std::vector<Individual<Genotype, FitnessT>>& population,
 *        std::vector<Individual<Genotype, FitnessT>> offspring, size_t mu)` -- pass
 *        GenerationalReplacement, MuPlusLambdaReplacement, or MuCommaLambdaReplacement
 *        (survivor_selection.hpp) directly, or any compatible callable.
 * @param thread_pool Optional -- see EvaluatePopulation.
 * @return The final generation's population (fitness values populated).
 */
template <typename Genotype, typename FitnessT, typename FitnessFn, typename OffspringFn,
          typename SurvivorFn>
std::vector<Individual<Genotype, FitnessT>> RunEvolutionaryLoop(
    std::vector<Individual<Genotype, FitnessT>> population, size_t num_generations,
    size_t lambda_size, FitnessFn fitness_fn, OffspringFn produce_offspring_genotype,
    SurvivorFn survivor_selector, DataThreadPool* thread_pool = nullptr) {
    const size_t mu = population.size();
    EvaluatePopulation(population, fitness_fn, thread_pool);

    for (size_t generation = 0; generation < num_generations; ++generation) {
        std::vector<Individual<Genotype, FitnessT>> offspring;
        offspring.reserve(lambda_size);
        for (size_t i = 0; i < lambda_size; ++i) {
            offspring.push_back(Individual<Genotype, FitnessT>{produce_offspring_genotype(population),
                                                                 FitnessT{}});
        }
        EvaluatePopulation(offspring, fitness_fn, thread_pool);
        population = survivor_selector(population, std::move(offspring), mu);
    }
    return population;
}

}  // namespace pulsatrix
