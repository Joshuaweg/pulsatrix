#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "pulsatrix/crossover.hpp"
#include "pulsatrix/data_thread_pool.hpp"
#include "pulsatrix/evolutionary_loop.hpp"
#include "pulsatrix/individual.hpp"
#include "pulsatrix/mutation.hpp"
#include "pulsatrix/selection.hpp"
#include "pulsatrix/survivor_selection.hpp"

namespace pulsatrix {
namespace {

// --- EvaluatePopulation: sequential vs. thread-pool-parallel equivalence ---

TEST(EvaluatePopulationTest, SequentialAndParallelProduceIdenticalResults) {
    auto sum_fitness = [](const std::vector<double>& genes) {
        double total = 0.0;
        for (double g : genes) {
            total += g;
        }
        return total;
    };

    std::vector<Individual<std::vector<double>, double>> population_sequential;
    for (int i = 0; i < 10; ++i) {
        population_sequential.push_back(
            Individual<std::vector<double>, double>{{1.0 * i, 2.0 * i, 3.0 * i}, 0.0});
    }
    auto population_parallel = population_sequential;

    EvaluatePopulation(population_sequential, sum_fitness, nullptr);

    DataThreadPool pool(4);
    EvaluatePopulation(population_parallel, sum_fitness, &pool);

    ASSERT_EQ(population_sequential.size(), population_parallel.size());
    for (size_t i = 0; i < population_sequential.size(); ++i) {
        EXPECT_DOUBLE_EQ(population_sequential[i].fitness, population_parallel[i].fitness);
        EXPECT_DOUBLE_EQ(population_sequential[i].fitness, sum_fitness(population_sequential[i].genes));
    }
}

// --- OneMax via the generational-replacement loop (Phase 1's own exit-gate benchmark) ---

TEST(EvolutionaryLoopTest, SolvesOneMaxViaGenerationalReplacement) {
    constexpr size_t kGenomeLength = 20;
    constexpr size_t kPopulationSize = 40;
    constexpr size_t kGenerations = 200;

    auto one_max_fitness = [](const std::vector<bool>& genes) {
        double total = 0.0;
        for (bool bit : genes) {
            total += bit ? 1.0 : 0.0;
        }
        return total;
    };

    for (unsigned seed : {1u, 2u}) {
        std::mt19937 rng(seed);
        std::bernoulli_distribution init_dist(0.5);

        std::vector<Individual<std::vector<bool>, double>> population;
        for (size_t i = 0; i < kPopulationSize; ++i) {
            std::vector<bool> genes(kGenomeLength);
            for (size_t b = 0; b < kGenomeLength; ++b) {
                genes[b] = init_dist(rng);
            }
            population.push_back(Individual<std::vector<bool>, double>{genes, 0.0});
        }

        auto produce_offspring =
            [&rng, kGenomeLength](const std::vector<Individual<std::vector<bool>, double>>& pop) {
                size_t i1 = TournamentSelect(pop, 3, rng);
                size_t i2 = TournamentSelect(pop, 3, rng);
                auto [child1, child2] = OnePointCrossover(pop[i1].genes, pop[i2].genes, rng);
                (void)child2;
                return BitFlipMutation(child1, 1.0 / kGenomeLength, rng);
            };

        auto result = RunEvolutionaryLoop(population, kGenerations, kPopulationSize,
                                           one_max_fitness, produce_offspring,
                                           GenerationalReplacement<std::vector<bool>, double>);

        double best = 0.0;
        for (const auto& ind : result) {
            best = std::max(best, ind.fitness);
        }
        EXPECT_DOUBLE_EQ(best, static_cast<double>(kGenomeLength)) << "seed=" << seed;
    }
}

// --- Sphere function via (mu+lambda) replacement -- proves a second survivor policy works ---

TEST(EvolutionaryLoopTest, ImprovesSphereFunctionViaMuPlusLambdaReplacement) {
    constexpr size_t kDimension = 5;
    constexpr size_t kPopulationSize = 30;
    constexpr size_t kGenerations = 150;

    auto negative_sphere_fitness = [](const std::vector<double>& genes) {
        double sum_sq = 0.0;
        for (double g : genes) {
            sum_sq += g * g;
        }
        return -sum_sq;  // maximize -sum(x_i^2) == minimize sum(x_i^2)
    };

    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> init_dist(-5.0, 5.0);

    std::vector<Individual<std::vector<double>, double>> population;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        std::vector<double> genes(kDimension);
        for (size_t d = 0; d < kDimension; ++d) {
            genes[d] = init_dist(rng);
        }
        population.push_back(Individual<std::vector<double>, double>{genes, 0.0});
    }

    double initial_best_sphere = 1e18;
    for (const auto& ind : population) {
        initial_best_sphere = std::min(initial_best_sphere, -negative_sphere_fitness(ind.genes));
    }

    auto produce_offspring =
        [&rng](const std::vector<Individual<std::vector<double>, double>>& pop) {
            size_t i1 = TournamentSelect(pop, 3, rng);
            size_t i2 = TournamentSelect(pop, 3, rng);
            auto [child1, child2] = BlendCrossover(pop[i1].genes, pop[i2].genes, 0.5, rng);
            (void)child2;
            return GaussianMutation(child1, 0.3, 0.3, rng);
        };

    auto result = RunEvolutionaryLoop(
        population, kGenerations, kPopulationSize, negative_sphere_fitness, produce_offspring,
        MuPlusLambdaReplacement<std::vector<double>, double>);

    double final_best_sphere = 1e18;
    for (const auto& ind : result) {
        final_best_sphere = std::min(final_best_sphere, -negative_sphere_fitness(ind.genes));
    }

    // A generous, non-tuned-to-barely-pass bar (per this project's own standing discipline):
    // at least a 10x reduction in the best individual's sphere value.
    EXPECT_LT(final_best_sphere, initial_best_sphere / 10.0);
}

}  // namespace
}  // namespace pulsatrix
