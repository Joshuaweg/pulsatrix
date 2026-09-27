#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/crossover.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/evolutionary_loop.hpp"
#include "pulsatrix/hpo_genotype.hpp"
#include "pulsatrix/individual.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/mutation.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/selection.hpp"
#include "pulsatrix/survivor_selection.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Phase 2's own exit-gate integration test: "GA-based HPO (including the CMA-ES ask-tell
// variant) measurably improves a validation metric over a fixed baseline hyperparameter
// configuration on a real training run." This test is the GA half; Mission 2 (CMA-ES) is the
// other. Wires entirely existing pieces (Phase 1's GA core + this phase's own
// DecodeGenotype), no new production code -- the same "wire manually, don't build a generic
// training abstraction" discipline this project's RL campaign established, applied here to
// composing an evolutionary loop with an HPO objective instead of a DL training loop.

double EvaluateLearningRate(float learning_rate, int epochs) {
    CPUBackend backend;
    XorNetwork net(&backend);
    AdamOptimizer optimizer(learning_rate, &backend);
    NoOpMetricsSink sink;

    std::vector<std::pair<Tensor, Tensor>> dataset;
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));

    int step = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto& [input, target] : dataset) {
            (void)net.train_step(input, target, optimizer, sink, step++);
        }
    }

    double total = 0.0;
    for (auto& [input, target] : dataset) {
        Tensor pred = net.forward(input);
        double diff = pred.data()[0] - target.data()[0];
        total += diff * diff;
    }
    return total / static_cast<double>(dataset.size());
}

TEST(GAHPOIntegrationTest, ImprovesLearningRateOverFixedBadDefault) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);

    constexpr int kEpochs = 40;  // established HPO calibration (see gp_bo_xor_integration_test.cpp)
    constexpr size_t kPopulationSize = 16;
    constexpr size_t kGenerations = 20;
    constexpr float kBadDefaultLearningRate = 2.0f;

    auto fitness_fn = [&](const std::vector<double>& genotype) {
        Configuration config = DecodeGenotype(space, genotype);
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        return -EvaluateLearningRate(lr, kEpochs);  // maximize negative loss
    };

    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    std::vector<Individual<std::vector<double>, double>> population;
    for (size_t i = 0; i < kPopulationSize; ++i) {
        population.push_back(Individual<std::vector<double>, double>{{unit_dist(rng)}, 0.0});
    }

    auto produce_offspring =
        [&](const std::vector<Individual<std::vector<double>, double>>& pop) {
            size_t i1 = TournamentSelect(pop, 3, rng);
            size_t i2 = TournamentSelect(pop, 3, rng);
            auto [child1, child2] = BlendCrossover(pop[i1].genes, pop[i2].genes, 0.3, rng);
            (void)child2;
            auto mutated = GaussianMutation(child1, 0.1, 0.3, rng);
            for (double& gene : mutated) {
                gene = std::clamp(gene, 0.0, 1.0);
            }
            return mutated;
        };

    auto result = RunEvolutionaryLoop(population, kGenerations, kPopulationSize, fitness_fn,
                                       produce_offspring,
                                       GenerationalReplacement<std::vector<double>, double>);

    double best_loss = 1e18;
    for (const auto& ind : result) {
        best_loss = std::min(best_loss, -ind.fitness);
    }
    double bad_default_loss = EvaluateLearningRate(kBadDefaultLearningRate, kEpochs);

    EXPECT_LT(best_loss, bad_default_loss / 2.0);
}

}  // namespace
}  // namespace pulsatrix
