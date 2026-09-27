#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cma_es.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hpo_genotype.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/trial.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Phase 2's own exit-gate requirement (CMA-ES half): "measurably improves a validation
// metric over a fixed baseline hyperparameter configuration on a real training run... CMA-ES's
// config representation verified interoperable with campaign_exai_dl_library_
// hyperparameter_optimization's SearchSpace/Trial types by direct integration test, not just
// shared documentation." This test explicitly constructs a SearchSpace, decodes every CMA-ES
// genotype through it via DecodeGenotype (Phase 2 Mission 0), and records every generation's
// best result into a real Trial -- not just a raw Configuration/double pair.

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

TEST(CMAESXorIntegrationTest, ImprovesLearningRateOverFixedBadDefaultViaSearchSpaceAndTrial) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);

    constexpr int kEpochs = 40;
    constexpr size_t kLambda = 10;
    constexpr int kGenerations = 15;
    constexpr float kBadDefaultLearningRate = 2.0f;

    // CMA-ES searches the unit-hypercube genotype directly (its own real-valued vector space);
    // every genotype is clamped to [0,1] and decoded through the *actual* SearchSpace type
    // (not a hand-rolled substitute) before being evaluated -- this clamp-then-decode step is
    // the concrete interoperability point the exit gate names.
    CMAES cmaes({0.5}, 0.2, kLambda);
    std::mt19937 rng(2026);

    std::vector<Trial> best_trial_per_generation;
    for (int generation = 0; generation < kGenerations; ++generation) {
        auto genotypes = cmaes.Ask(rng);
        std::vector<double> fitness(genotypes.size());
        std::vector<Configuration> configs(genotypes.size());

        for (size_t i = 0; i < genotypes.size(); ++i) {
            std::vector<double> clamped = genotypes[i];
            for (double& gene : clamped) {
                gene = std::clamp(gene, 0.0, 1.0);
            }
            configs[i] = DecodeGenotype(space, clamped);
            float lr = static_cast<float>(std::get<double>(configs[i].at("learning_rate")));
            fitness[i] = -EvaluateLearningRate(lr, kEpochs);
        }
        cmaes.Tell(genotypes, fitness);

        size_t best_idx = static_cast<size_t>(
            std::max_element(fitness.begin(), fitness.end()) - fitness.begin());
        Trial trial(configs[best_idx]);
        trial.RecordMetric("val_loss", -fitness[best_idx], generation);
        best_trial_per_generation.push_back(std::move(trial));
    }

    double best_loss = 1e18;
    for (const auto& trial : best_trial_per_generation) {
        best_loss = std::min(best_loss, *trial.BestMetric("val_loss", /*maximize=*/false));
    }
    double bad_default_loss = EvaluateLearningRate(kBadDefaultLearningRate, kEpochs);

    EXPECT_LT(best_loss, bad_default_loss / 2.0);
}

}  // namespace
}  // namespace pulsatrix
