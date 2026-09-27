#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/trial.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Phase 1's own exit-gate integration test: "grid and random search both successfully tune
// at least one real hyperparameter (e.g. learning rate) on a small network to a measurably
// better validation metric than an untuned default." Reuses Phase 1 (core-layers)'s own
// XorNetwork/training-loop pattern (examples/xor_demo.cpp) as the "real small network."

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

constexpr int kEpochs = 300;
constexpr float kBadDefaultLearningRate = 2.0f;  // deliberately too large for this network

TEST(HPOXorTuningIntegrationTest, GridSearchFindsLearningRateBeatingBadDefault) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);

    auto grid = GridSample(space, 6);
    ASSERT_FALSE(grid.empty());

    double best_loss = 1e18;
    for (const auto& config : grid) {
        Trial trial(config);
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        double loss = EvaluateLearningRate(lr, kEpochs);
        trial.RecordMetric("val_loss", loss, kEpochs);
        best_loss = std::min(best_loss, *trial.BestMetric("val_loss", /*maximize=*/false));
    }

    double bad_default_loss = EvaluateLearningRate(kBadDefaultLearningRate, kEpochs);
    EXPECT_LT(best_loss, bad_default_loss / 2.0);
}

TEST(HPOXorTuningIntegrationTest, RandomSearchFindsLearningRateBeatingBadDefault) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(2026);

    double best_loss = 1e18;
    constexpr int kDraws = 8;
    for (int i = 0; i < kDraws; ++i) {
        auto config = RandomSample(space, rng);
        Trial trial(config);
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        double loss = EvaluateLearningRate(lr, kEpochs);
        trial.RecordMetric("val_loss", loss, kEpochs);
        best_loss = std::min(best_loss, *trial.BestMetric("val_loss", /*maximize=*/false));
    }

    double bad_default_loss = EvaluateLearningRate(kBadDefaultLearningRate, kEpochs);
    EXPECT_LT(best_loss, bad_default_loss / 2.0);
}

}  // namespace
}  // namespace pulsatrix
