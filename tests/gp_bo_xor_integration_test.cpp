#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gp_bo.hpp"
#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/trial.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Phase 2's own exit-gate integration test: "GP-BO (with all three acquisition functions)...
// outperform Phase 1's random search on the same tuning task, within a defined trial budget."
// Same EvaluateLearningRate as tests/hpo_xor_tuning_integration_test.cpp (Phase 1 Mission 1)
// -- duplicated deliberately, for test-file independence, rather than shared.

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

constexpr int kEpochs = 40;
constexpr size_t kInitialRandom = 3;
constexpr size_t kIterations = 5;
constexpr size_t kTotalBudget = kInitialRandom + kIterations;  // 8 -- matches random search's draw count
constexpr size_t kNumCandidates = 100;
constexpr unsigned kSeeds[] = {1u, 2u, 3u};

double RunGPBOOnXor(AcquisitionKind acquisition, unsigned seed) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(seed);

    auto objective = [](const Configuration& config) {
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        return -EvaluateLearningRate(lr, kEpochs);  // maximize negative loss
    };

    auto trials = RunGPBOLoop(space, objective, kInitialRandom, kIterations, acquisition, kNumCandidates, rng);

    double best_loss = 1e18;
    for (const auto& trial : trials) {
        best_loss = std::min(best_loss, -*trial.LatestMetric("objective"));
    }
    return best_loss;
}

double RunRandomSearchOnXor(unsigned seed) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(seed);

    double best_loss = 1e18;
    for (size_t i = 0; i < kTotalBudget; ++i) {
        auto config = RandomSample(space, rng);
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        best_loss = std::min(best_loss, EvaluateLearningRate(lr, kEpochs));
    }
    return best_loss;
}

double AverageGPBOOverSeeds(AcquisitionKind acquisition) {
    double total = 0.0;
    for (unsigned seed : kSeeds) {
        total += RunGPBOOnXor(acquisition, seed);
    }
    return total / 3.0;
}

double AverageRandomSearchOverSeeds() {
    double total = 0.0;
    for (unsigned seed : kSeeds) {
        total += RunRandomSearchOnXor(seed);
    }
    return total / 3.0;
}

// Compared on average over several seeds (a single run is too noisy to be a fair,
// reproducible bar either way), with the *same total trial budget* on both sides.

TEST(GPBOXorIntegrationTest, ExpectedImprovementOutperformsRandomSearch) {
    double gp_bo_loss = AverageGPBOOverSeeds(AcquisitionKind::ExpectedImprovement);
    double random_loss = AverageRandomSearchOverSeeds();
    EXPECT_LT(gp_bo_loss, random_loss);
}

TEST(GPBOXorIntegrationTest, ProbabilityOfImprovementOutperformsRandomSearch) {
    double gp_bo_loss = AverageGPBOOverSeeds(AcquisitionKind::ProbabilityOfImprovement);
    double random_loss = AverageRandomSearchOverSeeds();
    EXPECT_LT(gp_bo_loss, random_loss);
}

TEST(GPBOXorIntegrationTest, UpperConfidenceBoundOutperformsRandomSearch) {
    double gp_bo_loss = AverageGPBOOverSeeds(AcquisitionKind::UpperConfidenceBound);
    double random_loss = AverageRandomSearchOverSeeds();
    EXPECT_LT(gp_bo_loss, random_loss);
}

}  // namespace
}  // namespace pulsatrix
