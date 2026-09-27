#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/successive_halving.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// A real, resumable XorNetwork trial: owns its own network/optimizer/dataset, trains
// incrementally across TrainForEpochs calls (continuing the same live network, not
// restarting), reports negative mean-squared loss (maximization convention).
class XorResumableTrial : public ResumableTrial {
public:
    explicit XorResumableTrial(float learning_rate) : net_(&backend_), optimizer_(learning_rate, &backend_) {
        dataset_.emplace_back(Tensor(Shape({1, 2}), &backend_, {0.0f, 0.0f}), Tensor(Shape({1, 1}), &backend_, {0.0f}));
        dataset_.emplace_back(Tensor(Shape({1, 2}), &backend_, {0.0f, 1.0f}), Tensor(Shape({1, 1}), &backend_, {1.0f}));
        dataset_.emplace_back(Tensor(Shape({1, 2}), &backend_, {1.0f, 0.0f}), Tensor(Shape({1, 1}), &backend_, {1.0f}));
        dataset_.emplace_back(Tensor(Shape({1, 2}), &backend_, {1.0f, 1.0f}), Tensor(Shape({1, 1}), &backend_, {0.0f}));
    }

    double TrainForEpochs(int num_epochs) override {
        for (int e = 0; e < num_epochs; ++e) {
            for (auto& [input, target] : dataset_) {
                (void)net_.train_step(input, target, optimizer_, sink_, step_++);
            }
        }
        double total = 0.0;
        for (auto& [input, target] : dataset_) {
            Tensor pred = net_.forward(input);
            double diff = pred.data()[0] - target.data()[0];
            total += diff * diff;
        }
        return -(total / static_cast<double>(dataset_.size()));  // maximize negative loss
    }

private:
    CPUBackend backend_;
    XorNetwork net_;
    AdamOptimizer optimizer_;
    NoOpMetricsSink sink_;
    std::vector<std::pair<Tensor, Tensor>> dataset_;
    int step_ = 0;
};

TrialFactory MakeXorTrialFactory() {
    return [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        return std::make_unique<XorResumableTrial>(lr);
    };
}

TEST(SuccessiveHalvingXorIntegrationTest, FindsAGoodLearningRateAmongRandomlySampledConfigs) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(2026);

    auto result = RunSuccessiveHalving(space, MakeXorTrialFactory(), /*num_configs=*/8,
                                        /*initial_epoch_budget=*/5, /*eta=*/2.0, rng);

    double final_loss = -result.best_metric;
    EXPECT_LT(final_loss, 0.1);  // a genuinely well-tuned learning rate should get well below this
}

TEST(SuccessiveHalvingXorIntegrationTest, UsesLessTotalComputeThanBruteForcingEveryConfigToMaxBudget) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(7);

    constexpr size_t kNumConfigs = 8;
    constexpr int kInitialBudget = 5;
    constexpr double kEta = 2.0;

    auto result = RunSuccessiveHalving(space, MakeXorTrialFactory(), kNumConfigs, kInitialBudget, kEta, rng);

    // The final rung's budget after 3 halvings (8 -> 4 -> 2 -> 1) is initial_budget * eta^3.
    int final_budget = static_cast<int>(kInitialBudget * kEta * kEta * kEta);
    size_t brute_force_cost = kNumConfigs * static_cast<size_t>(final_budget);

    EXPECT_LT(result.total_epochs_trained, brute_force_cost);
}

TEST(SuccessiveHalvingXorIntegrationTest, PicksTheGenuinelyGoodLearningRateOverDeliberatelyBadOnes) {
    // Explicit configs (the pure core, not RandomSample) -- one genuinely reasonable learning
    // rate, three deliberately pathological ones (too tiny to move at all in the rung budgets
    // used here, or large enough to destabilize training).
    std::vector<Configuration> configs = {
        Configuration{{"learning_rate", 0.05}},   // good
        Configuration{{"learning_rate", 1e-5}},   // far too small to make progress
        Configuration{{"learning_rate", 1e-5}},   // far too small to make progress
        Configuration{{"learning_rate", 3.0}},    // far too large, destabilizing
    };

    auto result = RunSuccessiveHalvingOnConfigs(configs, MakeXorTrialFactory(), /*initial_epoch_budget=*/5, /*eta=*/2.0);

    EXPECT_NEAR(std::get<double>(result.best_configuration.at("learning_rate")), 0.05, 1e-9);
}

}  // namespace
}  // namespace pulsatrix
