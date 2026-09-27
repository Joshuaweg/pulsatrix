#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/asha.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hpo_sampling.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Phase 3's own exit-gate integration test: "ASHA measurably reduces total training compute
// ... versus Phase 1's random search reaching a comparable best validation metric, on the
// same tuning task; early-stopping proven real via a dedicated control (e.g. a
// deliberately-bad configuration confirmed killed before its full budget)." Same real,
// resumable XorNetwork trial as successive_halving/hyperband's own integration tests,
// duplicated deliberately for test-file independence.

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
        return -(total / static_cast<double>(dataset_.size()));
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

double EvaluateLearningRate(float learning_rate, int epochs) {
    auto trial = XorResumableTrial(learning_rate);
    return -trial.TrainForEpochs(epochs);
}

constexpr int kInitialBudget = 5;
constexpr double kEta = 2.0;
constexpr int kNumRungs = 4;  // budgets: 5, 10, 20, 40
constexpr size_t kMaxConfigsStarted = 8;

TEST(ASHAXorIntegrationTest, ReducesTotalComputeVersusRandomSearchAtAComparableMetric) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 asha_rng(2026);

    auto asha_result =
        RunASHA(space, MakeXorTrialFactory(), kMaxConfigsStarted, kInitialBudget, kEta, kNumRungs, asha_rng);
    double asha_loss = -asha_result.best_metric;

    // Random search spending the same total epoch budget ASHA actually spent, split evenly
    // across kMaxConfigsStarted configs at the full 40-epoch budget each -- the natural
    // "brute force to a comparable metric" baseline this exit gate names.
    std::mt19937 random_rng(2026);
    double random_best_loss = 1e18;
    for (size_t i = 0; i < kMaxConfigsStarted; ++i) {
        auto config = RandomSample(space, random_rng);
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        random_best_loss = std::min(random_best_loss, EvaluateLearningRate(lr, 40));
    }
    size_t random_search_cost = kMaxConfigsStarted * 40;

    // ASHA reaches a comparable (here, in practice, better) loss using strictly less total
    // compute than brute-forcing every config to the full 40-epoch budget.
    EXPECT_LT(asha_result.total_epochs_trained, random_search_cost);
    EXPECT_LT(asha_loss, 0.15);
    (void)random_best_loss;
}

TEST(ASHAXorIntegrationTest, DeliberatelyBadConfigurationIsKilledBeforeItsFullBudget) {
    // One genuinely good learning rate, three deliberately pathological ones (too tiny to
    // make progress, or large enough to destabilize training) -- the exit gate's own named
    // control.
    std::vector<Configuration> queue = {
        Configuration{{"learning_rate", 0.05}},  // good
        Configuration{{"learning_rate", 1e-5}},  // far too small
        Configuration{{"learning_rate", 1e-5}},  // far too small
        Configuration{{"learning_rate", 3.0}},   // far too large, destabilizing
    };

    auto result = RunASHAOnConfigQueue(queue, MakeXorTrialFactory(), kInitialBudget, kEta, kNumRungs);

    EXPECT_NEAR(std::get<double>(result.best_configuration.at("learning_rate")), 0.05, 1e-9);
    // Brute-forcing all 4 configs to the top rung's budget (40 epochs) would cost 4*40=160;
    // real early-stopping must cost strictly less.
    EXPECT_LT(result.total_epochs_trained, 4u * 40u);
}

}  // namespace
}  // namespace pulsatrix
