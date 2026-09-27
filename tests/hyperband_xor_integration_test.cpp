#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hyperband.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/search_space.hpp"
#include "pulsatrix/xor_training_example.hpp"

namespace pulsatrix {
namespace {

// Same real, resumable XorNetwork trial as successive_halving_xor_integration_test.cpp --
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

TEST(HyperbandXorIntegrationTest, FindsAGoodLearningRateAcrossMultipleBrackets) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    std::mt19937 rng(2026);

    TrialFactory make_trial = [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        float lr = static_cast<float>(std::get<double>(config.at("learning_rate")));
        return std::make_unique<XorResumableTrial>(lr);
    };

    auto result = RunHyperband(space, make_trial, /*max_resource=*/16, /*eta=*/2.0, rng);

    double final_loss = -result.best_metric;
    EXPECT_LT(final_loss, 0.1);
}

}  // namespace
}  // namespace pulsatrix
