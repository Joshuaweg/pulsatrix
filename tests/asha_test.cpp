#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/asha.hpp"

namespace pulsatrix {
namespace {

// Same deterministic fake trial as successive_halving_test.cpp: metric = slope *
// cumulative_epochs_trained.
class FakeResumableTrial : public ResumableTrial {
public:
    explicit FakeResumableTrial(double slope) : slope_(slope) {}
    double TrainForEpochs(int num_epochs) override {
        cumulative_epochs_ += num_epochs;
        return slope_ * static_cast<double>(cumulative_epochs_);
    }

private:
    double slope_;
    int cumulative_epochs_ = 0;
};

// Hand-traced (see mission notes for the full step-by-step derivation): 4 configs, slopes
// 1/2/3/4 in queue order, initial_budget=1, eta=2.0, num_rungs=3 (budgets 1, 2, 4).
//
// Step-by-step: c1,c2 start at rung 0 (metrics 1, 2) since rung 0 has no history yet to
// promote from. Once rung_history[0] = [1,2] (size >= eta=2), c2 (the only one meeting the
// top-1/2 cutoff of 2) promotes to rung 1 (cumulative 1+1=2, metric 4). c3 starts (metric 3),
// then promotes (cumulative 1+1=2, metric 6, now the top-1/2 of [1,2,3]'s cutoff 3) to rung 1,
// then promotes again (cumulative 2+2=4, metric 12) to rung 2 once rung_history[1]=[4,6]
// makes it the top-1/2 (cutoff 6). c4 starts (metric 4), promotes to rung 1 (cumulative 2,
// metric 8, top-1/2 of [1,2,3,4]'s cutoff 3), then promotes to rung 2 (cumulative 4, metric
// 16, top-1/2 of [4,6,8]'s cutoff 8). Final state: c1 stuck at rung 0 (metric 1, only 1 of a
// possible 4 epochs), c2 stuck at rung 1 (metric 4, only 2 of a possible 4 epochs), c3 and c4
// both reach rung 2. Best: c4, metric 16.0. Total epochs trained: c1=1, c2=1+1=2,
// c3=1+1+2=4, c4=1+1+2=4 -> 1+2+4+4=11 (vs. 4*4=16 for brute-forcing every config to the
// max rung's budget).
TEST(RunASHAOnConfigQueueTest, ExactHandTracedPromotionAndFinalMetric) {
    std::vector<Configuration> queue = {
        Configuration{{"id", int64_t{1}}},
        Configuration{{"id", int64_t{2}}},
        Configuration{{"id", int64_t{3}}},
        Configuration{{"id", int64_t{4}}},
    };
    TrialFactory make_trial = [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        double slope = static_cast<double>(std::get<int64_t>(config.at("id")));
        return std::make_unique<FakeResumableTrial>(slope);
    };

    auto result = RunASHAOnConfigQueue(queue, make_trial, /*initial_epoch_budget=*/1, /*eta=*/2.0, /*num_rungs=*/3);

    EXPECT_EQ(std::get<int64_t>(result.best_configuration.at("id")), 4);
    EXPECT_NEAR(result.best_metric, 16.0, 1e-9);
    EXPECT_EQ(result.total_epochs_trained, 11u);
    EXPECT_EQ(result.num_configs_started, 4u);
}

TEST(RunASHAOnConfigQueueTest, ThrowsOnEmptyQueue) {
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunASHAOnConfigQueue({}, make_trial, 1, 2.0, 3), std::invalid_argument);
}

TEST(RunASHAOnConfigQueueTest, ThrowsOnInvalidParameters) {
    std::vector<Configuration> queue = {Configuration{{"id", int64_t{1}}}};
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunASHAOnConfigQueue(queue, make_trial, 0, 2.0, 3), std::invalid_argument);
    EXPECT_THROW(RunASHAOnConfigQueue(queue, make_trial, 1, 1.0, 3), std::invalid_argument);
    EXPECT_THROW(RunASHAOnConfigQueue(queue, make_trial, 1, 2.0, 1), std::invalid_argument);
}

TEST(RunASHATest, RNGWrapperThrowsOnZeroMaxConfigs) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunASHA(space, make_trial, 0, 1, 2.0, 3, rng), std::invalid_argument);
}

TEST(RunASHATest, RNGWrapperProducesAValidResult) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(42);
    TrialFactory make_trial = [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        double x = std::get<double>(config.at("x"));
        return std::make_unique<FakeResumableTrial>(x);
    };
    auto result = RunASHA(space, make_trial, 6, 1, 2.0, 3, rng);
    EXPECT_TRUE(result.best_configuration.count("x"));
    EXPECT_GT(result.total_epochs_trained, 0u);
    EXPECT_EQ(result.num_configs_started, 6u);
}

}  // namespace
}  // namespace pulsatrix
