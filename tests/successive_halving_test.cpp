#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/successive_halving.hpp"

namespace pulsatrix {
namespace {

// A fully deterministic fake trial: metric = slope * cumulative_epochs_trained. Lets the
// exact rung-by-rung promotion/survival be hand-computed in advance, independent of any real
// (float, RNG-driven) network training -- that's what the *_xor_integration_test.cpp
// (real XorNetwork) proves separately.
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

// Hand-derived trace (4 configs, slopes 1/2/3/4, initial_budget=1, eta=2.0):
//   Rung 1 (budget=1): metrics = [1,2,3,4] (cumulative=1 each). Survivors (top floor(4/2)=2):
//     slope 4 (metric 4), slope 3 (metric 3). total_epochs_trained so far: 4*1=4.
//   Rung 2 (budget=2): cumulative becomes 1+2=3. metrics: slope4->12, slope3->9. Survivors
//     (top floor(2/2)=1): slope 4. total_epochs_trained so far: 4 + 2*2=4 -> 8.
//   Rung 3 (budget=2*2=4): only slope 4 remains, trained anyway (the final rung trains the
//     survivor(s) to the increased budget, standard Successive Halving behavior) -- cumulative
//     becomes 3+4=7, metric = 4*7 = 28. total_epochs_trained: 8 + 1*4=4 -> 12. Loop stops
//     (candidates.size() == 1 after this rung's training).
// Expected: best slope = 4, best_metric = 28.0, total_epochs_trained = 12.
TEST(RunSuccessiveHalvingOnConfigsTest, ExactHandDerivedPromotionAndFinalMetric) {
    std::vector<Configuration> configs = {
        Configuration{{"id", int64_t{1}}},
        Configuration{{"id", int64_t{2}}},
        Configuration{{"id", int64_t{3}}},
        Configuration{{"id", int64_t{4}}},
    };
    TrialFactory make_trial = [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        double slope = static_cast<double>(std::get<int64_t>(config.at("id")));
        return std::make_unique<FakeResumableTrial>(slope);
    };

    auto result = RunSuccessiveHalvingOnConfigs(configs, make_trial, /*initial_epoch_budget=*/1, /*eta=*/2.0);

    EXPECT_EQ(std::get<int64_t>(result.best_configuration.at("id")), 4);
    EXPECT_NEAR(result.best_metric, 28.0, 1e-9);
    EXPECT_EQ(result.total_epochs_trained, 12u);
}

TEST(RunSuccessiveHalvingOnConfigsTest, SingleConfigTrainsOnceAndReturnsItsOwnMetric) {
    std::vector<Configuration> configs = {Configuration{{"id", int64_t{7}}}};
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    auto result = RunSuccessiveHalvingOnConfigs(configs, make_trial, 5, 3.0);
    EXPECT_NEAR(result.best_metric, 5.0, 1e-9);  // slope=1, cumulative=5 after one rung
    EXPECT_EQ(result.total_epochs_trained, 5u);
}

TEST(RunSuccessiveHalvingOnConfigsTest, ThrowsOnEmptyConfigs) {
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunSuccessiveHalvingOnConfigs({}, make_trial, 1, 2.0), std::invalid_argument);
}

TEST(RunSuccessiveHalvingOnConfigsTest, ThrowsOnNonPositiveInitialBudget) {
    std::vector<Configuration> configs = {Configuration{{"id", int64_t{1}}}};
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunSuccessiveHalvingOnConfigs(configs, make_trial, 0, 2.0), std::invalid_argument);
    EXPECT_THROW(RunSuccessiveHalvingOnConfigs(configs, make_trial, -1, 2.0), std::invalid_argument);
}

TEST(RunSuccessiveHalvingOnConfigsTest, ThrowsOnEtaNotGreaterThanOne) {
    std::vector<Configuration> configs = {Configuration{{"id", int64_t{1}}}};
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunSuccessiveHalvingOnConfigs(configs, make_trial, 1, 1.0), std::invalid_argument);
    EXPECT_THROW(RunSuccessiveHalvingOnConfigs(configs, make_trial, 1, 0.5), std::invalid_argument);
}

TEST(RunSuccessiveHalvingTest, RNGWrapperThrowsOnZeroConfigs) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeResumableTrial>(1.0);
    };
    EXPECT_THROW(RunSuccessiveHalving(space, make_trial, 0, 1, 2.0, rng), std::invalid_argument);
}

TEST(RunSuccessiveHalvingTest, RNGWrapperProducesAValidSurvivor) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(42);
    TrialFactory make_trial = [](const Configuration& config) -> std::unique_ptr<ResumableTrial> {
        double x = std::get<double>(config.at("x"));
        return std::make_unique<FakeResumableTrial>(x);  // higher x -> higher metric
    };
    auto result = RunSuccessiveHalving(space, make_trial, 8, 1, 2.0, rng);
    EXPECT_TRUE(result.best_configuration.count("x"));
    EXPECT_GT(result.total_epochs_trained, 0u);
}

}  // namespace
}  // namespace pulsatrix
