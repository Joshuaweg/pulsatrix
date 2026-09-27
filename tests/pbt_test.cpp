#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/pbt.hpp"
#include "pulsatrix/pbt_trial.hpp"
#include "pulsatrix/search_space.hpp"

namespace pulsatrix {
namespace {

TEST(ComputeTruncationGroupsTest, SelectsExactlyOneOnEachEndForFiveIndividualsAtTwentyPercent) {
    std::vector<double> metrics{1.0, 2.0, 3.0, 4.0, 5.0};

    PBTTruncationGroups groups = ComputeTruncationGroups(metrics, /*truncation_fraction=*/0.2);

    EXPECT_EQ(groups.top_indices, (std::vector<size_t>{4}));
    EXPECT_EQ(groups.bottom_indices, (std::vector<size_t>{0}));
}

TEST(ComputeTruncationGroupsTest, SelectsTwoOnEachEndForTenIndividualsAtTwentyPercent) {
    std::vector<double> metrics{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};

    PBTTruncationGroups groups = ComputeTruncationGroups(metrics, /*truncation_fraction=*/0.2);

    EXPECT_EQ(groups.top_indices, (std::vector<size_t>{9, 8}));
    // bottom_indices is the tail of the descending-sorted order (worst-metric-last), i.e.
    // [1, 0] (index 1's metric 20 comes before index 0's metric 10 in descending order) --
    // not [0, 1].
    EXPECT_EQ(groups.bottom_indices, (std::vector<size_t>{1, 0}));
}

TEST(ComputeTruncationGroupsTest, ThrowsOnFewerThanTwoMetrics) {
    EXPECT_THROW((void)ComputeTruncationGroups({1.0}, 0.2), std::invalid_argument);
}

TEST(ComputeTruncationGroupsTest, ThrowsOnNonPositiveTruncationFraction) {
    EXPECT_THROW((void)ComputeTruncationGroups({1.0, 2.0}, 0.0), std::invalid_argument);
}

TEST(ComputeTruncationGroupsTest, ThrowsOnTruncationFractionAboveOneHalf) {
    EXPECT_THROW((void)ComputeTruncationGroups({1.0, 2.0}, 0.6), std::invalid_argument);
}

TEST(ExploreConfigurationGivenFactorsTest, PerturbsContinuousParameterByExplicitFactor) {
    SearchSpace space;
    space.AddContinuous("lr", 0.0, 1.0);
    Configuration config{{"lr", 0.1}};

    Configuration up = ExploreConfigurationGivenFactors(config, space, {{"lr", 1.2}});
    Configuration down = ExploreConfigurationGivenFactors(config, space, {{"lr", 0.8}});

    EXPECT_NEAR(std::get<double>(up.at("lr")), 0.12, 1e-12);
    EXPECT_NEAR(std::get<double>(down.at("lr")), 0.08, 1e-12);
}

TEST(ExploreConfigurationGivenFactorsTest, ClampsToParameterBounds) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    Configuration config{{"x", 0.9}};

    Configuration result = ExploreConfigurationGivenFactors(config, space, {{"x", 1.2}});

    EXPECT_NEAR(std::get<double>(result.at("x")), 1.0, 1e-12);
}

TEST(ExploreConfigurationGivenFactorsTest, RoundsIntegerParameterAfterPerturbation) {
    SearchSpace space;
    space.AddInteger("batch_size", 1, 100);
    Configuration config{{"batch_size", static_cast<int64_t>(10)}};

    Configuration result = ExploreConfigurationGivenFactors(config, space, {{"batch_size", 1.2}});

    EXPECT_EQ(std::get<int64_t>(result.at("batch_size")), 12);
}

TEST(ExploreConfigurationGivenFactorsTest, LeavesCategoricalParameterUnchanged) {
    SearchSpace space;
    space.AddCategorical("opt", {"adam", "sgd"});
    Configuration config{{"opt", std::string("adam")}};

    Configuration result = ExploreConfigurationGivenFactors(config, space, {});

    EXPECT_EQ(std::get<std::string>(result.at("opt")), "adam");
}

TEST(ExploreConfigurationGivenFactorsTest, ThrowsOnMissingConfigParameter) {
    SearchSpace space;
    space.AddContinuous("lr", 0.0, 1.0);
    EXPECT_THROW((void)ExploreConfigurationGivenFactors({}, space, {{"lr", 1.2}}), std::invalid_argument);
}

TEST(ExploreConfigurationGivenFactorsTest, ThrowsOnMissingFactorForNonCategoricalParameter) {
    SearchSpace space;
    space.AddContinuous("lr", 0.0, 1.0);
    Configuration config{{"lr", 0.1}};
    EXPECT_THROW((void)ExploreConfigurationGivenFactors(config, space, {}), std::invalid_argument);
}

TEST(ExploreConfigurationTest, PerturbsByExactlyOneOfTheTwoStandardFactors) {
    SearchSpace space;
    space.AddContinuous("lr", 0.0, 10.0);
    Configuration config{{"lr", 1.0}};
    std::mt19937 rng(42);

    Configuration result = ExploreConfiguration(config, space, rng);

    double value = std::get<double>(result.at("lr"));
    EXPECT_TRUE(value == 0.8 || value == 1.2) << "value was " << value;
}

// A minimal PBTResumableTrial whose "weights" is a single running total and whose only
// hyperparameter ("rate") controls how fast that total grows -- lets RunPBTGeneration's own
// exploit/explore behavior be verified with a fully hand-derived fixture, independent of any
// real network.
class FakeTrial : public PBTResumableTrial {
public:
    explicit FakeTrial(double rate) : rate_(rate) {}

    double TrainForEpochs(int num_epochs) override {
        value_ += rate_ * static_cast<double>(num_epochs);
        return value_;
    }
    [[nodiscard]] std::vector<double> GetWeights() const override { return {value_}; }
    void SetWeights(const std::vector<double>& weights) override { value_ = weights.at(0); }
    [[nodiscard]] Configuration GetHyperparameters() const override { return {{"rate", rate_}}; }
    void SetHyperparameters(const Configuration& config) override { rate_ = std::get<double>(config.at("rate")); }

private:
    double rate_;
    double value_ = 0.0;
};

TEST(RunPBTGenerationTest, ExploitsBestPerformerOntoWorstPerformerAfterOneEpoch) {
    SearchSpace space;
    space.AddContinuous("rate", 0.0, 100.0);

    std::vector<std::unique_ptr<PBTResumableTrial>> trials;
    trials.push_back(std::make_unique<FakeTrial>(1.0));
    trials.push_back(std::make_unique<FakeTrial>(2.0));
    trials.push_back(std::make_unique<FakeTrial>(3.0));
    trials.push_back(std::make_unique<FakeTrial>(4.0));
    std::mt19937 rng(42);

    // After TrainForEpochs(1): metrics = [1, 2, 3, 4]. truncation_fraction=0.25 on 4
    // individuals selects exactly 1 on each end: top=[3] (rate 4.0), bottom=[0] (rate 1.0).
    // Trial 0 is exploited onto trial 3's post-training value (4.0), then its hyperparameter
    // is explored from the copied rate (4.0) by either 0.8 or 1.2.
    std::vector<double> metrics = RunPBTGeneration(trials, space, /*num_epochs=*/1,
                                                    /*truncation_fraction=*/0.25, rng);

    EXPECT_EQ(metrics, (std::vector<double>{4.0, 2.0, 3.0, 4.0}));
    Configuration exploited_hparams = trials[0]->GetHyperparameters();
    double exploited_rate = std::get<double>(exploited_hparams.at("rate"));
    EXPECT_TRUE(exploited_rate == 3.2 || exploited_rate == 4.8) << "rate was " << exploited_rate;
    // Trials outside both groups are untouched.
    EXPECT_EQ(std::get<double>(trials[1]->GetHyperparameters().at("rate")), 2.0);
    EXPECT_EQ(std::get<double>(trials[2]->GetHyperparameters().at("rate")), 3.0);
}

TEST(RunPBTGenerationTest, ThrowsOnEmptyTrials) {
    SearchSpace space;
    space.AddContinuous("rate", 0.0, 100.0);
    std::vector<std::unique_ptr<PBTResumableTrial>> trials;
    std::mt19937 rng(1);
    EXPECT_THROW((void)RunPBTGeneration(trials, space, 1, 0.2, rng), std::invalid_argument);
}

TEST(RunPBTGenerationTest, ThrowsOnNonPositiveEpochs) {
    SearchSpace space;
    space.AddContinuous("rate", 0.0, 100.0);
    std::vector<std::unique_ptr<PBTResumableTrial>> trials;
    trials.push_back(std::make_unique<FakeTrial>(1.0));
    std::mt19937 rng(1);
    EXPECT_THROW((void)RunPBTGeneration(trials, space, 0, 0.2, rng), std::invalid_argument);
}

TEST(RunPBTTest, ThrowsOnNonPositiveGenerations) {
    SearchSpace space;
    space.AddContinuous("rate", 0.0, 100.0);
    std::vector<std::unique_ptr<PBTResumableTrial>> trials;
    trials.push_back(std::make_unique<FakeTrial>(1.0));
    trials.push_back(std::make_unique<FakeTrial>(2.0));
    std::mt19937 rng(1);
    EXPECT_THROW((void)RunPBT(trials, space, 0, 1, 0.2, rng), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
