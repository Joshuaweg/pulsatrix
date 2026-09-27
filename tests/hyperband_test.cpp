#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>

#include "pulsatrix/hyperband.hpp"

namespace pulsatrix {
namespace {

// --- ComputeHyperbandBrackets ---

// Hand-derived for max_resource=8, eta=2: s_max = floor(log2(8)) = 3, B = (3+1)*8 = 32.
//   s=3: eta^s=8,  n=ceil((32/8)*(8/4))=ceil(4*2)=8,      r=round(8/8)=1
//   s=2: eta^s=4,  n=ceil((32/8)*(4/3))=ceil(4*1.333)=6,  r=round(8/4)=2
//   s=1: eta^s=2,  n=ceil((32/8)*(2/2))=ceil(4)=4,        r=round(8/2)=4
//   s=0: eta^s=1,  n=ceil((32/8)*(1/1))=ceil(4)=4,        r=round(8/1)=8
TEST(ComputeHyperbandBracketsTest, ExactHandDerivedScheduleForResourceEightEtaTwo) {
    auto brackets = ComputeHyperbandBrackets(8, 2.0);
    ASSERT_EQ(brackets.size(), 4u);

    EXPECT_EQ(brackets[0].s, 3);
    EXPECT_EQ(brackets[0].num_configs, 8u);
    EXPECT_EQ(brackets[0].initial_budget, 1);

    EXPECT_EQ(brackets[1].s, 2);
    EXPECT_EQ(brackets[1].num_configs, 6u);
    EXPECT_EQ(brackets[1].initial_budget, 2);

    EXPECT_EQ(brackets[2].s, 1);
    EXPECT_EQ(brackets[2].num_configs, 4u);
    EXPECT_EQ(brackets[2].initial_budget, 4);

    EXPECT_EQ(brackets[3].s, 0);
    EXPECT_EQ(brackets[3].num_configs, 4u);
    EXPECT_EQ(brackets[3].initial_budget, 8);
}

TEST(ComputeHyperbandBracketsTest, ThrowsOnNonPositiveMaxResource) {
    EXPECT_THROW(ComputeHyperbandBrackets(0, 2.0), std::invalid_argument);
    EXPECT_THROW(ComputeHyperbandBrackets(-1, 2.0), std::invalid_argument);
}

TEST(ComputeHyperbandBracketsTest, ThrowsOnEtaNotGreaterThanOne) {
    EXPECT_THROW(ComputeHyperbandBrackets(8, 1.0), std::invalid_argument);
    EXPECT_THROW(ComputeHyperbandBrackets(8, 0.5), std::invalid_argument);
}

// --- RunHyperband ---

// Every configuration behaves identically regardless of its sampled hyperparameter value --
// metric = min(cumulative_epochs_trained, cap) -- so the *only* way to reach the cap is for
// some bracket to train a candidate at least `cap` cumulative epochs. With max_resource=16
// (> cap=10), the s=0 bracket alone trains its survivors a full 16 epochs in its single
// round, guaranteeing the cap is reached -- a result knowable in advance regardless of RNG
// draws, since every candidate is behaviorally identical.
class FakeCappedTrial : public ResumableTrial {
public:
    explicit FakeCappedTrial(double cap) : cap_(cap) {}
    double TrainForEpochs(int num_epochs) override {
        cumulative_epochs_ += num_epochs;
        return std::min(static_cast<double>(cumulative_epochs_), cap_);
    }

private:
    double cap_;
    int cumulative_epochs_ = 0;
};

TEST(RunHyperbandTest, OverallBestReachesTheCapGuaranteedByTheFullBudgetBracket) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeCappedTrial>(10.0);
    };
    auto result = RunHyperband(space, make_trial, /*max_resource=*/16, /*eta=*/2.0, rng);
    EXPECT_NEAR(result.best_metric, 10.0, 1e-9);
    EXPECT_GT(result.total_epochs_trained, 0u);
}

TEST(RunHyperbandTest, ThrowsOnInvalidParameters) {
    SearchSpace space;
    space.AddContinuous("x", 0.0, 1.0);
    std::mt19937 rng(0);
    TrialFactory make_trial = [](const Configuration&) -> std::unique_ptr<ResumableTrial> {
        return std::make_unique<FakeCappedTrial>(1.0);
    };
    EXPECT_THROW(RunHyperband(space, make_trial, 0, 2.0, rng), std::invalid_argument);
    EXPECT_THROW(RunHyperband(space, make_trial, 8, 1.0, rng), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
