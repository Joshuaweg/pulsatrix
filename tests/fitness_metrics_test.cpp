// PLM-3: ProteinGym's metrics against scipy 1.x / scikit-learn and ProteinGym's own
// performance_DMS_benchmarks.py (calc_ndcg, calc_toprecall), run on the same 20 variants.
#include "pulsatrix/fitness_metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace pulsatrix {
namespace {

const std::vector<double> kTrue{0.1, -1.2, 0.5, 0.5, 2.0, -0.3, 1.1, 0.0, -2.5, 0.8, 1.7, -0.9, 0.3, 0.5, -1.6, 2.2, 0.9, -0.1, 1.4, -0.7};
const std::vector<double> kPred{0.4, -0.8, 0.2, 0.9, 1.5, -0.6, 0.7, 0.1, -2.0, 1.2, 0.6, -0.2, -0.4, 0.3, -1.1, 1.9, 1.0, 0.05, 2.1, -1.3};

std::vector<int> Bins() {
    std::vector<int> b;
    for (double t : kTrue) b.push_back(t >= 0.5 ? 1 : 0);
    return b;
}

TEST(FitnessMetrics, MatchScipySklearnAndProteinGym) {
    EXPECT_NEAR(SpearmanCorrelation(kTrue, kPred), 0.9156636890153327, 1e-12);  // tied true scores
    EXPECT_NEAR(RocAuc(Bins(), kPred), 0.98, 1e-12);
    EXPECT_NEAR(MedianSplitMcc(Bins(), kPred), 0.8, 1e-12);
    EXPECT_NEAR(Ndcg(kTrue, kPred), 0.9106250677764092, 1e-12);
    EXPECT_NEAR(TopRecall(kTrue, kPred), 0.5, 1e-12);
    EXPECT_NEAR(Ndcg(kTrue, kPred, 25), 0.9323768990463199, 1e-12);
    EXPECT_NEAR(TopRecall(kTrue, kPred, 25, 25), 0.6, 1e-12);
    EXPECT_NEAR(Percentile(kTrue, 90), 1.73, 1e-12);
    EXPECT_NEAR(Percentile(kTrue, 37.5), 0.0125, 1e-12);

    const FitnessMetrics m = EvaluateFitness(kTrue, Bins(), kPred);
    EXPECT_EQ(m.spearman, SpearmanCorrelation(kTrue, kPred));
    EXPECT_EQ(m.auc, RocAuc(Bins(), kPred));
    EXPECT_EQ(m.mcc, MedianSplitMcc(Bins(), kPred));
    EXPECT_EQ(m.ndcg, Ndcg(kTrue, kPred));
    EXPECT_EQ(m.top_recall, TopRecall(kTrue, kPred));
}

TEST(FitnessMetrics, EdgeCases) {
    EXPECT_NEAR(SpearmanCorrelation({1, 2, 3}, {30, 20, 10}), -1.0, 1e-12);
    EXPECT_TRUE(std::isnan(SpearmanCorrelation({1, 2, 3}, {5, 5, 5})));
    EXPECT_TRUE(std::isnan(RocAuc({1, 1, 1}, {0.1, 0.2, 0.3})));
    EXPECT_NEAR(RocAuc({0, 1, 0, 1}, {0.5, 0.5, 0.1, 0.9}), 0.875, 1e-12);  // a tie counts half
    EXPECT_EQ(MatthewsCorrelation({1, 1, 0, 0}, {1, 1, 1, 1}), 0.0);
    EXPECT_NEAR(MatthewsCorrelation({1, 0, 1, 0}, {1, 0, 1, 0}), 1.0, 1e-12);
    EXPECT_THROW((void)SpearmanCorrelation({1, 2}, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW((void)SpearmanCorrelation({1}, {1}), std::invalid_argument);
    EXPECT_THROW((void)RocAuc({0, 2}, {0.1, 0.2}), std::invalid_argument);
    EXPECT_THROW((void)Percentile({}, 50), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
