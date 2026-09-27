#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "pulsatrix/individual.hpp"
#include "pulsatrix/survivor_selection.hpp"

namespace pulsatrix {
namespace {

using IntInd = Individual<int, double>;

std::vector<IntInd> MakePop(std::initializer_list<double> fitnesses) {
    std::vector<IntInd> pop;
    int genes = 0;
    for (double f : fitnesses) {
        pop.push_back(IntInd{genes++, f});
    }
    return pop;
}

std::vector<double> FitnessValues(const std::vector<IntInd>& pop) {
    std::vector<double> values;
    for (const auto& ind : pop) {
        values.push_back(ind.fitness);
    }
    std::sort(values.begin(), values.end());
    return values;
}

// --- GenerationalReplacement ---

TEST(GenerationalReplacementTest, ReturnsOffspringUnchangedIgnoringPopulation) {
    auto population = MakePop({100.0, 200.0});  // deliberately much fitter than offspring
    auto offspring = MakePop({1.0, 2.0, 3.0});
    auto next = GenerationalReplacement(population, offspring, 3);
    EXPECT_EQ(FitnessValues(next), (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(GenerationalReplacementTest, ThrowsWhenOffspringSizeDoesNotEqualMu) {
    auto population = MakePop({1.0});
    auto offspring = MakePop({1.0, 2.0});
    EXPECT_THROW(GenerationalReplacement(population, offspring, 3), std::invalid_argument);
}

// --- MuPlusLambdaReplacement ---

TEST(MuPlusLambdaReplacementTest, KeepsFittestAcrossPopulationAndOffspring) {
    auto population = MakePop({5.0, 9.0});    // one parent (fitness 9) fitter than all offspring
    auto offspring = MakePop({1.0, 2.0, 3.0});
    auto next = MuPlusLambdaReplacement(population, offspring, 3);
    // Top 3 of {5, 9, 1, 2, 3} = {9, 5, 3}
    EXPECT_EQ(FitnessValues(next), (std::vector<double>{3.0, 5.0, 9.0}));
}

TEST(MuPlusLambdaReplacementTest, ThrowsWhenCombinedPoolSmallerThanMu) {
    auto population = MakePop({1.0});
    auto offspring = MakePop({2.0});
    EXPECT_THROW(MuPlusLambdaReplacement(population, offspring, 3), std::invalid_argument);
}

// --- MuCommaLambdaReplacement ---

TEST(MuCommaLambdaReplacementTest, KeepsFittestOffspringOnlyDiscardingParents) {
    // Parent fitness 1000 is far fitter than every offspring, but (mu,lambda) must discard it.
    auto population = MakePop({1000.0});
    auto offspring = MakePop({1.0, 2.0, 3.0, 4.0});
    auto next = MuCommaLambdaReplacement(population, offspring, 2);
    EXPECT_EQ(FitnessValues(next), (std::vector<double>{3.0, 4.0}));
}

TEST(MuCommaLambdaReplacementTest, ThrowsWhenOffspringSmallerThanMu) {
    auto population = MakePop({1.0});
    auto offspring = MakePop({2.0});
    EXPECT_THROW(MuCommaLambdaReplacement(population, offspring, 2), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
