#include <gtest/gtest.h>

#include <vector>

#include "pulsatrix/individual.hpp"

namespace pulsatrix {
namespace {

TEST(IndividualTest, StoresGenesAndFitness) {
    Individual<int, double> ind{42, 3.5};
    EXPECT_EQ(ind.genes, 42);
    EXPECT_DOUBLE_EQ(ind.fitness, 3.5);
}

TEST(IndividualTest, DefaultFitnessIsZeroInitialized) {
    Individual<int, double> ind{7, {}};
    EXPECT_DOUBLE_EQ(ind.fitness, 0.0);
}

TEST(IndividualTest, GenotypeCanBeAnArbitraryType) {
    Individual<std::vector<double>, double> ind{{1.0, 2.0, 3.0}, 0.0};
    EXPECT_EQ(ind.genes.size(), 3u);
}

}  // namespace
}  // namespace pulsatrix
