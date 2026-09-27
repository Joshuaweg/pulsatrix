#include <vector>

#include <gtest/gtest.h>

#include "pulsatrix/fixed_topology_xor_network.hpp"

namespace pulsatrix {
namespace {

TEST(FixedTopologyXORNetworkTest, AllZeroThetaScoresExactlyThree) {
    std::vector<double> theta(kFixedTopologyXORNumParams, 0.0);

    double fitness = FixedTopologyXORFitness(theta);

    EXPECT_NEAR(fitness, 3.0, 1e-12);
}

TEST(FixedTopologyXORNetworkTest, ChangingWeightsChangesFitness) {
    std::vector<double> theta{5.0, 5.0, -2.5, -5.0, -5.0, 7.5, 5.0, 5.0, -2.5};

    double fitness = FixedTopologyXORFitness(theta);

    EXPECT_NE(fitness, 3.0);
}

TEST(FixedTopologyXORNetworkTest, ForwardThrowsOnWrongThetaSize) {
    std::vector<double> theta{1.0, 2.0};
    EXPECT_THROW((void)FixedTopologyXORForward(theta, {0.0, 0.0}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
