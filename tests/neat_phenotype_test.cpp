#include <cmath>

#include <gtest/gtest.h>

#include "pulsatrix/neat_genome.hpp"
#include "pulsatrix/neat_phenotype.hpp"

namespace pulsatrix {
namespace {

TEST(NEATPhenotypeTest, SingleConnectionProducesExpectedSigmoidOutput) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, /*has_bias=*/false, tracker);
    genome.SetConnectionWeight(0, 2.0);

    std::vector<double> outputs = EvaluateNEATPhenotype(genome, {0.5});

    double expected = 1.0 / (1.0 + std::exp(-4.9 * 1.0));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], expected, 1e-12);
}

TEST(NEATPhenotypeTest, BiasConnectionContributesToSum) {
    InnovationTracker tracker(3);
    NEATGenome genome(1, 1, /*has_bias=*/true, tracker);
    // Genome(1 input, 1 output, has_bias=true) creates innov0 (in0->out) and innov1 (bias->out).
    genome.SetConnectionWeight(0, 1.0);
    genome.SetConnectionWeight(1, 0.5);

    std::vector<double> outputs = EvaluateNEATPhenotype(genome, {0.3});

    double expected = 1.0 / (1.0 + std::exp(-4.9 * 0.8));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], expected, 1e-12);
}

TEST(NEATPhenotypeTest, HiddenNodeFromSplitPropagatesThroughTwoLayers) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, /*has_bias=*/false, tracker);
    // Splitting innov0 creates hidden node id=2, innov1 (0->2, w=1.0 by convention),
    // innov2 (2->1, weight preserved from the original connection).
    genome.AddNodeSplitting(0, tracker);
    genome.SetConnectionWeight(2, 3.0);

    std::vector<double> outputs = EvaluateNEATPhenotype(genome, {0.2});

    double hidden = 1.0 / (1.0 + std::exp(-4.9 * 0.2));
    double expected = 1.0 / (1.0 + std::exp(-4.9 * 3.0 * hidden));
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_NEAR(outputs[0], expected, 1e-12);
}

TEST(NEATPhenotypeTest, ThrowsWhenInputSizeDoesNotMatchGenomeInputCount) {
    InnovationTracker tracker(3);
    NEATGenome genome(2, 1, /*has_bias=*/false, tracker);

    EXPECT_THROW((void)EvaluateNEATPhenotype(genome, {0.5}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
