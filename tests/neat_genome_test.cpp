#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <stdexcept>

#include "pulsatrix/neat_genome.hpp"

namespace pulsatrix {
namespace {

// --- InnovationTracker ---

TEST(InnovationTrackerTest, ReusesSameInnovationForSameConnection) {
    InnovationTracker tracker(0);
    int first = tracker.GetConnectionInnovation(1, 2);
    int second = tracker.GetConnectionInnovation(1, 2);
    EXPECT_EQ(first, second);
}

TEST(InnovationTrackerTest, AssignsDistinctIncreasingInnovationsForDifferentConnections) {
    InnovationTracker tracker(0);
    int a = tracker.GetConnectionInnovation(1, 2);
    int b = tracker.GetConnectionInnovation(1, 3);
    EXPECT_NE(a, b);
    EXPECT_LT(a, b);
}

TEST(InnovationTrackerTest, ReusesSameNodeIdForSameSplit) {
    InnovationTracker tracker(10);
    int first = tracker.GetNodeIdForSplit(5);
    int second = tracker.GetNodeIdForSplit(5);
    EXPECT_EQ(first, second);
}

TEST(InnovationTrackerTest, AssignsDistinctNodeIdsForDifferentSplits) {
    InnovationTracker tracker(10);
    int a = tracker.GetNodeIdForSplit(5);
    int b = tracker.GetNodeIdForSplit(7);
    EXPECT_NE(a, b);
}

// --- NEATGenome construction ---

TEST(NEATGenomeTest, MinimalTopologyIsFullyConnectedWithSequentialInnovations) {
    InnovationTracker tracker(3);  // 2 inputs + 1 bias => next_node_id starts at 3
    NEATGenome genome(2, 1, /*has_bias=*/true, tracker);

    ASSERT_EQ(genome.nodes().size(), 4u);  // 2 inputs + 1 bias + 1 output
    ASSERT_EQ(genome.connections().size(), 3u);  // every input(+bias) -> the one output

    for (const auto& c : genome.connections()) {
        EXPECT_TRUE(c.enabled);
        EXPECT_DOUBLE_EQ(c.weight, 0.0);
    }
    std::vector<int> innovations;
    for (const auto& c : genome.connections()) {
        innovations.push_back(c.innovation);
    }
    std::sort(innovations.begin(), innovations.end());
    EXPECT_EQ(innovations, (std::vector<int>{0, 1, 2}));
}

TEST(NEATGenomeTest, ThrowsOnNonPositiveInputsOrOutputs) {
    InnovationTracker tracker(0);
    EXPECT_THROW(NEATGenome(0, 1, false, tracker), std::invalid_argument);
    EXPECT_THROW(NEATGenome(1, 0, false, tracker), std::invalid_argument);
}

// --- AddConnectionBetween (pure core) ---

TEST(NEATGenomeTest, AddConnectionBetweenAddsExactConnectionWithCorrectInnovation) {
    InnovationTracker tracker(2);  // 1 input(id0) + 1 output(id1)
    NEATGenome genome(1, 1, false, tracker);
    // connections so far: {0,1,0.0,true,innov0}. The pure core doesn't enforce feedforward
    // direction (that's AddConnection's own wrapper's job) -- (1,0) is a genuinely new,
    // not-yet-existing pair, exercising the gene-creation/innovation-assignment mechanics
    // directly.
    genome.AddConnectionBetween(1, 0, 3.5, tracker);

    const auto& connections = genome.connections();
    auto it = std::find_if(connections.begin(), connections.end(),
                            [](const ConnectionGene& c) { return c.in_node == 1 && c.out_node == 0; });
    ASSERT_NE(it, connections.end());
    EXPECT_DOUBLE_EQ(it->weight, 3.5);
    EXPECT_TRUE(it->enabled);
    EXPECT_EQ(it->innovation, 1);
    EXPECT_EQ(connections.size(), 2u);
}

TEST(NEATGenomeTest, AddConnectionBetweenThrowsOnUnknownNode) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    EXPECT_THROW(genome.AddConnectionBetween(0, 99, 1.0, tracker), std::invalid_argument);
}

TEST(NEATGenomeTest, AddConnectionBetweenThrowsOnDuplicateConnection) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    EXPECT_THROW(genome.AddConnectionBetween(0, 1, 1.0, tracker), std::invalid_argument);
}

TEST(NEATGenomeTest, AddConnectionBetweenReusesInnovationAcrossGenomesSharingATracker) {
    InnovationTracker tracker(2);  // both genomes share the identical initial 1-input/1-output shape
    NEATGenome genome_a(1, 1, false, tracker);
    NEATGenome genome_b(1, 1, false, tracker);

    // Both genomes independently create the exact same new structural connection (with
    // different weights -- the innovation number tracks *structure*, not weight).
    genome_a.AddConnectionBetween(1, 0, 1.0, tracker);
    genome_b.AddConnectionBetween(1, 0, -2.0, tracker);

    auto find_reverse_innovation = [](const NEATGenome& g) {
        for (const auto& c : g.connections()) {
            if (c.in_node == 1 && c.out_node == 0) {
                return c.innovation;
            }
        }
        return -1;
    };
    EXPECT_EQ(find_reverse_innovation(genome_a), find_reverse_innovation(genome_b));
}

// --- AddNodeSplitting (pure core) ---

TEST(NEATGenomeTest, AddNodeSplittingDisablesOriginalAndPreservesWeightOnSecondSplit) {
    InnovationTracker tracker(2);  // 1 input(id0) + 1 output(id1)
    NEATGenome genome(1, 1, false, tracker);
    // connections: [{0,1,0.0,true,innov0}]

    genome.AddNodeSplitting(0, tracker);
    // disables innov0; creates hidden id=2; adds {0,2,1.0,true,innov1}, {2,1,0.0,true,innov2}
    // (0.0 preserved from innov0's own weight)

    genome.AddNodeSplitting(1, tracker);
    // splits {0,2,1.0,...,innov1} (weight 1.0 by the first split's own "in->new" convention):
    // disables innov1; creates hidden id=3; adds {0,3,1.0,true,innov3},
    // {3,2,1.0 [preserved from innov1's own weight],true,innov4}

    const auto& connections = genome.connections();
    auto find_by_innovation = [&](int innovation) {
        return std::find_if(connections.begin(), connections.end(),
                             [&](const ConnectionGene& c) { return c.innovation == innovation; });
    };

    auto in_to_new = find_by_innovation(3);
    ASSERT_NE(in_to_new, connections.end());
    EXPECT_EQ(in_to_new->in_node, 0);
    EXPECT_EQ(in_to_new->out_node, 3);
    EXPECT_DOUBLE_EQ(in_to_new->weight, 1.0);

    auto new_to_out = find_by_innovation(4);
    ASSERT_NE(new_to_out, connections.end());
    EXPECT_EQ(new_to_out->in_node, 3);
    EXPECT_EQ(new_to_out->out_node, 2);
    EXPECT_DOUBLE_EQ(new_to_out->weight, 1.0);  // preserved from innov1's own weight

    EXPECT_FALSE(find_by_innovation(0)->enabled);
    EXPECT_FALSE(find_by_innovation(1)->enabled);
    EXPECT_EQ(genome.nodes().size(), 4u);  // input, output, and two hidden nodes
}

TEST(NEATGenomeTest, AddNodeSplittingThrowsOnUnknownOrDisabledConnection) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    EXPECT_THROW(genome.AddNodeSplitting(99, tracker), std::invalid_argument);
    genome.AddNodeSplitting(0, tracker);
    EXPECT_THROW(genome.AddNodeSplitting(0, tracker), std::invalid_argument);  // now disabled
}

TEST(NEATGenomeTest, AddNodeSplittingReusesNewNodeIdAcrossGenomesSharingATracker) {
    InnovationTracker tracker(2);  // both genomes share the same initial 1-input/1-output shape
    NEATGenome genome_a(1, 1, false, tracker);
    NEATGenome genome_b(1, 1, false, tracker);

    genome_a.AddNodeSplitting(0, tracker);  // both start with the identical connection innov0
    genome_b.AddNodeSplitting(0, tracker);

    auto find_hidden = [](const NEATGenome& g) {
        for (const auto& n : g.nodes()) {
            if (n.type == NodeGene::Type::Hidden) {
                return n.id;
            }
        }
        return -1;
    };
    EXPECT_EQ(find_hidden(genome_a), find_hidden(genome_b));
}

// --- RNG-driven wrappers ---

TEST(NEATGenomeTest, AddConnectionWrapperAddsExactlyOneConnectionWhenACandidateExists) {
    // A 1-input/1-output genome, split once, is *already* fully connected in every
    // feedforward-valid direction (a disabled gene still counts as "exists") -- genuinely
    // found via this test's own first attempt failing, not assumed. A 2-input/1-output genome
    // split once leaves exactly one real candidate: in1(id1) has no connection to the new
    // hidden node (id3) yet, and hid3 has no path back to in1 (its only outgoing connection
    // goes to the output), so (in1 -> hid3) is feedforward-safe. Every other pair is either
    // already connected (including the disabled original) or would close a cycle. With
    // exactly one candidate, the outcome is deterministic regardless of RNG.
    InnovationTracker tracker(3);  // 2 inputs(0,1) + 1 output(2)
    NEATGenome genome(2, 1, false, tracker);
    genome.AddNodeSplitting(0, tracker);  // splits in0(0)->out(2); creates hidden id=3
    size_t before = genome.connections().size();

    std::mt19937 rng(0);
    bool added = genome.AddConnection(tracker, rng);
    ASSERT_TRUE(added);
    EXPECT_EQ(genome.connections().size(), before + 1);

    const auto& connections = genome.connections();
    auto it = std::find_if(connections.begin(), connections.end(),
                            [](const ConnectionGene& c) { return c.in_node == 1 && c.out_node == 3; });
    EXPECT_NE(it, connections.end()) << "expected the only valid candidate, in1(1) -> hidden(3)";
}

TEST(NEATGenomeTest, AddConnectionWrapperReturnsFalseWhenNoCandidateExists) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);  // already fully (and only) connected in0->out
    std::mt19937 rng(0);
    EXPECT_FALSE(genome.AddConnection(tracker, rng));
}

TEST(NEATGenomeTest, AddNodeWrapperSplitsAnEnabledConnection) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    std::mt19937 rng(0);
    EXPECT_TRUE(genome.AddNode(tracker, rng));
    EXPECT_EQ(genome.nodes().size(), 3u);
}

TEST(NEATGenomeTest, MutateWeightsThrowsOnInvalidParameters) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    std::mt19937 rng(0);
    EXPECT_THROW(genome.MutateWeights(-1.0, 0.5, rng), std::invalid_argument);
    EXPECT_THROW(genome.MutateWeights(1.0, 1.5, rng), std::invalid_argument);
}

TEST(NEATGenomeTest, MutateWeightsProbabilityZeroLeavesWeightsUnchanged) {
    InnovationTracker tracker(2);
    NEATGenome genome(1, 1, false, tracker);
    std::mt19937 rng(0);
    genome.MutateWeights(5.0, 0.0, rng);
    for (const auto& c : genome.connections()) {
        EXPECT_DOUBLE_EQ(c.weight, 0.0);
    }
}

}  // namespace
}  // namespace pulsatrix
