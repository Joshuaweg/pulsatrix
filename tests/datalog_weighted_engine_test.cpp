#include "pulsatrix/datalog_weighted_engine.hpp"

#include <gtest/gtest.h>

#include "pulsatrix/datalog_engine.hpp"

namespace pulsatrix::datalog {
namespace {

Term C(const std::string& value) { return Term::make_constant(value); }
Term V(const std::string& name) { return Term::make_variable(name); }

std::vector<Rule> ancestor_program() {
    // ancestor(X,Y) :- edge(X,Y).
    Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    // ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
    Rule recursive(Atom("ancestor", {V("X"), V("Y")}),
                    {Atom("edge", {V("X"), V("Z")}), Atom("ancestor", {V("Z"), V("Y")})});
    return {base, recursive};
}

// ---------------------------------------------------------------------------
// Weighted path-graph toy KB (single derivation path per ancestor fact) --
// Mission 0's exact edge/ancestor structure, now with "confidence" weights on `edge`.
//
// edge(a,b)=0.9  edge(b,c)=0.8  edge(c,d)=0.7
//
// Hand-derived expected weights under the real-valued (+, x) semiring, sum-of-products over
// all derivation paths (worked on paper before writing this test, per this project's
// derive-then-implement-then-test precedent). Every pair in this path graph has exactly ONE
// path, so each ancestor weight is exactly that single path's product -- no ⊕ accumulation is
// exercised by this fixture (that is WeightedDiamondKBTest's job, below):
//
//   ancestor(a,b) = edge(a,b)                               = 0.9
//   ancestor(b,c) = edge(b,c)                               = 0.8
//   ancestor(c,d) = edge(c,d)                                = 0.7
//   ancestor(a,c) = edge(a,b) * ancestor(b,c) = 0.9 * 0.8    = 0.72
//   ancestor(b,d) = edge(b,c) * ancestor(c,d) = 0.8 * 0.7    = 0.56
//   ancestor(a,d) = edge(a,b) * ancestor(b,d) = 0.9 * 0.56   = 0.504
//                 (equivalently edge(a,b)*edge(b,c)*edge(c,d) = 0.9*0.8*0.7 = 0.504)
// ---------------------------------------------------------------------------

class WeightedPathKBTest : public ::testing::Test {
protected:
    WeightedFactDatabase<double> edge_facts() const {
        WeightedFactDatabase<double> db;
        db.set(Atom("edge", {C("a"), C("b")}), 0.9);
        db.set(Atom("edge", {C("b"), C("c")}), 0.8);
        db.set(Atom("edge", {C("c"), C("d")}), 0.7);
        return db;
    }
};

TEST_F(WeightedPathKBTest, NaiveEvaluateWeightedMatchesHandDerivedWeightsExactly) {
    WeightedFactDatabase<double> result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());

    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("b")}), 0.0), 0.9);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("b"), C("c")}), 0.0), 0.8);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("c"), C("d")}), 0.0), 0.7);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("c")}), 0.0), 0.72);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("b"), C("d")}), 0.0), 0.56);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("d")}), 0.0), 0.504);

    // Original edge weights persist untouched.
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("edge", {C("a"), C("b")}), 0.0), 0.9);
}

TEST_F(WeightedPathKBTest, SinglePathWeightEqualsThatPathsProductExactlyNoSpuriousAccumulation) {
    // Boundary case explicitly named in the mission's deliverables: a fact with exactly one
    // derivation path must equal that path's product exactly -- not doubled, not summed with
    // anything else. ancestor(a,d) has exactly one path in this graph (a->b->d, itself built
    // from a->b->c->d), so its weight must equal the single product 0.9*0.8*0.7, not e.g.
    // 2 * 0.504 (which a "re-add every round" bug would have produced -- see
    // datalog_weighted_engine.hpp's Stage 3 decision 2 commentary).
    WeightedFactDatabase<double> result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("d")}), 0.0), 0.9 * 0.8 * 0.7);
}

TEST_F(WeightedPathKBTest, FactWithZeroDerivationPathsHasZeroWeight) {
    // ancestor(d, a) -- the reverse direction -- has no derivation at all in this DAG.
    WeightedFactDatabase<double> result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());

    EXPECT_FALSE(result.contains(Atom("ancestor", {C("d"), C("a")})));
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("d"), C("a")}), RealSemiring<double>::zero()), 0.0);
}

TEST_F(WeightedPathKBTest, SemiNaiveEvaluateWeightedProducesIdenticalResultToNaive) {
    WeightedFactDatabase<double> naive_result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());
    WeightedFactDatabase<double> semi_naive_result =
        semi_naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());

    EXPECT_EQ(semi_naive_result, naive_result);
}

// ---------------------------------------------------------------------------
// Weighted diamond-graph toy KB -- genuine multi-path accumulation.
//
//        b
//      /   \
//     a     d
//      \   /
//        c
//
// edge(a,b)=0.5  edge(a,c)=0.4  edge(b,d)=0.6  edge(c,d)=0.3
//
// Hand-derived expected weight of ancestor(a,d) (worked on paper before writing this test):
// there is no edge(a,d), so the base rule never fires for (a,d) -- it is only derivable via
// the recursive rule, through TWO distinct substitutions for Z (Z=b and Z=c), whose
// contributions must be ⊕-summed (not "first write wins", not "overwrite"):
//
//   ancestor(a,b) = edge(a,b)                                        = 0.5
//   ancestor(a,c) = edge(a,c)                                        = 0.4
//   ancestor(b,d) = edge(b,d)                                        = 0.6
//   ancestor(c,d) = edge(c,d)                                        = 0.3
//   ancestor(a,d) = edge(a,b)*ancestor(b,d) + edge(a,c)*ancestor(c,d)
//                 = (0.5 * 0.6) + (0.4 * 0.3)
//                 = 0.30 + 0.12
//                 = 0.42
// ---------------------------------------------------------------------------

class WeightedDiamondKBTest : public ::testing::Test {
protected:
    WeightedFactDatabase<double> edge_facts() const {
        WeightedFactDatabase<double> db;
        db.set(Atom("edge", {C("a"), C("b")}), 0.5);
        db.set(Atom("edge", {C("a"), C("c")}), 0.4);
        db.set(Atom("edge", {C("b"), C("d")}), 0.6);
        db.set(Atom("edge", {C("c"), C("d")}), 0.3);
        return db;
    }
};

TEST_F(WeightedDiamondKBTest, MultiplePathsAccumulateViaSemiringAddNotFirstWriteWinsNotOverwrite) {
    WeightedFactDatabase<double> result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());

    // The genuine correctness pitfall this mission flags: boolean ⊕=OR can't distinguish
    // "first write wins" from "accumulate", but real-valued ⊕=+ makes it observable. If the
    // engine only kept the first-found path (Z=b, 0.30) or the last-found (Z=c, 0.12) instead
    // of summing both, this assertion would fail with 0.30 or 0.12 rather than 0.42.
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("d")}), 0.0), 0.42);

    // Sanity-check the intermediate single-path facts too.
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("b")}), 0.0), 0.5);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("c")}), 0.0), 0.4);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("b"), C("d")}), 0.0), 0.6);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("c"), C("d")}), 0.0), 0.3);
}

TEST_F(WeightedDiamondKBTest, SemiNaiveEvaluateWeightedAlsoAccumulatesBothPaths) {
    WeightedFactDatabase<double> result = semi_naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("ancestor", {C("a"), C("d")}), 0.0), 0.42);
}

// ---------------------------------------------------------------------------
// Boolean-semiring regression: the generic weighted evaluator, instantiated with
// BooleanSemiring, must reproduce Mission 0's own naive_evaluate/semi_naive_evaluate
// fixpoint on the identical program/facts exactly -- not "also seems to produce 9 facts",
// a direct fact-by-fact translation check.
// ---------------------------------------------------------------------------

TEST(BooleanRegressionTest, GenericWeightedEvaluatorUnderBooleanSemiringMatchesMission0Fixpoint) {
    FactDatabase mission0_edges;
    mission0_edges.insert(Atom("edge", {C("a"), C("b")}));
    mission0_edges.insert(Atom("edge", {C("b"), C("c")}));
    mission0_edges.insert(Atom("edge", {C("c"), C("d")}));

    FactDatabase mission0_fixpoint = naive_evaluate(ancestor_program(), mission0_edges);
    ASSERT_EQ(mission0_fixpoint.size(), 9u);  // Mission 0's own hand-enumerated fixpoint size

    WeightedFactDatabase<bool> weighted_edges;
    weighted_edges.set(Atom("edge", {C("a"), C("b")}), true);
    weighted_edges.set(Atom("edge", {C("b"), C("c")}), true);
    weighted_edges.set(Atom("edge", {C("c"), C("d")}), true);

    WeightedFactDatabase<bool> weighted_result = naive_evaluate_weighted<BooleanSemiring>(ancestor_program(), weighted_edges);

    // Every atom in Mission 0's boolean fixpoint must be present here with weight true, and
    // the total count of "true"-weighted atoms must match exactly (no extra, no missing).
    std::size_t true_count = 0;
    for (const auto& [atom, weight] : weighted_result.facts()) {
        if (weight) ++true_count;
    }
    EXPECT_EQ(true_count, mission0_fixpoint.size());

    for (const Atom& fact : mission0_fixpoint.facts()) {
        EXPECT_TRUE(weighted_result.contains(fact)) << "missing fact in weighted regression result";
        EXPECT_TRUE(weighted_result.weight_of(fact, false)) << "fact present but weight is false, expected true";
    }
}

TEST(BooleanRegressionTest, SemiNaiveWeightedUnderBooleanSemiringAlsoMatchesMission0Fixpoint) {
    FactDatabase mission0_edges;
    mission0_edges.insert(Atom("edge", {C("a"), C("b")}));
    mission0_edges.insert(Atom("edge", {C("b"), C("c")}));
    mission0_edges.insert(Atom("edge", {C("c"), C("d")}));
    FactDatabase mission0_fixpoint = semi_naive_evaluate(ancestor_program(), mission0_edges);

    WeightedFactDatabase<bool> weighted_edges;
    weighted_edges.set(Atom("edge", {C("a"), C("b")}), true);
    weighted_edges.set(Atom("edge", {C("b"), C("c")}), true);
    weighted_edges.set(Atom("edge", {C("c"), C("d")}), true);

    WeightedFactDatabase<bool> weighted_result =
        semi_naive_evaluate_weighted<BooleanSemiring>(ancestor_program(), weighted_edges);

    for (const Atom& fact : mission0_fixpoint.facts()) {
        EXPECT_TRUE(weighted_result.weight_of(fact, false));
    }
}

// ---------------------------------------------------------------------------
// Adversarial/boundary tests
// ---------------------------------------------------------------------------

TEST(WeightedDatalogEngineAdversarialTest, EmptyFactDatabaseProducesEmptyFixpoint) {
    WeightedFactDatabase<double> empty;
    auto result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), empty);
    EXPECT_TRUE(result.empty());
}

TEST(WeightedDatalogEngineAdversarialTest, RuleWithNoMatchingFactsDerivesNothing) {
    WeightedFactDatabase<double> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), 0.9);

    Rule dead_rule(Atom("reachable", {V("X"), V("Y")}), {Atom("wormhole", {V("X"), V("Y")})});
    std::vector<Rule> rules{dead_rule};

    auto result = naive_evaluate_weighted<RealSemiring<double>>(rules, facts);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result.weight_of(Atom("edge", {C("a"), C("b")}), 0.0), 0.9);
}

TEST(WeightedDatalogEngineAdversarialTest, RecursiveAncestorRuleTerminatesUnderRealValuedSemiring) {
    WeightedFactDatabase<double> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), 0.9);
    facts.set(Atom("edge", {C("b"), C("c")}), 0.8);
    facts.set(Atom("edge", {C("c"), C("d")}), 0.7);

    auto result = naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), facts);
    // 3 edge facts + 6 ancestor facts, exactly Mission 0's own count -- termination proof
    // generalizes identically (finite constant space, monotone weight growth, bounded rounds).
    EXPECT_EQ(result.size(), 9u);
}

}  // namespace
}  // namespace pulsatrix::datalog
