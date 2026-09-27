#include "pulsatrix/datalog_lrp.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

#include "pulsatrix/datalog_weighted_engine.hpp"

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
// Mission 1's exact weighted diamond-graph toy KB (constant weights) -- the correctness
// oracle for this mission's own hand-derived LRP rule (see datalog_lrp.hpp's header comment
// for the full worked derivation this fixture's expected values come from).
//
// edge(a,b)=0.5  edge(a,c)=0.4  edge(b,d)=0.6  edge(c,d)=0.3
// ancestor(a,d) = edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d) = 0.30 + 0.12 = 0.42
// ---------------------------------------------------------------------------

class DatalogLRPTest : public ::testing::Test {
protected:
    WeightedFactDatabase<double> edge_facts() const {
        WeightedFactDatabase<double> db;
        db.set(Atom("edge", {C("a"), C("b")}), 0.5);
        db.set(Atom("edge", {C("a"), C("c")}), 0.4);
        db.set(Atom("edge", {C("b"), C("d")}), 0.6);
        db.set(Atom("edge", {C("c"), C("d")}), 0.3);
        return db;
    }

    WeightedFactDatabase<double> fixpoint() const {
        return naive_evaluate_weighted<RealSemiring<double>>(ancestor_program(), edge_facts());
    }
};

// ---------------------------------------------------------------------------
// Multi-path case: ancestor(a,d) has exactly two derivation paths (Z=b, Z=c). Relevance must
// split proportionally to each path's contribution, then split again (bilinearly) across each
// path's two factors, recursing through the single-path ancestor(b,d)/ancestor(c,d)
// intermediate facts down to the four base edge facts -- exactly datalog_lrp.hpp's own worked
// numeric example, reproduced here as the acceptance test for it.
// ---------------------------------------------------------------------------

TEST_F(DatalogLRPTest, MultiPathDiamondFactConservesRelevanceMatchingHandDerivedSplit) {
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result =
        propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("a"), C("d")}), 1.0, /*epsilon=*/0.0);

    EXPECT_NEAR(result.base_facts.at(Atom("edge", {C("a"), C("b")})), 0.357142857142857, 1e-9);
    EXPECT_NEAR(result.base_facts.at(Atom("edge", {C("b"), C("d")})), 0.357142857142857, 1e-9);
    EXPECT_NEAR(result.base_facts.at(Atom("edge", {C("a"), C("c")})), 0.142857142857143, 1e-9);
    EXPECT_NEAR(result.base_facts.at(Atom("edge", {C("c"), C("d")})), 0.142857142857143, 1e-9);

    double total = 0.0;
    for (const auto& [atom, r] : result.base_facts) total += r;
    EXPECT_NEAR(total, 1.0, 1e-9);  // exact at epsilon=0, per the hand derivation.
}

TEST_F(DatalogLRPTest, MultiPathDiamondFactConservesNearExactlyWithNonzeroEpsilon) {
    // Same query, default (nonzero) epsilon -- conservation should still hold within the
    // epsilon-stabilized tolerance class every other epsilon-rule test in this codebase uses.
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result = propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("a"), C("d")}), 5.0);

    double total = 0.0;
    for (const auto& [atom, r] : result.base_facts) total += r;
    EXPECT_NEAR(total, 5.0, 1e-2);
}

// ---------------------------------------------------------------------------
// Single-path boundary case: ancestor(a,b) is derivable only directly from edge(a,b) (one
// derivation, arity-1 body) -- relevance must flow entirely, unsplit, to edge(a,b).
// ---------------------------------------------------------------------------

TEST_F(DatalogLRPTest, SinglePathFactReceivesEntireRelevanceNoSplit) {
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result =
        propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("a"), C("b")}), 1.0, /*epsilon=*/0.0);

    ASSERT_EQ(result.base_facts.size(), 1u);
    EXPECT_DOUBLE_EQ(result.base_facts.at(Atom("edge", {C("a"), C("b")})), 1.0);
}

TEST_F(DatalogLRPTest, SinglePathIntermediateFactAlsoReceivesEntireRelevanceNoSplit) {
    // ancestor(b,d) is itself single-path (only edge(b,d)) -- same no-split guarantee one
    // level down the recursion, independent of ancestor(a,d)'s own multi-path structure.
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result =
        propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("b"), C("d")}), 1.0, /*epsilon=*/0.0);

    ASSERT_EQ(result.base_facts.size(), 1u);
    EXPECT_DOUBLE_EQ(result.base_facts.at(Atom("edge", {C("b"), C("d")})), 1.0);
}

// ---------------------------------------------------------------------------
// Adversarial/boundary tests.
// ---------------------------------------------------------------------------

TEST_F(DatalogLRPTest, BaseExtensionalFactQueriedDirectlyReceivesItsOwnSeedUnchanged) {
    // Querying a pure extensional fact directly (never a rule head) -- zero derivations found,
    // so it is its own (single) leaf.
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result = propagate_relevance_weighted(ancestor_program(), fp, Atom("edge", {C("a"), C("b")}), 3.0);

    EXPECT_DOUBLE_EQ(result.base_facts.at(Atom("edge", {C("a"), C("b")})), 3.0);
}

TEST_F(DatalogLRPTest, NonDerivableAtomTreatedAsLeafWithSeedRecordedButNoCrash) {
    // ancestor(d,a) -- the reverse direction -- has zero derivations in this DAG (same
    // boundary case datalog_weighted_engine_test.cpp's FactWithZeroDerivationPathsHasZeroWeight
    // exercises at the engine level). Nothing to redistribute into; the seed is simply
    // recorded at that atom as its own leaf, not a crash or an exception.
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result = propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("d"), C("a")}), 1.0);

    EXPECT_DOUBLE_EQ(result.base_facts.at(Atom("ancestor", {C("d"), C("a")})), 1.0);
}

TEST_F(DatalogLRPTest, ZeroSeedRelevancePropagatesAsAllZeros) {
    WeightedFactDatabase<double> fp = fixpoint();
    RelevanceResult result =
        propagate_relevance_weighted(ancestor_program(), fp, Atom("ancestor", {C("a"), C("d")}), 0.0, /*epsilon=*/0.0);

    for (const auto& [atom, r] : result.base_facts) {
        EXPECT_DOUBLE_EQ(r, 0.0);
    }
}

TEST(DatalogLRPArityBoundaryTest, RuleBodyWithArityGreaterThanTwoThrows) {
    // No toy program in this campaign needs a 3-atom rule body -- this is the honest,
    // logged scope boundary datalog_lrp.hpp's own header comment documents, verified here
    // rather than left as an untested claim.
    WeightedFactDatabase<double> facts;
    facts.set(Atom("p", {C("x")}), 0.5);
    facts.set(Atom("q", {C("x")}), 0.5);
    facts.set(Atom("r", {C("x")}), 0.5);

    Rule triple(Atom("s", {V("X")}), {Atom("p", {V("X")}), Atom("q", {V("X")}), Atom("r", {V("X")})});
    std::vector<Rule> rules{triple};

    WeightedFactDatabase<double> fp = naive_evaluate_weighted<RealSemiring<double>>(rules, facts);

    EXPECT_THROW(propagate_relevance_weighted(rules, fp, Atom("s", {C("x")}), 1.0), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix::datalog
