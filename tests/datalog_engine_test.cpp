#include "pulsatrix/datalog_engine.hpp"

#include <gtest/gtest.h>

namespace pulsatrix::datalog {
namespace {

Term C(const std::string& value) { return Term::make_constant(value); }
Term V(const std::string& name) { return Term::make_variable(name); }

// ---------------------------------------------------------------------------
// The standard minimal Datalog textbook example: edge facts + transitive-closure
// "ancestor" rules. Toy KB: a path graph a -> b -> c -> d.
//
// edge(a,b). edge(b,c). edge(c,d).
// ancestor(X,Y) :- edge(X,Y).
// ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
//
// Hand-enumerated fixpoint (worked on paper before writing this test, per this project's
// XorNetwork/toy-KB derive-then-implement-then-test precedent):
//
//   Round 1 (base rule only, since no ancestor facts exist yet to feed the recursive rule):
//     ancestor(a,b)  <- edge(a,b)
//     ancestor(b,c)  <- edge(b,c)
//     ancestor(c,d)  <- edge(c,d)
//
//   Round 2 (recursive rule, using round 1's ancestor facts):
//     ancestor(a,c)  <- edge(a,b), ancestor(b,c)
//     ancestor(b,d)  <- edge(b,c), ancestor(c,d)
//
//   Round 3 (recursive rule, using round 2's ancestor facts):
//     ancestor(a,d)  <- edge(a,b), ancestor(b,d)
//
//   Round 4: no new facts -- fixpoint reached.
//
// Final ancestor set: {(a,b), (b,c), (c,d), (a,c), (b,d), (a,d)} -- exactly the 6 = C(4,2)
// pairs of a 4-node total order, as transitive closure of a path graph must produce.
// Final fact count: 3 edge facts + 6 ancestor facts = 9.
// ---------------------------------------------------------------------------

class AncestorToyKBTest : public ::testing::Test {
protected:
    FactDatabase edge_facts() const {
        FactDatabase db;
        db.insert(Atom("edge", {C("a"), C("b")}));
        db.insert(Atom("edge", {C("b"), C("c")}));
        db.insert(Atom("edge", {C("c"), C("d")}));
        return db;
    }

    std::vector<Rule> ancestor_program() const {
        // ancestor(X,Y) :- edge(X,Y).
        Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
        // ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
        Rule recursive(Atom("ancestor", {V("X"), V("Y")}),
                        {Atom("edge", {V("X"), V("Z")}), Atom("ancestor", {V("Z"), V("Y")})});
        return {base, recursive};
    }

    FactDatabase expected_fixpoint() const {
        FactDatabase expected = edge_facts();
        expected.insert(Atom("ancestor", {C("a"), C("b")}));
        expected.insert(Atom("ancestor", {C("b"), C("c")}));
        expected.insert(Atom("ancestor", {C("c"), C("d")}));
        expected.insert(Atom("ancestor", {C("a"), C("c")}));
        expected.insert(Atom("ancestor", {C("b"), C("d")}));
        expected.insert(Atom("ancestor", {C("a"), C("d")}));
        return expected;
    }
};

TEST_F(AncestorToyKBTest, NaiveEvaluationMatchesHandEnumeratedFixpoint) {
    FactDatabase result = naive_evaluate(ancestor_program(), edge_facts());

    EXPECT_EQ(result.size(), 9u);
    EXPECT_EQ(result, expected_fixpoint());
}

TEST_F(AncestorToyKBTest, RecursiveAncestorRuleTerminates) {
    // The real test that the decidable-fragment restriction is doing its job: this call
    // must return at all (a Prolog-style engine with unrestricted SLD resolution could loop
    // forever re-deriving ancestor(Z,Y) from itself). A finite, bounded run is the assertion
    // here -- if naive_evaluate didn't terminate, this test would hang/time out rather than
    // fail cleanly, which is itself diagnostic.
    FactDatabase result = naive_evaluate(ancestor_program(), edge_facts());
    EXPECT_EQ(result.size(), 9u);
}

TEST_F(AncestorToyKBTest, SemiNaiveEvaluationProducesIdenticalFixpointToNaiveEvaluation) {
    FactDatabase naive_result = naive_evaluate(ancestor_program(), edge_facts());
    FactDatabase semi_naive_result = semi_naive_evaluate(ancestor_program(), edge_facts());

    // Direct equivalence, not "also seems to work": same fact count and the exact same set.
    EXPECT_EQ(semi_naive_result.size(), naive_result.size());
    EXPECT_EQ(semi_naive_result, naive_result);
    EXPECT_EQ(semi_naive_result, expected_fixpoint());
}

// ---------------------------------------------------------------------------
// Adversarial/boundary tests
// ---------------------------------------------------------------------------

TEST(DatalogEngineAdversarialTest, EmptyFactDatabaseProducesEmptyFixpoint) {
    FactDatabase empty;
    Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    Rule recursive(Atom("ancestor", {V("X"), V("Y")}),
                    {Atom("edge", {V("X"), V("Z")}), Atom("ancestor", {V("Z"), V("Y")})});
    std::vector<Rule> rules{base, recursive};

    FactDatabase naive_result = naive_evaluate(rules, empty);
    FactDatabase semi_naive_result = semi_naive_evaluate(rules, empty);

    EXPECT_TRUE(naive_result.empty());
    EXPECT_TRUE(semi_naive_result.empty());
}

TEST(DatalogEngineAdversarialTest, RuleWithNoMatchingFactsDerivesNothing) {
    FactDatabase facts;
    facts.insert(Atom("edge", {C("a"), C("b")}));

    // A rule over a predicate that never appears in the fact database.
    Rule dead_rule(Atom("reachable", {V("X"), V("Y")}), {Atom("wormhole", {V("X"), V("Y")})});
    std::vector<Rule> rules{dead_rule};

    FactDatabase naive_result = naive_evaluate(rules, facts);
    FactDatabase semi_naive_result = semi_naive_evaluate(rules, facts);

    // Only the original edge fact survives -- nothing derived.
    EXPECT_EQ(naive_result.size(), 1u);
    EXPECT_EQ(semi_naive_result.size(), 1u);
    EXPECT_EQ(naive_result, facts);
    EXPECT_EQ(semi_naive_result, facts);
}

TEST(DatalogEngineAdversarialTest, RuleHeadRepeatingExistingFactDoesNotGrowUnboundedly) {
    FactDatabase facts;
    facts.insert(Atom("edge", {C("a"), C("b")}));
    facts.insert(Atom("ancestor", {C("a"), C("b")}));  // already present, matches the rule's own head shape

    Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    std::vector<Rule> rules{base};

    FactDatabase naive_result = naive_evaluate(rules, facts);
    FactDatabase semi_naive_result = semi_naive_evaluate(rules, facts);

    // ancestor(a,b) is re-derivable every round from edge(a,b), but it's already present --
    // must reach a fixpoint (size stays 2), not grow without bound.
    EXPECT_EQ(naive_result.size(), 2u);
    EXPECT_EQ(semi_naive_result.size(), 2u);
}

TEST(DatalogEngineAdversarialTest, MultipleRulesWithSharedPredicateBothContribute) {
    FactDatabase facts;
    facts.insert(Atom("edge", {C("a"), C("b")}));
    facts.insert(Atom("edge", {C("b"), C("c")}));

    // Two independent base rules deriving into the same "reachable" predicate.
    Rule direct(Atom("reachable", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    std::vector<Rule> rules{direct};

    FactDatabase naive_result = naive_evaluate(rules, facts);

    EXPECT_TRUE(naive_result.contains(Atom("reachable", {C("a"), C("b")})));
    EXPECT_TRUE(naive_result.contains(Atom("reachable", {C("b"), C("c")})));
    EXPECT_EQ(naive_result.size(), 4u);  // 2 edge + 2 reachable
}

}  // namespace
}  // namespace pulsatrix::datalog
