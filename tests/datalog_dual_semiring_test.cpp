#include "pulsatrix/datalog_dual_semiring.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

#include "pulsatrix/datalog_weighted_engine.hpp"
#include "pulsatrix/datalog_weighted_fact_database.hpp"

namespace pulsatrix::datalog {
namespace {

using Dual = DualNumber<double>;
using Semiring = DualSemiring<double>;

Term C(const std::string& value) { return Term::make_constant(value); }
Term V(const std::string& name) { return Term::make_variable(name); }

// ---------------------------------------------------------------------------
// DualNumber/DualSemiring unit tests -- identity laws, matching RealSemiring<T>'s own
// SatisfiesSemiringIdentityLaws precedent (datalog_semiring_test.cpp), plus the sum/product
// derivative rules this semiring's whole point is.
// ---------------------------------------------------------------------------

TEST(DualNumberTest, EqualityComparesBothValueAndGradWithinEpsilon) {
    EXPECT_EQ((Dual{1.0, 2.0}), (Dual{1.0, 2.0}));
    EXPECT_NE((Dual{1.0, 2.0}), (Dual{1.0, 3.0}));
    EXPECT_NE((Dual{1.5, 2.0}), (Dual{1.0, 2.0}));
    // Within epsilon of each other (float noise) still compares equal.
    EXPECT_EQ((Dual{1.0, 2.0}), (Dual{1.0 + 1e-12, 2.0 - 1e-12}));
}

TEST(DualSemiringTest, ZeroAndOneMatchAdditiveAndMultiplicativeIdentity) {
    EXPECT_EQ(Semiring::zero(), (Dual{0.0, 0.0}));
    EXPECT_EQ(Semiring::one(), (Dual{1.0, 0.0}));
}

TEST(DualSemiringTest, SatisfiesSemiringIdentityLaws) {
    Dual x{3.0, 5.0};
    EXPECT_EQ(Semiring::add(Semiring::zero(), x), x);
    EXPECT_EQ(Semiring::mul(Semiring::one(), x), x);
    EXPECT_EQ(Semiring::mul(Semiring::zero(), x), Semiring::zero());
}

TEST(DualSemiringTest, AddImplementsSumRule) {
    // d(a+b) = da + db.
    Dual a{2.0, 3.0};
    Dual b{4.0, 5.0};
    Dual sum = Semiring::add(a, b);
    EXPECT_DOUBLE_EQ(sum.value, 6.0);
    EXPECT_DOUBLE_EQ(sum.grad, 8.0);
}

TEST(DualSemiringTest, MulImplementsProductRule) {
    // d(a*b) = da*b + a*db.
    Dual a{2.0, 3.0};
    Dual b{4.0, 5.0};
    Dual prod = Semiring::mul(a, b);
    EXPECT_DOUBLE_EQ(prod.value, 8.0);
    EXPECT_DOUBLE_EQ(prod.grad, 3.0 * 4.0 + 2.0 * 5.0);  // 22.0
}

// ---------------------------------------------------------------------------
// WeightedFactDatabase<DualNumber<double>> -- the additive third instantiation.
// ---------------------------------------------------------------------------

TEST(WeightedFactDatabaseDualTest, SetAndWeightOfRoundTrip) {
    WeightedFactDatabase<Dual> db;
    db.set(Atom("edge", {C("a"), C("b")}), Dual{0.9, 1.0});
    EXPECT_TRUE(db.contains(Atom("edge", {C("a"), C("b")})));
    EXPECT_EQ(db.weight_of(Atom("edge", {C("a"), C("b")}), Semiring::zero()), (Dual{0.9, 1.0}));
}

TEST(WeightedFactDatabaseDualTest, AbsentFactReportsCallerSuppliedDefault) {
    WeightedFactDatabase<Dual> db;
    EXPECT_EQ(db.weight_of(Atom("edge", {C("x"), C("y")}), Semiring::zero()), Semiring::zero());
}

TEST(WeightedFactDatabaseDualTest, SetThrowsOnNonGroundAtom) {
    WeightedFactDatabase<Dual> db;
    EXPECT_THROW(db.set(Atom("edge", {V("X"), C("b")}), Dual{0.9, 1.0}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// naive_evaluate_weighted<DualSemiring<double>> on Mission 1's own diamond toy KB -- the
// generic, engine-level forward-mode-AD correctness oracle this mission's bridge class
// (neuro_symbolic_datalog_bridge_test.cpp) builds on. Hand-derived before writing this test
// (mirrors this project's standing discipline):
//
//   edge(a,b)=0.5  edge(a,c)=0.4  edge(b,d)=0.6  edge(c,d)=0.3
//   ancestor(a,d) = edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d) = 0.5*0.6 + 0.4*0.3 = 0.42
//
// Seeding edge(a,b)'s derivative to 1 (every other fact's derivative 0):
//   d(ancestor(a,d))/d(edge(a,b)) = edge(b,d) = 0.6   (the other summand doesn't involve
//                                                       edge(a,b) at all, so its derivative
//                                                       contribution is exactly 0)
// Seeding edge(c,d)'s derivative to 1 instead:
//   d(ancestor(a,d))/d(edge(c,d)) = edge(a,c) = 0.4
// ---------------------------------------------------------------------------

std::vector<Rule> ancestor_program() {
    Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    Rule recursive(Atom("ancestor", {V("X"), V("Y")}),
                    {Atom("edge", {V("X"), V("Z")}), Atom("ancestor", {V("Z"), V("Y")})});
    return {base, recursive};
}

TEST(DualSemiringEngineTest, DerivativeWrtSeededBaseFactMatchesHandDerivedPartial) {
    WeightedFactDatabase<Dual> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), Dual{0.5, 1.0});  // seeded
    facts.set(Atom("edge", {C("a"), C("c")}), Dual{0.4, 0.0});
    facts.set(Atom("edge", {C("b"), C("d")}), Dual{0.6, 0.0});
    facts.set(Atom("edge", {C("c"), C("d")}), Dual{0.3, 0.0});

    WeightedFactDatabase<Dual> result = naive_evaluate_weighted<Semiring>(ancestor_program(), facts);
    Dual query = result.weight_of(Atom("ancestor", {C("a"), C("d")}), Semiring::zero());

    EXPECT_DOUBLE_EQ(query.value, 0.42);
    EXPECT_DOUBLE_EQ(query.grad, 0.6);
}

TEST(DualSemiringEngineTest, DerivativeWrtADifferentSeededFactMatchesItsOwnHandDerivedPartial) {
    WeightedFactDatabase<Dual> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), Dual{0.5, 0.0});
    facts.set(Atom("edge", {C("a"), C("c")}), Dual{0.4, 0.0});
    facts.set(Atom("edge", {C("b"), C("d")}), Dual{0.6, 0.0});
    facts.set(Atom("edge", {C("c"), C("d")}), Dual{0.3, 1.0});  // seeded instead

    WeightedFactDatabase<Dual> result = naive_evaluate_weighted<Semiring>(ancestor_program(), facts);
    Dual query = result.weight_of(Atom("ancestor", {C("a"), C("d")}), Semiring::zero());

    EXPECT_DOUBLE_EQ(query.value, 0.42);
    EXPECT_DOUBLE_EQ(query.grad, 0.4);
}

TEST(DualSemiringEngineTest, SinglePathFactDerivativeWrtItselfIsExactlyOne) {
    // ancestor(a,b) = edge(a,b) alone (base rule, single path) -- d/d(edge(a,b)) must be
    // exactly 1, the multiplicative-identity boundary this semiring's mul() must not corrupt.
    WeightedFactDatabase<Dual> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), Dual{0.5, 1.0});
    facts.set(Atom("edge", {C("a"), C("c")}), Dual{0.4, 0.0});
    facts.set(Atom("edge", {C("b"), C("d")}), Dual{0.6, 0.0});
    facts.set(Atom("edge", {C("c"), C("d")}), Dual{0.3, 0.0});

    WeightedFactDatabase<Dual> result = naive_evaluate_weighted<Semiring>(ancestor_program(), facts);
    Dual ancestor_ab = result.weight_of(Atom("ancestor", {C("a"), C("b")}), Semiring::zero());

    EXPECT_DOUBLE_EQ(ancestor_ab.value, 0.5);
    EXPECT_DOUBLE_EQ(ancestor_ab.grad, 1.0);
}

TEST(DualSemiringEngineTest, FactWithZeroDerivationPathsIsAbsentBoundary) {
    WeightedFactDatabase<Dual> facts;
    facts.set(Atom("edge", {C("a"), C("b")}), Dual{0.5, 1.0});
    WeightedFactDatabase<Dual> result = naive_evaluate_weighted<Semiring>(ancestor_program(), facts);
    EXPECT_FALSE(result.contains(Atom("ancestor", {C("d"), C("a")})));
}

}  // namespace
}  // namespace pulsatrix::datalog
