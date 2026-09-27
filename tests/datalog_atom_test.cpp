#include "pulsatrix/datalog_atom.hpp"

#include <gtest/gtest.h>

#include <unordered_set>

namespace pulsatrix::datalog {
namespace {

TEST(DatalogAtomTest, ReportsPredicateNameArityAndTerms) {
    Atom edge("edge", {Term::make_constant("a"), Term::make_constant("b")});
    EXPECT_EQ(edge.predicate_name(), "edge");
    EXPECT_EQ(edge.arity(), 2u);
    EXPECT_EQ(edge.terms()[0], Term::make_constant("a"));
    EXPECT_EQ(edge.terms()[1], Term::make_constant("b"));
}

TEST(DatalogAtomTest, GroundAtomIsGround) {
    Atom edge("edge", {Term::make_constant("a"), Term::make_constant("b")});
    EXPECT_TRUE(edge.is_ground());
}

TEST(DatalogAtomTest, AtomWithAnyVariableIsNotGround) {
    Atom ancestor("ancestor", {Term::make_variable("X"), Term::make_constant("b")});
    EXPECT_FALSE(ancestor.is_ground());
}

TEST(DatalogAtomTest, ZeroArityAtomIsGround) {
    Atom p("p", {});
    EXPECT_TRUE(p.is_ground());
}

TEST(DatalogAtomTest, EqualityComparesPredicateNameAndTerms) {
    Atom a1("edge", {Term::make_constant("a"), Term::make_constant("b")});
    Atom a2("edge", {Term::make_constant("a"), Term::make_constant("b")});
    Atom a3("edge", {Term::make_constant("b"), Term::make_constant("a")});
    Atom a4("ancestor", {Term::make_constant("a"), Term::make_constant("b")});

    EXPECT_EQ(a1, a2);
    EXPECT_NE(a1, a3);
    EXPECT_NE(a1, a4);
}

TEST(DatalogAtomTest, UsableAsUnorderedSetKey) {
    std::unordered_set<Atom, AtomHash> facts;
    facts.insert(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));
    facts.insert(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));  // duplicate
    facts.insert(Atom("edge", {Term::make_constant("b"), Term::make_constant("c")}));

    EXPECT_EQ(facts.size(), 2u);
}

}  // namespace
}  // namespace pulsatrix::datalog
