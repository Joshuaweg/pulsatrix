#include "pulsatrix/datalog_rule.hpp"

#include <gtest/gtest.h>

namespace pulsatrix::datalog {
namespace {

TEST(DatalogRuleTest, ConstructsWhenEveryHeadVariableAppearsInBody) {
    // ancestor(X,Y) :- edge(X,Y).
    Atom head("ancestor", {Term::make_variable("X"), Term::make_variable("Y")});
    std::vector<Atom> body{Atom("edge", {Term::make_variable("X"), Term::make_variable("Y")})};

    EXPECT_NO_THROW(Rule(head, body));
}

TEST(DatalogRuleTest, ConstructsRecursiveRuleWhereBodyHasTwoAtoms) {
    // ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
    Atom head("ancestor", {Term::make_variable("X"), Term::make_variable("Y")});
    std::vector<Atom> body{
        Atom("edge", {Term::make_variable("X"), Term::make_variable("Z")}),
        Atom("ancestor", {Term::make_variable("Z"), Term::make_variable("Y")}),
    };

    Rule rule(head, body);
    EXPECT_EQ(rule.head(), head);
    EXPECT_EQ(rule.body().size(), 2u);
}

TEST(DatalogRuleTest, ThrowsWhenHeadVariableDoesNotAppearInBody) {
    // p(X) :- q(a).  X is unbound -- unsafe.
    Atom head("p", {Term::make_variable("X")});
    std::vector<Atom> body{Atom("q", {Term::make_constant("a")})};

    EXPECT_THROW(Rule(head, body), std::invalid_argument);
}

TEST(DatalogRuleTest, ThrowsWhenHeadHasVariableAndBodyIsEmpty) {
    Atom head("p", {Term::make_variable("X")});
    std::vector<Atom> body{};

    EXPECT_THROW(Rule(head, body), std::invalid_argument);
}

TEST(DatalogRuleTest, ConstructsGroundFactAsRuleWithEmptyBody) {
    // edge(a,b) :- .  (a ground fact expressed as a rule; no variables, so no safety issue.)
    Atom head("edge", {Term::make_constant("a"), Term::make_constant("b")});
    std::vector<Atom> body{};

    EXPECT_NO_THROW(Rule(head, body));
}

}  // namespace
}  // namespace pulsatrix::datalog
