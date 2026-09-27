#include "pulsatrix/datalog_term.hpp"

#include <gtest/gtest.h>

#include <unordered_set>

namespace pulsatrix::datalog {
namespace {

TEST(DatalogTermTest, ConstantReportsKindAndValue) {
    Term t = Term::make_constant("a");
    EXPECT_TRUE(t.is_constant());
    EXPECT_FALSE(t.is_variable());
    EXPECT_EQ(t.value(), "a");
}

TEST(DatalogTermTest, VariableReportsKindAndValue) {
    Term t = Term::make_variable("X");
    EXPECT_TRUE(t.is_variable());
    EXPECT_FALSE(t.is_constant());
    EXPECT_EQ(t.value(), "X");
}

TEST(DatalogTermTest, EqualityComparesKindAndValue) {
    EXPECT_EQ(Term::make_constant("a"), Term::make_constant("a"));
    EXPECT_NE(Term::make_constant("a"), Term::make_constant("b"));
    // Same string, different kind -- must not compare equal (a constant "X" is not the
    // variable "X").
    EXPECT_NE(Term::make_constant("X"), Term::make_variable("X"));
}

TEST(DatalogTermTest, UsableAsUnorderedSetKey) {
    std::unordered_set<Term, TermHash> terms;
    terms.insert(Term::make_constant("a"));
    terms.insert(Term::make_variable("X"));
    terms.insert(Term::make_constant("a"));  // duplicate

    EXPECT_EQ(terms.size(), 2u);
    EXPECT_EQ(terms.count(Term::make_constant("a")), 1u);
    EXPECT_EQ(terms.count(Term::make_variable("X")), 1u);
}

}  // namespace
}  // namespace pulsatrix::datalog
