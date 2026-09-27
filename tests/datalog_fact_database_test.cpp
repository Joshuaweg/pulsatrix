#include "pulsatrix/datalog_fact_database.hpp"

#include <gtest/gtest.h>

namespace pulsatrix::datalog {
namespace {

TEST(DatalogFactDatabaseTest, StartsEmpty) {
    FactDatabase db;
    EXPECT_TRUE(db.empty());
    EXPECT_EQ(db.size(), 0u);
}

TEST(DatalogFactDatabaseTest, InsertAddsGroundFact) {
    FactDatabase db;
    Atom edge_ab("edge", {Term::make_constant("a"), Term::make_constant("b")});

    EXPECT_TRUE(db.insert(edge_ab));
    EXPECT_TRUE(db.contains(edge_ab));
    EXPECT_EQ(db.size(), 1u);
}

TEST(DatalogFactDatabaseTest, InsertingSameFactTwiceDoesNotGrowDatabase) {
    FactDatabase db;
    Atom edge_ab("edge", {Term::make_constant("a"), Term::make_constant("b")});

    EXPECT_TRUE(db.insert(edge_ab));
    EXPECT_FALSE(db.insert(edge_ab));  // already present
    EXPECT_EQ(db.size(), 1u);
}

TEST(DatalogFactDatabaseTest, InsertThrowsOnNonGroundAtom) {
    FactDatabase db;
    Atom not_ground("ancestor", {Term::make_variable("X"), Term::make_constant("b")});

    EXPECT_THROW(db.insert(not_ground), std::invalid_argument);
    EXPECT_TRUE(db.empty());
}

TEST(DatalogFactDatabaseTest, ConstructingFromSetWithNonGroundAtomThrows) {
    FactSet facts{Atom("ancestor", {Term::make_variable("X"), Term::make_constant("b")})};
    EXPECT_THROW(FactDatabase{facts}, std::invalid_argument);
}

TEST(DatalogFactDatabaseTest, EqualityComparesFactSets) {
    FactDatabase a;
    a.insert(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));

    FactDatabase b;
    b.insert(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));

    FactDatabase c;
    c.insert(Atom("edge", {Term::make_constant("b"), Term::make_constant("c")}));

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

}  // namespace
}  // namespace pulsatrix::datalog
