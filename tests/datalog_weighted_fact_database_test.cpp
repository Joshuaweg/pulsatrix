#include "pulsatrix/datalog_weighted_fact_database.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace pulsatrix::datalog {
namespace {

Term C(const std::string& value) { return Term::make_constant(value); }
Term V(const std::string& name) { return Term::make_variable(name); }

TEST(WeightedFactDatabaseTest, StartsEmpty) {
    WeightedFactDatabase<double> db;
    EXPECT_TRUE(db.empty());
    EXPECT_EQ(db.size(), 0u);
}

TEST(WeightedFactDatabaseTest, SetAddsFactWithWeight) {
    WeightedFactDatabase<double> db;
    db.set(Atom("edge", {C("a"), C("b")}), 0.9);

    EXPECT_TRUE(db.contains(Atom("edge", {C("a"), C("b")})));
    EXPECT_DOUBLE_EQ(db.weight_of(Atom("edge", {C("a"), C("b")}), 0.0), 0.9);
    EXPECT_EQ(db.size(), 1u);
}

TEST(WeightedFactDatabaseTest, WeightOfAbsentFactReturnsDefault) {
    WeightedFactDatabase<double> db;
    EXPECT_FALSE(db.contains(Atom("edge", {C("a"), C("b")})));
    EXPECT_DOUBLE_EQ(db.weight_of(Atom("edge", {C("a"), C("b")}), 0.0), 0.0);
}

TEST(WeightedFactDatabaseTest, SetOverwritesExistingWeight) {
    WeightedFactDatabase<double> db;
    Atom fact("edge", {C("a"), C("b")});
    db.set(fact, 0.5);
    db.set(fact, 0.9);

    EXPECT_EQ(db.size(), 1u);  // overwrite, not a second entry
    EXPECT_DOUBLE_EQ(db.weight_of(fact, 0.0), 0.9);
}

TEST(WeightedFactDatabaseTest, SetThrowsOnNonGroundAtom) {
    WeightedFactDatabase<double> db;
    EXPECT_THROW(db.set(Atom("edge", {V("X"), C("b")}), 0.9), std::invalid_argument);
}

TEST(WeightedFactDatabaseTest, ConstructingFromMapWithNonGroundAtomThrows) {
    WeightedFactSet<double> facts{{Atom("edge", {V("X"), C("b")}), 0.9}};
    EXPECT_THROW(WeightedFactDatabase<double> db(facts), std::invalid_argument);
}

TEST(WeightedFactDatabaseTest, EqualityComparesFactWeightMaps) {
    WeightedFactDatabase<double> a;
    a.set(Atom("edge", {C("a"), C("b")}), 0.9);
    WeightedFactDatabase<double> b;
    b.set(Atom("edge", {C("a"), C("b")}), 0.9);
    WeightedFactDatabase<double> c;
    c.set(Atom("edge", {C("a"), C("b")}), 0.5);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(WeightedFactDatabaseTest, WorksWithBooleanValueType) {
    WeightedFactDatabase<bool> db;
    db.set(Atom("edge", {C("a"), C("b")}), true);
    EXPECT_TRUE(db.weight_of(Atom("edge", {C("a"), C("b")}), false));
    EXPECT_FALSE(db.weight_of(Atom("edge", {C("b"), C("c")}), false));
}

}  // namespace
}  // namespace pulsatrix::datalog
