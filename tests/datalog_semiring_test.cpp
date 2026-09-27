#include "pulsatrix/datalog_semiring.hpp"

#include <gtest/gtest.h>

namespace pulsatrix::datalog {
namespace {

TEST(BooleanSemiringTest, ZeroAndOneAreFalseAndTrue) {
    EXPECT_EQ(BooleanSemiring::zero(), false);
    EXPECT_EQ(BooleanSemiring::one(), true);
}

TEST(BooleanSemiringTest, AddIsLogicalOr) {
    EXPECT_EQ(BooleanSemiring::add(false, false), false);
    EXPECT_EQ(BooleanSemiring::add(true, false), true);
    EXPECT_EQ(BooleanSemiring::add(false, true), true);
    EXPECT_EQ(BooleanSemiring::add(true, true), true);
}

TEST(BooleanSemiringTest, MulIsLogicalAnd) {
    EXPECT_EQ(BooleanSemiring::mul(false, false), false);
    EXPECT_EQ(BooleanSemiring::mul(true, false), false);
    EXPECT_EQ(BooleanSemiring::mul(false, true), false);
    EXPECT_EQ(BooleanSemiring::mul(true, true), true);
}

TEST(BooleanSemiringTest, SatisfiesSemiringIdentityLaws) {
    // add(zero(), x) == x, mul(one(), x) == x -- both semiring's own identity laws, load-bearing
    // for the weighted engine's single-derivation-path correctness (see datalog_weighted_engine.hpp).
    for (bool x : {false, true}) {
        EXPECT_EQ(BooleanSemiring::add(BooleanSemiring::zero(), x), x);
        EXPECT_EQ(BooleanSemiring::mul(BooleanSemiring::one(), x), x);
    }
}

TEST(RealSemiringTest, ZeroAndOneAreZeroPointZeroAndOnePointZero) {
    EXPECT_DOUBLE_EQ(RealSemiring<double>::zero(), 0.0);
    EXPECT_DOUBLE_EQ(RealSemiring<double>::one(), 1.0);
}

TEST(RealSemiringTest, AddIsFloatingPointAddition) {
    EXPECT_DOUBLE_EQ(RealSemiring<double>::add(0.3, 0.4), 0.7);
}

TEST(RealSemiringTest, MulIsFloatingPointMultiplication) {
    EXPECT_DOUBLE_EQ(RealSemiring<double>::mul(0.9, 0.8), 0.72);
}

TEST(RealSemiringTest, SatisfiesSemiringIdentityLaws) {
    EXPECT_DOUBLE_EQ(RealSemiring<double>::add(RealSemiring<double>::zero(), 0.42), 0.42);
    EXPECT_DOUBLE_EQ(RealSemiring<double>::mul(RealSemiring<double>::one(), 0.42), 0.42);
}

// Adversarial/boundary: RealSemiring<float> is a distinct instantiation, per the mission's
// stated "double/float" scope -- confirm it also behaves correctly, not just RealSemiring<double>.
TEST(RealSemiringTest, FloatInstantiationBehavesIdenticallyInShape) {
    EXPECT_FLOAT_EQ(RealSemiring<float>::zero(), 0.0f);
    EXPECT_FLOAT_EQ(RealSemiring<float>::one(), 1.0f);
    EXPECT_FLOAT_EQ(RealSemiring<float>::add(0.3f, 0.4f), 0.7f);
    EXPECT_FLOAT_EQ(RealSemiring<float>::mul(0.9f, 0.8f), 0.72f);
}

}  // namespace
}  // namespace pulsatrix::datalog
