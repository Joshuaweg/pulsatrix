/** @file learnable_scalar_test.cpp
 *  @brief LearnableScalar construction, gradient accumulation, and plain-SGD step.
 */
#include <gtest/gtest.h>

#include "pulsatrix/learnable_scalar.hpp"

namespace pulsatrix {
namespace {

TEST(LearnableScalarTest, DefaultsToZero) {
    LearnableScalar s;
    EXPECT_FLOAT_EQ(s.value(), 0.0f);
    EXPECT_FLOAT_EQ(s.grad(), 0.0f);
}

TEST(LearnableScalarTest, ConstructsWithInitialValue) {
    LearnableScalar s(3.5f);
    EXPECT_FLOAT_EQ(s.value(), 3.5f);
    EXPECT_FLOAT_EQ(s.grad(), 0.0f);
}

TEST(LearnableScalarTest, AccumulateGradSums) {
    LearnableScalar s;
    s.accumulate_grad(2.0f);
    s.accumulate_grad(1.5f);
    EXPECT_FLOAT_EQ(s.grad(), 3.5f);
}

TEST(LearnableScalarTest, ZeroGradResetsGradNotValue) {
    LearnableScalar s(1.0f);
    s.accumulate_grad(4.0f);
    s.zero_grad();
    EXPECT_FLOAT_EQ(s.grad(), 0.0f);
    EXPECT_FLOAT_EQ(s.value(), 1.0f);
}

TEST(LearnableScalarTest, StepAppliesPlainSGDUpdate) {
    // Hand-derived: value=2.0, grad=4.0, lr=0.1 -> value -= 0.1*4.0 = 2.0 - 0.4 = 1.6.
    LearnableScalar s(2.0f);
    s.accumulate_grad(4.0f);
    s.step(0.1f);
    EXPECT_NEAR(s.value(), 1.6f, 1e-6f);
}

TEST(LearnableScalarTest, StepDoesNotResetGrad) {
    LearnableScalar s(1.0f);
    s.accumulate_grad(1.0f);
    s.step(0.1f);
    EXPECT_FLOAT_EQ(s.grad(), 1.0f);
}

TEST(LearnableScalarTest, TwoStepsCompoundCorrectly) {
    // value=0, grad=1 -> step(1.0) -> value=-1. zero_grad, grad=2 -> step(1.0) -> value=-3.
    LearnableScalar s;
    s.accumulate_grad(1.0f);
    s.step(1.0f);
    EXPECT_NEAR(s.value(), -1.0f, 1e-6f);
    s.zero_grad();
    s.accumulate_grad(2.0f);
    s.step(1.0f);
    EXPECT_NEAR(s.value(), -3.0f, 1e-6f);
}

}  // namespace
}  // namespace pulsatrix
