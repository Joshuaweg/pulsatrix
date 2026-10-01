#include "pulsatrix/negation_module.hpp"

#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class NegationModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    NegationModule negation{&backend};
};

using NegationModuleDeathTest = NegationModuleTest;

// ---------------------------------------------------------------------------
// Forward correctness -- y = 1 - x
// ---------------------------------------------------------------------------

TEST_F(NegationModuleTest, ForwardComputesOneMinusX) {
    Tensor x(Shape({5}), &backend, {0.0f, 0.25f, 0.5f, 1.0f, -3.0f});
    Tensor y = negation.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.75f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.5f);
    EXPECT_FLOAT_EQ(y.data()[3], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[4], 4.0f);  // out-of-[0,1] input: not enforced, formula still holds
}

// ---------------------------------------------------------------------------
// Backward -- dy/dx = -1
// ---------------------------------------------------------------------------

TEST_F(NegationModuleTest, BackwardNegatesGradOutput) {
    Tensor x(Shape({3}), &backend, {0.2f, 0.6f, 0.9f});
    (void)negation.forward(x);

    Tensor grad_y(Shape({3}), &backend, {1.0f, -2.0f, 0.5f});
    Tensor grad_x = negation.backward(grad_y);

    EXPECT_FLOAT_EQ(grad_x.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[2], -0.5f);
}

TEST_F(NegationModuleTest, BackwardBeforeForwardThrows) {
    Tensor grad_y(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)negation.backward(grad_y); }, std::logic_error);
}

TEST_F(NegationModuleTest, BackwardShapeMismatchThrows) {
    Tensor x(Shape({3}), &backend, {0.2f, 0.6f, 0.9f});
    (void)negation.forward(x);
    Tensor grad_y(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)negation.backward(grad_y); }, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// LRP -- pass-through conservation
// ---------------------------------------------------------------------------

TEST_F(NegationModuleTest, PropagateRelevancePassesThroughUnchanged) {
    Tensor x(Shape({3}), &backend, {0.1f, 0.4f, 0.9f});
    (void)negation.forward(x);

    Tensor relevance_out(Shape({3}), &backend, {0.3f, -1.0f, 2.0f});
    Tensor relevance_in = negation.propagate_relevance(relevance_out, LRPRuleConfig{});

    EXPECT_FLOAT_EQ(relevance_in.data()[0], relevance_out.data()[0]);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], relevance_out.data()[1]);
    EXPECT_FLOAT_EQ(relevance_in.data()[2], relevance_out.data()[2]);
}

TEST_F(NegationModuleTest, PropagateRelevanceBeforeForwardThrows) {
    Tensor relevance_out(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)negation.propagate_relevance(relevance_out, LRPRuleConfig{}); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

// Degenerate values 0 and 1 -- exact boundary of the intended [0,1] truth-value range.
TEST_F(NegationModuleTest, ForwardHandlesDegenerateZeroAndOne) {
    Tensor x(Shape({2}), &backend, {0.0f, 1.0f});
    Tensor y = negation.forward(x);
    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.0f);
}

// Empty-tensor construction -- Module::forward()'s own NVI precondition (external boundary,
// escalated to throw per module.hpp's own documented convention). Not re-implemented here;
// this test confirms NegationModule inherits it correctly.
TEST_F(NegationModuleTest, ForwardRejectsEmptyInput) {
    Tensor x(Shape({0}), &backend);
    EXPECT_THROW({ (void)negation.forward(x); }, std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
