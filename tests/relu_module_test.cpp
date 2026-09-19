#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/relu_module.hpp"

namespace exai {
namespace {

class ReluModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    ReluModule relu{&backend};
};

TEST_F(ReluModuleTest, ForwardClampsNegativeValuesToZero) {
    Tensor x(Shape({5}), &backend, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    Tensor y = relu.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(y.data()[3], 0.5f);
    EXPECT_FLOAT_EQ(y.data()[4], 2.0f);
}

TEST_F(ReluModuleTest, BackwardPassesGradientThroughPositiveInputsOnly) {
    Tensor x(Shape({4}), &backend, {-1.0f, 0.0f, 1.0f, 2.0f});
    (void)relu.forward(x);  // must forward first -- backward needs the cached input

    Tensor grad_y(Shape({4}), &backend, {10.0f, 10.0f, 10.0f, 10.0f});
    Tensor grad_x = relu.backward(grad_y);

    EXPECT_FLOAT_EQ(grad_x.data()[0], 0.0f);   // x=-1: blocked
    EXPECT_FLOAT_EQ(grad_x.data()[1], 0.0f);   // x=0: project convention -- treated as blocked, not passed
    EXPECT_FLOAT_EQ(grad_x.data()[2], 10.0f);  // x=1: passed through
    EXPECT_FLOAT_EQ(grad_x.data()[3], 10.0f);  // x=2: passed through
}

// Pass-through is a stronger claim than mere conservation-of-sum -- this asserts exact
// per-element equality, not just that the totals match. See mission_activations.md's
// exit gate for why this is the correct rule, not a placeholder.
TEST_F(ReluModuleTest, PropagateRelevancePassesThroughUnchanged) {
    Tensor x(Shape({3}), &backend, {-1.0f, 2.0f, 3.0f});
    (void)relu.forward(x);

    Tensor relevance_out(Shape({3}), &backend, {0.1f, 5.0f, -2.5f});
    LRPRuleConfig config;  // epsilon is irrelevant to a pass-through rule -- default is fine

    Tensor relevance_in = relu.propagate_relevance(relevance_out, config);

    EXPECT_FLOAT_EQ(relevance_in.data()[0], relevance_out.data()[0]);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], relevance_out.data()[1]);
    EXPECT_FLOAT_EQ(relevance_in.data()[2], relevance_out.data()[2]);
}

using ReluModuleDeathTest = ReluModuleTest;

// backward() dereferences Tensor::data() directly in a raw host loop -- undefined behavior
// on a CUDA-backed Tensor. Phase 1.5 Mission 2 (mission_host_loop_guards.md) guards it with
// EXAI_ASSERT. No real GPU needed: DeviceType::Cuda over real CPUBackend memory triggers
// the guard identically to a genuine CUDA tensor (see LinearModuleDeathTest for the same
// pattern). propagate_relevance() is NOT guarded -- it's a plain Tensor copy, which routes
// through DeviceBackend::copy() and is already backend-safe.
TEST_F(ReluModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor x(Shape({4}), &backend, {-1.0f, 0.0f, 1.0f, 2.0f});
    (void)relu.forward(x);

    Tensor grad_y(Shape({4}), &backend, {10.0f, 10.0f, 10.0f, 10.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)relu.backward(grad_y); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
