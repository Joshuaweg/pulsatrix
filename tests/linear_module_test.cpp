#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"
#include "exai/lrp_rule_config.hpp"

namespace exai {
namespace {

class LinearModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(LinearModuleTest, ConstructionSetsWeightAndBiasShapes) {
    LinearModule linear(3, 2, &backend);
    EXPECT_EQ(linear.weight().shape(), Shape({3, 2}));
    EXPECT_EQ(linear.bias().shape(), Shape({2}));
}

// std::vector overloads exist alongside the initializer_list ones for runtime-sized
// callers (Phase 5 Python bindings, or any C++ caller loading weights from a file) --
// see tensor.hpp's vector-based constructor overload this delegates to.
TEST_F(LinearModuleTest, SetWeightAcceptsVector) {
    LinearModule linear(2, 1, &backend);
    linear.set_weight(std::vector<float>{1.0f, 2.0f});
    EXPECT_FLOAT_EQ(linear.weight().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(linear.weight().data()[1], 2.0f);
}

TEST_F(LinearModuleTest, SetBiasAcceptsVector) {
    LinearModule linear(2, 1, &backend);
    linear.set_bias(std::vector<float>{5.0f});
    EXPECT_FLOAT_EQ(linear.bias().data()[0], 5.0f);
}

// Every internally-constructed Tensor defaulted to DeviceType::Cpu regardless of the
// backend actually passed to the constructor -- a real gap Phase 1.5 Mission 3 (Objective
// 2) fixes by threading an explicit device parameter through. No real GPU needed for this
// specific check: it only verifies which DeviceType tag lands on each member, using the
// same mislabeled-Tensor trick as Mission 2's guard tests.
TEST_F(LinearModuleTest, ConstructionDefaultsEveryTensorMemberToCpu) {
    LinearModule linear(3, 2, &backend);
    EXPECT_EQ(linear.weight().device(), DeviceType::Cpu);
    EXPECT_EQ(linear.bias().device(), DeviceType::Cpu);
    EXPECT_EQ(linear.weight_grad().device(), DeviceType::Cpu);
    EXPECT_EQ(linear.bias_grad().device(), DeviceType::Cpu);
}

TEST_F(LinearModuleTest, ConstructionThreadsExplicitDeviceToEveryTensorMember) {
    LinearModule linear(3, 2, &backend, DeviceType::Cuda);
    EXPECT_EQ(linear.weight().device(), DeviceType::Cuda);
    EXPECT_EQ(linear.bias().device(), DeviceType::Cuda);
    EXPECT_EQ(linear.weight_grad().device(), DeviceType::Cuda);
    EXPECT_EQ(linear.bias_grad().device(), DeviceType::Cuda);
}

TEST_F(LinearModuleTest, WeightAndBiasGradientsStartAtZero) {
    LinearModule linear(3, 2, &backend);
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 0.0f);
}

TEST_F(LinearModuleTest, ForwardComputesHandVerifiedOutput) {
    // in_features=2, out_features=2. W = [[1,2],[3,4]] (in x out, row-major), b = [0.5, -0.5].
    // x = [1, 1]. y = x @ W + b = [1*1+1*3, 1*2+1*4] + [0.5,-0.5] = [4,6] + [0.5,-0.5] = [4.5, 5.5]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.5f, -0.5f});

    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor y = linear.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 4.5f);
    EXPECT_FLOAT_EQ(y.data()[1], 5.5f);
}

// Adversarial hardening (campaign_exai_dl_library_adversarial_hardening, Mission 1):
// forward_impl previously had zero input-shape validation -- a wrong-sized input fed
// straight into gemm as if it had in_features_ elements, a real heap OOB read on the
// input buffer if it was shorter. External boundary per Mission 0's classification
// table (this input can originate from Phase 5's Python bindings) -- throw, not assert.
TEST_F(LinearModuleTest, ForwardThrowsOnWrongNumel) {
    LinearModule linear(3, 2, &backend);
    Tensor wrong_size(Shape({2}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)linear.forward(wrong_size); }, std::invalid_argument);
}

TEST_F(LinearModuleTest, ForwardThrowsOnWrongRank) {
    LinearModule linear(4, 2, &backend);
    Tensor wrong_rank(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW({ (void)linear.forward(wrong_rank); }, std::invalid_argument);
}

// Finding 12: backward()/propagate_relevance() silently computed a meaningless answer
// from zero-initialized cached state if called before any forward() -- now a real error.
TEST_F(LinearModuleTest, BackwardThrowsIfCalledBeforeForward) {
    LinearModule linear(2, 2, &backend);
    Tensor grad(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)linear.backward(grad); }, std::logic_error);
}

TEST_F(LinearModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    LinearModule linear(2, 2, &backend);
    Tensor relevance(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)linear.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(LinearModuleTest, BackwardComputesHandVerifiedInputGradient) {
    // Same W as above. grad_y = [1, 1]. grad_x = grad_y @ W^T.
    // W^T = [[1,3],[2,4]]. grad_x = [1*1+1*2, 1*3+1*4] = [3, 7]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
    linear.set_bias({0.0f, 0.0f});

    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);  // must forward first -- backward needs the cached input

    Tensor grad_y(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor grad_x = linear.backward(grad_y);

    EXPECT_FLOAT_EQ(grad_x.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(grad_x.data()[1], 7.0f);
}

TEST_F(LinearModuleTest, BackwardAccumulatesWeightGradientAsOuterProduct) {
    // grad_W = outer(x, grad_y). x=[1,2], grad_y=[3,4] -> grad_W = [[1*3,1*4],[2*3,2*4]] = [[3,4],[6,8]]
    LinearModule linear(2, 2, &backend);
    linear.set_weight({0.0f, 0.0f, 0.0f, 0.0f});
    linear.set_bias({0.0f, 0.0f});

    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {3.0f, 4.0f});
    (void)linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 3.0f);  // W[0][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[1], 4.0f);  // W[0][1]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[2], 6.0f);  // W[1][0]
    EXPECT_FLOAT_EQ(linear.weight_grad().data()[3], 8.0f);  // W[1][1]
}

TEST_F(LinearModuleTest, BackwardAccumulatesBiasGradientAsGradOutput) {
    LinearModule linear(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);
    Tensor grad_y(Shape({2}), &backend, {2.5f, -1.5f});
    (void)linear.backward(grad_y);

    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.5f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[1], -1.5f);
}

TEST_F(LinearModuleTest, GradientsAccumulateAcrossTwoBackwardCalls) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({1.0f});
    linear.set_bias({0.0f});

    Tensor x(Shape({1}), &backend, {2.0f});
    Tensor grad_y(Shape({1}), &backend, {1.0f});

    (void)linear.forward(x);
    (void)linear.backward(grad_y);  // grad_W += 2*1 = 2
    (void)linear.forward(x);
    (void)linear.backward(grad_y);  // grad_W += 2*1 = 2 again -> total 4

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 4.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 2.0f);
}

// Bias is deliberately excluded from the LRP rule's z_j (the pre-bias linear output is
// used, not the forward() return value) -- this is what makes exact conservation possible;
// see this mission's Notes for the full rationale (bias has no associated input feature to
// redistribute relevance to, a well-known LRP simplification).
TEST_F(LinearModuleTest, PropagateRelevanceMatchesHandDerivedValues) {
    // W = [[2],[3]] (2 in, 1 out). x = [1, 2]. z_0 (pre-bias) = 1*2 + 2*3 = 8.
    LinearModule linear(2, 1, &backend);
    linear.set_weight({2.0f, 3.0f});
    linear.set_bias({100.0f});  // deliberately large/irrelevant -- must not affect the result

    Tensor x(Shape({2}), &backend, {1.0f, 2.0f});
    (void)linear.forward(x);

    Tensor relevance_out(Shape({1}), &backend, {10.0f});
    LRPRuleConfig config;
    config.epsilon = 0.0f;  // exact -- no absorption term, isolates the rule's core correctness

    Tensor relevance_in = linear.propagate_relevance(relevance_out, config);

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 2.5f);  // (1*2/8)*10
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 7.5f);  // (2*3/8)*10
}

TEST_F(LinearModuleTest, PropagateRelevanceConservesTotalRelevance) {
    // The mission's real acceptance criterion for the LRP half -- see mission_module_linear.md.
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, -2.0f, 3.0f, 0.5f, 2.5f, -1.0f});
    linear.set_bias({0.1f, -0.2f});

    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 0.5f});
    (void)linear.forward(x);

    Tensor relevance_out(Shape({2}), &backend, {4.0f, 6.0f});
    LRPRuleConfig config;  // default epsilon (1e-6) -- proves conservation holds in normal use, not just the epsilon=0 special case

    Tensor relevance_in = linear.propagate_relevance(relevance_out, config);

    float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
    float sum_out = relevance_out.data()[0] + relevance_out.data()[1];
    EXPECT_NEAR(sum_in, sum_out, 1e-3f);
}

using LinearModuleDeathTest = LinearModuleTest;

// backward()/propagate_relevance() dereference Tensor::data() in raw host loops (via
// backward()'s transpose() helper, and propagate_relevance()'s own loop) -- undefined
// behavior on a CUDA-backed Tensor. Phase 1.5 Mission 2
// (mission_host_loop_guards.md) guards both with EXAI_ASSERT. No real GPU needed to test
// this: Tensor::device() is metadata decoupled from which DeviceBackend* actually
// allocated its buffer, so a Tensor tagged DeviceType::Cuda over real CPUBackend memory
// triggers the guard just like a genuine CUDA tensor would.
TEST_F(LinearModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LinearModule linear(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);

    Tensor grad_output(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)linear.backward(grad_output); }, "EXAI_ASSERT failed");
}

TEST_F(LinearModuleDeathTest, PropagateRelevanceAbortsOnNonCpuRelevanceOut) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LinearModule linear(2, 2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)linear.forward(x);

    Tensor relevance_out(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    LRPRuleConfig config;
    EXPECT_DEATH({ (void)linear.propagate_relevance(relevance_out, config); }, "EXAI_ASSERT failed");
}

TEST_F(LinearModuleTest, ParametersExposesWeightAndBiasByPointer) {
    LinearModule linear(2, 2, &backend);
    auto params = linear.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].value, &linear.weight());
    EXPECT_EQ(params[0].grad, &linear.weight_grad());
    EXPECT_EQ(params[1].value, &linear.bias());
    EXPECT_EQ(params[1].grad, &linear.bias_grad());
}

TEST_F(LinearModuleTest, MutatingThroughParametersChangesModuleState) {
    LinearModule linear(1, 1, &backend);
    linear.set_weight({5.0f});

    auto params = linear.parameters();
    params[0].value->data()[0] = 99.0f;  // mutate through the returned pointer

    EXPECT_FLOAT_EQ(linear.weight().data()[0], 99.0f);  // module's own state reflects it
}

}  // namespace
}  // namespace exai
