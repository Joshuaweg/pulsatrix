#include <gtest/gtest.h>

#include <cmath>

#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/linear_module.hpp"

namespace exai {
namespace {

// At t=1, Adam's bias correction makes m_hat == grad and v_hat == grad^2 EXACTLY:
// m_1 = (1-beta1)*grad, m_hat = m_1/(1-beta1^1) = grad. Same for v. This is the one step
// tractable to verify by hand -- later steps trust the formula once this is confirmed.
TEST(AdamOptimizerTest, FirstStepMatchesHandDerivedBiasCorrection) {
    CPUBackend backend;
    LinearModule linear(1, 1, &backend);
    linear.set_weight({2.0f});
    linear.set_bias({1.0f});

    auto params = linear.parameters();
    params[0].grad->data()[0] = 4.0f;   // weight_grad
    params[1].grad->data()[0] = -2.0f;  // bias_grad

    AdamOptimizer opt(0.1f, &backend);  // default beta1=0.9, beta2=0.999, eps=1e-8
    opt.step(linear);

    // update = lr * m_hat/(sqrt(v_hat)+eps) = lr * grad/(|grad|+eps), computed the same way
    // the implementation should, so this checks the formula is applied correctly, not that
    // two independently-typed-out floats happen to match.
    float expected_weight = 2.0f - 0.1f * (4.0f / (std::sqrt(16.0f) + 1e-8f));
    float expected_bias = 1.0f - 0.1f * (-2.0f / (std::sqrt(4.0f) + 1e-8f));

    EXPECT_NEAR(linear.weight().data()[0], expected_weight, 1e-6f);
    EXPECT_NEAR(linear.bias().data()[0], expected_bias, 1e-6f);
}

TEST(AdamOptimizerTest, StateIsIsolatedPerParameter) {
    // If weight and bias' Adam state (m, v, t) were accidentally shared or swapped, this
    // first-step case (which depends only on each parameter's OWN gradient) would fail --
    // weight's update would leak bias's gradient sign or vice versa.
    CPUBackend backend;
    LinearModule linear(1, 1, &backend);
    linear.set_weight({0.0f});
    linear.set_bias({0.0f});

    auto params = linear.parameters();
    params[0].grad->data()[0] = 10.0f;  // weight: large positive gradient
    params[1].grad->data()[0] = -10.0f;  // bias: large negative gradient

    AdamOptimizer opt(0.1f, &backend);
    opt.step(linear);

    // Same-magnitude, opposite-sign gradients at t=1 -> equal-magnitude, opposite-sign updates.
    EXPECT_LT(linear.weight().data()[0], 0.0f);  // positive grad -> weight decreases
    EXPECT_GT(linear.bias().data()[0], 0.0f);    // negative grad -> bias increases
    EXPECT_NEAR(linear.weight().data()[0], -linear.bias().data()[0], 1e-6f);
}

TEST(AdamOptimizerTest, ZeroGradResetsAllGradientsToZero) {
    CPUBackend backend;
    LinearModule linear(1, 1, &backend);
    auto params = linear.parameters();
    params[0].grad->data()[0] = 5.0f;
    params[1].grad->data()[0] = 3.0f;

    AdamOptimizer opt(0.1f, &backend);
    opt.zero_grad(linear);

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 0.0f);
}

TEST(AdamOptimizerTest, StepOnParameterlessModuleIsSafeNoOp) {
    class NoParamModule : public Module {
    public:
        Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return Tensor(r); }

    protected:
        Tensor forward_impl(const Tensor& input) override { return Tensor(input); }
    };

    CPUBackend backend;
    NoParamModule m;
    AdamOptimizer opt(0.1f, &backend);
    EXPECT_NO_THROW(opt.step(m));
    EXPECT_NO_THROW(opt.zero_grad(m));
}

}  // namespace
}  // namespace exai
