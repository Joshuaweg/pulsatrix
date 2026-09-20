#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/layer_norm_module.hpp"
#include "exai/lrp_rule_config.hpp"

namespace exai {
namespace {

class LayerNormModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    static float loss(LayerNormModule& norm, const Tensor& x, const Tensor& grad_output) {
        Tensor y = norm.forward(x);
        float total = 0.0f;
        for (int64_t i = 0; i < y.numel(); ++i) {
            total += grad_output.data()[i] * y.data()[i];
        }
        return total;
    }
};

TEST_F(LayerNormModuleTest, ConstructionThrowsOnZeroNumFeatures) {
    EXPECT_THROW({ LayerNormModule norm(0, &backend); }, std::invalid_argument);
}

TEST_F(LayerNormModuleTest, ConstructionThrowsOnNegativeNumFeatures) {
    EXPECT_THROW({ LayerNormModule norm(-2, &backend); }, std::invalid_argument);
}

TEST_F(LayerNormModuleTest, GammaAndBetaGradientsStartAtZero) {
    LayerNormModule norm(3, &backend);
    EXPECT_FLOAT_EQ(norm.gamma_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 0.0f);
}

// x=[0,4] -> mu=2, var=mean([4,4])=4, std=2 (eps=0). xhat=[-1,1].
// gamma=[3,3], beta=[10,10] -> y = [3*-1+10, 3*1+10] = [7,13].
TEST_F(LayerNormModuleTest, ForwardComputesHandVerifiedOutput) {
    LayerNormModule norm(2, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({3.0f, 3.0f});
    norm.set_beta({10.0f, 10.0f});

    Tensor x(Shape({1, 2}), &backend, {0.0f, 4.0f});
    Tensor y = norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({1, 2}));
    EXPECT_FLOAT_EQ(y.data()[0], 7.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 13.0f);
}

// Same x/xhat, per-element gamma/beta -- proves both are applied elementwise, not as a
// shared scalar (the uniform case above can't distinguish that).
TEST_F(LayerNormModuleTest, ForwardScalesAndShiftsPerElement) {
    LayerNormModule norm(2, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 5.0f});
    norm.set_beta({1.0f, -1.0f});

    Tensor x(Shape({1, 2}), &backend, {0.0f, 4.0f});
    Tensor y = norm.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], -1.0f);  // 2*-1 + 1
    EXPECT_FLOAT_EQ(y.data()[1], 4.0f);   // 5*1 + -1
}

// The real acceptance criterion for the batch migration's forward pass -- N=2, each row
// with a different mu/std, proving rows are normalized independently.
TEST_F(LayerNormModuleTest, ForwardComputesIndependentStatsPerRowAcrossTwoExampleBatch) {
    // Row 0: x=[0,4] -> y=[7,13] (as above). Row 1: x=[10,20] -> mu=15, var=mean([25,25])=25,
    // std=5. xhat=[-1,1]. y = [3*-1+10, 3*1+10] = [7,13] too (LayerNorm's shift/scale
    // invariance -- same normalized result for any affine-transformed row). Use a
    // genuinely different row 1 instead: x=[0,2] -> mu=1, var=1, std=1. xhat=[-1,1].
    // y=[7,13] again by the same invariance -- LayerNorm inherently produces the same xhat
    // for any two-point row regardless of scale/shift, so distinguish rows via a
    // three-element case instead.
    LayerNormModule norm(3, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({1.0f, 1.0f, 1.0f});
    norm.set_beta({0.0f, 0.0f, 0.0f});

    // Row 0: x=[0,3,6] -> mu=3, var=mean([9,0,9])=6, std=sqrt(6). xhat=[-3,0,3]/sqrt(6).
    // Row 1: x=[0,1,2] -> mu=1, var=mean([1,0,1])=2/3, std=sqrt(2/3). xhat=[-1,0,1]/sqrt(2/3).
    Tensor x(Shape({2, 3}), &backend, {0.0f, 3.0f, 6.0f, 0.0f, 1.0f, 2.0f});
    Tensor y = norm.forward(x);

    float std0 = std::sqrt(6.0f);
    float std1 = std::sqrt(2.0f / 3.0f);
    EXPECT_NEAR(y.data()[0], -3.0f / std0, 1e-5f);
    EXPECT_NEAR(y.data()[1], 0.0f, 1e-5f);
    EXPECT_NEAR(y.data()[2], 3.0f / std0, 1e-5f);
    EXPECT_NEAR(y.data()[3], -1.0f / std1, 1e-5f);
    EXPECT_NEAR(y.data()[4], 0.0f, 1e-5f);
    EXPECT_NEAR(y.data()[5], 1.0f / std1, 1e-5f);
}

TEST_F(LayerNormModuleTest, ForwardThrowsOnWrongRank) {
    LayerNormModule norm(4, &backend);
    Tensor wrong_rank(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(LayerNormModuleTest, ForwardThrowsOnWrongNumel) {
    LayerNormModule norm(4, &backend);
    Tensor wrong_size(Shape({1, 2}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_size); }, std::invalid_argument);
}

TEST_F(LayerNormModuleTest, BackwardThrowsIfCalledBeforeForward) {
    LayerNormModule norm(3, &backend);
    Tensor grad(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(grad); }, std::logic_error);
}

TEST_F(LayerNormModuleTest, BackwardThrowsOnWrongGradOutputSize) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);
    Tensor wrong_size_grad(Shape({1, 2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_size_grad); }, std::invalid_argument);
}

TEST_F(LayerNormModuleTest, BackwardThrowsOnBatchSizeMismatch) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);
    Tensor wrong_batch_grad(Shape({2, 3}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_batch_grad); }, std::invalid_argument);
}

// dL/dbeta_i = dL/dy_i exactly (y is affine in beta with unit coefficient), summed across
// the batch since beta is shared -- an exact algebraic identity, not a finite-difference
// approximation.
TEST_F(LayerNormModuleTest, BackwardBetaGradientSumsGradOutputAcrossTwoExampleBatch) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({2, 3}), &backend, {1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({2, 3}), &backend, {2.5f, -1.5f, 0.5f, 1.0f, 0.5f, -0.5f});
    (void)norm.backward(grad_output);

    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 3.5f);   // 2.5 + 1.0
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[1], -1.0f);  // -1.5 + 0.5
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[2], 0.0f);   // 0.5 + -0.5
}

// The real acceptance criterion for backward()'s input-gradient correctness -- central
// finite differences against the analytic gradient. N=2 -- proves each row's gradient is
// computed independently from its own cached mu/std.
TEST_F(LayerNormModuleTest, BackwardInputGradientMatchesFiniteDifferenceAcrossTwoExampleBatch) {
    LayerNormModule norm(3, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({1.0f, 1.0f, 1.0f});
    norm.set_beta({0.0f, 0.0f, 0.0f});

    std::vector<float> x_vals = {1.0f, 2.0f, 5.0f, 0.0f, 1.0f, 2.0f};
    Tensor x(Shape({2, 3}), &backend, x_vals);
    Tensor grad_output(Shape({2, 3}), &backend, {1.0f, 0.5f, -0.25f, 0.3f, -0.2f, 0.1f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(6);
    for (int64_t i = 0; i < 6; ++i) {
        std::vector<float> x_plus = x_vals;
        std::vector<float> x_minus = x_vals;
        x_plus[static_cast<size_t>(i)] += h;
        x_minus[static_cast<size_t>(i)] -= h;
        Tensor xp(Shape({2, 3}), &backend, x_plus);
        Tensor xm(Shape({2, 3}), &backend, x_minus);
        float lp = loss(norm, xp, grad_output);
        float lm = loss(norm, xm, grad_output);
        fd_grad[static_cast<size_t>(i)] = (lp - lm) / (2.0f * h);
    }

    (void)norm.forward(x);  // restore cached state to the unperturbed input before backward()
    Tensor grad_input = norm.backward(grad_output);

    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad_input.data()[i], fd_grad[static_cast<size_t>(i)], 1e-2f);
    }
}

TEST_F(LayerNormModuleTest, BackwardGammaGradientSumsAcrossTwoExampleBatch) {
    LayerNormModule norm(3, &backend, DeviceType::Cpu, 0.0f);
    std::vector<float> gamma_vals = {1.0f, 1.5f, 2.0f};
    norm.set_gamma(gamma_vals);
    norm.set_beta({0.0f, 0.0f, 0.0f});

    Tensor x(Shape({2, 3}), &backend, {1.0f, 2.0f, 5.0f, 0.0f, 1.0f, 2.0f});
    Tensor grad_output(Shape({2, 3}), &backend, {1.0f, 0.5f, -0.25f, 0.3f, -0.2f, 0.1f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(3);
    for (int64_t i = 0; i < 3; ++i) {
        std::vector<float> g_plus = gamma_vals;
        std::vector<float> g_minus = gamma_vals;
        g_plus[static_cast<size_t>(i)] += h;
        g_minus[static_cast<size_t>(i)] -= h;

        norm.set_gamma(g_plus);
        float lp = loss(norm, x, grad_output);
        norm.set_gamma(g_minus);
        float lm = loss(norm, x, grad_output);
        fd_grad[static_cast<size_t>(i)] = (lp - lm) / (2.0f * h);
    }

    norm.set_gamma(gamma_vals);
    (void)norm.forward(x);
    (void)norm.backward(grad_output);

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(norm.gamma_grad().data()[i], fd_grad[static_cast<size_t>(i)], 1e-2f);
    }
}

TEST_F(LayerNormModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    LayerNormModule norm(3, &backend);
    Tensor relevance(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(LayerNormModuleTest, PropagateRelevanceThrowsOnWrongSize) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);
    Tensor wrong_size_relevance(Shape({1, 2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(wrong_size_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

// The mission's real acceptance criterion for the LRP half -- identity rule (AttnLRP,
// Achtibat et al. 2024), same citation and treatment as RMSNormModule's identical rule.
TEST_F(LayerNormModuleTest, PropagateRelevanceIsIdentityPassThrough) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({1, 3}), &backend, {4.0f, -1.0f, 2.5f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

TEST_F(LayerNormModuleTest, PropagateRelevanceConservesTotalRelevance) {
    LayerNormModule norm(3, &backend);
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({1, 3}), &backend, {4.0f, -1.0f, 2.5f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
    float sum_out = relevance_out.data()[0] + relevance_out.data()[1] + relevance_out.data()[2];
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

TEST_F(LayerNormModuleTest, ParametersExposesGammaAndBetaByPointer) {
    LayerNormModule norm(2, &backend);
    auto params = norm.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].value, &norm.gamma());
    EXPECT_EQ(params[0].grad, &norm.gamma_grad());
    EXPECT_EQ(params[1].value, &norm.beta());
    EXPECT_EQ(params[1].grad, &norm.beta_grad());
}

using LayerNormModuleDeathTest = LayerNormModuleTest;

TEST_F(LayerNormModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    LayerNormModule norm(2, &backend);
    Tensor x(Shape({1, 2}), &backend, {1.0f, 1.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({1, 2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)norm.backward(grad_output); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
