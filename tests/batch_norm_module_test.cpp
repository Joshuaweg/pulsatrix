#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class BatchNormModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    static float loss(BatchNormModule& norm, const Tensor& x, const Tensor& grad_output) {
        Tensor y = norm.forward(x);
        float total = 0.0f;
        for (int64_t i = 0; i < y.numel(); ++i) {
            total += grad_output.data()[i] * y.data()[i];
        }
        return total;
    }
};

TEST_F(BatchNormModuleTest, ConstructionThrowsOnZeroNumChannels) {
    EXPECT_THROW({ BatchNormModule norm(0, &backend); }, std::invalid_argument);
}

TEST_F(BatchNormModuleTest, ConstructionThrowsOnNegativeNumChannels) {
    EXPECT_THROW({ BatchNormModule norm(-2, &backend); }, std::invalid_argument);
}

TEST_F(BatchNormModuleTest, GammaAndBetaGradientsStartAtZero) {
    BatchNormModule norm(2, &backend);
    EXPECT_FLOAT_EQ(norm.gamma_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 0.0f);
}

// The real acceptance criterion for BatchNorm's defining statistic -- computed PER CHANNEL
// ACROSS THE BATCH (not per example like GroupNorm/LayerNorm). N=2, C=2, H=W=1: channel 0
// values across the batch are [1,3] (mean=2, var=1, std=1 with eps=0) -> xhat=[-1,1].
// Channel 1 values are [0,4] (mean=2, var=4, std=2) -> xhat=[-1,1].
// gamma=[2,3], beta=[10,20] -> y: ch0=[8,12], ch1=[17,23].
TEST_F(BatchNormModuleTest, ForwardComputesHandVerifiedOutputAcrossTheBatch) {
    BatchNormModule norm(2, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 3.0f});
    norm.set_beta({10.0f, 20.0f});

    // Row-major (N,C,H,W)=(2,2,1,1): [n0c0, n0c1, n1c0, n1c1] = [1, 0, 3, 4].
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    Tensor y = norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({2, 2, 1, 1}));
    EXPECT_FLOAT_EQ(y.data()[0], 8.0f);   // n0c0
    EXPECT_FLOAT_EQ(y.data()[1], 17.0f);  // n0c1
    EXPECT_FLOAT_EQ(y.data()[2], 12.0f);  // n1c0
    EXPECT_FLOAT_EQ(y.data()[3], 23.0f);  // n1c1
}

TEST_F(BatchNormModuleTest, ForwardThrowsOnWrongRank) {
    BatchNormModule norm(2, &backend);
    Tensor wrong_rank(Shape({2, 1, 1}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(BatchNormModuleTest, ForwardThrowsOnWrongChannelCount) {
    BatchNormModule norm(2, &backend);
    Tensor wrong_channels(Shape({2, 3, 1, 1}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_channels); }, std::invalid_argument);
}

TEST_F(BatchNormModuleTest, BackwardThrowsIfCalledBeforeForward) {
    BatchNormModule norm(2, &backend);
    Tensor grad(Shape({2, 2, 1, 1}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(grad); }, std::logic_error);
}

TEST_F(BatchNormModuleTest, BackwardThrowsOnMismatchedGradOutputShape) {
    BatchNormModule norm(2, &backend);
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    (void)norm.forward(x);
    Tensor wrong_shape_grad(Shape({1, 2, 1, 1}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_shape_grad); }, std::invalid_argument);
}

// Exact algebraic identity (y is affine in beta_c with unit coefficient per (n,h,w)
// position within channel c) -- not a finite-difference approximation.
TEST_F(BatchNormModuleTest, BackwardBetaGradientEqualsPerChannelSumOfGradOutputAcrossBatch) {
    BatchNormModule norm(2, &backend);
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({2, 2, 1, 1}), &backend, {1.0f, 2.0f, 0.5f, 0.5f});
    (void)norm.backward(grad_output);

    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 1.5f);  // ch0: n0=1.0 + n1=0.5
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[1], 2.5f);  // ch1: n0=2.0 + n1=0.5
}

// The real acceptance criterion for backward()'s input/gamma gradient correctness --
// central finite differences, per this project's finite-difference discipline. N=3,
// C=2, H=1, W=1 -- proves the per-channel statistic genuinely pools across all 3 batch
// rows (not accidentally per-row like GroupNorm/LayerNorm).
TEST_F(BatchNormModuleTest, BackwardInputGradientMatchesFiniteDifferenceAcrossThreeExampleBatch) {
    BatchNormModule norm(2, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 3.0f});
    norm.set_beta({0.0f, 0.0f});

    std::vector<float> x_vals = {1.0f, 0.0f, 3.0f, 4.0f, -1.0f, 2.0f};  // (3,2,1,1)
    Tensor x(Shape({3, 2, 1, 1}), &backend, x_vals);
    Tensor grad_output(Shape({3, 2, 1, 1}), &backend, {1.0f, 0.5f, -0.25f, 0.75f, 0.2f, -0.4f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(6);
    for (int64_t i = 0; i < 6; ++i) {
        std::vector<float> x_plus = x_vals;
        std::vector<float> x_minus = x_vals;
        x_plus[static_cast<size_t>(i)] += h;
        x_minus[static_cast<size_t>(i)] -= h;
        Tensor xp(Shape({3, 2, 1, 1}), &backend, x_plus);
        Tensor xm(Shape({3, 2, 1, 1}), &backend, x_minus);
        float lp = loss(norm, xp, grad_output);
        float lm = loss(norm, xm, grad_output);
        fd_grad[static_cast<size_t>(i)] = (lp - lm) / (2.0f * h);
    }

    (void)norm.forward(x);  // restore cached state to the unperturbed input before backward()
    Tensor grad_input = norm.backward(grad_output);

    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(grad_input.data()[i], fd_grad[static_cast<size_t>(i)], 1.5e-2f);
    }
}

TEST_F(BatchNormModuleTest, BackwardGammaGradientSumsAcrossThreeExampleBatch) {
    BatchNormModule norm(2, &backend, DeviceType::Cpu, 0.0f);
    std::vector<float> gamma_vals = {2.0f, 3.0f};
    norm.set_gamma(gamma_vals);
    norm.set_beta({0.0f, 0.0f});

    Tensor x(Shape({3, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f, -1.0f, 2.0f});
    Tensor grad_output(Shape({3, 2, 1, 1}), &backend, {1.0f, 0.5f, -0.25f, 0.75f, 0.2f, -0.4f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(2);
    for (int64_t i = 0; i < 2; ++i) {
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

    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(norm.gamma_grad().data()[i], fd_grad[static_cast<size_t>(i)], 1.5e-2f);
    }
}

TEST_F(BatchNormModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    BatchNormModule norm(2, &backend);
    Tensor relevance(Shape({2, 2, 1, 1}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(BatchNormModuleTest, PropagateRelevanceThrowsOnWrongSize) {
    BatchNormModule norm(2, &backend);
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    (void)norm.forward(x);
    Tensor wrong_size_relevance(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(wrong_size_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

TEST_F(BatchNormModuleTest, PropagateRelevanceIsIdentityPassThrough) {
    BatchNormModule norm(2, &backend);
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({2, 2, 1, 1}), &backend, {4.0f, -1.0f, 2.5f, 0.0f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

TEST_F(BatchNormModuleTest, PropagateRelevanceConservesTotalRelevance) {
    BatchNormModule norm(2, &backend);
    Tensor x(Shape({2, 2, 1, 1}), &backend, {1.0f, 0.0f, 3.0f, 4.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({2, 2, 1, 1}), &backend, {4.0f, -1.0f, 2.5f, 0.0f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f, sum_out = 0.0f;
    for (int64_t i = 0; i < 4; ++i) {
        sum_in += relevance_in.data()[i];
        sum_out += relevance_out.data()[i];
    }
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

TEST_F(BatchNormModuleTest, ParametersExposesGammaAndBetaByPointer) {
    BatchNormModule norm(2, &backend);
    auto params = norm.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].value, &norm.gamma());
    EXPECT_EQ(params[0].grad, &norm.gamma_grad());
    EXPECT_EQ(params[1].value, &norm.beta());
    EXPECT_EQ(params[1].grad, &norm.beta_grad());
}

using BatchNormModuleDeathTest = BatchNormModuleTest;

}  // namespace
}  // namespace pulsatrix
