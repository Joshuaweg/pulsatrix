#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/group_norm_module.hpp"
#include "exai/lrp_rule_config.hpp"

namespace exai {
namespace {

class GroupNormModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    static float loss(GroupNormModule& norm, const Tensor& x, const Tensor& grad_output) {
        Tensor y = norm.forward(x);
        float total = 0.0f;
        for (int64_t i = 0; i < y.numel(); ++i) {
            total += grad_output.data()[i] * y.data()[i];
        }
        return total;
    }
};

TEST_F(GroupNormModuleTest, ConstructionThrowsOnZeroNumGroups) {
    EXPECT_THROW({ GroupNormModule norm(0, 4, &backend); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, ConstructionThrowsOnZeroNumChannels) {
    EXPECT_THROW({ GroupNormModule norm(2, 0, &backend); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, ConstructionThrowsOnNegativeNumGroups) {
    EXPECT_THROW({ GroupNormModule norm(-1, 4, &backend); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, ConstructionThrowsWhenChannelsNotDivisibleByGroups) {
    EXPECT_THROW({ GroupNormModule norm(3, 4, &backend); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, GammaAndBetaGradientsStartAtZero) {
    GroupNormModule norm(2, 4, &backend);
    EXPECT_FLOAT_EQ(norm.gamma_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 0.0f);
}

// N=1 batch, 2 groups, 4 channels, H=1 W=2. Group 0 = channels {0,1}, group 1 = channels {2,3}.
// x (row-major N,C,H,W): c0=[1,1] c1=[3,3] c2=[0,0] c3=[4,4].
// Group0: mean=2, var=1, std=1 (eps=0) -> xhat: c0=-1, c1=+1.
// Group1: mean=2, var=4, std=2 (eps=0) -> xhat: c2=-1, c3=+1.
// gamma=[2,3,4,5], beta=[10,20,30,40] -> y: c0=8, c1=23, c2=26, c3=45 (each repeated over W=2).
TEST_F(GroupNormModuleTest, ForwardComputesHandVerifiedOutputAcrossTwoGroups) {
    GroupNormModule norm(2, 4, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 3.0f, 4.0f, 5.0f});
    norm.set_beta({10.0f, 20.0f, 30.0f, 40.0f});

    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    Tensor y = norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({1, 4, 1, 2}));
    std::vector<float> expected = {8.0f, 8.0f, 23.0f, 23.0f, 26.0f, 26.0f, 45.0f, 45.0f};
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(y.data()[i], expected[static_cast<size_t>(i)], 1e-4f);
    }
}

// The real acceptance criterion for the batch migration's forward pass -- N=2, each batch
// row with different group statistics, proving rows are normalized independently.
TEST_F(GroupNormModuleTest, ForwardComputesIndependentGroupStatsPerRowAcrossTwoExampleBatch) {
    GroupNormModule norm(2, 4, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 3.0f, 4.0f, 5.0f});
    norm.set_beta({10.0f, 20.0f, 30.0f, 40.0f});

    // Row 0: as above -> y = [8,8, 23,23, 26,26, 45,45].
    // Row 1: c0=[2,2] c1=[6,6] c2=[1,1] c3=[3,3]. Group0: mean=4, var=4, std=2 (eps=0) ->
    // xhat: c0=-1, c1=+1. Group1: mean=2, var=1, std=1 -> xhat: c2=-1, c3=+1. Same xhat as
    // row 0 by construction (mean-centered symmetric pairs) -> same y as row 0: [8,23,26,45]
    // repeated over W=2 -- proves per-row independence without needing new arithmetic
    // (row 1's absolute values are unrelated to row 0's, only the normalized xhat coincides).
    Tensor x(Shape({2, 4, 1, 2}), &backend,
             {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f, 2.0f, 2.0f, 6.0f, 6.0f, 1.0f, 1.0f, 3.0f, 3.0f});
    Tensor y = norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({2, 4, 1, 2}));
    std::vector<float> expected_row = {8.0f, 8.0f, 23.0f, 23.0f, 26.0f, 26.0f, 45.0f, 45.0f};
    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(y.data()[i], expected_row[static_cast<size_t>(i)], 1e-4f);
        EXPECT_NEAR(y.data()[8 + i], expected_row[static_cast<size_t>(i)], 1e-4f);
    }
}

TEST_F(GroupNormModuleTest, ForwardThrowsOnWrongRank) {
    GroupNormModule norm(2, 4, &backend);
    Tensor wrong_rank(Shape({4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, ForwardThrowsOnWrongChannelCount) {
    GroupNormModule norm(2, 4, &backend);
    Tensor wrong_channels(Shape({1, 2, 1, 2}), &backend, {1.0f, 1.0f, 2.0f, 2.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_channels); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, BackwardThrowsIfCalledBeforeForward) {
    GroupNormModule norm(2, 4, &backend);
    Tensor grad(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(grad); }, std::logic_error);
}

TEST_F(GroupNormModuleTest, BackwardThrowsOnMismatchedGradOutputShape) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);
    Tensor wrong_shape_grad(Shape({1, 4, 1, 1}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, BackwardThrowsOnBatchSizeMismatch) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);
    Tensor wrong_batch_grad(Shape({2, 4, 1, 2}), &backend,
                             {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_batch_grad); }, std::invalid_argument);
}

// Exact algebraic identity (y is affine in beta_c with unit coefficient per spatial
// position within channel c), summed across both spatial AND batch since beta is shared.
TEST_F(GroupNormModuleTest, BackwardBetaGradientEqualsPerChannelSumOfGradOutputAcrossBatch) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({2, 4, 1, 2}), &backend,
             {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f, 2.0f, 2.0f, 6.0f, 6.0f, 1.0f, 1.0f, 3.0f, 3.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({2, 4, 1, 2}), &backend,
                        {1.0f, 2.0f, 0.5f, 0.5f, -1.0f, 1.0f, 3.0f, 0.0f, 0.5f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                         1.0f});
    (void)norm.backward(grad_output);

    EXPECT_FLOAT_EQ(norm.beta_grad().data()[0], 4.0f);  // (1+2) + (0.5+0.5)
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[1], 3.0f);  // (0.5+0.5) + (1+1)
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[2], 0.0f);  // (-1+1) + (0+0)
    EXPECT_FLOAT_EQ(norm.beta_grad().data()[3], 5.0f);  // (3+0) + (1+1)
}

// The real acceptance criterion for backward()'s input/gamma gradient correctness --
// central finite differences, per this project's finite-difference discipline. N=2 --
// proves each row's gradient is computed independently from its own cached group stats.
TEST_F(GroupNormModuleTest, BackwardInputGradientMatchesFiniteDifferenceAcrossTwoExampleBatch) {
    GroupNormModule norm(2, 4, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 3.0f, 4.0f, 5.0f});
    norm.set_beta({0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<float> x_vals = {1.0f, 1.5f, 3.0f, 2.5f, 0.0f, -1.0f, 4.0f, 2.0f,
                                  2.0f, 1.0f, 5.0f, 3.5f, -1.0f, 0.5f, 2.0f, 3.0f};
    Tensor x(Shape({2, 4, 1, 2}), &backend, x_vals);
    Tensor grad_output(Shape({2, 4, 1, 2}), &backend,
                        {1.0f, 0.5f, -0.25f, 0.75f, 0.2f, -0.4f, 0.1f, 0.3f, -0.5f, 0.6f, 0.2f, -0.1f, 0.4f, 0.3f,
                         -0.2f, 0.1f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(16);
    for (int64_t i = 0; i < 16; ++i) {
        std::vector<float> x_plus = x_vals;
        std::vector<float> x_minus = x_vals;
        x_plus[static_cast<size_t>(i)] += h;
        x_minus[static_cast<size_t>(i)] -= h;
        Tensor xp(Shape({2, 4, 1, 2}), &backend, x_plus);
        Tensor xm(Shape({2, 4, 1, 2}), &backend, x_minus);
        float lp = loss(norm, xp, grad_output);
        float lm = loss(norm, xm, grad_output);
        fd_grad[static_cast<size_t>(i)] = (lp - lm) / (2.0f * h);
    }

    (void)norm.forward(x);  // restore cached state to the unperturbed input before backward()
    Tensor grad_input = norm.backward(grad_output);

    for (int64_t i = 0; i < 16; ++i) {
        EXPECT_NEAR(grad_input.data()[i], fd_grad[static_cast<size_t>(i)], 1.5e-2f);
    }
}

TEST_F(GroupNormModuleTest, BackwardGammaGradientSumsAcrossTwoExampleBatch) {
    GroupNormModule norm(2, 4, &backend, DeviceType::Cpu, 0.0f);
    std::vector<float> gamma_vals = {2.0f, 3.0f, 4.0f, 5.0f};
    norm.set_gamma(gamma_vals);
    norm.set_beta({0.0f, 0.0f, 0.0f, 0.0f});

    Tensor x(Shape({2, 4, 1, 2}), &backend,
             {1.0f, 1.5f, 3.0f, 2.5f, 0.0f, -1.0f, 4.0f, 2.0f, 2.0f, 1.0f, 5.0f, 3.5f, -1.0f, 0.5f, 2.0f, 3.0f});
    Tensor grad_output(Shape({2, 4, 1, 2}), &backend,
                        {1.0f, 0.5f, -0.25f, 0.75f, 0.2f, -0.4f, 0.1f, 0.3f, -0.5f, 0.6f, 0.2f, -0.1f, 0.4f, 0.3f,
                         -0.2f, 0.1f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(4);
    for (int64_t i = 0; i < 4; ++i) {
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

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(norm.gamma_grad().data()[i], fd_grad[static_cast<size_t>(i)], 1.5e-2f);
    }
}

TEST_F(GroupNormModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    GroupNormModule norm(2, 4, &backend);
    Tensor relevance(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(GroupNormModuleTest, PropagateRelevanceThrowsOnWrongSize) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);
    Tensor wrong_size_relevance(Shape({4}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(wrong_size_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

TEST_F(GroupNormModuleTest, PropagateRelevanceIsIdentityPassThrough) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({1, 4, 1, 2}), &backend, {4.0f, -1.0f, 2.5f, 0.0f, 1.0f, 1.0f, -2.0f, 3.0f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

TEST_F(GroupNormModuleTest, PropagateRelevanceConservesTotalRelevance) {
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({1, 4, 1, 2}), &backend, {4.0f, -1.0f, 2.5f, 0.0f, 1.0f, 1.0f, -2.0f, 3.0f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f, sum_out = 0.0f;
    for (int64_t i = 0; i < 8; ++i) {
        sum_in += relevance_in.data()[i];
        sum_out += relevance_out.data()[i];
    }
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

TEST_F(GroupNormModuleTest, ParametersExposesGammaAndBetaByPointer) {
    GroupNormModule norm(2, 4, &backend);
    auto params = norm.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0].value, &norm.gamma());
    EXPECT_EQ(params[0].grad, &norm.gamma_grad());
    EXPECT_EQ(params[1].value, &norm.beta());
    EXPECT_EQ(params[1].grad, &norm.beta_grad());
}

using GroupNormModuleDeathTest = GroupNormModuleTest;

TEST_F(GroupNormModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    GroupNormModule norm(2, 4, &backend);
    Tensor x(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 3.0f, 3.0f, 0.0f, 0.0f, 4.0f, 4.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({1, 4, 1, 2}), &backend, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                        DeviceType::Cuda);
    EXPECT_DEATH({ (void)norm.backward(grad_output); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
