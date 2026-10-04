#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pulsatrix/batch_norm_fold.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

std::vector<float> patterned(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(static_cast<int>((i * 7) % 13) - 6);
    return v;
}

class BatchNormEvalTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// --- Running statistics ----------------------------------------------------------------

TEST_F(BatchNormEvalTest, RunningStatisticsStartAtZeroMeanUnitVariance) {
    BatchNormModule bn(2, &backend);
    EXPECT_EQ(values_of(bn.running_mean()), (std::vector<float>{0.0f, 0.0f}));
    EXPECT_EQ(values_of(bn.running_var()), (std::vector<float>{1.0f, 1.0f}));
}

TEST_F(BatchNormEvalTest, TrainingForwardUpdatesRunningStatisticsLikePyTorch) {
    BatchNormModule bn(1, &backend);
    // One channel over (N=2, H=1, W=2): mean 4, biased variance 5, unbiased 20/3.
    (void)bn.forward(Tensor(Shape({2, 1, 1, 2}), &backend, {1, 3, 5, 7}));
    EXPECT_NEAR(bn.running_mean().data()[0], 0.1f * 4.0f, 1e-6f);
    EXPECT_NEAR(bn.running_var().data()[0], 0.9f + 0.1f * 20.0f / 3.0f, 1e-6f);
}

TEST_F(BatchNormEvalTest, MomentumIsConfigurable) {
    BatchNormModule bn(1, &backend, backend.device(), 1e-6f, /*momentum=*/0.5f);
    (void)bn.forward(Tensor(Shape({2, 1, 1, 2}), &backend, {1, 3, 5, 7}));
    EXPECT_NEAR(bn.running_mean().data()[0], 2.0f, 1e-6f);
    EXPECT_THROW(BatchNormModule(1, &backend, backend.device(), 1e-6f, 0.0f), std::invalid_argument);
    EXPECT_THROW(BatchNormModule(1, &backend, backend.device(), 1e-6f, 1.5f), std::invalid_argument);
}

TEST_F(BatchNormEvalTest, SingleValuePerChannelLeavesRunningVarianceUnchanged) {
    BatchNormModule bn(1, &backend);
    (void)bn.forward(Tensor(Shape({1, 1, 1, 1}), &backend, {3}));
    EXPECT_NEAR(bn.running_mean().data()[0], 0.3f, 1e-6f);
    EXPECT_EQ(bn.running_var().data()[0], 1.0f);  // one value says nothing about variance
}

TEST_F(BatchNormEvalTest, EvalForwardDoesNotUpdateRunningStatistics) {
    BatchNormModule bn(1, &backend);
    bn.set_training(false);
    (void)bn.forward(Tensor(Shape({2, 1, 1, 2}), &backend, {1, 3, 5, 7}));
    EXPECT_EQ(bn.running_mean().data()[0], 0.0f);
    EXPECT_EQ(bn.running_var().data()[0], 1.0f);
}

TEST_F(BatchNormEvalTest, RunningStatisticSettersValidate) {
    BatchNormModule bn(2, &backend);
    bn.set_running_mean({0.5f, -1.0f});
    bn.set_running_var({2.0f, 0.0f});
    EXPECT_EQ(values_of(bn.running_mean()), (std::vector<float>{0.5f, -1.0f}));
    EXPECT_EQ(values_of(bn.running_var()), (std::vector<float>{2.0f, 0.0f}));
    EXPECT_THROW(bn.set_running_mean({1.0f}), std::invalid_argument);
    EXPECT_THROW(bn.set_running_var({1.0f, -0.5f}), std::invalid_argument);
    EXPECT_THROW(bn.set_running_var({1.0f, std::numeric_limits<float>::quiet_NaN()}), std::invalid_argument);
}

// --- Eval mode -------------------------------------------------------------------------

TEST_F(BatchNormEvalTest, EvalForwardNormalizesWithRunningStatistics) {
    BatchNormModule bn(1, &backend);
    bn.set_gamma({2.0f});
    bn.set_beta({1.0f});
    bn.set_running_mean({4.0f});
    bn.set_running_var({3.0f});
    bn.set_training(false);
    Tensor y = bn.forward(Tensor(Shape({1, 1, 1, 3}), &backend, {1, 4, 7}));
    const float std_dev = std::sqrt(3.0f + 1e-6f);
    EXPECT_NEAR(y.data()[0], 2.0f * (1.0f - 4.0f) / std_dev + 1.0f, 1e-5f);
    EXPECT_NEAR(y.data()[1], 1.0f, 1e-6f);
    EXPECT_NEAR(y.data()[2], 2.0f * (7.0f - 4.0f) / std_dev + 1.0f, 1e-5f);
}

// The bug this fixes (lrp_issues #8): in eval mode a sample's output must not depend on
// what else is in its batch.
TEST_F(BatchNormEvalTest, EvalOutputOfASampleIsIndependentOfItsBatch) {
    BatchNormModule bn(3, &backend);
    bn.set_gamma({1.0f, 0.5f, -2.0f});
    bn.set_beta({0.1f, 0.2f, 0.3f});
    bn.set_running_mean({0.2f, -0.1f, 0.4f});
    bn.set_running_var({1.5f, 0.7f, 2.0f});
    bn.set_training(false);
    std::vector<float> batch = patterned(4 * 3 * 2 * 2, 0.3f);
    Tensor batched_out = bn.forward(Tensor(Shape({4, 3, 2, 2}), &backend, batch));
    std::vector<float> first(batch.begin(), batch.begin() + 12);
    Tensor alone_out = bn.forward(Tensor(Shape({1, 3, 2, 2}), &backend, first));
    std::vector<float> batched_first(batched_out.data(), batched_out.data() + 12);
    EXPECT_EQ(values_of(alone_out), batched_first);
}

TEST_F(BatchNormEvalTest, EvalBackwardIsTheAffineGradient) {
    BatchNormModule bn(2, &backend);
    bn.set_gamma({1.5f, -0.5f});
    bn.set_beta({0.0f, 0.0f});
    bn.set_running_mean({0.3f, -0.2f});
    bn.set_running_var({0.8f, 2.5f});
    bn.set_training(false);
    std::vector<float> x = patterned(2 * 2 * 1 * 3, 0.4f);
    std::vector<float> g = patterned(2 * 2 * 1 * 3, 0.1f);
    (void)bn.forward(Tensor(Shape({2, 2, 1, 3}), &backend, x));
    Tensor grad_in = bn.backward(Tensor(Shape({2, 2, 1, 3}), &backend, g));
    const float mean[2] = {0.3f, -0.2f}, var[2] = {0.8f, 2.5f}, gamma[2] = {1.5f, -0.5f};
    float gamma_grad[2] = {0, 0}, beta_grad[2] = {0, 0};
    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 2; ++c) {
            const float std_dev = std::sqrt(var[c] + 1e-6f);
            for (int s = 0; s < 3; ++s) {
                const int idx = n * 6 + c * 3 + s;
                EXPECT_NEAR(grad_in.data()[idx], g[idx] * gamma[c] / std_dev, 1e-6f) << idx;
                gamma_grad[c] += g[idx] * (x[idx] - mean[c]) / std_dev;
                beta_grad[c] += g[idx];
            }
        }
    }
    for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(bn.gamma_grad().data()[c], gamma_grad[c], 1e-5f);
        EXPECT_NEAR(bn.beta_grad().data()[c], beta_grad[c], 1e-5f);
    }
}

TEST_F(BatchNormEvalTest, TrainingModeStillUsesBatchStatistics) {
    BatchNormModule bn(1, &backend);
    bn.set_gamma({1.0f});
    bn.set_running_mean({100.0f});  // would dominate if (wrongly) used in training mode
    Tensor y = bn.forward(Tensor(Shape({2, 1, 1, 2}), &backend, {1, 3, 5, 7}));
    float sum = 0.0f;
    for (float v : values_of(y)) sum += v;
    EXPECT_NEAR(sum, 0.0f, 1e-5f);
}

// --- Folding BatchNorm into the preceding Conv2D (the LRP canonizer) ---------------------

class BatchNormFoldTest : public BatchNormEvalTest {
protected:
    Conv2DModule conv{2, 3, 2, 2, &backend};
    BatchNormModule bn{3, &backend};
    Tensor x{Shape({2, 2, 4, 4}), &backend, patterned(2 * 2 * 4 * 4, 0.25f)};

    void SetUp() override {
        conv.set_kernel(patterned(3 * 2 * 2 * 2, 0.2f));
        conv.set_bias({0.1f, -0.3f, 0.2f});
        bn.set_gamma({1.2f, -0.7f, 0.5f});
        bn.set_beta({0.05f, 0.4f, -0.2f});
        bn.set_running_mean({0.3f, -0.6f, 0.1f});
        bn.set_running_var({0.9f, 2.0f, 0.4f});
        bn.set_training(false);
    }

    Tensor pair_forward() { return bn.forward(conv.forward(x)); }
};

TEST_F(BatchNormFoldTest, FoldedPairComputesTheSameFunction) {
    Tensor reference = pair_forward();
    BatchNormFold fold(conv, bn);
    Tensor folded = pair_forward();
    std::vector<float> a = values_of(reference), b = values_of(folded);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-5f) << i;
}

TEST_F(BatchNormFoldTest, FoldedBatchNormIsAnExactIdentity) {
    BatchNormFold fold(conv, bn);
    Tensor z(Shape({1, 3, 2, 2}), &backend, patterned(12, 0.7f));
    Tensor out = bn.forward(z);
    EXPECT_EQ(values_of(out), values_of(z));
    Tensor r = bn.propagate_relevance(z, LRPRuleConfig{});
    EXPECT_EQ(values_of(r), values_of(z));
}

// LRP through the folded pair equals LRP through one Conv2D carrying the folded weights,
// computed here independently: w' = w * s, b' = (b - mean) * s + beta, s = gamma / std.
TEST_F(BatchNormFoldTest, LRPThroughTheFoldedPairEqualsLRPThroughTheMergedConv) {
    std::vector<float> kernel = values_of(conv.kernel()), bias = values_of(conv.bias());
    const float gamma[3] = {1.2f, -0.7f, 0.5f}, beta[3] = {0.05f, 0.4f, -0.2f};
    const float mean[3] = {0.3f, -0.6f, 0.1f}, var[3] = {0.9f, 2.0f, 0.4f};
    for (int o = 0; o < 3; ++o) {
        const float s = gamma[o] / std::sqrt(var[o] + 1e-6f);
        for (int i = 0; i < 8; ++i) kernel[o * 8 + i] *= s;
        bias[o] = (bias[o] - mean[o]) * s + beta[o];
    }
    Conv2DModule merged(2, 3, 2, 2, &backend);
    merged.set_kernel(kernel);
    merged.set_bias(bias);
    Tensor merged_out = merged.forward(x);
    Tensor expected = merged.propagate_relevance(merged_out, LRPRuleConfig{});

    BatchNormFold fold(conv, bn);
    SequentialModule pair({&conv, &bn});
    Tensor out = pair.forward(x);
    Tensor actual = pair.propagate_relevance(out, LRPRuleConfig{});
    std::vector<float> a = values_of(expected), b = values_of(actual);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-5f) << i;
}

TEST_F(BatchNormFoldTest, LeavingTheScopeRestoresBothModulesExactly) {
    const std::vector<float> kernel = values_of(conv.kernel()), bias = values_of(conv.bias());
    Tensor reference = pair_forward();
    {
        BatchNormFold fold(conv, bn);
        (void)pair_forward();
    }
    EXPECT_EQ(values_of(conv.kernel()), kernel);
    EXPECT_EQ(values_of(conv.bias()), bias);
    EXPECT_EQ(values_of(pair_forward()), values_of(reference));
}

TEST_F(BatchNormFoldTest, RefusesTrainingModeMismatchedChannelsAndDoubleFolds) {
    bn.set_training(true);
    EXPECT_THROW(BatchNormFold(conv, bn), std::invalid_argument);
    bn.set_training(false);
    BatchNormModule wrong(4, &backend);
    wrong.set_training(false);
    EXPECT_THROW(BatchNormFold(conv, wrong), std::invalid_argument);
    BatchNormFold fold(conv, bn);
    EXPECT_THROW(BatchNormFold(conv, bn), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
