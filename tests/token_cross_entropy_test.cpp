#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

constexpr float kIgnore = static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex);

// -log softmax(row)[target], in double, written independently of the code under test.
double reference_ce(const float* row, int64_t classes, int64_t target) {
    double max = row[0];
    for (int64_t k = 1; k < classes; ++k) max = std::max(max, static_cast<double>(row[k]));
    double sum = 0.0;
    for (int64_t k = 0; k < classes; ++k) sum += std::exp(row[k] - max);
    return -(row[target] - max - std::log(sum));
}

class TokenCrossEntropyTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(TokenCrossEntropyTest, MeanOverTokensSkippingIgnoredOnes) {
    const std::vector<float> logits = random_values(4 * 5, 1);
    Tensor x(Shape({4, 5}), &backend, logits);
    Tensor targets(Shape({4}), &backend, {2, kIgnore, 0, 4});
    TokenCrossEntropyLoss loss(&backend);
    const float value = loss.forward(x, targets);
    const double expected =
        (reference_ce(&logits[0], 5, 2) + reference_ce(&logits[10], 5, 0) + reference_ce(&logits[15], 5, 4)) / 3.0;
    EXPECT_NEAR(value, expected, 1e-5);
    EXPECT_EQ(loss.num_tokens(), 3);
}

TEST_F(TokenCrossEntropyTest, GradientIsSoftmaxMinusOneHotOverTheTokenCount) {
    const std::vector<float> logits = random_values(3 * 4, 2);
    Tensor x(Shape({3, 4}), &backend, logits);
    Tensor targets(Shape({3}), &backend, {1, kIgnore, 3});
    TokenCrossEntropyLoss loss(&backend);
    (void)loss.forward(x, targets);
    const std::vector<float> g = values_of(loss.backward());
    const int64_t target_of[3] = {1, -1, 3};
    for (int64_t r = 0; r < 3; ++r) {
        double max = logits[r * 4], sum = 0.0;
        for (int k = 1; k < 4; ++k) max = std::max(max, static_cast<double>(logits[r * 4 + k]));
        for (int k = 0; k < 4; ++k) sum += std::exp(logits[r * 4 + k] - max);
        for (int64_t k = 0; k < 4; ++k) {
            const double p = std::exp(logits[r * 4 + k] - max) / sum;
            const double expected = target_of[r] < 0 ? 0.0 : (p - (k == target_of[r] ? 1.0 : 0.0)) / 2.0;
            EXPECT_NEAR(g[r * 4 + k], expected, 1e-6) << r << "," << k;
        }
    }
}

TEST_F(TokenCrossEntropyTest, AcceptsSequenceShapedLogits) {
    const std::vector<float> logits = random_values(2 * 3 * 4, 3);
    TokenCrossEntropyLoss loss(&backend);
    const float value =
        loss.forward(Tensor(Shape({2, 3, 4}), &backend, logits), Tensor(Shape({2, 3}), &backend, {0, 1, 2, 3, 0, kIgnore}));
    double sum = 0.0;
    const int64_t targets[5] = {0, 1, 2, 3, 0};
    for (int64_t r = 0; r < 5; ++r) sum += reference_ce(&logits[r * 4], 4, targets[r]);
    EXPECT_NEAR(value, sum / 5.0, 1e-5);
    EXPECT_EQ(loss.backward().shape(), Shape({2, 3, 4}));
}

TEST_F(TokenCrossEntropyTest, AnExplicitNormalizerDividesTheSum) {
    const std::vector<float> logits = random_values(2 * 3, 4);
    Tensor x(Shape({2, 3}), &backend, logits);
    Tensor targets(Shape({2}), &backend, {0, 2});
    TokenCrossEntropyLoss loss(&backend);
    const double sum = reference_ce(&logits[0], 3, 0) + reference_ce(&logits[3], 3, 2);
    EXPECT_NEAR(loss.forward(x, targets, /*normalizer=*/10.0f), sum / 10.0, 1e-6);
}

// Roadmap TRN-5's falsifier: 4 accumulated micro-batches of 8 must equal 1 batch of 32 when the
// batches are ragged (different numbers of real tokens). Dividing each micro-batch's sum by
// the token count of the whole accumulation window gets that right; averaging the
// micro-batches' own means (the bug Hugging Face Trainer fixed in v4.46) doesn't.
TEST_F(TokenCrossEntropyTest, AccumulatingRaggedMicroBatchesMatchesTheFullBatch) {
    constexpr int64_t kIn = 6, kClasses = 5, kRows = 32, kMicro = 8;
    const std::vector<float> inputs = random_values(kRows * kIn, 5);
    std::vector<float> targets(kRows);
    std::mt19937 rng(6);
    // Micro-batch k keeps 8 - 2k real tokens (8, 6, 4, 2): the rest are padding.
    for (int64_t r = 0; r < kRows; ++r) {
        const int64_t k = r / kMicro, pos = r % kMicro;
        targets[r] = pos < kMicro - 2 * k ? static_cast<float>(rng() % kClasses) : kIgnore;
    }
    auto make_model = [&] {
        LinearModule m(kIn, kClasses, &backend);
        m.set_weight(random_values(kIn * kClasses, 7));
        m.set_bias(random_values(kClasses, 8));
        return m;
    };

    // Full batch, mean over all real tokens.
    LinearModule full = make_model();
    TokenCrossEntropyLoss full_loss(&backend);
    (void)full_loss.forward(full.forward(Tensor(Shape({kRows, kIn}), &backend, inputs)),
                            Tensor(Shape({kRows}), &backend, targets));
    (void)full.backward(full_loss.backward());

    // Four micro-batches, each divided by the window's total token count.
    Tensor all_targets(Shape({kRows}), &backend, targets);
    const int64_t total = CountTargetTokens(all_targets);
    EXPECT_EQ(total, 8 + 6 + 4 + 2);
    LinearModule accumulated = make_model(), naive = make_model();
    for (int64_t k = 0; k < kRows / kMicro; ++k) {
        Tensor xk(Shape({kMicro, kIn}), &backend,
                  std::vector<float>(inputs.begin() + k * kMicro * kIn, inputs.begin() + (k + 1) * kMicro * kIn));
        Tensor tk(Shape({kMicro}), &backend,
                  std::vector<float>(targets.begin() + k * kMicro, targets.begin() + (k + 1) * kMicro));
        TokenCrossEntropyLoss lk(&backend);
        (void)lk.forward(accumulated.forward(xk), tk, static_cast<float>(total));
        (void)accumulated.backward(lk.backward());

        // The bug: each micro-batch's own mean, averaged over the 4 micro-batches.
        TokenCrossEntropyLoss nk(&backend);
        (void)nk.forward(naive.forward(xk), tk, static_cast<float>(CountTargetTokens(tk)) * 4.0f);
        (void)naive.backward(nk.backward());
    }

    const std::vector<float> gf = values_of(full.weight_grad()), ga = values_of(accumulated.weight_grad()),
                             gn = values_of(naive.weight_grad());
    double naive_gap = 0.0;
    for (size_t i = 0; i < gf.size(); ++i) {
        EXPECT_NEAR(ga[i], gf[i], 1e-6) << i;
        naive_gap = std::max(naive_gap, std::fabs(static_cast<double>(gn[i]) - gf[i]));
    }
    EXPECT_GT(naive_gap, 1e-3);  // the naive average really is wrong here
}

TEST_F(TokenCrossEntropyTest, AllTokensIgnoredGivesZeroNotNaN) {
    TokenCrossEntropyLoss loss(&backend);
    EXPECT_EQ(loss.forward(Tensor(Shape({2, 3}), &backend, random_values(6, 9)),
                           Tensor(Shape({2}), &backend, {kIgnore, kIgnore})),
              0.0f);
    EXPECT_EQ(loss.num_tokens(), 0);
    EXPECT_EQ(values_of(loss.backward()), std::vector<float>(6, 0.0f));
}

TEST_F(TokenCrossEntropyTest, RejectsBadInput) {
    TokenCrossEntropyLoss loss(&backend);
    Tensor x(Shape({2, 3}), &backend, random_values(6, 10));
    EXPECT_THROW((void)loss.backward(), std::logic_error);
    EXPECT_THROW((void)loss.forward(x, Tensor(Shape({2}), &backend, {0, 3})), std::invalid_argument);    // out of range
    EXPECT_THROW((void)loss.forward(x, Tensor(Shape({2}), &backend, {0, 1.5f})), std::invalid_argument);  // fractional
    EXPECT_THROW((void)loss.forward(x, Tensor(Shape({3}), &backend, {0, 1, 2})), std::invalid_argument);  // shape
    EXPECT_THROW((void)loss.forward(x, Tensor(Shape({2}), &backend, {0, 1}), -1.0f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(Tensor(Shape({6}), &backend, random_values(6, 11)), Tensor(Shape({}), &backend, {0})),
                 std::invalid_argument);  // logits need a class dimension and at least one row
}

}  // namespace
}  // namespace pulsatrix
