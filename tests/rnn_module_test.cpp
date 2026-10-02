#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/rnn_module.hpp"

namespace pulsatrix {
namespace {

class RNNModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(RNNModuleTest, ConstructionThrowsOnZeroInputSize) {
    EXPECT_THROW({ RNNModule rnn(0, 2, &backend); }, std::invalid_argument);
}

TEST_F(RNNModuleTest, ConstructionThrowsOnNegativeHiddenSize) {
    EXPECT_THROW({ RNNModule rnn(2, -1, &backend); }, std::invalid_argument);
}

TEST_F(RNNModuleTest, ForwardThrowsOnWrongRank) {
    RNNModule rnn(2, 2, &backend);
    Tensor wrong_rank(Shape({1, 2}), &backend, {0.1f, 0.2f});
    EXPECT_THROW({ (void)rnn.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(RNNModuleTest, ForwardThrowsOnMismatchedInputSize) {
    RNNModule rnn(2, 2, &backend);
    Tensor wrong_input(Shape({1, 2, 3}), &backend);
    EXPECT_THROW({ (void)rnn.forward(wrong_input); }, std::invalid_argument);
}

TEST_F(RNNModuleTest, BackwardThrowsIfCalledBeforeForward) {
    RNNModule rnn(1, 1, &backend);
    Tensor grad(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)rnn.backward(grad); }, std::logic_error);
}

TEST_F(RNNModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    RNNModule rnn(1, 1, &backend);
    Tensor relevance(Shape({1, 2, 1}), &backend);
    EXPECT_THROW({ (void)rnn.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(RNNModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    RNNModule rnn(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)rnn.forward(input);
    Tensor wrong_shape_grad(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)rnn.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(RNNModuleTest, PropagateRelevanceThrowsOnShapeMismatch) {
    RNNModule rnn(1, 1, &backend);
    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    (void)rnn.forward(input);
    Tensor wrong_shape_relevance(Shape({1, 3, 1}), &backend);
    EXPECT_THROW({ (void)rnn.propagate_relevance(wrong_shape_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

// Independent reference recomputation of the recurrence (h_0 = 0, tied weights across
// timesteps) -- catches shape/indexing bugs, not just formula bugs baked into both.
TEST_F(RNNModuleTest, ForwardMatchesReferenceRecurrence) {
    RNNModule rnn(1, 1, &backend);
    rnn.set_weight_xh({2.0f});
    rnn.set_weight_hh({0.5f});
    rnn.set_bias({0.1f});

    Tensor input(Shape({1, 2, 1}), &backend, {1.0f, 0.5f});
    Tensor output = rnn.forward(input);

    float h0_expected = std::tanh(1.0f * 2.0f + 0.0f * 0.5f + 0.1f);
    float h1_expected = std::tanh(0.5f * 2.0f + h0_expected * 0.5f + 0.1f);

    EXPECT_EQ(output.shape(), Shape({1, 2, 1}));
    EXPECT_NEAR(output.at({0, 0, 0}), h0_expected, 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), h1_expected, 1e-5f);
}

// N=2 -- proves each batch row is processed independently through the recurrence.
TEST_F(RNNModuleTest, ForwardHandlesMultiBatch) {
    RNNModule rnn(1, 1, &backend);
    rnn.set_weight_xh({2.0f});
    rnn.set_weight_hh({0.5f});
    rnn.set_bias({0.1f});

    Tensor input(Shape({2, 2, 1}), &backend, {1.0f, 0.5f, -0.3f, 0.8f});
    Tensor output = rnn.forward(input);

    float row0_h0 = std::tanh(1.0f * 2.0f + 0.1f);
    float row0_h1 = std::tanh(0.5f * 2.0f + row0_h0 * 0.5f + 0.1f);
    float row1_h0 = std::tanh(-0.3f * 2.0f + 0.1f);
    float row1_h1 = std::tanh(0.8f * 2.0f + row1_h0 * 0.5f + 0.1f);

    EXPECT_NEAR(output.at({0, 0, 0}), row0_h0, 1e-5f);
    EXPECT_NEAR(output.at({0, 1, 0}), row0_h1, 1e-5f);
    EXPECT_NEAR(output.at({1, 0, 0}), row1_h0, 1e-5f);
    EXPECT_NEAR(output.at({1, 1, 0}), row1_h1, 1e-5f);
}

namespace {
float compute_loss(RNNModule& rnn, const Tensor& input, const Tensor& grad_seed) {
    Tensor out = rnn.forward(input);
    float total = 0.0f;
    for (int64_t i = 0; i < out.numel(); ++i) total += out.data()[i] * grad_seed.data()[i];
    return total;
}
}  // namespace

// BPTT correctness can't be reliably hand-derived past L=1 -- central finite differences
// is this mission's actual proof, same discipline RMSNormModule's Mission 0 used.
TEST_F(RNNModuleTest, BackwardGradientMatchesFiniteDifferenceForWxh) {
    const float h = 1e-3f;
    Tensor input(Shape({1, 2, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f});
    Tensor grad_seed(Shape({1, 2, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f});
    std::vector<float> wxh = {0.1f, -0.2f, 0.3f, 0.15f};
    std::vector<float> whh = {0.05f, -0.1f, 0.2f, 0.05f};
    std::vector<float> b = {0.01f, -0.02f};

    RNNModule rnn(2, 2, &backend);
    rnn.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
    rnn.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
    rnn.set_bias({b[0], b[1]});
    (void)rnn.forward(input);
    (void)rnn.backward(grad_seed);

    for (int64_t idx = 0; idx < 4; ++idx) {
        std::vector<float> w_plus = wxh, w_minus = wxh;
        w_plus[static_cast<size_t>(idx)] += h;
        w_minus[static_cast<size_t>(idx)] -= h;

        RNNModule plus(2, 2, &backend);
        plus.set_weight_xh({w_plus[0], w_plus[1], w_plus[2], w_plus[3]});
        plus.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
        plus.set_bias({b[0], b[1]});

        RNNModule minus(2, 2, &backend);
        minus.set_weight_xh({w_minus[0], w_minus[1], w_minus[2], w_minus[3]});
        minus.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
        minus.set_bias({b[0], b[1]});

        float numeric_grad = (compute_loss(plus, input, grad_seed) - compute_loss(minus, input, grad_seed)) / (2 * h);
        EXPECT_NEAR(rnn.weight_xh_grad().data()[idx], numeric_grad, 1e-2f) << "Wxh index " << idx;
    }
}

TEST_F(RNNModuleTest, BackwardGradientMatchesFiniteDifferenceForWhh) {
    const float h = 1e-3f;
    Tensor input(Shape({1, 2, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f});
    Tensor grad_seed(Shape({1, 2, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f});
    std::vector<float> wxh = {0.1f, -0.2f, 0.3f, 0.15f};
    std::vector<float> whh = {0.05f, -0.1f, 0.2f, 0.05f};
    std::vector<float> b = {0.01f, -0.02f};

    RNNModule rnn(2, 2, &backend);
    rnn.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
    rnn.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
    rnn.set_bias({b[0], b[1]});
    (void)rnn.forward(input);
    (void)rnn.backward(grad_seed);

    for (int64_t idx = 0; idx < 4; ++idx) {
        std::vector<float> w_plus = whh, w_minus = whh;
        w_plus[static_cast<size_t>(idx)] += h;
        w_minus[static_cast<size_t>(idx)] -= h;

        RNNModule plus(2, 2, &backend);
        plus.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
        plus.set_weight_hh({w_plus[0], w_plus[1], w_plus[2], w_plus[3]});
        plus.set_bias({b[0], b[1]});

        RNNModule minus(2, 2, &backend);
        minus.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
        minus.set_weight_hh({w_minus[0], w_minus[1], w_minus[2], w_minus[3]});
        minus.set_bias({b[0], b[1]});

        float numeric_grad = (compute_loss(plus, input, grad_seed) - compute_loss(minus, input, grad_seed)) / (2 * h);
        EXPECT_NEAR(rnn.weight_hh_grad().data()[idx], numeric_grad, 1e-2f) << "Whh index " << idx;
    }
}

TEST_F(RNNModuleTest, BackwardGradientMatchesFiniteDifferenceForBias) {
    const float h = 1e-3f;
    Tensor input(Shape({1, 2, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f});
    Tensor grad_seed(Shape({1, 2, 2}), &backend, {1.0f, -0.5f, 0.3f, 0.8f});
    std::vector<float> wxh = {0.1f, -0.2f, 0.3f, 0.15f};
    std::vector<float> whh = {0.05f, -0.1f, 0.2f, 0.05f};
    std::vector<float> b = {0.01f, -0.02f};

    RNNModule rnn(2, 2, &backend);
    rnn.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
    rnn.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
    rnn.set_bias({b[0], b[1]});
    (void)rnn.forward(input);
    (void)rnn.backward(grad_seed);

    for (int64_t idx = 0; idx < 2; ++idx) {
        std::vector<float> b_plus = b, b_minus = b;
        b_plus[static_cast<size_t>(idx)] += h;
        b_minus[static_cast<size_t>(idx)] -= h;

        RNNModule plus(2, 2, &backend);
        plus.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
        plus.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
        plus.set_bias({b_plus[0], b_plus[1]});

        RNNModule minus(2, 2, &backend);
        minus.set_weight_xh({wxh[0], wxh[1], wxh[2], wxh[3]});
        minus.set_weight_hh({whh[0], whh[1], whh[2], whh[3]});
        minus.set_bias({b_minus[0], b_minus[1]});

        float numeric_grad = (compute_loss(plus, input, grad_seed) - compute_loss(minus, input, grad_seed)) / (2 * h);
        EXPECT_NEAR(rnn.bias_grad().data()[idx], numeric_grad, 1e-2f) << "bias index " << idx;
    }
}

// propagate_relevance: end-to-end conservation, exact given h_0 == 0 (see class-level
// note) -- not just a tolerance-bounded approximation.
TEST_F(RNNModuleTest, PropagateRelevanceConservesExactly) {
    RNNModule rnn(2, 2, &backend);
    rnn.set_weight_xh({0.1f, -0.2f, 0.3f, 0.15f});
    rnn.set_weight_hh({0.05f, -0.1f, 0.2f, 0.05f});
    rnn.set_bias({0.01f, -0.02f});

    Tensor input(Shape({1, 3, 2}), &backend, {0.3f, -0.2f, 0.6f, 0.1f, -0.4f, 0.5f});
    (void)rnn.forward(input);

    Tensor relevance_out(Shape({1, 3, 2}), &backend, {1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 0.8f});
    Tensor relevance_in = rnn.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
    float sum_out = 0.0f;
    for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

}  // namespace
}  // namespace pulsatrix
