#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

class DropoutModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(DropoutModuleTest, ConstructionThrowsOnNegativeP) {
    EXPECT_THROW({ DropoutModule d(-0.1f, &backend); }, std::invalid_argument);
}

TEST_F(DropoutModuleTest, ConstructionThrowsOnPEqualsOne) {
    EXPECT_THROW({ DropoutModule d(1.0f, &backend); }, std::invalid_argument);
}

TEST_F(DropoutModuleTest, ConstructionThrowsOnPGreaterThanOne) {
    EXPECT_THROW({ DropoutModule d(1.5f, &backend); }, std::invalid_argument);
}

TEST_F(DropoutModuleTest, ConstructionAcceptsPZero) {
    EXPECT_NO_THROW({ DropoutModule d(0.0f, &backend); });
}

TEST_F(DropoutModuleTest, ForwardIsIdentityWhenPIsZero) {
    DropoutModule d(0.0f, &backend);
    Tensor input(Shape({5}), &backend, {1.0f, -2.0f, 3.0f, 0.0f, 5.0f});
    Tensor output = d.forward(input);
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], input.data()[i]);
    }
}

TEST_F(DropoutModuleTest, ForwardIsIdentityInEvalModeRegardlessOfP) {
    DropoutModule d(0.9f, &backend);
    d.set_training(false);
    Tensor input(Shape({5}), &backend, {1.0f, -2.0f, 3.0f, 0.0f, 5.0f});
    Tensor output = d.forward(input);
    for (int64_t i = 0; i < input.numel(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], input.data()[i]);
    }
}

TEST_F(DropoutModuleTest, BackwardThrowsIfCalledBeforeForward) {
    DropoutModule d(0.5f, &backend);
    Tensor grad(Shape({3}), &backend);
    EXPECT_THROW({ (void)d.backward(grad); }, std::logic_error);
}

TEST_F(DropoutModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    DropoutModule d(0.5f, &backend);
    Tensor relevance(Shape({3}), &backend);
    EXPECT_THROW({ (void)d.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(DropoutModuleTest, BackwardThrowsOnShapeMismatchedGradOutput) {
    DropoutModule d(0.5f, &backend);
    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)d.forward(input);
    Tensor wrong_shape_grad(Shape({4}), &backend);
    EXPECT_THROW({ (void)d.backward(wrong_shape_grad); }, std::invalid_argument);
}

TEST_F(DropoutModuleTest, PropagateRelevanceThrowsOnShapeMismatch) {
    DropoutModule d(0.5f, &backend);
    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)d.forward(input);
    Tensor wrong_shape_relevance(Shape({4}), &backend);
    EXPECT_THROW({ (void)d.propagate_relevance(wrong_shape_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

// Property-based, not exact-value: every output element is either exactly 0 or exactly
// input*scale (scale = 1/(1-p) = 2.0 for p=0.5), and the observed drop fraction falls in a
// wide statistical tolerance band. Fixed seed for reproducibility across runs.
TEST_F(DropoutModuleTest, ForwardTrainingModeProducesOnlyZeroOrScaledValues) {
    DropoutModule d(0.5f, &backend, /*seed=*/7);
    const int64_t n = 2000;
    std::vector<float> values(static_cast<size_t>(n), 3.0f);
    Tensor input(Shape({n}), &backend, values);
    Tensor output = d.forward(input);

    int64_t dropped = 0;
    for (int64_t i = 0; i < n; ++i) {
        float v = output.data()[i];
        if (v == 0.0f) {
            ++dropped;
        } else {
            EXPECT_FLOAT_EQ(v, 3.0f * 2.0f);
        }
    }
    float drop_fraction = static_cast<float>(dropped) / static_cast<float>(n);
    EXPECT_GT(drop_fraction, 0.35f);
    EXPECT_LT(drop_fraction, 0.65f);
}

// backward(): grad_input_i / grad_output_i matches 0 or scale consistent with the same
// seeded forward pass's own outcome for that element.
TEST_F(DropoutModuleTest, BackwardScalesGradientConsistentlyWithForwardMask) {
    DropoutModule d(0.5f, &backend, /*seed=*/7);
    const int64_t n = 100;
    std::vector<float> values(static_cast<size_t>(n), 1.0f);
    Tensor input(Shape({n}), &backend, values);
    Tensor output = d.forward(input);

    std::vector<float> grad_values(static_cast<size_t>(n), 1.0f);
    Tensor grad_output(Shape({n}), &backend, grad_values);
    Tensor grad_input = d.backward(grad_output);

    for (int64_t i = 0; i < n; ++i) {
        if (output.data()[i] == 0.0f) {
            EXPECT_FLOAT_EQ(grad_input.data()[i], 0.0f);
        } else {
            EXPECT_FLOAT_EQ(grad_input.data()[i], 2.0f);
        }
    }
}

// propagate_relevance: exact unconditional identity, regardless of mask/training mode --
// explicitly NOT following backward()'s masked rule.
TEST_F(DropoutModuleTest, PropagateRelevanceIsUnconditionalIdentity) {
    DropoutModule d(0.5f, &backend, /*seed=*/7);
    const int64_t n = 50;
    std::vector<float> values(static_cast<size_t>(n), 1.0f);
    Tensor input(Shape({n}), &backend, values);
    (void)d.forward(input);

    std::vector<float> relevance_values(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) relevance_values[static_cast<size_t>(i)] = static_cast<float>(i) + 1.0f;
    Tensor relevance_out(Shape({n}), &backend, relevance_values);
    Tensor relevance_in = d.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

using DropoutModuleDeathTest = DropoutModuleTest;

// GPU-native-kernels Mission 1b fixed: backward after an eval-mode forward used to multiply
// by 1/(1-p) although the forward was the identity.
TEST_F(DropoutModuleTest, BackwardAfterEvalForwardIsIdentity) {
    DropoutModule d(0.5f, &backend, /*seed=*/7);
    d.set_training(false);
    Tensor input(Shape({4}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    (void)d.forward(input);
    Tensor grad(Shape({4}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});
    Tensor grad_input = d.backward(grad);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(grad_input.data()[i], grad.data()[i]);
    }
}

}  // namespace
}  // namespace pulsatrix
