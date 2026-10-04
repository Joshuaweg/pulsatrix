#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/param_groups.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class ParamGroupsTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // A Linear(2 -> 2) with known weights and gradients: weight = 1, 2, 3, 4; bias = 5, 6;
    // grad = 0.5 everywhere.
    LinearModule make_linear() {
        LinearModule m(2, 2, &backend);
        m.set_weight({1.0f, 2.0f, 3.0f, 4.0f});
        m.set_bias({5.0f, 6.0f});
        for (ParamRef p : m.parameters()) p.grad->fill(0.5f);
        return m;
    }
    static const Tensor& weight(LinearModule& m) { return *m.named_parameters()[0].ref.value; }
    static const Tensor& bias(LinearModule& m) { return *m.named_parameters()[1].ref.value; }
};

// --- Selectors ---------------------------------------------------------------------------

TEST_F(ParamGroupsTest, NamePrefixMatchesWholeSegments) {
    Tensor t(Shape({1}), &backend);
    ParamSelector q = param_select::name_prefix("mha.q_proj");
    EXPECT_TRUE(q("mha.q_proj.weight", t));
    EXPECT_TRUE(q("mha.q_proj", t));
    EXPECT_FALSE(q("mha.q_projection.weight", t));
    EXPECT_FALSE(q("xmha.q_proj.weight", t));
}

TEST_F(ParamGroupsTest, OneDimensionalSelectsBiasesAndNormScales) {
    TransformerBlock block(4, 2, 8, &backend);
    ParamSelector one_d = param_select::one_dimensional();
    for (const NamedParamRef& p : block.named_parameters()) {
        const bool expected = p.name.find("bias") != std::string::npos || p.name.find("norm") != std::string::npos;
        EXPECT_EQ(one_d(p.name, *p.ref.value), expected) << p.name;
    }
}

// --- SGD ---------------------------------------------------------------------------------

TEST_F(ParamGroupsTest, SGDWithoutGroupsIsUnchanged) {
    LinearModule m = make_linear();
    SGDOptimizer sgd(0.1f);
    sgd.step(m);
    EXPECT_EQ(values_of(weight(m)), (std::vector<float>{1.0f - 0.05f, 2.0f - 0.05f, 3.0f - 0.05f, 4.0f - 0.05f}));
    EXPECT_EQ(values_of(bias(m)), (std::vector<float>{5.0f - 0.05f, 6.0f - 0.05f}));
}

TEST_F(ParamGroupsTest, SGDGroupsSetLearningRateAndWeightDecayPerParameter) {
    LinearModule m = make_linear();
    SGDOptimizer sgd(0.1f);
    sgd.set_param_groups({{"no-decay biases", param_select::one_dimensional(), 0.2f, 0.0f}});
    sgd.set_weight_decay(0.5f);  // the default group: everything no group selects
    sgd.step(m);
    // Weight (default group): w <- (1 - lr*wd) w - lr g = 0.95 w - 0.05.
    const std::vector<float> w = values_of(weight(m));
    const float w0[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(w[static_cast<size_t>(i)], 0.95f * w0[i] - 0.05f) << i;
    // Bias (its own group): lr 0.2, no decay.
    EXPECT_EQ(values_of(bias(m)), (std::vector<float>{5.0f - 0.1f, 6.0f - 0.1f}));
}

TEST_F(ParamGroupsTest, FirstMatchingGroupWins) {
    LinearModule m = make_linear();
    SGDOptimizer sgd(0.0f);
    sgd.set_param_groups({{"bias", param_select::name_prefix("bias"), 1.0f, 0.0f},
                          {"everything", [](const std::string&, const Tensor&) { return true; }, 2.0f, 0.0f}});
    sgd.step(m);
    EXPECT_EQ(values_of(bias(m)), (std::vector<float>{5.0f - 0.5f, 6.0f - 0.5f}));
    EXPECT_EQ(values_of(weight(m))[0], 1.0f - 1.0f);
}

TEST_F(ParamGroupsTest, GroupSettingsCanChangeBetweenSteps) {
    LinearModule m = make_linear();
    SGDOptimizer sgd(0.0f);
    sgd.set_param_groups({{"bias", param_select::name_prefix("bias"), 0.0f, 0.0f}});
    sgd.step(m);
    EXPECT_EQ(values_of(bias(m))[0], 5.0f);
    sgd.param_groups()[0].learning_rate = 1.0f;  // what a scheduler will do (TRN-4)
    sgd.step(m);
    EXPECT_EQ(values_of(bias(m))[0], 4.5f);
}

// --- Adam --------------------------------------------------------------------------------

TEST_F(ParamGroupsTest, AdamWeightDecayIsL2AddedToTheGradient) {
    // PyTorch's Adam(weight_decay=wd): the step uses g + wd * w. So Adam with decay must match
    // plain Adam fed that gradient directly.
    LinearModule decayed = make_linear(), manual = make_linear();
    AdamOptimizer with_decay(0.01f, &backend);
    with_decay.set_weight_decay(0.1f);
    AdamOptimizer without(0.01f, &backend);
    for (int step = 0; step < 3; ++step) {
        for (ParamRef p : decayed.parameters()) p.grad->fill(0.5f);
        std::vector<ParamRef> mp = manual.parameters();
        for (ParamRef p : mp) {
            std::vector<float> g(static_cast<size_t>(p.value->numel()));
            for (size_t i = 0; i < g.size(); ++i) g[i] = 0.5f + 0.1f * p.value->data()[i];
            *p.grad = Tensor(p.grad->shape(), &backend, g);
        }
        with_decay.step(decayed);
        without.step(manual);
    }
    const std::vector<float> a = values_of(weight(decayed)), b = values_of(weight(manual));
    for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], b[i]) << i;
}

TEST_F(ParamGroupsTest, WeightDecayLeavesTheStoredGradientAlone) {
    LinearModule m = make_linear();
    AdamOptimizer adam(0.01f, &backend);
    adam.set_weight_decay(0.1f);
    adam.step(m);
    EXPECT_EQ(values_of(*m.named_parameters()[0].ref.grad), std::vector<float>(4, 0.5f));
    SGDOptimizer sgd(0.01f);
    sgd.set_weight_decay(0.1f);
    sgd.step(m);
    EXPECT_EQ(values_of(*m.named_parameters()[0].ref.grad), std::vector<float>(4, 0.5f));
}

TEST_F(ParamGroupsTest, AdamGroupsUseTheirOwnLearningRate) {
    LinearModule fast = make_linear(), slow = make_linear();
    AdamOptimizer grouped(0.001f, &backend);
    grouped.set_param_groups({{"bias", param_select::name_prefix("bias"), 0.1f, 0.0f}});
    AdamOptimizer reference(0.1f, &backend);
    grouped.step(fast);
    reference.step(slow);
    EXPECT_EQ(values_of(bias(fast)), values_of(bias(slow)));      // bias: lr 0.1 in both
    EXPECT_NE(values_of(weight(fast)), values_of(weight(slow)));  // weight: 0.001 vs 0.1
}

// The standard transformer recipe: decay matrices, not biases or norm scales.
TEST_F(ParamGroupsTest, NoDecayOnBiasesAndNormsRecipe) {
    TransformerBlock block(4, 2, 8, &backend);
    for (ParamRef p : block.parameters()) {
        p.value->fill(1.0f);
        p.grad->fill(0.0f);  // isolate weight decay
    }
    SGDOptimizer sgd(0.1f);
    sgd.set_weight_decay(0.5f);
    sgd.set_param_groups({{"no decay", param_select::one_dimensional(), 0.1f, 0.0f}});
    sgd.step(block);
    for (const NamedParamRef& p : block.named_parameters()) {
        const float expected = p.ref.value->rank() <= 1 ? 1.0f : 1.0f - 0.1f * 0.5f;
        EXPECT_FLOAT_EQ(p.ref.value->data()[0], expected) << p.name;
    }
}

// --- Validation --------------------------------------------------------------------------

TEST_F(ParamGroupsTest, RejectsInvalidSettings) {
    SGDOptimizer sgd(0.1f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(sgd.set_param_groups({{"g", nullptr, 0.1f, 0.0f}}), std::invalid_argument);
    EXPECT_THROW(sgd.set_param_groups({{"g", param_select::one_dimensional(), -0.1f, 0.0f}}), std::invalid_argument);
    EXPECT_THROW(sgd.set_param_groups({{"g", param_select::one_dimensional(), 0.1f, -1.0f}}), std::invalid_argument);
    EXPECT_THROW(sgd.set_param_groups({{"g", param_select::one_dimensional(), nan, 0.0f}}), std::invalid_argument);
    EXPECT_THROW(sgd.set_weight_decay(-0.5f), std::invalid_argument);
    AdamOptimizer adam(0.1f, &backend);
    EXPECT_THROW(adam.set_weight_decay(nan), std::invalid_argument);
    // Values set directly on a group are checked when step() uses them.
    LinearModule m = make_linear();
    sgd.set_param_groups({{"g", param_select::one_dimensional(), 0.1f, 0.0f}});
    sgd.param_groups()[0].weight_decay = -1.0f;
    EXPECT_THROW(sgd.step(m), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
