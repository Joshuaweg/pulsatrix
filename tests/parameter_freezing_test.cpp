#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mamba_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

// Deterministic, sign-varying fill so no gradient is trivially zero.
Tensor patterned(const Shape& shape, DeviceBackend* backend, float scale = 0.1f) {
    std::vector<float> v(static_cast<size_t>(shape.numel()));
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = scale * static_cast<float>(static_cast<int>((i * 7) % 11) - 5);
    }
    return Tensor(shape, backend, v);
}

void zero_grads(Module& module) {
    for (ParamRef p : module.parameters()) {
        p.grad->fill(0.0f);
    }
}

bool all_zero(const Tensor& t) {
    for (float x : values_of(t)) {
        if (x != 0.0f) {
            return false;
        }
    }
    return true;
}

// Roadmap FND-2's falsifier, per module: freezing every parameter must leave the input
// gradient bit-identical to the trainable run, and leave every parameter gradient at zero.
void expect_frozen_backward_matches(Module& module, const Tensor& input) {
    Tensor output = module.forward(input);
    Tensor grad_output = patterned(output.shape(), input.backend(), 0.05f);
    zero_grads(module);
    Tensor trainable_grad_input = module.backward(grad_output);
    bool any_param_grad = false;
    for (ParamRef p : module.parameters()) {
        any_param_grad = any_param_grad || !all_zero(*p.grad);
    }
    ASSERT_TRUE(any_param_grad) << "test input too degenerate to show parameter gradients";

    zero_grads(module);
    module.set_requires_grad(false);
    Tensor again = module.forward(input);
    EXPECT_EQ(values_of(again), values_of(output));
    Tensor frozen_grad_input = module.backward(grad_output);

    EXPECT_EQ(values_of(frozen_grad_input), values_of(trainable_grad_input));
    for (const NamedParamRef& p : module.named_parameters()) {
        EXPECT_FALSE(p.ref.value->requires_grad()) << p.name;
        EXPECT_TRUE(all_zero(*p.ref.grad)) << p.name << " accumulated a gradient while frozen";
    }
}

class ParameterFreezingTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// --- Tensor flag semantics -------------------------------------------------------------

TEST_F(ParameterFreezingTest, TensorRequiresGradDefaultsToTrue) {
    Tensor t(Shape({2}), &backend);
    EXPECT_TRUE(t.requires_grad());
    t.set_requires_grad(false);
    EXPECT_FALSE(t.requires_grad());
}

TEST_F(ParameterFreezingTest, ConstructionFromAnotherTensorCopiesTheFlag) {
    Tensor frozen(Shape({2}), &backend, {1.0f, 2.0f});
    frozen.set_requires_grad(false);
    Tensor copy(frozen);
    EXPECT_FALSE(copy.requires_grad());
    Tensor moved(std::move(copy));
    EXPECT_FALSE(moved.requires_grad());
}

// The flag belongs to the slot, not the value: loading new weights into a frozen parameter
// (weight_ = loaded) must not silently unfreeze it.
TEST_F(ParameterFreezingTest, AssignmentKeepsTheDestinationsFlag) {
    Tensor slot(Shape({2}), &backend, {0.0f, 0.0f});
    slot.set_requires_grad(false);
    Tensor trainable(Shape({2}), &backend, {1.0f, 2.0f});
    slot = trainable;
    EXPECT_FALSE(slot.requires_grad());
    EXPECT_EQ(values_of(slot), (std::vector<float>{1.0f, 2.0f}));
    slot = Tensor(Shape({2}), &backend, {3.0f, 4.0f});
    EXPECT_FALSE(slot.requires_grad());
    EXPECT_EQ(values_of(slot), (std::vector<float>{3.0f, 4.0f}));

    Tensor trainable_slot(Shape({2}), &backend);
    Tensor frozen_value(Shape({2}), &backend, {5.0f, 6.0f});
    frozen_value.set_requires_grad(false);
    trainable_slot = frozen_value;
    EXPECT_TRUE(trainable_slot.requires_grad());
}

// --- Every leaf module: skip dW, keep dX -----------------------------------------------

TEST_F(ParameterFreezingTest, Linear) {
    LinearModule m(3, 4, &backend);
    expect_frozen_backward_matches(m, patterned(Shape({2, 3}), &backend));
}

TEST_F(ParameterFreezingTest, Conv2D) {
    Conv2DModule m(2, 3, 2, 2, &backend);
    expect_frozen_backward_matches(m, patterned(Shape({2, 2, 4, 4}), &backend));
}

TEST_F(ParameterFreezingTest, Embedding) {
    EmbeddingModule m(5, 3, &backend);
    expect_frozen_backward_matches(m, Tensor(Shape({2, 3}), &backend, {0.0f, 1.0f, 4.0f, 2.0f, 2.0f, 3.0f}));
}

TEST_F(ParameterFreezingTest, Norms) {
    LayerNormModule layer_norm(4, &backend);
    RMSNormModule rms_norm(4, &backend);
    GroupNormModule group_norm(2, 4, &backend);
    BatchNormModule batch_norm(4, &backend);
    expect_frozen_backward_matches(layer_norm, patterned(Shape({3, 4}), &backend));
    expect_frozen_backward_matches(rms_norm, patterned(Shape({3, 4}), &backend));
    expect_frozen_backward_matches(group_norm, patterned(Shape({2, 4, 2, 2}), &backend));
    expect_frozen_backward_matches(batch_norm, patterned(Shape({2, 4, 2, 2}), &backend));
}

TEST_F(ParameterFreezingTest, Recurrent) {
    RNNModule rnn(3, 4, &backend);
    GRUModule gru(3, 4, &backend);
    LSTMModule lstm(3, 4, &backend);
    expect_frozen_backward_matches(rnn, patterned(Shape({2, 3, 3}), &backend));
    expect_frozen_backward_matches(gru, patterned(Shape({2, 3, 3}), &backend));
    expect_frozen_backward_matches(lstm, patterned(Shape({2, 3, 3}), &backend));
}

TEST_F(ParameterFreezingTest, SequenceMixers) {
    MambaModule mamba(4, 2, &backend);
    RWKVModule rwkv(4, &backend);
    RetNetModule retnet(4, 4, 0.9f, &backend);
    expect_frozen_backward_matches(mamba, patterned(Shape({2, 3, 4}), &backend));
    expect_frozen_backward_matches(rwkv, patterned(Shape({2, 3, 4}), &backend));
    expect_frozen_backward_matches(retnet, patterned(Shape({2, 3, 4}), &backend));
}

TEST_F(ParameterFreezingTest, TransformerBlock) {
    TransformerBlock m(4, 2, 8, &backend);
    expect_frozen_backward_matches(m, patterned(Shape({2, 3, 4}), &backend));
}

TEST_F(ParameterFreezingTest, FreezingOnlyTheWeightStillTrainsTheBias) {
    LinearModule m(3, 2, &backend);
    m.set_requires_grad(false, "weight");
    Tensor out = m.forward(patterned(Shape({2, 3}), &backend));
    (void)m.backward(patterned(out.shape(), &backend));
    std::vector<NamedParamRef> params = m.named_parameters();
    EXPECT_TRUE(all_zero(*params[0].ref.grad));
    EXPECT_FALSE(all_zero(*params[1].ref.grad));
}

// --- Selecting by name -----------------------------------------------------------------

TEST_F(ParameterFreezingTest, PrefixSelectsASubtreeOnDotBoundaries) {
    TransformerBlock block(4, 2, 8, &backend);
    block.set_requires_grad(false);
    block.set_requires_grad(true, "mha.q_proj");
    for (const NamedParamRef& p : block.named_parameters()) {
        const bool expected = p.name == "mha.q_proj.weight" || p.name == "mha.q_proj.bias";
        EXPECT_EQ(p.ref.value->requires_grad(), expected) << p.name;
    }
}

TEST_F(ParameterFreezingTest, PrefixMustMatchAWholeSegment) {
    TransformerBlock block(4, 2, 8, &backend);
    EXPECT_THROW(block.set_requires_grad(false, "mha.q"), std::invalid_argument);
    EXPECT_THROW(block.set_requires_grad(false, "nonexistent"), std::invalid_argument);
    for (const NamedParamRef& p : block.named_parameters()) {
        EXPECT_TRUE(p.ref.value->requires_grad()) << p.name << " changed by a rejected call";
    }
}

TEST_F(ParameterFreezingTest, ExactLeafNameSelectsOneParameter) {
    TransformerBlock block(4, 2, 8, &backend);
    block.set_requires_grad(false, "norm1.weight");
    int frozen = 0;
    for (const NamedParamRef& p : block.named_parameters()) {
        frozen += p.ref.value->requires_grad() ? 0 : 1;
    }
    EXPECT_EQ(frozen, 1);
}

// --- Optimizers never move a frozen parameter -------------------------------------------

void expect_optimizer_skips_frozen(const std::function<void(Module&)>& step, DeviceBackend* backend) {
    LinearModule m(3, 2, backend);
    m.set_requires_grad(false, "weight");
    std::vector<NamedParamRef> params = m.named_parameters();
    const std::vector<float> frozen_before = values_of(*params[0].ref.value);
    const std::vector<float> bias_before = values_of(*params[1].ref.value);
    // A nonzero gradient already sitting in the frozen slot (e.g. accumulated before
    // freezing) must still be ignored.
    params[0].ref.grad->fill(1.0f);
    params[1].ref.grad->fill(1.0f);
    for (int i = 0; i < 5; ++i) {
        step(m);
    }
    EXPECT_EQ(values_of(*params[0].ref.value), frozen_before);
    EXPECT_NE(values_of(*params[1].ref.value), bias_before);
}

TEST_F(ParameterFreezingTest, SGDSkipsFrozenParameters) {
    SGDOptimizer sgd(0.1f);
    expect_optimizer_skips_frozen([&](Module& m) { sgd.step(m); }, &backend);
}

TEST_F(ParameterFreezingTest, AdamSkipsFrozenParameters) {
    AdamOptimizer adam(0.1f, &backend);
    expect_optimizer_skips_frozen([&](Module& m) { adam.step(m); }, &backend);
}

}  // namespace
}  // namespace pulsatrix
