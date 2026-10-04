#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

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
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/swiglu_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

using Names = std::vector<std::string>;

Names names_of(Module& module) {
    Names names;
    for (const NamedParamRef& p : module.named_parameters()) {
        names.push_back(p.name);
    }
    return names;
}

// The invariant everything downstream (optimizers, checkpoints, freezing by name) relies on:
// parameters() is named_parameters() with the names dropped -- same tensors, same order -- and
// every name is non-empty, dot-separated without empty segments, and unique within the tree.
void expect_consistent(Module& module) {
    std::vector<NamedParamRef> named = module.named_parameters();
    std::vector<ParamRef> plain = module.parameters();
    ASSERT_EQ(named.size(), plain.size());
    std::set<std::string> seen;
    for (size_t i = 0; i < named.size(); ++i) {
        const std::string& name = named[i].name;
        EXPECT_EQ(named[i].ref.value, plain[i].value) << name;
        EXPECT_EQ(named[i].ref.grad, plain[i].grad) << name;
        EXPECT_NE(named[i].ref.value, nullptr) << name;
        EXPECT_NE(named[i].ref.grad, nullptr) << name;
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name.front(), '.') << name;
        EXPECT_NE(name.back(), '.') << name;
        EXPECT_EQ(name.find(".."), std::string::npos) << name;
        EXPECT_TRUE(seen.insert(name).second) << "duplicate name " << name;
    }
}

class NamedParametersTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(NamedParametersTest, ParameterlessModuleHasNone) {
    ReluModule relu(&backend);
    EXPECT_TRUE(relu.named_parameters().empty());
    EXPECT_TRUE(relu.parameters().empty());
}

TEST_F(NamedParametersTest, LinearUsesPyTorchNames) {
    LinearModule linear(3, 2, &backend);
    EXPECT_EQ(names_of(linear), (Names{"weight", "bias"}));
    expect_consistent(linear);
}

TEST_F(NamedParametersTest, Conv2DKernelIsNamedWeight) {
    Conv2DModule conv(1, 2, 3, 3, &backend);
    EXPECT_EQ(names_of(conv), (Names{"weight", "bias"}));
    expect_consistent(conv);
}

TEST_F(NamedParametersTest, EmbeddingUsesPyTorchName) {
    EmbeddingModule embedding(10, 4, &backend);
    EXPECT_EQ(names_of(embedding), (Names{"weight"}));
    expect_consistent(embedding);
}

TEST_F(NamedParametersTest, NormGammaBetaAreNamedWeightBias) {
    LayerNormModule layer_norm(4, &backend);
    GroupNormModule group_norm(2, 4, &backend);
    BatchNormModule batch_norm(4, &backend);
    RMSNormModule rms_norm(4, &backend);
    EXPECT_EQ(names_of(layer_norm), (Names{"weight", "bias"}));
    EXPECT_EQ(names_of(group_norm), (Names{"weight", "bias"}));
    EXPECT_EQ(names_of(batch_norm), (Names{"weight", "bias"}));
    EXPECT_EQ(names_of(rms_norm), (Names{"weight"}));
    expect_consistent(layer_norm);
    expect_consistent(group_norm);
    expect_consistent(batch_norm);
    expect_consistent(rms_norm);
}

TEST_F(NamedParametersTest, RecurrentModulesUseMemberNames) {
    RNNModule rnn(3, 4, &backend);
    GRUModule gru(3, 4, &backend);
    LSTMModule lstm(3, 4, &backend);
    EXPECT_EQ(names_of(rnn), (Names{"weight_xh", "weight_hh", "bias"}));
    EXPECT_EQ(names_of(gru), (Names{"weight_xz", "weight_hz", "bias_z", "weight_xr", "weight_hr", "bias_r",
                                    "weight_xn", "weight_hn", "bias_n"}));
    EXPECT_EQ(names_of(lstm),
              (Names{"weight_xi", "weight_hi", "bias_i", "weight_xf", "weight_hf", "bias_f", "weight_xg",
                     "weight_hg", "bias_g", "weight_xo", "weight_ho", "bias_o"}));
    expect_consistent(rnn);
    expect_consistent(gru);
    expect_consistent(lstm);
}

TEST_F(NamedParametersTest, SequenceMixersUseMemberNames) {
    MambaModule mamba(4, 2, &backend);
    RWKVModule rwkv(4, &backend);
    RetNetModule retnet(4, 4, 0.9f, &backend);
    EXPECT_EQ(names_of(mamba), (Names{"w_delta", "bias_delta", "w_b", "w_c", "a", "d"}));
    EXPECT_EQ(names_of(rwkv), (Names{"w_r", "w_k", "w_v", "w_o", "w", "u", "mu_r", "mu_k", "mu_v"}));
    EXPECT_EQ(names_of(retnet), (Names{"w_q", "w_k", "w_v"}));
    expect_consistent(mamba);
    expect_consistent(rwkv);
    expect_consistent(retnet);
}

TEST_F(NamedParametersTest, MultiHeadAttentionPrefixesProjections) {
    MultiHeadAttentionModule mha(4, 2, &backend);
    EXPECT_EQ(names_of(mha), (Names{"q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias", "v_proj.weight",
                                    "v_proj.bias", "out_proj.weight", "out_proj.bias"}));
    expect_consistent(mha);
}

TEST_F(NamedParametersTest, MultiHeadAttentionWithQkNormAddsNormWeights) {
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/true, /*use_qk_norm=*/true);
    Names names = names_of(mha);
    ASSERT_EQ(names.size(), 10u);
    EXPECT_EQ(names[8], "q_norm.weight");
    EXPECT_EQ(names[9], "k_norm.weight");
    expect_consistent(mha);
}

TEST_F(NamedParametersTest, SwiGLUPrefixesProjections) {
    SwiGLUModule swiglu(4, 8, &backend);
    EXPECT_EQ(names_of(swiglu), (Names{"gate_proj.weight", "gate_proj.bias", "up_proj.weight", "up_proj.bias",
                                       "down_proj.weight", "down_proj.bias"}));
    expect_consistent(swiglu);
}

TEST_F(NamedParametersTest, TransformerBlockNamesAreHierarchical) {
    TransformerBlock block(4, 2, 8, &backend);
    Names names = names_of(block);
    ASSERT_EQ(names.size(), 1u + 8u + 1u + 6u);
    EXPECT_EQ(names.front(), "norm1.weight");
    EXPECT_EQ(names[1], "mha.q_proj.weight");
    EXPECT_EQ(names[8], "mha.out_proj.bias");
    EXPECT_EQ(names[9], "norm2.weight");
    EXPECT_EQ(names[10], "swiglu.gate_proj.weight");
    EXPECT_EQ(names.back(), "swiglu.down_proj.bias");
    expect_consistent(block);
}

TEST_F(NamedParametersTest, SequentialPrefixesLayerIndexIncludingParameterlessLayers) {
    LinearModule first(3, 4, &backend);
    ReluModule relu(&backend);
    LinearModule second(4, 2, &backend);
    SequentialModule seq({&first, &relu, &second});
    EXPECT_EQ(names_of(seq), (Names{"0.weight", "0.bias", "2.weight", "2.bias"}));
    expect_consistent(seq);
}

TEST_F(NamedParametersTest, ResidualPrefixesInner) {
    LinearModule linear(4, 4, &backend);
    ResidualModule residual(&linear, &backend);
    EXPECT_EQ(names_of(residual), (Names{"inner.weight", "inner.bias"}));
    expect_consistent(residual);
}

TEST_F(NamedParametersTest, NestedContainersComposeNames) {
    TransformerBlock block0(4, 2, 8, &backend);
    TransformerBlock block1(4, 2, 8, &backend);
    SequentialModule blocks({&block0, &block1});
    Names names = names_of(blocks);
    EXPECT_EQ(names.front(), "0.norm1.weight");
    EXPECT_EQ(names.back(), "1.swiglu.down_proj.bias");
    EXPECT_NE(std::find(names.begin(), names.end(), "1.mha.q_proj.weight"), names.end());
    expect_consistent(blocks);
}

TEST_F(NamedParametersTest, AppendNamedParametersPrefixesWithDot) {
    LinearModule linear(2, 2, &backend);
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "layer", linear.named_parameters());
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].name, "layer.weight");
    EXPECT_EQ(out[1].name, "layer.bias");
    EXPECT_EQ(out[0].ref.value, linear.parameters()[0].value);
}

// A pre-FND-1 user subclass that overrides only parameters() must keep compiling and training:
// the optimizer still sees its parameters; it just has no names.
class LegacyParamsModule : public Module {
public:
    explicit LegacyParamsModule(DeviceBackend* backend)
        : weight_(Shape({1}), backend, {1.0f}), weight_grad_(Shape({1}), backend, {0.5f}) {}
    [[nodiscard]] std::vector<ParamRef> parameters() override { return {{&weight_, &weight_grad_}}; }
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) override {
        return relevance_out;
    }
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override { return grad_output; }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] const Tensor& weight() const { return weight_; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override { return input; }

private:
    Tensor weight_;
    Tensor weight_grad_;
};

TEST_F(NamedParametersTest, LegacyParametersOnlyOverrideStillTrains) {
    LegacyParamsModule legacy(&backend);
    EXPECT_TRUE(legacy.named_parameters().empty());
    SGDOptimizer sgd(0.1f);
    sgd.step(legacy);
    EXPECT_FLOAT_EQ(legacy.weight().data()[0], 1.0f - 0.1f * 0.5f);
}

}  // namespace
}  // namespace pulsatrix
