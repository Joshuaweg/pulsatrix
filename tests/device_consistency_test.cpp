#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/calibration_loss.hpp"
#include "pulsatrix/conjunction_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/cross_entropy_loss.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/dqn_loss.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/group_norm_module.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/kl_divergence_loss.hpp"
#include "pulsatrix/layer_norm_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mamba_module.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/policy_gradient_loss.hpp"
#include "pulsatrix/ppo_clipped_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/retnet_module.hpp"
#include "pulsatrix/rms_norm_module.hpp"
#include "pulsatrix/rnn_module.hpp"
#include "pulsatrix/rope_module.hpp"
#include "pulsatrix/rwkv_module.hpp"
#include "pulsatrix/satisfaction_loss.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/softmax_module.hpp"
#include "pulsatrix/swiglu_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

// Host memory, but tagged Hip: lets a CPU-only build check device bookkeeping without a GPU.
// A missing check would let a mismatched call run and the test would see no exception.
class FakeGpuBackend : public CPUBackend {
public:
    [[nodiscard]] DeviceType device() const noexcept override { return DeviceType::Hip; }
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override {
        directions.push_back(dir);
        CPUBackend::copy(dst, src, bytes, dir);
    }
    std::vector<CopyDirection> directions;
};

// Values in (0, 1): valid for every module here, including the fuzzy-logic ones.
Tensor filled(const Shape& shape, DeviceBackend* backend) {
    std::vector<float> v(static_cast<size_t>(shape.numel()));
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.1f + 0.8f * static_cast<float>((i * 7) % 11) / 11.0f;
    return Tensor(shape, backend, v);
}

struct Case {
    std::string name;
    std::function<std::unique_ptr<Module>(DeviceBackend*)> make;
    Shape input;
};

std::vector<std::unique_ptr<Module>>& keep_alive() {
    static std::vector<std::unique_ptr<Module>> modules;
    return modules;
}

std::vector<Case> cases() {
    using M = std::unique_ptr<Module>;
    return {
        {"Linear", [](DeviceBackend* b) -> M { return std::make_unique<LinearModule>(3, 4, b); }, Shape({2, 3})},
        {"Conv2D", [](DeviceBackend* b) -> M { return std::make_unique<Conv2DModule>(2, 3, 2, 2, b); }, Shape({1, 2, 4, 4})},
        {"ReLU", [](DeviceBackend* b) -> M { return std::make_unique<ReluModule>(b); }, Shape({2, 3})},
        {"Flatten", [](DeviceBackend* b) -> M { return std::make_unique<FlattenModule>(b); }, Shape({2, 2, 2})},
        {"Softmax", [](DeviceBackend* b) -> M { return std::make_unique<SoftmaxModule>(b); }, Shape({2, 3})},
        {"Dropout", [](DeviceBackend* b) -> M { return std::make_unique<DropoutModule>(0.5f, b, 1); }, Shape({2, 3})},
        {"MaxPool2D", [](DeviceBackend* b) -> M { return std::make_unique<MaxPool2DModule>(2, 2, b); }, Shape({1, 1, 4, 4})},
        {"AvgPool2D", [](DeviceBackend* b) -> M { return std::make_unique<AvgPool2DModule>(2, 2, b); }, Shape({1, 1, 4, 4})},
        {"LayerNorm", [](DeviceBackend* b) -> M { return std::make_unique<LayerNormModule>(4, b); }, Shape({3, 4})},
        {"RMSNorm", [](DeviceBackend* b) -> M { return std::make_unique<RMSNormModule>(4, b); }, Shape({3, 4})},
        {"GroupNorm", [](DeviceBackend* b) -> M { return std::make_unique<GroupNormModule>(2, 4, b); }, Shape({2, 4, 2, 2})},
        {"BatchNorm", [](DeviceBackend* b) -> M { return std::make_unique<BatchNormModule>(4, b); }, Shape({2, 4, 2, 2})},
        {"RNN", [](DeviceBackend* b) -> M { return std::make_unique<RNNModule>(3, 4, b); }, Shape({2, 3, 3})},
        {"GRU", [](DeviceBackend* b) -> M { return std::make_unique<GRUModule>(3, 4, b); }, Shape({2, 3, 3})},
        {"LSTM", [](DeviceBackend* b) -> M { return std::make_unique<LSTMModule>(3, 4, b); }, Shape({2, 3, 3})},
        {"Mamba", [](DeviceBackend* b) -> M { return std::make_unique<MambaModule>(4, 2, b); }, Shape({2, 3, 4})},
        {"RWKV", [](DeviceBackend* b) -> M { return std::make_unique<RWKVModule>(4, b); }, Shape({2, 3, 4})},
        {"RetNet", [](DeviceBackend* b) -> M { return std::make_unique<RetNetModule>(4, 4, 0.9f, b); }, Shape({2, 3, 4})},
        {"RoPE", [](DeviceBackend* b) -> M { return std::make_unique<RoPEModule>(4, b); }, Shape({2, 3, 4})},
        {"MultiHeadAttention", [](DeviceBackend* b) -> M { return std::make_unique<MultiHeadAttentionModule>(4, 2, b); },
         Shape({2, 3, 4})},
        {"SwiGLU", [](DeviceBackend* b) -> M { return std::make_unique<SwiGLUModule>(4, 8, b); }, Shape({2, 3, 4})},
        {"TransformerBlock", [](DeviceBackend* b) -> M { return std::make_unique<TransformerBlock>(4, 2, 8, b); },
         Shape({2, 3, 4})},
        {"Residual",
         [](DeviceBackend* b) -> M {
             keep_alive().push_back(std::make_unique<LinearModule>(4, 4, b));
             return std::make_unique<ResidualModule>(keep_alive().back().get(), b);
         },
         Shape({2, 4})},
        {"Negation", [](DeviceBackend* b) -> M { return std::make_unique<NegationModule>(b); }, Shape({2, 3})},
        {"Aggregator", [](DeviceBackend* b) -> M { return std::make_unique<AggregatorModule>(b); }, Shape({2, 3})},
        {"Conjunction", [](DeviceBackend* b) -> M { return std::make_unique<ConjunctionModule>(b); }, Shape({2, 3})},
        {"Disjunction", [](DeviceBackend* b) -> M { return std::make_unique<DisjunctionModule>(b); }, Shape({2, 3})},
    };
}

class DeviceConsistencyTest : public ::testing::Test {
protected:
    CPUBackend cpu;
    FakeGpuBackend gpu;
    // Residual's inner layers free through these backends, so release them first.
    void TearDown() override { keep_alive().clear(); }
};

// gpu_review #1: a CPU tensor handed to a GPU module used to reach the GPU kernels as a host
// pointer (an uncatchable HSA fault on gfx1151), and the reverse segfaulted.
TEST_F(DeviceConsistencyTest, ForwardRejectsAnInputOnAnotherDevice) {
    for (const Case& c : cases()) {
        SCOPED_TRACE(c.name);
        auto on_gpu = c.make(&gpu);
        EXPECT_THROW((void)on_gpu->forward(filled(c.input, &cpu)), std::invalid_argument);
        auto on_cpu = c.make(&cpu);
        EXPECT_THROW((void)on_cpu->forward(filled(c.input, &gpu)), std::invalid_argument);
        EXPECT_NO_THROW((void)on_cpu->forward(filled(c.input, &cpu)));
    }
}

TEST_F(DeviceConsistencyTest, BackwardAndLRPRejectATensorOnAnotherDevice) {
    for (const Case& c : cases()) {
        SCOPED_TRACE(c.name);
        auto m = c.make(&gpu);
        Tensor out = m->forward(filled(c.input, &gpu));
        EXPECT_THROW((void)m->backward(filled(out.shape(), &cpu)), std::invalid_argument);
        EXPECT_THROW((void)m->propagate_relevance(filled(out.shape(), &cpu), LRPRuleConfig{}), std::invalid_argument);
    }
}

// Embedding reads its indices through the indices' own backend, so CPU indices into a GPU
// embedding are supported; its gradients must still be on the module's device.
TEST_F(DeviceConsistencyTest, EmbeddingAcceptsIndicesFromAnyDeviceButNotGradients) {
    EmbeddingModule embedding(5, 3, &gpu);
    Tensor indices(Shape({2, 2}), &cpu, {0, 1, 4, 2});
    Tensor out = embedding.forward(indices);
    EXPECT_EQ(out.device(), DeviceType::Hip);
    EXPECT_THROW((void)embedding.backward(filled(out.shape(), &cpu)), std::invalid_argument);
    EXPECT_THROW((void)embedding.propagate_relevance(filled(out.shape(), &cpu), LRPRuleConfig{}),
                 std::invalid_argument);
}

TEST_F(DeviceConsistencyTest, ContainersRejectThroughTheirLayers) {
    LinearModule a(3, 4, &gpu), b(4, 2, &gpu);
    ReluModule relu(&gpu);
    SequentialModule seq({&a, &relu, &b});
    EXPECT_THROW((void)seq.forward(filled(Shape({2, 3}), &cpu)), std::invalid_argument);
    Tensor out = seq.forward(filled(Shape({2, 3}), &gpu));
    EXPECT_THROW((void)seq.backward(filled(out.shape(), &cpu)), std::invalid_argument);
}

// A user module written without compute_device() is not checked, and keeps working.
class UncheckedModule : public Module {
public:
    [[nodiscard]] Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return r; }
    [[nodiscard]] Tensor backward(const Tensor& g) override { return g; }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override { return input; }
};

TEST_F(DeviceConsistencyTest, ModulesWithoutAComputeDeviceAreNotChecked) {
    UncheckedModule m;
    EXPECT_NO_THROW((void)m.forward(filled(Shape({2}), &gpu)));
    EXPECT_NO_THROW((void)m.forward(filled(Shape({2}), &cpu)));
}

TEST_F(DeviceConsistencyTest, LossesRejectInputsOnAnotherDevice) {
    const Tensor pred_cpu = filled(Shape({2, 3}), &cpu), pred_gpu = filled(Shape({2, 3}), &gpu);
    MSELoss mse(&gpu);
    EXPECT_THROW((void)mse.forward(pred_cpu, pred_cpu), std::invalid_argument);
    EXPECT_NO_THROW((void)mse.forward(pred_gpu, pred_gpu));
    BCEWithLogitsLoss bce(&gpu);
    EXPECT_THROW((void)bce.forward(pred_cpu, pred_cpu), std::invalid_argument);
    KLDivergenceLoss kl(&gpu);
    EXPECT_THROW((void)kl.forward(pred_cpu, pred_cpu), std::invalid_argument);
    CrossEntropyLoss ce(&gpu);
    EXPECT_THROW((void)ce.forward(filled(Shape({1, 3}), &cpu), 1), std::invalid_argument);
    CalibrationLoss cal(&gpu);
    EXPECT_THROW((void)cal.forward(pred_cpu, Tensor(Shape({2}), &cpu, {0, 1})), std::invalid_argument);
    SatisfactionLoss sat(&gpu);
    EXPECT_THROW((void)sat.forward(pred_cpu), std::invalid_argument);
    const Tensor actions(Shape({2}), &cpu, {0, 2}), per_row(Shape({2}), &cpu, {0.5f, -0.5f});
    DQNLoss dqn(&gpu);
    EXPECT_THROW((void)dqn.forward(pred_cpu, actions, per_row), std::invalid_argument);
    PolicyGradientLoss pg(&gpu);
    EXPECT_THROW((void)pg.forward(pred_cpu, actions, per_row), std::invalid_argument);
    PPOClippedLoss ppo(&gpu);
    EXPECT_THROW((void)ppo.forward(pred_cpu, actions, per_row, per_row, 0.2f), std::invalid_argument);
}

// gpu_review #2: Stack allocated through the destination backend but tagged the result with
// the sources' device, and chose the copy direction from that wrong tag.
TEST_F(DeviceConsistencyTest, StackOntoAnotherBackendTagsAndCopiesForTheDestination) {
    std::vector<Tensor> rows = {filled(Shape({3}), &cpu), filled(Shape({3}), &cpu)};
    Tensor stacked = Tensor::Stack(rows, &gpu);
    EXPECT_EQ(stacked.device(), DeviceType::Hip);
    ASSERT_EQ(gpu.directions.size(), 2u);
    EXPECT_EQ(gpu.directions[0], CopyDirection::HostToDevice);
    EXPECT_EQ(gpu.directions[1], CopyDirection::HostToDevice);
    std::vector<float> expected(rows[0].data(), rows[0].data() + 3);
    expected.insert(expected.end(), rows[1].data(), rows[1].data() + 3);
    EXPECT_EQ(std::vector<float>(stacked.data(), stacked.data() + 6), expected);
}

TEST_F(DeviceConsistencyTest, StackOfGpuTensorsOntoTheCpuCopiesDeviceToHostThroughTheSource) {
    std::vector<Tensor> rows = {filled(Shape({2}), &gpu), filled(Shape({2}), &gpu)};
    gpu.directions.clear();
    Tensor stacked = Tensor::Stack(rows, &cpu);
    EXPECT_EQ(stacked.device(), DeviceType::Cpu);
    ASSERT_EQ(gpu.directions.size(), 2u);  // the source's backend performed the reads
    EXPECT_EQ(gpu.directions[0], CopyDirection::DeviceToHost);
}

}  // namespace
}  // namespace pulsatrix
