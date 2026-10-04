#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Conv2D -> BatchNorm -> ReLU -> Flatten -> Linear: parameters in two layers, buffers in one.
struct Net {
    explicit Net(DeviceBackend* b, unsigned seed)
        : conv(1, 2, 2, 2, b), bn(2, b), relu(b), flatten(b), fc(2 * 3 * 3, 2, b), seq({&conv, &bn, &relu, &flatten, &fc}) {
        for (ParamRef p : seq.parameters()) {
            *p.value = Tensor(p.value->shape(), b, random_values(static_cast<size_t>(p.value->numel()), seed++));
        }
    }
    Conv2DModule conv;
    BatchNormModule bn;
    ReluModule relu;
    FlattenModule flatten;
    LinearModule fc;
    SequentialModule seq;
};

class CheckpointTest : public ::testing::Test {
protected:
    CPUBackend backend;
    Tensor x{Shape({4, 1, 4, 4}), &backend, random_values(4 * 16, 100)};
    Tensor y{Shape({4, 2}), &backend, random_values(4 * 2, 101)};
    std::string path = ::testing::TempDir() + "pulsatrix_checkpoint_test.safetensors";

    void TearDown() override {
        std::remove(path.c_str());
        std::remove(OptimizerStatePath(path).c_str());
    }

    // One full training step; returns the loss.
    float train_step(Net& net, AdamOptimizer& adam) {
        MSELoss loss(&backend);
        adam.zero_grad(net.seq);
        float l = loss.forward(net.seq.forward(x), y);
        (void)net.seq.backward(loss.backward());
        adam.step(net.seq);
        return l;
    }
};

// --- named_buffers -----------------------------------------------------------------------

TEST_F(CheckpointTest, BatchNormRunningStatisticsAreNamedBuffers) {
    BatchNormModule bn(3, &backend);
    std::vector<NamedBufferRef> buffers = bn.named_buffers();
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0].name, "running_mean");
    EXPECT_EQ(buffers[0].value, &bn.running_mean());
    EXPECT_EQ(buffers[1].name, "running_var");
    LinearModule linear(2, 2, &backend);
    EXPECT_TRUE(linear.named_buffers().empty());
}

TEST_F(CheckpointTest, ContainersPrefixTheirLayersBuffers) {
    Net net(&backend, 1);
    std::vector<NamedBufferRef> buffers = net.seq.named_buffers();
    ASSERT_EQ(buffers.size(), 2u);
    EXPECT_EQ(buffers[0].name, "1.running_mean");
    EXPECT_EQ(buffers[1].name, "1.running_var");
    BatchNormModule bn(2, &backend);
    ResidualModule residual(&bn, &backend);
    EXPECT_EQ(residual.named_buffers()[0].name, "inner.running_mean");
}

// --- Model round trip --------------------------------------------------------------------

TEST_F(CheckpointTest, LoadRestoresParametersBuffersAndTheForwardPassBitForBit) {
    Net a(&backend, 1);
    AdamOptimizer adam(0.01f, &backend);
    for (int i = 0; i < 3; ++i) (void)train_step(a, adam);  // running statistics move off their defaults
    SaveCheckpoint(path, a.seq, {{"run", "unit-test"}});

    Net b(&backend, 50);
    LoadCheckpoint(path, b.seq);
    std::vector<NamedParamRef> pa = a.seq.named_parameters(), pb = b.seq.named_parameters();
    for (size_t i = 0; i < pa.size(); ++i) EXPECT_EQ(values_of(*pa[i].ref.value), values_of(*pb[i].ref.value)) << pa[i].name;
    EXPECT_EQ(values_of(a.bn.running_mean()), values_of(b.bn.running_mean()));
    EXPECT_EQ(values_of(a.bn.running_var()), values_of(b.bn.running_var()));
    a.seq.set_training(false);
    b.seq.set_training(false);
    EXPECT_EQ(values_of(a.seq.forward(x)), values_of(b.seq.forward(x)));
}

TEST_F(CheckpointTest, FileCarriesFormatVersionAndCallerMetadata) {
    Net a(&backend, 1);
    SaveCheckpoint(path, a.seq, {{"run", "unit-test"}});
    SafetensorsFile f = SafetensorsFile::Read(path);
    EXPECT_EQ(f.metadata().at("format"), "pulsatrix");
    EXPECT_EQ(f.metadata().at("format_version"), std::to_string(kCheckpointFormatVersion));
    EXPECT_EQ(f.metadata().at("run"), "unit-test");
    EXPECT_TRUE(f.contains("0.weight"));
    EXPECT_TRUE(f.contains("1.running_var"));
}

TEST_F(CheckpointTest, LoadingKeepsFrozenParametersFrozen) {
    Net a(&backend, 1);
    SaveCheckpoint(path, a.seq);
    Net b(&backend, 50);
    b.seq.set_requires_grad(false, "0");
    LoadCheckpoint(path, b.seq);
    EXPECT_FALSE(b.conv.kernel().requires_grad());
    EXPECT_TRUE(b.fc.weight().requires_grad());
}

// --- The roadmap's falsifier: resumed training reproduces the loss curve -----------------

TEST_F(CheckpointTest, ResumedTrainingReproducesTheLossCurveBitForBit) {
    Net a(&backend, 1);
    AdamOptimizer adam_a(0.01f, &backend);
    for (int i = 0; i < 3; ++i) (void)train_step(a, adam_a);
    SaveCheckpoint(path, a.seq, adam_a);
    std::vector<float> curve_a;
    for (int i = 0; i < 4; ++i) curve_a.push_back(train_step(a, adam_a));

    Net b(&backend, 50);
    AdamOptimizer adam_b(0.01f, &backend);
    LoadCheckpoint(path, b.seq, adam_b);
    std::vector<float> curve_b;
    for (int i = 0; i < 4; ++i) curve_b.push_back(train_step(b, adam_b));

    EXPECT_EQ(curve_a, curve_b);
    EXPECT_EQ(values_of(a.fc.weight()), values_of(b.fc.weight()));
}

TEST_F(CheckpointTest, WithoutOptimizerStateTheCurveDiverges) {
    // The control for the test above: same weights, fresh Adam moments, different curve.
    Net a(&backend, 1);
    AdamOptimizer adam_a(0.01f, &backend);
    for (int i = 0; i < 3; ++i) (void)train_step(a, adam_a);
    SaveCheckpoint(path, a.seq);
    (void)train_step(a, adam_a);
    Net b(&backend, 50);
    AdamOptimizer fresh(0.01f, &backend);
    LoadCheckpoint(path, b.seq);
    (void)train_step(b, fresh);  // the first loss matches; the update after it doesn't
    EXPECT_NE(values_of(a.fc.weight()), values_of(b.fc.weight()));
}

TEST_F(CheckpointTest, OptimizerStateLivesInASiblingFile) {
    EXPECT_EQ(OptimizerStatePath("/x/model.safetensors"), "/x/model.optim.safetensors");
    EXPECT_EQ(OptimizerStatePath("/x/model.bin"), "/x/model.bin.optim.safetensors");
    Net a(&backend, 1);
    AdamOptimizer adam(0.01f, &backend);
    SaveCheckpoint(path, a.seq);  // no optimizer: no sibling file
    AdamOptimizer other(0.01f, &backend);
    EXPECT_THROW(LoadCheckpoint(path, a.seq, other), std::runtime_error);
}

TEST_F(CheckpointTest, RefusesAnOptimizerFileFromAnotherSaveOfTheModel) {
    Net a(&backend, 1);
    AdamOptimizer adam(0.01f, &backend);
    (void)train_step(a, adam);
    SaveCheckpoint(path, a.seq, adam);
    (void)train_step(a, adam);
    SaveCheckpoint(path, a.seq);  // overwrites the model file only: the old .optim file is now stale
    AdamOptimizer fresh(0.01f, &backend);
    EXPECT_THROW(LoadCheckpoint(path, a.seq, fresh), std::invalid_argument);
}

// --- Strictness and versions -----------------------------------------------------------

TEST_F(CheckpointTest, StrictLoadRejectsMissingUnexpectedAndMisshapenEntries) {
    LinearModule small(2, 3, &backend), other(2, 3, &backend), wrong_shape(3, 3, &backend);
    SequentialModule two({&small, &other});
    SaveCheckpoint(path, two);

    SequentialModule three({&small, &other, &wrong_shape});
    EXPECT_THROW(LoadCheckpoint(path, three), std::invalid_argument);  // 2.weight missing from the file
    SequentialModule one({&small});
    EXPECT_THROW(LoadCheckpoint(path, one), std::invalid_argument);  // 1.weight unexpected
    LinearModule a(2, 3, &backend), b(3, 3, &backend);
    SequentialModule misshapen({&a, &b});
    EXPECT_THROW(LoadCheckpoint(path, misshapen), std::invalid_argument);  // 1.weight is (2, 3) in the file

    CheckpointLoadOptions lenient;
    lenient.strict = false;
    EXPECT_NO_THROW(LoadCheckpoint(path, one, lenient));
    EXPECT_NO_THROW(LoadCheckpoint(path, three, lenient));
    EXPECT_THROW(LoadCheckpoint(path, misshapen, lenient), std::invalid_argument);  // shapes always checked
}

TEST_F(CheckpointTest, PlainSafetensorsWithoutAVersionLoadThroughTheMigrationTable) {
    LinearModule saved(2, 3, &backend);
    saved.set_weight(random_values(6, 7));
    WriteSafetensors(path, {{"weight", &saved.weight()}, {"bias", &saved.bias()}});  // version 0: no metadata
    LinearModule loaded(2, 3, &backend);
    LoadCheckpoint(path, loaded);
    EXPECT_EQ(values_of(loaded.weight()), values_of(saved.weight()));
}

TEST_F(CheckpointTest, RejectsFilesFromNewerVersionsAndOtherFormats) {
    LinearModule m(2, 3, &backend);
    WriteSafetensors(path, {{"weight", &m.weight()}, {"bias", &m.bias()}},
                     {{"format", "pulsatrix"}, {"format_version", std::to_string(kCheckpointFormatVersion + 1)}});
    EXPECT_THROW(LoadCheckpoint(path, m), std::invalid_argument);
    WriteSafetensors(path, {{"weight", &m.weight()}, {"bias", &m.bias()}}, {{"format", "pt"}});
    EXPECT_THROW(LoadCheckpoint(path, m), std::invalid_argument);  // a PyTorch export needs name mapping (IO-4)
    WriteSafetensors(path, {{"weight", &m.weight()}, {"bias", &m.bias()}},
                     {{"format", "pulsatrix"}, {"format_version", "one"}});
    EXPECT_THROW(LoadCheckpoint(path, m), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
