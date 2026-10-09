// KS-9: torchvision ResNet and VGG built from pulsatrix layers -- structure, weight loading, and
// the logits and Zennit heatmaps of tools/golden/make_vision_golden.py (vision_golden_cases.hpp).
#include <gtest/gtest.h>

#include "pulsatrix/adaptive_avg_pool2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "vision_golden_cases.hpp"

namespace pulsatrix {
namespace {

using vision_cases::Fixture;

TEST(VisionModels, TinyResNetMatchesTorchvisionAndZennit) {
    CPUBackend cpu;
    vision_cases::TinyResNetMatchesTorchvisionAndZennit(&cpu);
}

TEST(VisionModels, TinyVGGMatchesTorchvisionAndZennit) {
    CPUBackend cpu;
    vision_cases::TinyVGGMatchesTorchvisionAndZennit(&cpu);
}

TEST(VisionModels, PublishedResNet18) {
    CPUBackend cpu;
    vision_cases::PublishedModelMatches("resnet18", &cpu);
}

TEST(VisionModels, PublishedVGG16) {
    CPUBackend cpu;
    vision_cases::PublishedModelMatches("vgg16", &cpu);
}

TEST(VisionModels, ParameterNamesAreTorchvisions) {
    CPUBackend cpu;
    TorchvisionResNet resnet(vision_cases::TinyResNetConfig(), &cpu);
    std::vector<std::string> names;
    for (const NamedParamRef& p : resnet.named_parameters()) names.push_back(p.name);
    for (const char* want : {"conv1.weight", "bn1.weight", "layer1.0.conv2.weight", "layer2.0.downsample.0.weight",
                             "layer2.0.downsample.1.bias", "layer2.1.bn1.weight", "fc.bias"}) {
        EXPECT_NE(std::find(names.begin(), names.end(), want), names.end()) << want;
    }
    // layer1.0 keeps its shape, so it has no downsample; layer2.1 neither.
    EXPECT_EQ(std::find(names.begin(), names.end(), "layer1.0.downsample.0.weight"), names.end());
    // conv1, bn1, relu, maxpool, three blocks, avgpool, flatten, fc.
    EXPECT_EQ(resnet.layers().size(), 10u);

    TorchvisionVGG vgg(vision_cases::TinyVGGConfig(), &cpu);
    names.clear();
    for (const NamedParamRef& p : vgg.named_parameters()) names.push_back(p.name);
    EXPECT_EQ(names, (std::vector<std::string>{"features.0.weight", "features.0.bias", "features.3.weight", "features.3.bias",
                                               "features.5.weight", "features.5.bias", "classifier.0.weight", "classifier.0.bias",
                                               "classifier.3.weight", "classifier.3.bias", "classifier.6.weight",
                                               "classifier.6.bias"}));
}

TEST(VisionModels, LoadingReportsAbsentConvolutionBiasesAndRefusesMismatches) {
    CPUBackend cpu;
    TorchvisionResNet resnet(vision_cases::TinyResNetConfig(), &cpu);
    const TorchvisionLoadReport report = LoadTorchvisionWeights(resnet, Fixture("tiny_resnet.safetensors"));
    EXPECT_GT(report.loaded, 0);
    EXPECT_NE(std::find(report.left_at_default.begin(), report.left_at_default.end(), "conv1.bias"),
              report.left_at_default.end());
    // The VGG file into a ResNet: names the model doesn't have.
    EXPECT_THROW(LoadTorchvisionWeights(resnet, Fixture("tiny_vgg.safetensors")), std::invalid_argument);
    // A wider ResNet: shapes that don't fit.
    ResNetConfig wide = vision_cases::TinyResNetConfig();
    wide.width = 8;
    TorchvisionResNet wider(wide, &cpu);
    EXPECT_THROW(LoadTorchvisionWeights(wider, Fixture("tiny_resnet.safetensors")), std::invalid_argument);
}

TEST(VisionModels, FoldingNeedsEvalModeAndBadConfigsAreRefused) {
    CPUBackend cpu;
    TorchvisionResNet resnet(vision_cases::TinyResNetConfig(), &cpu);
    EXPECT_THROW(static_cast<void>(resnet.fold_batch_norms()), std::invalid_argument);
    ResNetConfig empty;
    empty.blocks.clear();
    EXPECT_THROW(TorchvisionResNet(empty, &cpu), std::invalid_argument);
    VGGConfig pools_only;
    pools_only.features = {0, 0};
    EXPECT_THROW(TorchvisionVGG(pools_only, &cpu), std::invalid_argument);
}

// ---- The layers KS-9 needed ---------------------------------------------------------------------

// MaxPool2d(3, stride 2, padding 1) on a 1x1x4x4 ramp, by hand: windows overlap, and the
// gradient of an input that wins two windows is the sum of theirs.
TEST(VisionModels, OverlappingPaddedMaxPool) {
    CPUBackend cpu;
    MaxPool2DModule pool(3, 3, 2, 2, 1, 1, &cpu);
    std::vector<float> v(16);
    for (size_t i = 0; i < 16; ++i) v[i] = static_cast<float>(i);
    v[5] = 100.0f;  // (1, 1) wins every window that holds it
    const Tensor y = pool.forward(Tensor(Shape({1, 1, 4, 4}), &cpu, v));
    ASSERT_EQ(y.shape(), Shape({1, 1, 2, 2}));
    // Windows: rows/cols {-1,0,1}, {1,2,3}. (0,0) and (0,1), (1,0) hold (1,1); (1,1) is rows 1..3 x cols 1..3.
    EXPECT_EQ(y.to_host_vector(), (std::vector<float>{100, 100, 100, 100}));
    const Tensor g = pool.backward(Tensor(Shape({1, 1, 2, 2}), &cpu, {1, 2, 3, 4}));
    std::vector<float> want(16, 0.0f);
    want[5] = 10.0f;
    EXPECT_EQ(g.to_host_vector(), want);
    const Tensor r = pool.propagate_relevance(Tensor(Shape({1, 1, 2, 2}), &cpu, {1, 2, 3, 4}), LRPRuleConfig{});
    EXPECT_EQ(r.to_host_vector(), want);

    // Without the planted maximum, the bottom-right of each window wins; padding never does.
    for (size_t i = 0; i < 16; ++i) v[i] = -static_cast<float>(16 - i);
    EXPECT_EQ(pool.forward(Tensor(Shape({1, 1, 4, 4}), &cpu, v)).to_host_vector(), (std::vector<float>{-11, -9, -3, -1}));

    EXPECT_THROW(MaxPool2DModule(3, 3, 0, 2, 1, 1, &cpu), std::invalid_argument);
    EXPECT_THROW(MaxPool2DModule(3, 3, 2, 2, 2, 1, &cpu), std::invalid_argument);
}

TEST(VisionModels, AdaptiveAvgPoolNeedsEvenWindows) {
    CPUBackend cpu;
    AdaptiveAvgPool2DModule pool(1, 2, &cpu);
    const Tensor y = pool.forward(Tensor(Shape({1, 1, 2, 4}), &cpu, {1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(y.to_host_vector(), (std::vector<float>{3.5f, 5.5f}));
    EXPECT_THROW(static_cast<void>(pool.forward(Tensor(Shape({1, 1, 2, 3}), &cpu))), std::invalid_argument);
    EXPECT_THROW(AdaptiveAvgPool2DModule(0, 1, &cpu), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
