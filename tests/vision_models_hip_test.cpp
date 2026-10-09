// KS-9: the torchvision ResNet and VGG goldens (vision_golden_cases.hpp) on the Hip backend.
#include <gtest/gtest.h>

#include "pulsatrix/hip_backend.hpp"
#include "vision_golden_cases.hpp"

namespace pulsatrix {
namespace {

TEST(VisionModelsHip, TinyResNetMatchesTorchvisionAndZennit) {
    HIPBackend gpu;
    vision_cases::TinyResNetMatchesTorchvisionAndZennit(&gpu);
}

TEST(VisionModelsHip, TinyVGGMatchesTorchvisionAndZennit) {
    HIPBackend gpu;
    vision_cases::TinyVGGMatchesTorchvisionAndZennit(&gpu);
}

TEST(VisionModelsHip, PublishedResNet18) {
    HIPBackend gpu;
    vision_cases::PublishedModelMatches("resnet18", &gpu);
}

TEST(VisionModelsHip, PublishedVGG16) {
    HIPBackend gpu;
    vision_cases::PublishedModelMatches("vgg16", &gpu);
}

}  // namespace
}  // namespace pulsatrix
