// KS-9: the torchvision ResNet and VGG goldens (vision_golden_cases.hpp) on the Cuda backend.
#include <gtest/gtest.h>

#include "pulsatrix/cuda_backend.hpp"
#include "vision_golden_cases.hpp"

namespace pulsatrix {
namespace {

TEST(VisionModelsCuda, TinyResNetMatchesTorchvisionAndZennit) {
    CUDABackend gpu;
    vision_cases::TinyResNetMatchesTorchvisionAndZennit(&gpu);
}

TEST(VisionModelsCuda, TinyVGGMatchesTorchvisionAndZennit) {
    CUDABackend gpu;
    vision_cases::TinyVGGMatchesTorchvisionAndZennit(&gpu);
}

TEST(VisionModelsCuda, PublishedResNet18) {
    CUDABackend gpu;
    vision_cases::PublishedModelMatches("resnet18", &gpu);
}

TEST(VisionModelsCuda, PublishedVGG16) {
    CUDABackend gpu;
    vision_cases::PublishedModelMatches("vgg16", &gpu);
}

}  // namespace
}  // namespace pulsatrix
