/** @file grad_cam_walkthrough.cpp
 *  @brief Recipe: Grad-CAM's mechanics (graph wiring, activation/gradient caching, per-channel
 *         weighting) on a small synthetic Conv2DModule network. Paired with
 *         docs/recipes/interpretability/grad_cam_walkthrough.md.
 *  @note Smaller scale than examples/grad_cam_mnist_demo.cpp (12x12 instead of MNIST's
 *        28x28) purely to keep the printed heatmap compact -- the mechanics are identical.
 *        Same caveat applies: randomly-initialized/untrained network, synthetic input, so
 *        the heatmap demonstrates Grad-CAM's pipeline, not "what a real model learned."
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    constexpr int64_t kImageSize = 12;
    constexpr int64_t kNumClasses = 3;
    constexpr int64_t kConvOutChannels = 4;
    constexpr int64_t kKernelSize = 3;
    constexpr int64_t kConvOutSize = kImageSize - kKernelSize + 1;  // 10
    constexpr int64_t kFlattenSize = kConvOutChannels * kConvOutSize * kConvOutSize;

    Conv2DModule conv(1, kConvOutChannels, kKernelSize, kKernelSize, &backend);
    ReluModule relu(&backend);
    FlattenModule flatten(&backend);
    LinearModule classifier(kFlattenSize, kNumClasses, &backend);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weight_dist(-0.1f, 0.1f);
    auto randomize = [&](Tensor* t) {
        for (int64_t i = 0; i < t->numel(); ++i) {
            t->data()[i] = weight_dist(rng);
        }
    };
    for (const auto& p : conv.parameters()) randomize(p.value);
    for (const auto& p : classifier.parameters()) randomize(p.value);

    // Synthetic filled-circle "digit" on a 12x12 canvas.
    Tensor input(Shape({1, 1, kImageSize, kImageSize}), &backend);
    constexpr float kCenter = static_cast<float>(kImageSize) / 2.0f;
    constexpr float kRadius = 3.5f;
    for (int64_t h = 0; h < kImageSize; ++h) {
        for (int64_t w = 0; w < kImageSize; ++w) {
            float dy = static_cast<float>(h) - kCenter;
            float dw = static_cast<float>(w) - kCenter;
            input.at({0, 0, h, w}) = (std::sqrt(dy * dy + dw * dw) < kRadius) ? 1.0f : 0.0f;
        }
    }

    ExplainerContext ctx({&conv, &relu, &flatten, &classifier});
    Tensor output = ctx.forward_pass(input);

    std::printf("Grad-CAM recipe -- Conv(1,%lld,%lld,%lld)->ReLU->Flatten->Linear(%lld,%lld), SYNTHETIC/untrained\n\n",
                static_cast<long long>(kConvOutChannels), static_cast<long long>(kKernelSize),
                static_cast<long long>(kKernelSize), static_cast<long long>(kFlattenSize),
                static_cast<long long>(kNumClasses));

    int64_t predicted = 0;
    float best = output.data()[0];
    for (int64_t c = 0; c < kNumClasses; ++c) {
        std::printf("class %lld score: %8.4f\n", static_cast<long long>(c), output.data()[c]);
        if (output.data()[c] > best) {
            best = output.data()[c];
            predicted = c;
        }
    }
    std::printf("\npredicted class %lld (meaningless on an untrained network)\n\n", static_cast<long long>(predicted));

    GradCAM gradcam;
    Attribution attr = gradcam.explain(ctx, input, predicted, &backend);

    int64_t cam_h = attr.values.shape().dim(1);
    int64_t cam_w = attr.values.shape().dim(2);
    float max_val = 0.0f;
    for (int64_t i = 0; i < attr.values.numel(); ++i) max_val = std::max(max_val, attr.values.data()[i]);

    std::printf("Grad-CAM heatmap (%lldx%lld), ASCII-scaled 0-9:\n", static_cast<long long>(cam_h),
                static_cast<long long>(cam_w));
    for (int64_t h = 0; h < cam_h; ++h) {
        for (int64_t w = 0; w < cam_w; ++w) {
            float v = attr.values.at({0, h, w});
            int level = max_val > 0.0f ? static_cast<int>((v / max_val) * 9.0f) : 0;
            std::putchar(static_cast<char>('0' + level));
        }
        std::putchar('\n');
    }

    std::printf(
        "\nGrad-CAM finds the last Conv node, averages its cached gradient spatially per\n"
        "channel to get per-channel weights alpha_k, then computes ReLU(sum_k alpha_k * A_k).\n");

    return 0;
}
