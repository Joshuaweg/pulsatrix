/** @file grad_cam_mnist_demo.cpp
 *  @brief Grad-CAM demo at MNIST's actual input shape/scale (1x28x28 = 784 pixels,
 *         10-class output) -- Conv2DModule -> ReluModule -> FlattenModule ->
 *         LinearModule, explained via GradCAM (Phase 2 Mission 3).
 *  @note This is a SYNTHETIC demo, not a real digit classifier: this codebase has no
 *        dataset loader (no IDX/ubyte parser, no MNIST files) and no
 *        softmax/cross-entropy loss yet (only MSELoss), so there is no trained model to
 *        explain. Weights are randomly initialized and the input is a hand-drawn-shaped
 *        synthetic blob, not a real digit. The resulting heatmap is not meaningful in the
 *        "this is what the model actually learned to look at" sense -- it only
 *        demonstrates Grad-CAM's pipeline mechanics (graph wiring, activation/gradient
 *        caching, per-channel weighting) at MNIST's real input shape and a real
 *        classifier-head network depth. Training on real MNIST data is a genuinely larger
 *        future task (data loader + a real loss function + a real training loop), not
 *        built here.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/flatten_module.hpp"
#include "exai/grad_cam.hpp"
#include "exai/linear_module.hpp"
#include "exai/relu_module.hpp"

int main() {
    using namespace exai;

    CPUBackend backend;

    constexpr int64_t kImageSize = 28;  // MNIST: 28x28 = 784 pixels
    constexpr int64_t kNumClasses = 10;
    constexpr int64_t kConvOutChannels = 8;
    constexpr int64_t kKernelSize = 5;
    constexpr int64_t kConvOutSize = kImageSize - kKernelSize + 1;  // 24
    constexpr int64_t kFlattenSize = kConvOutChannels * kConvOutSize * kConvOutSize;

    Conv2DModule conv(1, kConvOutChannels, kKernelSize, kKernelSize, &backend);
    ReluModule relu(&backend);
    FlattenModule flatten(&backend);
    LinearModule classifier(kFlattenSize, kNumClasses, &backend);

    // Randomly initialize every parameter -- synthetic/untrained, see file-level @note.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weight_dist(-0.1f, 0.1f);
    auto randomize = [&](Tensor* t) {
        for (int64_t i = 0; i < t->numel(); ++i) {
            t->data()[i] = weight_dist(rng);
        }
    };
    for (const auto& p : conv.parameters()) {
        randomize(p.value);
    }
    for (const auto& p : classifier.parameters()) {
        randomize(p.value);
    }

    // Synthetic "hand-drawn digit"-shaped input: a filled circle on a 28x28 canvas -- not
    // real MNIST data.
    Tensor input(Shape({1, kImageSize, kImageSize}), &backend);
    constexpr float kCenter = static_cast<float>(kImageSize) / 2.0f;
    constexpr float kRadius = 8.0f;
    for (int64_t h = 0; h < kImageSize; ++h) {
        for (int64_t w = 0; w < kImageSize; ++w) {
            float dy = static_cast<float>(h) - kCenter;
            float dw = static_cast<float>(w) - kCenter;
            float dist = std::sqrt(dy * dy + dw * dw);
            input.at({0, h, w}) = (dist < kRadius) ? 1.0f : 0.0f;
        }
    }

    ExplainerContext ctx({&conv, &relu, &flatten, &classifier});
    Tensor output = ctx.forward_pass(input);

    std::printf("Grad-CAM demo -- MNIST-shaped input (1x%lldx%lld = %lld pixels), SYNTHETIC/untrained\n",
                static_cast<long long>(kImageSize), static_cast<long long>(kImageSize),
                static_cast<long long>(kImageSize * kImageSize));
    std::printf("Conv(1,%lld,%lld,%lld) -> ReLU -> Flatten -> Linear(%lld,%lld)\n\n",
                static_cast<long long>(kConvOutChannels), static_cast<long long>(kKernelSize),
                static_cast<long long>(kKernelSize), static_cast<long long>(kFlattenSize),
                static_cast<long long>(kNumClasses));

    std::printf("Class scores:\n");
    int64_t predicted = 0;
    float best = output.data()[0];
    for (int64_t c = 0; c < kNumClasses; ++c) {
        std::printf("  class %lld: %8.4f\n", static_cast<long long>(c), output.data()[c]);
        if (output.data()[c] > best) {
            best = output.data()[c];
            predicted = c;
        }
    }
    std::printf("\npredicted class (highest score, meaningless on an untrained network): %lld\n\n",
                static_cast<long long>(predicted));

    GradCAM gradcam;
    Attribution attr = gradcam.explain(ctx, input, predicted, &backend);

    int64_t cam_h = attr.values.shape().dim(0);
    int64_t cam_w = attr.values.shape().dim(1);
    std::printf("Grad-CAM heatmap (%lldx%lld, from the last conv layer), ASCII-scaled 0-9:\n",
                static_cast<long long>(cam_h), static_cast<long long>(cam_w));

    float max_val = 0.0f;
    for (int64_t i = 0; i < attr.values.numel(); ++i) {
        max_val = std::max(max_val, attr.values.data()[i]);
    }
    for (int64_t h = 0; h < cam_h; ++h) {
        for (int64_t w = 0; w < cam_w; ++w) {
            float v = attr.values.at({h, w});
            int level = max_val > 0.0f ? static_cast<int>((v / max_val) * 9.0f) : 0;
            std::putchar(static_cast<char>('0' + level));
        }
        std::putchar('\n');
    }

    return 0;
}
