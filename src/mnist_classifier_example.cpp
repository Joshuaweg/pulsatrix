#include "exai/mnist_classifier_example.hpp"

#include <random>
#include <vector>

namespace exai {

namespace {
constexpr int64_t kImageSize = 28;
constexpr int64_t kConvOutChannels = 8;
constexpr int64_t kKernelSize = 5;
constexpr int64_t kConvOutSize = kImageSize - kKernelSize + 1;  // 24
constexpr int64_t kFlattenSize = kConvOutChannels * kConvOutSize * kConvOutSize;  // 4608
constexpr int64_t kNumClasses = 10;
}  // namespace

MnistConvNet::MnistConvNet(DeviceBackend* backend, unsigned seed)
    : backend_(backend),
      conv_(1, kConvOutChannels, kKernelSize, kKernelSize, backend),
      relu_(backend),
      flatten_(backend),
      classifier_(kFlattenSize, kNumClasses, backend),
      loss_(backend) {
    // Random, not hand-picked, initial weights -- see header @note. Small uniform range
    // (matching grad_cam_mnist_demo.cpp's own precedent) is enough to break the all-zero
    // symmetry XorNetwork's own note explains in detail for the same underlying reason.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> weight_dist(-0.1f, 0.1f);
    auto randomize = [&](Tensor* t) {
        for (int64_t i = 0; i < t->numel(); ++i) {
            t->data()[i] = weight_dist(rng);
        }
    };
    for (const auto& p : conv_.parameters()) {
        randomize(p.value);
    }
    for (const auto& p : classifier_.parameters()) {
        randomize(p.value);
    }
}

Tensor MnistConvNet::forward(const Tensor& image) {
    Tensor conv_out = conv_.forward(image);
    Tensor relu_out = relu_.forward(conv_out);
    Tensor flat = flatten_.forward(relu_out);
    return classifier_.forward(flat);
}

int64_t MnistConvNet::predict(const Tensor& image) {
    Tensor logits = forward(image);
    int64_t best_class = 0;
    float best_value = logits.data()[0];
    for (int64_t i = 1; i < logits.numel(); ++i) {
        if (logits.data()[i] > best_value) {
            best_value = logits.data()[i];
            best_class = i;
        }
    }
    return best_class;
}

float MnistConvNet::train_step(const Tensor& image, int64_t target_class, AdamOptimizer& optimizer, MetricsSink& sink,
                                int step) {
    // Same zero_grad()-before-forward() discipline XorNetwork's own AAR flagged as a real
    // footgun (gradients otherwise accumulate unboundedly across the whole training run).
    optimizer.zero_grad(conv_);
    optimizer.zero_grad(classifier_);

    Tensor conv_out = conv_.forward(image);
    Tensor relu_out = relu_.forward(conv_out);
    Tensor flat = flatten_.forward(relu_out);
    Tensor logits = classifier_.forward(flat);

    float loss_value = loss_.forward(logits, target_class);

    Tensor grad_logits = loss_.backward();
    Tensor grad_flat = classifier_.backward(grad_logits);
    Tensor grad_relu_out = flatten_.backward(grad_flat);
    Tensor grad_conv_out = relu_.backward(grad_relu_out);
    (void)conv_.backward(grad_conv_out);  // input has no further upstream to receive this

    optimizer.step(conv_);
    optimizer.step(classifier_);
    // relu_/flatten_ have no parameters (Module::parameters() default) -- nothing to step.

    sink.log_scalar("loss", static_cast<double>(loss_value), step);
    return loss_value;
}

}  // namespace exai
