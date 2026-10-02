#include <gtest/gtest.h>

#include <random>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/mnist_classifier_example.hpp"

// Unit-level correctness for MnistConvNet's wiring (forward shape, weight updates, sink
// logging, single-example overfitting) using synthetic-but-realistically-shaped
// (N=1,1,28,28) images -- fast, deterministic, no real MNIST data dependency. This mission's actual
// "trained on real data" acceptance criterion (real measured test-set accuracy well above
// chance) is examples/mnist_training_demo.cpp's job (Objective 5), not this unit suite --
// training on the real ~60k-image dataset here would make this test suite far too slow.
namespace pulsatrix {
namespace {

Tensor SyntheticImage(DeviceBackend* backend, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);
    Tensor image(Shape({1, 1, 28, 28}), backend);
    for (int64_t i = 0; i < image.numel(); ++i) {
        image.data()[i] = pixel_dist(rng);
    }
    return image;
}

class MnistConvNetTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(MnistConvNetTest, ForwardProducesTenClassLogits) {
    MnistConvNet net(&backend);
    Tensor image = SyntheticImage(&backend, 1);

    Tensor logits = net.forward(image);

    EXPECT_EQ(logits.shape(), Shape({1, 10}));
}

TEST_F(MnistConvNetTest, ModulesBuildAnExplainerContextMatchingForward) {
    // modules() exposes the trained layers so explainers run on exactly the model
    // train_step() learned -- an ExplainerContext over them must reproduce forward().
    MnistConvNet net(&backend);
    Tensor image = SyntheticImage(&backend, 3);

    std::vector<Module*> modules = net.modules();
    ASSERT_EQ(modules.size(), 4u);
    ExplainerContext ctx(modules);
    Tensor via_context = ctx.forward_pass(image);
    Tensor via_net = net.forward(image);

    ASSERT_EQ(via_context.shape(), via_net.shape());
    for (int64_t i = 0; i < via_net.numel(); ++i) {
        EXPECT_FLOAT_EQ(via_context.data()[i], via_net.data()[i]);
    }
}

TEST_F(MnistConvNetTest, PredictReturnsValidClassIndex) {
    MnistConvNet net(&backend);
    Tensor image = SyntheticImage(&backend, 1);

    int64_t predicted = net.predict(image);

    EXPECT_GE(predicted, 0);
    EXPECT_LT(predicted, 10);
}

TEST_F(MnistConvNetTest, TrainStepChangesClassifierWeights) {
    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.01f, &backend);
    NoOpMetricsSink sink;

    Tensor weight_before(net.classifier_weight());  // copy, before training
    Tensor image = SyntheticImage(&backend, 1);

    (void)net.train_step(image, /*target_class=*/3, optimizer, sink, 0);

    bool any_changed = false;
    for (int64_t i = 0; i < weight_before.numel(); ++i) {
        if (weight_before.data()[i] != net.classifier_weight().data()[i]) {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST_F(MnistConvNetTest, TrainStepLogsLossThroughMetricsSink) {
    class CapturingSink : public MetricsSink {
    public:
        void log_scalar(const std::string& tag, double value, int step) override {
            last_tag = tag;
            last_value = value;
            last_step = step;
            call_count++;
        }
        void log_histogram(const std::string&, const Tensor&, int) override {}

        std::string last_tag;
        double last_value = 0.0;
        int last_step = -1;
        int call_count = 0;
    };

    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.01f, &backend);
    CapturingSink sink;
    Tensor image = SyntheticImage(&backend, 1);

    float loss = net.train_step(image, /*target_class=*/5, optimizer, sink, 7);

    EXPECT_EQ(sink.call_count, 1);
    EXPECT_EQ(sink.last_tag, "loss");
    EXPECT_EQ(sink.last_step, 7);
    EXPECT_NEAR(sink.last_value, static_cast<double>(loss), 1e-6);
}

// Single-example overfitting: repeated train_step on one fixed image must drive the loss
// toward zero and eventually predict its target class -- proves the full chain (forward,
// cross-entropy loss, backward through Conv2D/ReLU/Flatten/Linear, Adam updates on both
// parameterized layers) is wired correctly, the same acceptance-criterion shape as
// XorTrainingExampleTest.TrainingConvergesOnXOR.
TEST_F(MnistConvNetTest, TrainStepOverfitsASingleExample) {
    MnistConvNet net(&backend);
    AdamOptimizer optimizer(0.01f, &backend);
    NoOpMetricsSink sink;
    Tensor image = SyntheticImage(&backend, 1);
    constexpr int64_t target_class = 6;

    float initial_loss = net.train_step(image, target_class, optimizer, sink, 0);

    float final_loss = initial_loss;
    for (int step = 1; step < 200; ++step) {
        final_loss = net.train_step(image, target_class, optimizer, sink, step);
    }

    EXPECT_LT(final_loss, initial_loss * 0.05f) << "loss did not substantially decrease";
    EXPECT_EQ(net.predict(image), target_class);
}

}  // namespace
}  // namespace pulsatrix
