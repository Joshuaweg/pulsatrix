#include <gtest/gtest.h>

#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/metrics_sink.hpp"
#include "exai/xor_training_example.hpp"

namespace exai {
namespace {

class XorTrainingExampleTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(XorTrainingExampleTest, ForwardProducesOneDimensionalOutput) {
    XorNetwork net(&backend);
    Tensor input(Shape({2}), &backend, {0.0f, 1.0f});
    Tensor output = net.forward(input);
    EXPECT_EQ(output.shape(), Shape({1}));
}

TEST_F(XorTrainingExampleTest, TrainStepChangesWeights) {
    XorNetwork net(&backend);
    AdamOptimizer optimizer(0.1f, &backend);
    NoOpMetricsSink sink;

    Tensor w1_before(net.linear1_weight());  // copy, before training

    Tensor input(Shape({2}), &backend, {0.0f, 1.0f});
    Tensor target(Shape({1}), &backend, {1.0f});
    (void)net.train_step(input, target, optimizer, sink, 0);

    // At least one weight must have changed -- proves the full chain (forward, loss,
    // backward through both Linear layers and ReLU, optimizer step) is wired correctly.
    bool any_changed = false;
    for (int64_t i = 0; i < w1_before.numel(); ++i) {
        if (w1_before.data()[i] != net.linear1_weight().data()[i]) {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST_F(XorTrainingExampleTest, TrainStepLogsLossThroughMetricsSink) {
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

    XorNetwork net(&backend);
    AdamOptimizer optimizer(0.1f, &backend);
    CapturingSink sink;

    Tensor input(Shape({2}), &backend, {1.0f, 1.0f});
    Tensor target(Shape({1}), &backend, {0.0f});
    float loss = net.train_step(input, target, optimizer, sink, 7);

    EXPECT_EQ(sink.call_count, 1);
    EXPECT_EQ(sink.last_tag, "loss");
    EXPECT_EQ(sink.last_step, 7);
    EXPECT_NEAR(sink.last_value, static_cast<double>(loss), 1e-6);
}

}  // namespace
}  // namespace exai
