// Trains the XOR network from an installed pulsatrix and checks the loss falls. Exits
// non-zero on failure, so the install test fails with it.
#include <cstdio>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/xor_training_example.hpp"
#ifdef CONSUMER_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    XorNetwork net(&backend);
    AdamOptimizer optimizer(0.01f, &backend);
    NoOpMetricsSink sink;

    std::vector<std::pair<Tensor, Tensor>> dataset;
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
    dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));

    auto mean_loss = [&]() {
        float total = 0.0f;
        for (auto& [input, target] : dataset) {
            float diff = net.forward(input).data()[0] - target.data()[0];
            total += diff * diff;
        }
        return total / static_cast<float>(dataset.size());
    };

    const float before = mean_loss();
    int step = 0;
    for (int epoch = 0; epoch < 200; ++epoch) {
        for (auto& [input, target] : dataset) {
            (void)net.train_step(input, target, optimizer, sink, step++);
        }
    }
    const float after = mean_loss();

    std::printf("pulsatrix %s: XOR loss %.4f -> %.4f\n", EXPECTED_VERSION, before, after);
    if (!(after < before)) {
        std::fprintf(stderr, "loss did not fall\n");
        return 1;
    }
    return 0;
}
