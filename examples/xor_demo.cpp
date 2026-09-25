/** @file xor_demo.cpp
 *  @brief Standalone demo: trains XorNetwork on XOR and prints loss/predictions to stdout.
 *  @note Not a test -- tests/xor_training_example_test.cpp is the actual TDD acceptance
 *        criterion (mission_training_loop.md). This exists purely so a human can watch the
 *        same training loop converge, run manually via the `xor_demo` target.
 */
#include <cstdio>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/xor_training_example.hpp"

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
            Tensor pred = net.forward(input);
            float diff = pred.data()[0] - target.data()[0];
            total += diff * diff;
        }
        return total / static_cast<float>(dataset.size());
    };

    std::printf("XOR training demo -- Linear(2,4)->ReLU->Linear(4,1), Adam(lr=0.01)\n");
    std::printf("epoch %5d | mean loss %.6f\n", 0, mean_loss());

    constexpr int kEpochs = 1000;
    int step = 0;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        for (auto& [input, target] : dataset) {
            (void)net.train_step(input, target, optimizer, sink, step++);
        }
        if (epoch % 100 == 0) {
            std::printf("epoch %5d | mean loss %.6f\n", epoch, mean_loss());
        }
    }

    std::printf("\nFinal predictions:\n");
    for (auto& [input, target] : dataset) {
        Tensor pred = net.forward(input);
        float rounded = (pred.data()[0] > 0.5f) ? 1.0f : 0.0f;
        std::printf("  (%.0f, %.0f) -> raw %.4f, rounded %.0f (target %.0f)%s\n", input.data()[0], input.data()[1],
                    pred.data()[0], rounded, target.data()[0], rounded == target.data()[0] ? "" : "  <-- WRONG");
    }

    return 0;
}
