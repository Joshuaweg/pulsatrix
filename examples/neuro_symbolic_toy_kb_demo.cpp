/** @file neuro_symbolic_toy_kb_demo.cpp
 *  @brief Standalone demo: trains ToyKnowledgeBase's rule `A(x) -> not(B(x))` end-to-end via
 *         ordinary gradient descent and prints satisfaction/predicate values to stdout --
 *         Phase 1 Mission 2 of campaign_exai_dl_library_neuro_symbolic.
 *  @note Not a test -- tests/neuro_symbolic_toy_kb_test.cpp is the actual TDD acceptance
 *        criterion (mission_2_toy_kb_training_demo.md). This exists purely so a human can
 *        watch the same training loop converge, run manually via the
 *        `neuro_symbolic_toy_kb_demo` target -- same convention as examples/xor_demo.cpp.
 */
#include <cstdio>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_toy_kb.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    ToyKnowledgeBase kb(&backend, 2.0f);
    SGDOptimizer optimizer(0.1f);

    std::vector<float> x_values{-1.5f, -0.5f, 0.5f, 1.5f};
    Tensor x(Shape({static_cast<int64_t>(x_values.size()), 1}), &backend, x_values);

    std::printf(
        "Neuro-symbolic toy KB demo -- rule A(x) -> not(B(x)), De Morgan-expanded to\n"
        "not(A(x)) or not(B(x)) via NegationModule/DisjunctionModule (Product), SGD(lr=0.1)\n\n");

    auto print_state = [&](int epoch, float loss) {
        std::printf("epoch %5d | loss (1-satisfaction) %.6f | satisfaction %.6f\n", epoch, loss, 1.0f - loss);
    };

    float loss = kb.forward(x);
    print_state(0, loss);

    constexpr int kEpochs = 500;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        loss = kb.train_step(x, optimizer);
        if (epoch % 50 == 0) {
            print_state(epoch, loss);
        }
    }

    std::printf("\nFinal per-grounding truth degrees:\n");
    (void)kb.forward(x);
    const Tensor& a = kb.last_a();
    const Tensor& b = kb.last_b();
    const Tensor& rule = kb.last_rule();
    for (size_t i = 0; i < x_values.size(); ++i) {
        std::printf("  x=%.2f -> A=%.4f, B=%.4f, rule(not A or not B)=%.4f\n", x_values[i], a.data()[i], b.data()[i],
                    rule.data()[static_cast<int64_t>(i)]);
    }

    return 0;
}
