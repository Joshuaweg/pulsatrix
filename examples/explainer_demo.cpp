/** @file explainer_demo.cpp
 *  @brief Standalone demo tying together everything Phase 2 has shipped so far:
 *         Module -> ComputationGraph/Autograd wiring (Mission 0), ExplainerContext
 *         (Mission 1), and the Saliency/IntegratedGradients explainers (Mission 2) --
 *         run against a small Linear->ReLU->Linear network on XOR-shaped inputs.
 *  @note Not a test -- the GoogleTest suites (module_graph_wiring_test.cpp,
 *        explainer_context_test.cpp, saliency_test.cpp, integrated_gradients_test.cpp)
 *        remain the real acceptance criteria. This exists so a human can watch real
 *        attribution scores come out, not just read test assertions.
 */
#include <cstdio>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/saliency.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    // Same small MLP shape as xor_demo's XorNetwork -- Linear(2,4) -> ReLU -> Linear(4,1)
    // -- built directly here (not via XorNetwork) since ExplainerContext needs the
    // modules exposed as a plain sequence, and XorNetwork encapsulates its layers
    // privately.
    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});

    ExplainerContext ctx({&linear1, &relu, &linear2});

    // --- Graph structure (Mission 0/1) ---
    Tensor probe_input(Shape({1, 2}), &backend, {0.0f, 0.0f});
    (void)ctx.forward_pass(probe_input);
    std::printf("ExAI demo -- Linear(2,4)->ReLU->Linear(4,1)\n\n");
    std::printf("ComputationGraph: %zu nodes (1 input + 3 module outputs)\n", ctx.graph().node_count());
    std::printf("  Linear nodes:     %zu\n", ctx.graph().nodes_by_op_type(OpType::Linear).size());
    std::printf("  Activation nodes: %zu\n", ctx.graph().nodes_by_op_type(OpType::Activation).size());
    std::printf("\n");

    // --- Attribution scores (Mission 2) over every XOR-shaped input ---
    std::vector<std::pair<float, float>> inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    Saliency saliency;
    IntegratedGradients ig;
    Tensor baseline(Shape({1, 2}), &backend);  // zero-initialized -- the "no signal" baseline

    for (auto [a, b] : inputs) {
        Tensor input(Shape({1, 2}), &backend, {a, b});
        Tensor output = ctx.forward_pass(input);
        std::printf("input (%.0f, %.0f) -> output %.4f\n", a, b, output.data()[0]);

        Attribution sal = saliency.explain(ctx, input, /*target_index=*/0, &backend);
        std::printf("  saliency:             d(out)/d(in) = [%.4f, %.4f]\n", sal.values.data()[0],
                    sal.values.data()[1]);

        Tensor output_baseline = ctx.forward_pass(baseline);
        float f_baseline = output_baseline.data()[0];

        Attribution ig_attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/200, &backend);
        float sum_ig = ig_attr.values.data()[0] + ig_attr.values.data()[1];
        std::printf("  integrated gradients: IG = [%.4f, %.4f], sum=%.4f (F(x)-F(baseline)=%.4f)\n",
                    ig_attr.values.data()[0], ig_attr.values.data()[1], sum_ig, output.data()[0] - f_baseline);
        std::printf("\n");
    }

    return 0;
}
