/** @file saliency_and_integrated_gradients.cpp
 *  @brief Recipe: Saliency and Integrated Gradients on the same small Linear->ReLU->Linear
 *         network, for one input. Paired with
 *         docs/recipes/interpretability/saliency_and_integrated_gradients.md.
 *  @note Trimmed version of examples/explainer_demo.cpp (a single input instead of all four
 *        XOR-shaped inputs, plus the ComputationGraph node-count structure it also prints).
 */
#include <cstdio>

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

    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});

    ExplainerContext ctx({&linear1, &relu, &linear2});

    Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});
    Tensor baseline(Shape({1, 2}), &backend);  // zero-initialized -- the "no signal" baseline

    Tensor output = ctx.forward_pass(input);
    std::printf("Saliency and Integrated Gradients recipe -- Linear(2,4)->ReLU->Linear(4,1)\n\n");
    std::printf("input (%.0f, %.0f) -> output %.4f\n\n", input.data()[0], input.data()[1], output.data()[0]);

    Saliency saliency;
    Attribution sal = saliency.explain(ctx, input, /*target_index=*/0, &backend);
    std::printf("saliency:             d(out)/d(in) = [%.4f, %.4f]\n", sal.values.data()[0], sal.values.data()[1]);

    Tensor output_baseline = ctx.forward_pass(baseline);
    float f_baseline = output_baseline.data()[0];

    IntegratedGradients ig;
    Attribution ig_attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/200, &backend);
    float sum_ig = ig_attr.values.data()[0] + ig_attr.values.data()[1];
    std::printf("integrated gradients: IG = [%.4f, %.4f], sum=%.4f (F(x)-F(baseline)=%.4f)\n",
                ig_attr.values.data()[0], ig_attr.values.data()[1], sum_ig, output.data()[0] - f_baseline);

    std::printf(
        "\nSaliency is the raw gradient at x -- cheap, but blind to any region where the\n"
        "network's gradient has saturated. Integrated Gradients instead averages the gradient\n"
        "along the straight-line path from baseline to x, which is what gives it the\n"
        "completeness guarantee (sum(IG) == F(x) - F(baseline), checked above).\n");

    return 0;
}
